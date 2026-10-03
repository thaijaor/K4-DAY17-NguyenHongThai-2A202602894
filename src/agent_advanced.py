from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_baseline import _message_text
from config import LabConfig, load_config
from memory_store import FACT_LABELS, CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates
from model_provider import build_chat_model
from offline_responder import compose_reply

try:  # needed at module level: @tool resolves the ToolRuntime annotation from globals
    from langchain.tools import ToolRuntime
except ImportError:  # offline-only install
    ToolRuntime = Any

ADVANCED_SYSTEM_PROMPT = (
    "Bạn là trợ lý tiếng Việt có bộ nhớ dài hạn. Hồ sơ người dùng (User.md) và tóm tắt hội thoại cũ được đưa vào "
    "ngữ cảnh. Luôn ưu tiên fact mới nhất trong User.md khi có đính chính, bỏ qua thông tin chỉ là câu đùa hoặc "
    "nơi đi công tác. Trả lời theo style trong hồ sơ."
)


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B: short-term thread memory + persistent User.md + compact memory."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.langchain_agent = None
        if not force_offline and self.config.model.is_configured:
            try:
                self.langchain_agent = self._maybe_build_langchain_agent()
            except Exception as exc:
                print(f"[advanced] live agent unavailable, using offline mode: {exc}")

    @property
    def mode(self) -> str:
        return "live" if self.langchain_agent is not None else "offline"

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is not None:
            return self._reply_live(user_id, thread_id, message)
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _remember(self, user_id: str, thread_id: str, message: str) -> tuple[dict[str, str], int]:
        """Steps shared by offline and live: persist facts, append to short-term memory."""

        saved: dict[str, str] = {}
        for key, value in extract_profile_updates(message, self.config.profile_confidence_threshold).items():
            if self.profile_store.upsert_fact(user_id, key, value):
                saved[key] = self.profile_store.facts(user_id)[key]
        summary_before = self.compact_memory.context(thread_id)["summary_tokens_generated"]
        self.compact_memory.append(thread_id, "user", message)
        # Memory writes and summaries are model output in a live system, so they count as agent tokens.
        overhead = sum(estimate_tokens(f"- {k}: {v}") for k, v in saved.items())
        overhead += self.compact_memory.context(thread_id)["summary_tokens_generated"] - summary_before
        return saved, overhead

    def _record(self, thread_id: str, message: str, response: str, prompt_tokens: int, overhead: int) -> None:
        summary_before = self.compact_memory.context(thread_id)["summary_tokens_generated"]
        self.compact_memory.append(thread_id, "assistant", response)
        overhead += self.compact_memory.context(thread_id)["summary_tokens_generated"] - summary_before
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        self.thread_tokens[thread_id] = (
            self.thread_tokens.get(thread_id, 0) + estimate_tokens(message) + estimate_tokens(response) + overhead
        )

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        saved, overhead = self._remember(user_id, thread_id, message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        response = self._offline_response(user_id, thread_id, message, saved)
        self._record(thread_id, message, response, prompt_tokens, overhead)
        return {"response": response, "prompt_tokens": prompt_tokens, "profile_updates": saved, "mode": "offline"}

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        saved, overhead = self._remember(user_id, thread_id, message)
        context = AgentContext(user_id=user_id, memory_path=str(self.profile_store.path_for(user_id)))
        result = self.langchain_agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": thread_id}},
            context=context,
        )
        response = _message_text(result["messages"][-1])
        usage = getattr(result["messages"][-1], "usage_metadata", None) or {}
        prompt_tokens = usage.get("input_tokens") or self._estimate_prompt_context_tokens(user_id, thread_id)
        self._record(thread_id, message, response, prompt_tokens, overhead)
        return {"response": response, "prompt_tokens": prompt_tokens, "profile_updates": saved, "mode": "live"}

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """System prompt + User.md + compact summary + recent kept messages."""

        state = self.compact_memory.context(thread_id)
        return (
            estimate_tokens(ADVANCED_SYSTEM_PROMPT)
            + estimate_tokens(self.profile_store.read_text(user_id))
            + estimate_tokens(state["summary"])
            + sum(estimate_tokens(m["content"]) for m in state["messages"])
        )

    def _offline_response(self, user_id: str, thread_id: str, message: str, saved: dict[str, str] | None = None) -> str:
        return compose_reply(message, self.profile_store.facts(user_id), saved)

    def _maybe_build_langchain_agent(self):
        from langchain.agents import create_agent
        from langchain.agents.middleware import ModelRequest, SummarizationMiddleware, dynamic_prompt
        from langchain.tools import tool
        from langgraph.checkpoint.memory import InMemorySaver

        store = self.profile_store
        model = build_chat_model(self.config.model)

        @tool
        def read_user_memory(runtime: ToolRuntime[AgentContext]) -> str:
            """Đọc toàn bộ User.md của người dùng hiện tại."""
            return store.read_text(runtime.context.user_id)

        @tool
        def save_user_fact(key: str, value: str, runtime: ToolRuntime[AgentContext]) -> str:
            """Lưu hoặc ghi đè một fact ổn định vào User.md. key thuộc: name, location, profession,
            response_style, favorite_drink, favorite_food, pet, interests."""
            if key not in FACT_LABELS:
                return f"Từ chối: key '{key}' không nằm trong schema."
            changed = store.upsert_fact(runtime.context.user_id, key, value.strip())
            return "Đã cập nhật." if changed else "Không đổi."

        @dynamic_prompt
        def inject_profile(request: ModelRequest) -> str:
            profile = store.read_text(request.runtime.context.user_id)
            return f"{ADVANCED_SYSTEM_PROMPT}\n\n<user_memory>\n{profile}\n</user_memory>"

        return create_agent(
            model,
            tools=[read_user_memory, save_user_fact],
            middleware=[
                inject_profile,
                SummarizationMiddleware(
                    model,
                    trigger=("tokens", self.config.compact_threshold_tokens),
                    keep=("messages", self.config.compact_keep_messages),
                ),
            ],
            context_schema=AgentContext,
            checkpointer=InMemorySaver(),
        )
