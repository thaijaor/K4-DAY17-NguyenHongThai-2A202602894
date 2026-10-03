from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens, extract_profile_updates
from model_provider import build_chat_model
from offline_responder import compose_reply

BASELINE_SYSTEM_PROMPT = (
    "Bạn là trợ lý tiếng Việt. Chỉ dùng thông tin trong cuộc trò chuyện hiện tại. "
    "Nếu người dùng hỏi điều chưa được nói trong cuộc trò chuyện này, hãy nói là bạn chưa có thông tin."
)


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A: within-thread memory only. No User.md, so a new thread starts from zero."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None
        if not force_offline and self.config.model.is_configured:
            try:
                self.langchain_agent = self._maybe_build_langchain_agent()
            except Exception as exc:  # missing SDK / bad key -> stay offline
                print(f"[baseline] live agent unavailable, using offline mode: {exc}")

    @property
    def mode(self) -> str:
        return "live" if self.langchain_agent is not None else "offline"

    def _session(self, thread_id: str) -> SessionState:
        return self.sessions.setdefault(thread_id, SessionState())

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is not None:
            return self._reply_live(thread_id, message)
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self._session(thread_id).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self._session(thread_id).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def memory_file_size(self, user_id: str) -> int:
        # Baseline has no persistent memory file.
        return 0

    def _prompt_tokens(self, session: SessionState) -> int:
        # The whole thread is re-sent on every turn.
        return estimate_tokens(BASELINE_SYSTEM_PROMPT) + sum(estimate_tokens(m["content"]) for m in session.messages)

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self._session(thread_id)
        session.messages.append({"role": "user", "content": message})
        prompt_tokens = self._prompt_tokens(session)

        # Facts are re-read from this thread's messages only, never from another thread.
        thread_facts: dict[str, str] = {}
        for item in session.messages:
            if item["role"] == "user":
                thread_facts.update(extract_profile_updates(item["content"], self.config.profile_confidence_threshold))
        response = compose_reply(message, thread_facts)

        session.messages.append({"role": "assistant", "content": response})
        session.prompt_tokens_processed += prompt_tokens
        session.token_usage += estimate_tokens(message) + estimate_tokens(response)
        return {"response": response, "prompt_tokens": prompt_tokens, "mode": "offline"}

    def _reply_live(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self._session(thread_id)
        session.messages.append({"role": "user", "content": message})
        result = self.langchain_agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": thread_id}},
        )
        response = _message_text(result["messages"][-1])
        usage = getattr(result["messages"][-1], "usage_metadata", None) or {}
        prompt_tokens = usage.get("input_tokens") or self._prompt_tokens(session)

        session.messages.append({"role": "assistant", "content": response})
        session.prompt_tokens_processed += prompt_tokens
        session.token_usage += estimate_tokens(message) + estimate_tokens(response)
        return {"response": response, "prompt_tokens": prompt_tokens, "mode": "live"}

    def _maybe_build_langchain_agent(self):
        from langchain.agents import create_agent
        from langgraph.checkpoint.memory import InMemorySaver

        return create_agent(
            build_chat_model(self.config.model),
            tools=[],
            system_prompt=BASELINE_SYSTEM_PROMPT,
            middleware=[live_retry_middleware()],
            checkpointer=InMemorySaver(),
        )


def live_retry_middleware():
    """Retry transient model errors (dropped connection, 429) instead of aborting a long benchmark."""

    from langchain.agents.middleware import ModelRetryMiddleware

    return ModelRetryMiddleware(max_retries=4, initial_delay=10.0, max_delay=60.0, on_failure="error")


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, list):
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return str(content)
