from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from benchmark import recall_points
from config import load_config
from memory_store import CompactMemoryManager, UserProfileStore, extract_profile_updates, validate_fact_write

ROOT = Path(__file__).resolve().parent.parent


def make_config(tmp_path: Path, threshold: int = 120, keep: int = 2):
    """Isolated config: state in tmp_path, small compact threshold so compaction happens fast."""

    return replace(
        load_config(ROOT),
        state_dir=tmp_path / "state",
        compact_threshold_tokens=threshold,
        compact_keep_messages=keep,
    )


def load_stress() -> dict:
    return json.loads((ROOT / "data" / "advanced_long_context.json").read_text(encoding="utf-8"))[0]


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """User.md read/write/edit, plus what is allowed to be written into it."""

    store = UserProfileStore(tmp_path / "profiles")
    assert store.file_size("dungct") == 0
    assert "# User Profile" in store.read_text("dungct")  # default profile before any write
    assert (tmp_path / "profiles") in store.path_for("../../etc/passwd").parents  # user id is sanitized

    path = store.write_text("dungct", "# User Profile\n\n## Facts\n- name: DũngCT\n- location: Đà Nẵng\n")
    assert path.name == "User.md" and path.exists()
    assert store.facts("dungct") == {"name": "DũngCT", "location": "Đà Nẵng"}

    assert store.edit_text("dungct", "location: Đà Nẵng", "location: Huế") is True
    assert store.edit_text("dungct", "không tồn tại", "x") is False
    assert store.facts("dungct")["location"] == "Huế"
    assert store.file_size("dungct") == path.stat().st_size > 0

    # Conflict handling: a correction overwrites the old fact instead of keeping both.
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    agent.reply("u1", "t1", "Mình ở Đà Nẵng và đang làm backend engineer cho startup AI.")
    agent.reply("u1", "t1", "Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer.")
    agent.reply("u1", "t1", "À, mình đính chính: giờ mình đang ở Huế chứ không còn ở Đà Nẵng mỗi ngày nữa.")
    profile = agent.profile_store.read_text("u1")
    assert "MLOps engineer" in profile and "backend engineer" not in profile
    assert "Huế" in profile and "Đà Nẵng" not in profile

    # Noise, questions and hypotheticals are not stored.
    assert extract_profile_updates("Hà Nội chỉ là nơi mình vừa bay ra họp hai ngày với đối tác.") == {}
    assert extract_profile_updates("Có lúc mình đùa rằng hay là chuyển sang product manager cho đỡ mệt.") == {}
    assert extract_profile_updates("Bạn thử nhớ lại xem đồ uống yêu thích của mình là gì.") == {}
    assert extract_profile_updates("Nếu sau này mình nói ở Đà Nẵng thì đừng tin nhé.") == {}

    # LLM tool writes must be grounded in what the user said (values Gemini wrote in a live run).
    said = load_stress()["turns"][:10]
    assert validate_fact_write("interests", "Python, Energy policy, so sánh CAPEX mở rộng", said)[0] == "Python"
    assert validate_fact_write("location", "Hà Nội", said)[0] is None
    assert validate_fact_write("profession", "product manager", said)[0] is None
    assert validate_fact_write("location", "Đà Nẵng", said)[0] == "Đà Nẵng"
    assert validate_fact_write("response_style", "ngắn gọn, 3 bullet", said)[0] == "ngắn gọn, 3 bullet"
    pet_turns = ["Mình nuôi một bé corgi tên Bơ."]
    assert validate_fact_write("pet", "corgi tên Bơ (thường dùng làm ví dụ test case)", pet_turns)[0] is None
    assert validate_fact_write("pet", "corgi tên Bơ", pet_turns)[0] == "corgi tên Bơ"
    assert validate_fact_write("salary", "1000", pet_turns)[0] is None


def test_compact_trigger(tmp_path: Path) -> None:
    manager = CompactMemoryManager(threshold_tokens=100, keep_messages=2)
    for i in range(6):
        manager.append("t", "user", f"Tin số {i}: " + "nội dung khá dài về benchmark memory. " * 5)
    state = manager.context("t")
    assert manager.compaction_count("t") >= 1
    assert len(state["messages"]) <= 3
    assert "Tin số 0" in state["summary"]  # old content survives as summary, not verbatim

    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    stress = load_stress()
    for turn in stress["turns"]:
        agent.reply(stress["user_id"], "stress", turn)
    assert agent.compaction_count("stress") >= 2


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path, threshold=800, keep=4)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    turns = [
        "Chào bạn, mình tên là DũngCT.",
        "Mình ở Đà Nẵng và đang làm backend engineer cho startup AI.",
        "Đồ uống yêu thích là cà phê sữa đá.",
    ]
    for agent in (baseline, advanced):
        for turn in turns:
            agent.reply("dungct", "session-1", turn)

    question = "Mình tên gì và đồ uống yêu thích là gì?"
    expected = ["DũngCT", "cà phê sữa đá"]
    # Same thread: both agents can answer.
    assert recall_points(baseline.reply("dungct", "session-1", question)["response"], expected) == 1.0
    # New thread: only the advanced agent remembers.
    assert recall_points(baseline.reply("dungct", "session-2", question)["response"], expected) == 0.0
    assert recall_points(advanced.reply("dungct", "session-2", question)["response"], expected) == 1.0

    # A fresh agent process reading the same User.md still remembers (persistence on disk).
    reloaded = AdvancedAgent(config, force_offline=True)
    assert recall_points(reloaded.reply("dungct", "session-3", question)["response"], expected) == 1.0


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path, threshold=800, keep=4)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    stress = load_stress()

    for turn in stress["turns"]:
        baseline.reply(stress["user_id"], "long", turn)
        advanced.reply(stress["user_id"], "long", turn)

    assert advanced.compaction_count("long") > 0
    assert advanced.prompt_token_usage("long") < 0.7 * baseline.prompt_token_usage("long")
    # The last turn's context stays bounded instead of growing with the thread.
    last_baseline = baseline.reply(stress["user_id"], "long", "Tóm tắt về mình?")["prompt_tokens"]
    last_advanced = advanced.reply(stress["user_id"], "long", "Tóm tắt về mình?")["prompt_tokens"]
    assert last_advanced < last_baseline / 2

    # Counter-case: on a short thread nothing is compacted and User.md is pure overhead.
    for turn in ["Mình tên là DũngCT.", "Mình thích Python.", "Hôm nay trời đẹp."]:
        baseline.reply("u", "short", turn)
        advanced.reply("u", "short", turn)
    assert advanced.compaction_count("short") == 0
    assert advanced.prompt_token_usage("short") > baseline.prompt_token_usage("short")
