"""Deterministic reply generation shared by both agents in offline mode.

Both agents use the same composer; they differ only in which facts they can see
(baseline: facts said in the current thread, advanced: User.md).
"""

from __future__ import annotations

import re

from memory_store import FACT_LABELS

INTENT_KEYWORDS = {
    "name": ("tên", "là ai", "tóm tắt"),
    "location": ("ở đâu", "nơi ở", "còn ở"),
    "profession": ("nghề", "làm gì", "tóm tắt"),
    "response_style": ("style", "kiểu trả lời", "phong cách"),
    "favorite_drink": ("đồ uống", "uống gì"),
    "favorite_food": ("món ăn", "ăn gì"),
    "pet": ("nuôi", "thú cưng"),
    "interests": ("quan tâm", "là ai", "tóm tắt"),
}
RECALL_RE = re.compile(r"\?|nhắc lại|nhớ lại|tóm tắt", re.IGNORECASE)


def detect_intents(message: str) -> list[str]:
    low = message.lower()
    return [key for key, words in INTENT_KEYWORDS.items() if any(w in low for w in words)]


def is_recall_request(message: str) -> bool:
    return bool(RECALL_RE.search(message)) and bool(detect_intents(message))


def compose_reply(message: str, facts: dict[str, str], saved: dict[str, str] | None = None) -> str:
    if is_recall_request(message):
        lines = []
        for key in detect_intents(message):
            value = facts.get(key)
            lines.append(f"- {FACT_LABELS[key]}: {value if value else 'mình chưa có thông tin này'}")
        return "\n".join(lines)
    if saved:
        noted = "; ".join(f"{FACT_LABELS.get(k, k)} = {v}" for k, v in saved.items())
        return f"Đã ghi nhận. Cập nhật hồ sơ: {noted}."
    return "Đã ghi nhận."
