from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

PROFILE_HEADER = "# User Profile"
FACTS_HEADER = "## Facts"
FACT_LINE_RE = re.compile(r"^- (?P<key>[a-z_]+): (?P<value>.+)$")

# Field order + Vietnamese labels, shared by User.md rendering and offline replies.
FACT_LABELS = {
    "name": "Tên",
    "location": "Nơi ở hiện tại",
    "profession": "Nghề nghiệp hiện tại",
    "response_style": "Style trả lời",
    "favorite_drink": "Đồ uống yêu thích",
    "favorite_food": "Món ăn yêu thích",
    "pet": "Thú cưng",
    "interests": "Mối quan tâm kỹ thuật",
}

MAX_INTERESTS = 6


def estimate_tokens(text: str) -> int:
    """Heuristic token count: ~4 characters per token, 0 for empty text."""

    text = (text or "").strip()
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


# ---------------------------------------------------------------------------
# Persistent memory: User.md
# ---------------------------------------------------------------------------


@dataclass
class UserProfileStore:
    """One `User.md` per user under `root_dir/<user_id>/User.md`."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", user_id.strip()).strip("._") or "anonymous"
        return Path(self.root_dir) / slug / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return f"{PROFILE_HEADER}\n\n{FACTS_HEADER}\n"

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        content = self.read_text(user_id)
        if not search_text or search_text not in content:
            return False
        self.write_text(user_id, content.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        facts: dict[str, str] = {}
        for line in self.read_text(user_id).splitlines():
            match = FACT_LINE_RE.match(line.strip())
            if match:
                facts[match["key"]] = match["value"].strip()
        return facts

    def upsert_fact(self, user_id: str, key: str, value: str) -> bool:
        """Insert or overwrite one fact. Returns True when User.md changed.

        Overwriting (instead of appending) is the conflict-handling policy: a correction
        replaces the old value, so the profile never holds two locations at once.
        """

        facts = self.facts(user_id)
        if key == "interests":
            value = merge_interests(facts.get("interests", ""), value)
        elif key == "response_style":
            value = merge_style(facts.get("response_style", ""), value)
        if facts.get(key) == value:
            return False
        facts[key] = value
        self.write_text(user_id, render_profile(facts))
        return True


def render_profile(facts: dict[str, str]) -> str:
    ordered = [k for k in FACT_LABELS if k in facts] + sorted(k for k in facts if k not in FACT_LABELS)
    lines = [PROFILE_HEADER, "", FACTS_HEADER] + [f"- {key}: {facts[key]}" for key in ordered]
    return "\n".join(lines) + "\n"


def merge_interests(old: str, new: str) -> str:
    """Most recently mentioned first, capped at MAX_INTERESTS (recency-based decay)."""

    items = [x.strip() for x in new.split(",") if x.strip()]
    items += [x.strip() for x in old.split(",") if x.strip() and x.strip() not in items]
    return ", ".join(items[:MAX_INTERESTS])


# ---------------------------------------------------------------------------
# Response style as structured slots
# ---------------------------------------------------------------------------

STYLE_SLOTS = ("length", "format", "examples", "emphasis")


def _style_slot(part: str) -> str | None:
    if part == "ngắn gọn":
        return "length"
    if "bullet" in part or part == "có cấu trúc":
        return "format"
    if part.startswith("có ví dụ"):
        return "examples"
    if "trade-off" in part:
        return "emphasis"
    return None


def parse_style(sentence: str) -> str:
    """Turn a style instruction into normalized parts, e.g. 'ngắn gọn, 3 bullet'."""

    low = sentence.lower()
    slots: dict[str, str] = {}
    if "ngắn" in low or "gọn" in low:
        slots["length"] = "ngắn gọn"
    bullet = re.search(r"(\d+)\s*bullet", low)
    if bullet:
        slots["format"] = f"{bullet.group(1)} bullet"
    elif "bullet" in low:
        slots["format"] = "bullet"
    elif "cấu trúc" in low:
        slots["format"] = "có cấu trúc"
    if "ví dụ thực chiến" in low:
        slots["examples"] = "có ví dụ thực chiến"
    elif "ví dụ thực tế" in low:
        slots["examples"] = "có ví dụ thực tế"
    if "trade-off" in low:
        slots["emphasis"] = "nhấn trade-off"
    return ", ".join(slots[s] for s in STYLE_SLOTS if s in slots)


def merge_style(old: str, new: str) -> str:
    """Slot-wise overwrite: a new bullet count replaces the old one, other slots are kept."""

    slots: dict[str, str] = {}
    for text in (old, new):
        for part in (p.strip() for p in text.split(",")):
            slot = _style_slot(part)
            # A vague format ("bullet", "có cấu trúc") never overrides a specific "3 bullet".
            if slot == "format" and re.match(r"\d+ bullet", slots.get("format", "")) and not re.match(r"\d+ bullet", part):
                continue
            if slot:
                slots[slot] = part
    return ", ".join(slots[s] for s in STYLE_SLOTS if s in slots)


# ---------------------------------------------------------------------------
# Fact extraction with confidence threshold
# ---------------------------------------------------------------------------

CITIES = ["Hồ Chí Minh", "Sài Gòn", "Hà Nội", "Đà Nẵng", "Huế", "Hải Phòng", "Cần Thơ", "Nha Trang", "Đà Lạt", "Quy Nhơn"]
_CITY_ALT = "|".join(re.escape(c) for c in CITIES)
LOCATION_RE = re.compile(rf"(?:\bở|\bsang|\bvề|nơi ở(?: hiện tại)? là)\s+(?P<v>{_CITY_ALT})", re.IGNORECASE)
PROFESSION_RE = re.compile(
    r"(?:\blàm|\blà|\bsang|\bnghề)\s+(?P<v>(?:[A-Za-z][A-Za-z-]*\s+){0,2}"
    r"(?:engineer|developer|manager|scientist|designer|analyst|researcher))\b",
    re.IGNORECASE,
)
NAME_RES = [
    re.compile(r"(?:[Mm]ình|[Tt]ôi|[Ee]m)\s+tên\s+(?:là\s+)?(?P<v>\S+(?:\s+\S+){0,3})"),
    re.compile(r"[Tt]ên\s+(?:của\s+)?(?:mình|tôi|em)\s+là\s+(?P<v>\S+(?:\s+\S+){0,3})"),
    re.compile(r"(?:^|[:,]\s*)[Tt]ên\s+(?P<v>\S+(?:\s+\S+){0,3})"),
]
DRINK_RE = re.compile(r"đồ uống yêu thích(?: của mình)? là\s+(?P<v>[^.,!?;]+)", re.IGNORECASE)
FOOD_RE = re.compile(r"món(?: ăn)? yêu thích(?: của mình)? là\s+(?P<v>[^.,!?;]+)", re.IGNORECASE)
PET_RE = re.compile(r"\bnuôi\s+(?:một\s+)?(?:bé\s+|con\s+)?(?P<kind>[^\s,.!?]+)(?:\s+tên\s+(?P<name>[^\s,.!?]+))?", re.IGNORECASE)
TECH_INTERESTS = ["Python", "AI ứng dụng", "AI agent", "MLOps", "RAG", "LLM", "LangChain", "LangGraph"]

QUESTION_RE = re.compile(r"\?\s*$|\b(gì|ở đâu|thế nào|là ai)\b|^\s*(nhắc lại|tóm tắt)(?!\s+lần cuối)|nhớ lại xem", re.IGNORECASE)
# Clause-level cues meaning "this mention is NOT the current fact".
NEGATION_CUES = ("không còn", "không phải", "chỉ là", "đùa", "trước đó", "lúc đầu", "đừng", "bay ra", "họp", "ví dụ cũ")
CORRECTION_CUES = ("đính chính", "giờ", "hiện tại", "cập nhật", "thực ra", "từ tuần này", "vẫn")
STYLE_TRIGGERS = ("trả lời", "style", "giải thích", "trình bày")
INTEREST_TRIGGERS = ("thích", "quan tâm", "học thêm")
NAME_STOPWORDS = {"Mình", "Tôi", "Bạn"}


@dataclass
class ExtractedFact:
    key: str
    value: str
    confidence: float
    evidence: str


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _clauses(sentence: str) -> list[str]:
    return [c.strip() for c in re.split(r"[,;:]|\s(?:nhưng|chứ|dù)\s", sentence) if c.strip()]


def is_question(sentence: str) -> bool:
    return bool(QUESTION_RE.search(sentence.strip()))


def _leading_capitalized(chunk: str) -> str:
    words: list[str] = []
    for token in chunk.split():
        word = token.strip(",.!?;:\"'()")
        if not word or not word[0].isupper() or word in NAME_STOPWORDS:
            break
        words.append(word)
        if token != word:  # trailing punctuation ends the name
            break
    return " ".join(words)


def _clause_confidence(clause: str, sentence: str) -> float:
    low_clause, low_sentence = clause.lower(), sentence.lower()
    if any(cue in low_clause for cue in NEGATION_CUES):
        return 0.0
    confidence = 0.8
    if re.search(r"\bnếu\b", low_clause):
        confidence = 0.4  # hypothetical, not a statement of fact
    elif any(cue in low_sentence for cue in CORRECTION_CUES):
        confidence = 0.95
    return confidence


def extract_profile_facts(message: str) -> list[ExtractedFact]:
    """All candidate facts with a confidence score. Question sentences are skipped."""

    facts: list[ExtractedFact] = []
    for sentence in _sentences(message):
        if is_question(sentence):
            continue
        low = sentence.lower()

        for pattern in NAME_RES:
            match = pattern.search(sentence)
            name = _leading_capitalized(match["v"]) if match else ""
            if name:
                facts.append(ExtractedFact("name", name, 0.9, sentence))
                break

        for clause in _clauses(sentence):
            confidence = _clause_confidence(clause, sentence)
            for match in LOCATION_RE.finditer(clause):
                city = next(c for c in CITIES if c.lower() == match["v"].lower())
                facts.append(ExtractedFact("location", city, confidence, clause))
            for match in PROFESSION_RE.finditer(clause):
                facts.append(ExtractedFact("profession", match["v"].strip(), confidence, clause))

        for key, pattern in (("favorite_drink", DRINK_RE), ("favorite_food", FOOD_RE)):
            match = pattern.search(sentence)
            if match:
                value = re.sub(r"\s+(nhé|nha|ạ)$", "", match["v"].strip())
                facts.append(ExtractedFact(key, value, 0.9, sentence))

        pet = PET_RE.search(sentence)
        if pet:
            value = pet["kind"] + (f" tên {pet['name']}" if pet["name"] else "")
            facts.append(ExtractedFact("pet", value, 0.85, sentence))

        if any(t in low for t in STYLE_TRIGGERS):
            style = parse_style(sentence)
            # Only a length/format instruction counts as a style preference.
            if any(_style_slot(p) in ("length", "format") for p in style.split(", ")):
                facts.append(ExtractedFact("response_style", style, 0.85, sentence))

        if any(t in low for t in INTEREST_TRIGGERS):
            found = [t for t in TECH_INTERESTS if re.search(rf"(?<!\w){re.escape(t)}(?!\w)", sentence)]
            if found:
                facts.append(ExtractedFact("interests", ", ".join(found), 0.8, sentence))
    return facts


def extract_profile_updates(message: str, min_confidence: float = 0.6) -> dict[str, str]:
    """Stable profile facts above the confidence threshold; later mentions win."""

    updates: dict[str, str] = {}
    for fact in extract_profile_facts(message):
        if fact.confidence < min_confidence:
            continue
        if fact.key == "response_style" and fact.key in updates:
            updates[fact.key] = merge_style(updates[fact.key], fact.value)
        elif fact.key == "interests" and fact.key in updates:
            updates[fact.key] = merge_interests(updates[fact.key], fact.value)
        else:
            updates[fact.key] = fact.value
    return updates


# ---------------------------------------------------------------------------
# Guardrail for LLM-proposed writes (live tool `save_user_fact`)
# ---------------------------------------------------------------------------

TEMPORARY_CUES = ("tạm thời", "cho cuộc benchmark này", "hôm nay", "tuần này", "tối nay", "chiều nay", "sáng nay")
MAX_FACT_CHARS = 60


def grounding_confidence(value: str, user_messages: list[str]) -> float:
    """Highest confidence of a user clause that literally contains `value`, 0 if none.

    The same cues as the extractor apply: questions, negated/joke clauses, hypotheticals
    and explicitly temporary context do not ground a long-term fact.
    """

    needle = value.strip().lower()
    best = 0.0
    for message in user_messages:
        for sentence in _sentences(message):
            low = sentence.lower()
            if needle not in low or is_question(sentence) or any(c in low for c in TEMPORARY_CUES):
                continue
            for clause in _clauses(sentence) or [sentence]:
                if needle in clause.lower():
                    best = max(best, _clause_confidence(clause, sentence))
    return best


def validate_fact_write(
    key: str, value: str, user_messages: list[str], min_confidence: float = 0.6
) -> tuple[str | None, str]:
    """Return (value to store, reason). The value is None when the write is rejected.

    An LLM-proposed fact must be grounded in what the user actually said in this thread,
    so the model cannot store its own paraphrases, news topics or temporary context.
    """

    value = " ".join(value.split())
    if key not in FACT_LABELS:
        return None, f"key '{key}' không nằm trong schema"
    if not value or len(value) > MAX_FACT_CHARS or "(" in value:
        return None, "value rỗng, quá dài hoặc có chú thích"

    if key == "response_style":
        said = ", ".join(parse_style(s) for m in user_messages for s in _sentences(m) if not is_question(s))
        parts = [p for p in parse_style(value).split(", ") if p]
        kept = [p for p in parts if p in said]
        return (", ".join(kept), "ok") if kept else (None, "style không khớp lời người dùng")

    if key == "interests":
        items = [i.strip() for i in value.split(",") if i.strip()]
        canonical = {t.lower(): t for t in TECH_INTERESTS}
        kept = [
            canonical[i.lower()]
            for i in items
            if i.lower() in canonical and grounding_confidence(i, user_messages) >= min_confidence
        ]
        rejected = [i for i in items if i.lower() not in {k.lower() for k in kept}]
        if not kept:
            return None, f"không có mối quan tâm kỹ thuật nào được người dùng nói rõ: {', '.join(rejected)}"
        return ", ".join(kept), "ok" if not rejected else f"bỏ: {', '.join(rejected)}"

    if grounding_confidence(value, user_messages) < min_confidence:
        return None, "người dùng không nói fact này như một thông tin hiện tại"
    return value, "ok"


# ---------------------------------------------------------------------------
# Compact memory
# ---------------------------------------------------------------------------


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Heuristic summary: first sentence of each older user message, newest kept."""

    lines: list[str] = []
    for message in messages:
        content = message.get("content", "").strip()
        if message.get("role") == "summary":
            lines.extend(line for line in content.splitlines() if line.strip())
            continue
        if message.get("role") != "user" or not content:
            continue
        first = _sentences(content)[0] if _sentences(content) else content
        if len(first) > 120:
            first = first[:117].rstrip() + "..."
        lines.append(f"- {first}")
    return "\n".join(lines[-max_items:])


@dataclass
class CompactMemoryManager:
    """Keeps recent messages verbatim and folds older ones into a bounded summary."""

    threshold_tokens: int
    keep_messages: int
    max_summary_items: int = 6
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _thread(self, thread_id: str) -> dict[str, object]:
        return self.state.setdefault(
            thread_id, {"messages": [], "summary": "", "compactions": 0, "summary_tokens_generated": 0}
        )

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread = self._thread(thread_id)
        thread["messages"].append({"role": role, "content": content})
        if self.message_tokens(thread_id) > self.threshold_tokens and len(thread["messages"]) > self.keep_messages:
            self._compact(thread)

    def _compact(self, thread: dict[str, object]) -> None:
        messages: list[dict[str, str]] = thread["messages"]
        split = len(messages) - self.keep_messages
        older, recent = messages[:split], messages[split:]
        previous = [{"role": "summary", "content": thread["summary"]}] if thread["summary"] else []
        thread["summary"] = summarize_messages(previous + older, max_items=self.max_summary_items)
        thread["messages"] = recent
        thread["compactions"] += 1
        thread["summary_tokens_generated"] += estimate_tokens(thread["summary"])

    def message_tokens(self, thread_id: str) -> int:
        return sum(estimate_tokens(m["content"]) for m in self._thread(thread_id)["messages"])

    def context(self, thread_id: str) -> dict[str, object]:
        return self._thread(thread_id)

    def compaction_count(self, thread_id: str) -> int:
        return int(self._thread(thread_id)["compactions"])
