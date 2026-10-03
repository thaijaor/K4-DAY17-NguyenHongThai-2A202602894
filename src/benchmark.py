from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config

HEADERS = [
    "Agent",
    "Agent tokens only",
    "Prompt tokens processed",
    "Cross-session recall",
    "Response quality",
    "Memory growth (bytes)",
    "Compactions",
]


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _found(answer: str, expected: list[str]) -> int:
    low = answer.lower()
    return sum(1 for item in expected if item.lower() in low)


def recall_points(answer: str, expected: list[str]) -> float:
    """1 if every expected fact appears, 0.5 if some do, 0 if none."""

    if not expected:
        return 1.0
    found = _found(answer, expected)
    if found == len(expected):
        return 1.0
    return 0.5 if found else 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """0..1: fact coverage (70%) + concise (20%) + structured as bullets (10%). No facts -> 0."""

    coverage = _found(answer, expected) / len(expected) if expected else 1.0
    if coverage == 0:
        return 0.0
    concise = 1.0 if len(answer) <= 400 else 0.0
    structured = 1.0 if re.search(r"^\s*[-*•]", answer, re.MULTILINE) else 0.0
    return round(0.7 * coverage + 0.2 * concise + 0.1 * structured, 4)


def judge_quality(judge, question: str, answer: str, expected: list[str]) -> float:
    """LLM-as-judge score 0..1 (live mode only). Falls back to the heuristic on parse errors."""

    prompt = (
        "Chấm câu trả lời của trợ lý trên thang 1-5 theo: đúng fact, không dùng fact cũ đã bị đính chính, "
        "ngắn gọn. Chỉ trả về một số.\n"
        f"Câu hỏi: {question}\nFact kỳ vọng: {', '.join(expected)}\nCâu trả lời: {answer}"
    )
    try:
        raw = judge.invoke(prompt).content
        score = float(re.search(r"[1-5](?:\.\d+)?", str(raw)).group())
        return round((score - 1) / 4, 4)
    except Exception:
        return heuristic_quality(answer, expected)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config, judge=None) -> BenchmarkRow:
    user_ids = sorted({conv["user_id"] for conv in conversations})
    size_before = sum(agent.memory_file_size(uid) for uid in user_ids)
    threads: list[str] = []
    recall_scores: list[float] = []
    quality_scores: list[float] = []

    for conv in conversations:
        print(f"  [{agent_name}] {conv['id']}", file=sys.stderr, flush=True)
        thread_id = conv["id"]
        threads.append(thread_id)
        for turn in conv["turns"]:
            agent.reply(conv["user_id"], thread_id, turn)

        for index, item in enumerate(conv["recall_questions"]):
            recall_thread = f"{conv['id']}-recall-{index}"  # fresh thread = new session
            threads.append(recall_thread)
            answer = agent.reply(conv["user_id"], recall_thread, item["question"])["response"]
            recall_scores.append(recall_points(answer, item["expected_contains"]))
            if judge is not None:
                quality_scores.append(judge_quality(judge, item["question"], answer, item["expected_contains"]))
            else:
                quality_scores.append(heuristic_quality(answer, item["expected_contains"]))

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=sum(agent.token_usage(t) for t in threads),
        prompt_tokens_processed=sum(agent.prompt_token_usage(t) for t in threads),
        recall_score=round(sum(recall_scores) / len(recall_scores), 4),
        response_quality=round(sum(quality_scores) / len(quality_scores), 4),
        memory_growth_bytes=sum(agent.memory_file_size(uid) for uid in user_ids) - size_before,
        compactions=sum(agent.compaction_count(t) for t in threads),
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    table = [
        [
            r.agent_name,
            r.agent_tokens_only,
            r.prompt_tokens_processed,
            f"{r.recall_score:.2f}",
            f"{r.response_quality:.2f}",
            r.memory_growth_bytes,
            r.compactions,
        ]
        for r in rows
    ]
    try:
        from tabulate import tabulate

        return tabulate(table, headers=HEADERS, tablefmt="github")
    except ImportError:
        lines = ["| " + " | ".join(HEADERS) + " |", "|" + "---|" * len(HEADERS)]
        lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in table]
        return "\n".join(lines)


def _delta(baseline: BenchmarkRow, advanced: BenchmarkRow) -> str:
    def pct(new: int, old: int) -> str:
        return f"{(new - old) / old * 100:+.1f}%" if old else "n/a"

    return (
        f"Advanced vs Baseline: agent tokens {pct(advanced.agent_tokens_only, baseline.agent_tokens_only)}, "
        f"prompt tokens {pct(advanced.prompt_tokens_processed, baseline.prompt_tokens_processed)}, "
        f"recall {advanced.recall_score - baseline.recall_score:+.2f}"
    )


def run_suite(title: str, dataset: Path, config, live: bool) -> list[BenchmarkRow]:
    suite_dir = config.state_dir / "benchmark" / dataset.stem
    shutil.rmtree(suite_dir, ignore_errors=True)  # fresh User.md every run -> reproducible
    suite_config = replace(config, state_dir=suite_dir)
    conversations = load_conversations(dataset)

    judge = None
    if live and config.judge_model.is_configured:
        from model_provider import build_chat_model

        judge = build_chat_model(config.judge_model)

    rows = [
        run_agent_benchmark("Baseline", BaselineAgent(suite_config, force_offline=not live), conversations, suite_config, judge),
        run_agent_benchmark("Advanced", AdvancedAgent(suite_config, force_offline=not live), conversations, suite_config, judge),
    ]
    turns = sum(len(c["turns"]) for c in conversations)
    questions = sum(len(c["recall_questions"]) for c in conversations)
    print(f"\n## {title}\n")
    print(f"Dataset: {dataset.name} ({len(conversations)} conversations, {turns} turns, {questions} recall questions)\n")
    print(format_rows(rows))
    print(f"\n{_delta(rows[0], rows[1])}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 17 memory benchmark: Baseline vs Advanced")
    parser.add_argument("--live", action="store_true", help="use the configured LLM provider instead of offline mode")
    args = parser.parse_args()

    config = load_config(Path(__file__).resolve().parent.parent)
    mode = f"live ({config.model.provider}:{config.model.model_name})" if args.live else "offline (deterministic)"
    print(f"# Memory benchmark - mode: {mode}")
    print(f"Compact threshold: {config.compact_threshold_tokens} tokens, keep {config.compact_keep_messages} messages")

    run_suite("Standard Benchmark", config.data_dir / "conversations.json", config, args.live)
    run_suite("Long-Context Stress Benchmark", config.data_dir / "advanced_long_context.json", config, args.live)


if __name__ == "__main__":
    main()
