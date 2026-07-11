"""Offline and LLM-as-judge evaluation for financial RAG answers.

Input records are JSON Lines objects with ``generated_answer`` and either
``reference_answer`` or ``retrieved_context``.  To run the paid judge, source
chunks must be supplied through ``source_chunks`` (a list of strings) or
``retrieved_context``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import time
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Iterable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .groq_client import DEFAULT_GROQ_MODEL, GroqJudgeClient


LOGGER = logging.getLogger(__name__)
MAX_JUDGE_ATTEMPTS = 3
MODEL_IDENTIFIERS = re.compile(
    r"\b(?:minimax|kimi|gpt[ _-]?oss(?:[ _-]?\d+b)?|claude|gemini|llama)\b",
    re.IGNORECASE,
)

JUDGE_SYSTEM_PROMPT = """You are a strict evaluator for a financial-document RAG system.
Evaluate the GENERATED ANSWER using only the RETRIEVED SOURCE CHUNKS as evidence.
Groundedness is the most important dimension: heavily penalize unsupported or invented
figures, dates, entities, calculations, recommendations, and assertions. Do not reward
plausible claims that are absent from the source chunks.

Score each integer dimension from 1 (very poor) to 5 (excellent):
- groundedness: every material claim is traceable to source chunks.
- numerical_accuracy: figures, units, dates, percentages, and calculations match exactly.
- relevance: directly answers the user's financial question.
- clarity: well structured and unambiguous for a financial decision context.
Set overall to a numeric 1.0-5.0 summary, weighted most heavily toward groundedness.
List each material unsupported claim verbatim or as a concise paraphrase. Use [] when none.
Give a concise rationale (at most two sentences).
Return STRICT JSON ONLY: no Markdown, explanation, or code fences. The JSON object must
have exactly these keys: groundedness, numerical_accuracy, relevance, clarity, overall,
unsupported_claims, rationale."""


class JudgeScore(BaseModel):
    """Validated contract returned by the financial LLM judge."""

    model_config = ConfigDict(extra="forbid", strict=True)

    groundedness: int = Field(ge=1, le=5)
    numerical_accuracy: int = Field(ge=1, le=5)
    relevance: int = Field(ge=1, le=5)
    clarity: int = Field(ge=1, le=5)
    overall: float = Field(ge=1.0, le=5.0)
    unsupported_claims: list[str]
    rationale: str = Field(max_length=1000)


def clean_text(text: Any) -> str:
    """Remove punctuation/extra spaces and lowercase text for overlap metrics."""
    if not text:
        return ""
    text = re.sub(r"[^\w\s]", " ", str(text).lower())
    return " ".join(text.split())


def calculate_rag_metrics(retrieved_context: Any, generated_answer: Any) -> tuple[float, float, float]:
    """Return token-set precision, recall, and F1 for a generated answer."""
    context_words = set(clean_text(retrieved_context).split())
    answer_words = set(clean_text(generated_answer).split())
    if not answer_words or not context_words:
        return 0.0, 0.0, 0.0
    hits = len(context_words.intersection(answer_words))
    precision = hits / len(answer_words)
    recall = hits / len(context_words)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return round(precision, 2), round(recall, 2), round(f1, 2)


def _redact_model_identifiers(value: Any) -> str:
    return MODEL_IDENTIFIERS.sub("[redacted-model]", str(value or ""))


def _source_text(source_chunks: Any) -> str:
    if isinstance(source_chunks, list):
        return "\n\n".join(f"[Chunk {i + 1}]\n{chunk}" for i, chunk in enumerate(source_chunks))
    return str(source_chunks or "")


def _judge_user_prompt(question: str, source_chunks: Any, generated_answer: str) -> str:
    return (
        "USER QUESTION:\n"
        f"{_redact_model_identifiers(question)}\n\n"
        "RETRIEVED SOURCE CHUNKS:\n"
        f"{_redact_model_identifiers(_source_text(source_chunks))}\n\n"
        "GENERATED ANSWER:\n"
        f"{_redact_model_identifiers(generated_answer)}"
    )


class FinancialRAGJudge:
    """Calls a deterministic Groq judge with parsing, schema checks, and retries."""

    def __init__(
        self,
        client: GroqJudgeClient | Any | None = None,
        *,
        max_attempts: int = MAX_JUDGE_ATTEMPTS,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.client = client or GroqJudgeClient()
        self.max_attempts = max_attempts
        self.sleep = sleep

    def score(self, *, question: str, source_chunks: Any, generated_answer: str) -> tuple[JudgeScore | None, str | None]:
        """Return a score and raw final response; failures are represented by ``None``."""
        raw_response: str | None = None
        prompt = _judge_user_prompt(question, source_chunks, generated_answer)
        for attempt in range(1, self.max_attempts + 1):
            try:
                raw_response = self.client.complete(JUDGE_SYSTEM_PROMPT, prompt)
                payload = json.loads(raw_response)
                return JudgeScore.model_validate(payload), raw_response
            except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
                LOGGER.warning("Invalid judge response (attempt %s/%s): %s", attempt, self.max_attempts, exc)
            except Exception as exc:  # API/network errors must not terminate an evaluation run.
                LOGGER.warning("Judge API failure (attempt %s/%s): %s", attempt, self.max_attempts, exc)
            if attempt < self.max_attempts:
                self.sleep(2 ** (attempt - 1))
        LOGGER.error("judge_failed after %s attempts; raw response: %r", self.max_attempts, raw_response)
        return None, raw_response


def evaluate_records(records: Iterable[dict[str, Any]], *, with_judge: bool = False, judge: FinancialRAGJudge | None = None) -> list[dict[str, Any]]:
    """Add overlap metrics and, optionally, judge fields to each evaluation record."""
    if with_judge and judge is None:
        judge = FinancialRAGJudge()
    results: list[dict[str, Any]] = []
    for record in records:
        row = dict(record)
        answer = str(row.get("generated_answer", ""))
        overlap_target = row.get("reference_answer", row.get("retrieved_context", row.get("source_chunks", "")))
        precision, recall, f1 = calculate_rag_metrics(overlap_target, answer)
        row.update({"token_precision": precision, "token_recall": recall, "token_f1": f1})
        if with_judge:
            source_chunks = row.get("source_chunks", row.get("retrieved_context", ""))
            score, raw = judge.score(question=str(row.get("question", "")), source_chunks=source_chunks, generated_answer=answer)
            if score is None:
                row.update({"judge_status": "judge_failed", "judge_raw_response": raw})
            else:
                row.update({"judge_status": "ok", **score.model_dump()})
        results.append(row)
    return results


def aggregate_judge_scores(rows: Iterable[dict[str, Any]], group_by: str = "model") -> list[dict[str, Any]]:
    """Average successful judge scores per model (or another supplied record field)."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("judge_status") == "ok":
            groups.setdefault(str(row.get(group_by, "unknown")), []).append(row)
    dimensions = ("groundedness", "numerical_accuracy", "relevance", "clarity", "overall")
    return [
        {group_by: key, "judge_rows": len(group), **{name: round(fmean(float(r[name]) for r in group), 3) for name in dimensions}}
        for key, group in sorted(groups.items())
    ]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_results(rows: list[dict[str, Any]], path: Path) -> None:
    if path.suffix.lower() == ".csv":
        fieldnames = sorted({key for row in rows for key in row})
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()})
        return
    with path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)


def _load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE pairs when python-dotenv is unavailable."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


def main(argv: list[str] | None = None) -> int:
    # Match the original evaluation script's .env behavior, but do not fail
    # until the optional paid judge is actually requested.
    env_path = Path(__file__).resolve().parents[3] / ".env"
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path)
    except ImportError:
        _load_env_file(env_path)
    parser = argparse.ArgumentParser(description="Evaluate financial RAG answers from a JSONL file.")
    parser.add_argument("--input", type=Path, required=True, help="JSONL records containing question, generated_answer, and source_chunks.")
    parser.add_argument("--output", type=Path, required=True, help="Output .json or .csv path.")
    parser.add_argument("--with-judge", action="store_true", help="Enable paid Groq LLM-as-judge scores.")
    parser.add_argument("--judge-model", default=DEFAULT_GROQ_MODEL, help="Groq judge model (default: %(default)s).")
    args = parser.parse_args(argv)
    judge = FinancialRAGJudge(GroqJudgeClient(model=args.judge_model)) if args.with_judge else None
    rows = evaluate_records(_read_jsonl(args.input), with_judge=args.with_judge, judge=judge)
    _write_results(rows, args.output)
    if args.with_judge:
        LOGGER.info("Judge aggregate by model: %s", aggregate_judge_scores(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
