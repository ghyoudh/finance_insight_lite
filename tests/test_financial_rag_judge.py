import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finance_insight_lite.modules.eval import (
    FinancialRAGJudge,
    aggregate_judge_scores,
    evaluate_records,
)


VALID_SCORE = json.dumps(
    {
        "groundedness": 5,
        "numerical_accuracy": 5,
        "relevance": 4,
        "clarity": 4,
        "overall": 4.5,
        "unsupported_claims": [],
        "rationale": "The answer is supported by the supplied source chunks.",
    }
)


class FakeJudgeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def complete(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


class FinancialRAGJudgeTests(unittest.TestCase):
    def test_valid_json_is_parsed_and_model_identity_is_redacted(self):
        client = FakeJudgeClient([VALID_SCORE])
        judge = FinancialRAGJudge(client, sleep=lambda _: None)

        score, raw = judge.score(
            question="What did kimi report?",
            source_chunks=["minimax revenue was SAR 100 million"],
            generated_answer="gpt-oss says revenue was SAR 100 million",
        )

        self.assertEqual(score.groundedness, 5)
        self.assertEqual(score.overall, 4.5)
        self.assertEqual(raw, VALID_SCORE)
        self.assertNotIn("kimi", client.calls[0][1].lower())
        self.assertNotIn("minimax", client.calls[0][1].lower())
        self.assertNotIn("gpt-oss", client.calls[0][1].lower())

    def test_malformed_and_api_failures_retry_then_fail_gracefully(self):
        client = FakeJudgeClient(["not json", RuntimeError("temporary API error"), "{}"])
        pauses = []
        judge = FinancialRAGJudge(client, sleep=pauses.append)

        score, raw = judge.score(question="Q", source_chunks=["Source"], generated_answer="Answer")

        self.assertIsNone(score)
        self.assertEqual(raw, "{}")
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(pauses, [1, 2])

    def test_fixture_evaluation_and_aggregation_stay_in_sane_bounds(self):
        fixture = Path(__file__).parent / "fixtures" / "financial_rag_samples.jsonl"
        records = [json.loads(line) for line in fixture.read_text().splitlines()]
        client = FakeJudgeClient([VALID_SCORE] * len(records))
        rows = evaluate_records(records, with_judge=True, judge=FinancialRAGJudge(client, sleep=lambda _: None))
        aggregates = {row["model"]: row for row in aggregate_judge_scores(rows)}

        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["judge_status"] == "ok" for row in rows))
        self.assertTrue(all(0 <= row["token_f1"] <= 1 for row in rows))
        self.assertEqual(aggregates["candidate_a"]["judge_rows"], 2)
        self.assertEqual(aggregates["candidate_b"]["judge_rows"], 1)
        for aggregate in aggregates.values():
            for metric in ("groundedness", "numerical_accuracy", "relevance", "clarity", "overall"):
                self.assertGreaterEqual(aggregate[metric], 1)
                self.assertLessEqual(aggregate[metric], 5)


if __name__ == "__main__":
    unittest.main()
