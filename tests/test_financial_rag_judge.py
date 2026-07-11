"""Tests for FinancialRAGJudge and the record evaluation/aggregation pipeline."""

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


# Model identity terms that must never leak into a judge prompt, since the
# judge is meant to score answers blind to which model produced them.
REDACTED_MODEL_TERMS = ("kimi", "minimax", "gpt-oss")

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

JUDGE_METRICS = ("groundedness", "numerical_accuracy", "relevance", "clarity", "overall")

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "financial_rag_samples.jsonl"


class FakeJudgeClient:
    """A stand-in judge client that replays canned responses in order.

    Each entry in `responses` is either a raw response string to return,
    or an Exception instance to raise -- letting tests simulate API
    failures interleaved with successful/malformed responses.
    """

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def complete(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response

    def prompts_seen(self):
        """All prompt text (system + user) sent across every call, joined."""
        return "\n".join(f"{s}\n{u}" for s, u in self.calls)


class FinancialRAGJudgeTests(unittest.TestCase):
    def make_judge(self, responses):
        """Build a FinancialRAGJudge wired to a FakeJudgeClient with no real sleep."""
        client = FakeJudgeClient(responses)
        judge = FinancialRAGJudge(client, sleep=lambda _: None)
        return judge, client

    def test_valid_json_is_parsed_correctly(self):
        judge, client = self.make_judge([VALID_SCORE])

        score, raw = judge.score(
            question="What did the model report?",
            source_chunks=["Revenue was SAR 100 million"],
            generated_answer="Reported revenue was SAR 100 million",
        )

        self.assertEqual(score.groundedness, 5)
        self.assertEqual(score.overall, 4.5)
        self.assertEqual(raw, VALID_SCORE)
        self.assertEqual(len(client.calls), 1)

    def test_model_identity_is_redacted_from_prompt(self):
        """None of the model names involved should ever reach the judge prompt."""
        judge, client = self.make_judge([VALID_SCORE])

        judge.score(
            question="What did kimi report?",
            source_chunks=["minimax revenue was SAR 100 million"],
            generated_answer="gpt-oss says revenue was SAR 100 million",
        )

        combined_prompt = client.prompts_seen().lower()
        for term in REDACTED_MODEL_TERMS:
            with self.subTest(term=term):
                self.assertNotIn(term, combined_prompt)

    def test_malformed_and_api_failures_retry_then_fail_gracefully(self):
        judge, client = self.make_judge(
            ["not json", RuntimeError("temporary API error"), "{}"]
        )
        pauses = []
        judge.sleep = pauses.append

        score, raw = judge.score(
            question="Q", source_chunks=["Source"], generated_answer="Answer"
        )

        self.assertIsNone(score)
        self.assertEqual(raw, "{}")
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(pauses, [1, 2])

    def test_all_attempts_failing_still_returns_gracefully(self):
        """Even if every retry raises, score() should not propagate the exception."""
        judge, client = self.make_judge(
            [
                RuntimeError("api down"),
                RuntimeError("api down"),
                RuntimeError("api down"),
            ]
        )

        score, raw = judge.score(
            question="Q", source_chunks=["Source"], generated_answer="Answer"
        )

        self.assertIsNone(score)
        self.assertEqual(len(client.calls), 3)

    def test_no_sleep_is_invoked_when_first_attempt_succeeds(self):
        pauses = []
        client = FakeJudgeClient([VALID_SCORE])
        judge = FinancialRAGJudge(client, sleep=pauses.append)

        score, _ = judge.score(
            question="Q", source_chunks=["Source"], generated_answer="Answer"
        )

        self.assertIsNotNone(score)
        self.assertEqual(pauses, [])

    def test_boundary_scores_are_parsed_without_clamping_or_rejection(self):
        """A judge should trust the model's own min (1) and max (5) scores."""
        boundary_score = json.dumps(
            {
                "groundedness": 1,
                "numerical_accuracy": 1,
                "relevance": 5,
                "clarity": 5,
                "overall": 3,
                "unsupported_claims": ["The Q3 revenue figure is not in any source chunk"],
                "rationale": "Mixed: strong clarity but a fabricated figure.",
            }
        )
        judge, _ = self.make_judge([boundary_score])

        score, _ = judge.score(
            question="Q", source_chunks=["Source"], generated_answer="Answer"
        )

        self.assertEqual(score.groundedness, 1)
        self.assertEqual(score.relevance, 5)
        self.assertEqual(score.overall, 3)

    def test_unsupported_claims_and_rationale_are_preserved(self):
        """The judge's qualitative reasoning shouldn't be dropped during parsing."""
        claims = ["The 12% growth figure is not present in any source chunk"]
        rationale = "The headline number is fabricated; everything else checks out."
        annotated_score = json.dumps(
            {
                "groundedness": 3,
                "numerical_accuracy": 2,
                "relevance": 4,
                "clarity": 4,
                "overall": 3,
                "unsupported_claims": claims,
                "rationale": rationale,
            }
        )
        judge, _ = self.make_judge([annotated_score])

        score, _ = judge.score(
            question="Q", source_chunks=["Source"], generated_answer="Answer"
        )

        self.assertEqual(score.unsupported_claims, claims)
        self.assertEqual(score.rationale, rationale)

    def test_json_missing_required_field_is_treated_as_malformed(self):
        """A response missing a required metric (e.g. 'overall') should fail like bad JSON,
        not raise or silently default a score."""
        incomplete = json.dumps(
            {
                "groundedness": 5,
                "numerical_accuracy": 5,
                "relevance": 4,
                "clarity": 4,
                # "overall" omitted
                "unsupported_claims": [],
                "rationale": "Missing overall score.",
            }
        )
        judge, client = self.make_judge([incomplete, incomplete, incomplete])

        score, raw = judge.score(
            question="Q", source_chunks=["Source"], generated_answer="Answer"
        )

        self.assertIsNone(score)
        self.assertEqual(len(client.calls), 3)

    def test_non_object_json_is_treated_as_malformed(self):
        """A syntactically valid JSON value that isn't a score object (e.g. a bare list)
        should be rejected the same way malformed text is."""
        judge, client = self.make_judge(["[1, 2, 3]", "[1, 2, 3]", "[1, 2, 3]"])

        score, raw = judge.score(
            question="Q", source_chunks=["Source"], generated_answer="Answer"
        )

        self.assertIsNone(score)
        self.assertEqual(len(client.calls), 3)

    def test_redaction_is_case_insensitive(self):
        """Model names should be stripped from the prompt regardless of casing."""
        judge, client = self.make_judge([VALID_SCORE])

        judge.score(
            question="What did KIMI and Gpt-Oss disagree on?",
            source_chunks=["MINIMAX reported SAR 100 million"],
            generated_answer="Revenue was SAR 100 million",
        )

        combined_prompt = client.prompts_seen().lower()
        for term in REDACTED_MODEL_TERMS:
            with self.subTest(term=term):
                self.assertNotIn(term, combined_prompt)


class EvaluateAndAggregateTests(unittest.TestCase):
    """Tests covering the fixture-driven evaluate_records/aggregate_judge_scores flow."""

    @classmethod
    def setUpClass(cls):
        cls.records = [
            json.loads(line) for line in FIXTURE_PATH.read_text().splitlines() if line.strip()
        ]

    def evaluate_fixture(self):
        client = FakeJudgeClient([VALID_SCORE] * len(self.records))
        judge = FinancialRAGJudge(client, sleep=lambda _: None)
        rows = evaluate_records(self.records, with_judge=True, judge=judge)
        aggregates = {row["model"]: row for row in aggregate_judge_scores(rows)}
        return rows, aggregates

    def test_fixture_evaluation_produces_one_row_per_record(self):
        rows, _ = self.evaluate_fixture()

        self.assertEqual(len(rows), len(self.records))
        self.assertTrue(all(row["judge_status"] == "ok" for row in rows))

    def test_token_f1_scores_are_normalized(self):
        rows, _ = self.evaluate_fixture()

        for row in rows:
            with self.subTest(row=row.get("model")):
                self.assertGreaterEqual(row["token_f1"], 0)
                self.assertLessEqual(row["token_f1"], 1)

    def test_aggregate_row_counts_per_model(self):
        _, aggregates = self.evaluate_fixture()

        self.assertEqual(aggregates["candidate_a"]["judge_rows"], 2)
        self.assertEqual(aggregates["candidate_b"]["judge_rows"], 1)

    def test_aggregate_judge_metrics_stay_in_valid_range(self):
        _, aggregates = self.evaluate_fixture()

        for model, aggregate in aggregates.items():
            for metric in JUDGE_METRICS:
                with self.subTest(model=model, metric=metric):
                    self.assertGreaterEqual(aggregate[metric], 1)
                    self.assertLessEqual(aggregate[metric], 5)

    def test_aggregation_excludes_failed_judge_calls(self):
        """A judge failure on one record should drop that row out of the aggregate
        counts rather than being silently averaged in as if it succeeded.

        This is a stronger check than bounds-checking on identical repeated scores:
        it verifies aggregate_judge_scores is actually keying off judge_status.
        """
        # First record's judge call fails outright across all retries; the rest succeed.
        responses = [RuntimeError("api down"), RuntimeError("api down"), RuntimeError("api down")]
        responses += [VALID_SCORE] * (len(self.records) - 1)
        client = FakeJudgeClient(responses)
        judge = FinancialRAGJudge(client, sleep=lambda _: None)

        rows = evaluate_records(self.records, with_judge=True, judge=judge)
        aggregates = {row["model"]: row for row in aggregate_judge_scores(rows)}

        ok_row_count = sum(1 for row in rows if row["judge_status"] == "ok")
        self.assertEqual(ok_row_count, len(self.records) - 1)

        # Every row should still be present -- a judge failure marks the row,
        # it doesn't drop it from the evaluation output.
        self.assertEqual(len(rows), len(self.records))

        # The failed row must not be counted toward any model's aggregate.
        total_aggregated_rows = sum(a["judge_rows"] for a in aggregates.values())
        self.assertEqual(total_aggregated_rows, ok_row_count)


if __name__ == "__main__":
    unittest.main()