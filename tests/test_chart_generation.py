"""Focused tests for chart intent routing and Plotly generation."""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import plotly.graph_objects as go
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finance_insight_lite.modules.rag_agent import (
    ChartGenerator,
    FinancialDataExtractor,
    FinancialRAGAgent,
)


class FakeVectorDB:
    def __init__(self, documents):
        self.documents = documents
        self.queries = []
        self.index = type("FakeIndex", (), {"ntotal": len(documents)})()

    def similarity_search(self, query, k):
        self.queries.append((query, k))
        return self.documents[:k]


class FailIfInvokedLLM:
    def invoke(self, _messages):
        raise AssertionError("Structured improvement rows should not require the LLM")


class ChartGenerationTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {"label": ["Q1", "Q2", "Q3"], "value": [100.0, 125.5, 140.0]}
        )
        self.agent = FinancialRAGAgent.__new__(FinancialRAGAgent)

    def test_arabic_chart_request_is_recognized(self):
        self.assertTrue(
            self.agent._needs_chart("ابي رسم بياني يوضح لي نسبة التحسن")
        )

    def test_non_chart_arabic_question_is_not_recognized_as_chart(self):
        self.assertFalse(self.agent._needs_chart("ما هي نسبة التحسن؟"))

    def test_arabic_chart_types_are_selected(self):
        cases = {
            "ارسم مخططاً دائرياً لتوزيع المصروفات": "pie",
            "ارسم مخططاً خطياً للتطور الشهري": "line",
            "ارسم مخطط أعمدة للمقارنة": "bar",
            "ارسم مخططاً مبعثراً للإيرادات والأرباح": "scatter",
            "ارسم مخططاً مساحياً للتدفق النقدي": "area",
        }

        for query, expected in cases.items():
            with self.subTest(query=query):
                self.assertTrue(self.agent._needs_chart(query))
                self.assertEqual(
                    self.agent._suggest_chart_type(query, self.df), expected
                )

    def test_improvement_percentage_uses_bar_chart(self):
        query = "ابي رسم بياني يوضح لي نسبة التحسن"

        self.assertEqual(self.agent._suggest_chart_type(query, self.df), "bar")

    def test_improvement_extraction_keeps_only_comparable_percentages(self):
        documents = [
            Document(page_content=(
                "البُعد المقاس: معرفتي بمفهوم مواطن القوّة | الفرق: 1.95 | "
                "نسبة التحسّن: +120%"
            )),
            Document(page_content=(
                "البُعد المقاس: بداية استشعاري لمواهبي | الفرق: 1.17 | "
                "نسبة التحسّن: +55%"
            )),
            Document(page_content=(
                "المؤشر: الرضا العام | المعدّل (1-4): 3.57"
            )),
        ]
        vector_db = FakeVectorDB(documents)
        extractor = FinancialDataExtractor(vector_db, FailIfInvokedLLM())

        result = extractor.extract_data_from_query(
            "ابي رسم بياني يوضح لي نسبة التحسن", k=10
        )

        self.assertEqual(result["value"].tolist(), [120.0, 55.0])
        self.assertEqual(result["currency"].tolist(), ["%", "%"])
        self.assertNotIn(3.57, result["value"].tolist())
        self.assertEqual(vector_db.queries[0][0], "نسبة التحسّن البُعد المقاس")
        self.assertEqual(vector_db.queries[0][1], len(documents))

    def test_line_chart_survives_ui_json_round_trip(self):
        chart_json = ChartGenerator.create_line_chart(
            self.df, x="label", y="value", title="Quarterly trend"
        ).to_json()

        rebuilt = go.Figure(json.loads(chart_json))

        self.assertEqual(len(rebuilt.data), 1)
        self.assertEqual(rebuilt.data[0].type, "scatter")
        self.assertEqual(rebuilt.layout.xaxis.title.font.size, 14)
        self.assertEqual(rebuilt.layout.yaxis.title.font.size, 14)

    def test_percentage_chart_labels_axis_and_values_as_percentages(self):
        percentage_df = pd.DataFrame({
            "label": ["الوعي", "التطبيق"],
            "value": [55.0, 80.0],
            "currency": ["%", "%"],
            "suggestion": ["", ""],
        })
        self.agent.vector_db = object()
        self.agent.fast_llm = object()

        with patch.object(
            FinancialDataExtractor, "extract_data_from_query", return_value=percentage_df
        ):
            result = self.agent._build_chart("ارسم نسبة التحسن")

        figure = go.Figure(json.loads(result["chart"]))
        self.assertTrue(result["success"])
        self.assertEqual(figure.layout.yaxis.title.text, "النسبة (%)")
        self.assertEqual(figure.data[0].texttemplate, "%{y:g}%")


if __name__ == "__main__":
    unittest.main()
