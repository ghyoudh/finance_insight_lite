import datetime
import unittest

from finance_insight_lite.modules.processor import (
    _is_probable_header_row,
    _looks_like_data_value,
)


class ExcelHeaderDetectionTests(unittest.TestCase):
    def test_financial_identifiers_and_dates_are_data_values(self):
        self.assertTrue(_looks_like_data_value("E-101"))
        self.assertTrue(_looks_like_data_value("INV-2026-101"))
        self.assertTrue(_looks_like_data_value("2026-05-02"))
        self.assertTrue(_looks_like_data_value(datetime.date(2026, 5, 2)))

    def test_ledger_rows_are_not_mistaken_for_headers(self):
        header = [
            "entry_id",
            "date",
            "description",
            "counterparty",
            "amount_sar",
            "vat_number",
            "invoice_number",
        ]
        transaction = [
            "E-101",
            datetime.datetime(2026, 5, 2),
            "Office rent May",
            "Al Rajhi Properties",
            15000,
            "310123456700003",
            "INV-2026-101",
        ]

        self.assertTrue(_is_probable_header_row(header))
        self.assertFalse(_is_probable_header_row(transaction))


if __name__ == "__main__":
    unittest.main()
