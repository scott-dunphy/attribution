"""Tests for the template_generator module."""

import pytest
import pandas as pd
import numpy as np
from io import BytesIO

from attribution.template_generator import generate_template, PORTFOLIO_COLUMNS


# ---------------------------------------------------------------------------
# Helper: build a minimal benchmark DataFrame
# ---------------------------------------------------------------------------

def _make_benchmark_df(
    property_types=("Office", "Industrial", "Retail"),
    cbsas=("New York-Newark-Jersey City", "Los Angeles-Long Beach-Anaheim", "Chicago-Naperville-Elgin"),
    periods=((2025, 1), (2025, 2), (2025, 3), (2025, 4), (2026, 1)),
):
    """
    Build a small but realistic benchmark DataFrame with rows at every
    Query_Level needed by generate_template:
      - Total
      - By_PropertyType
      - By_CBSA
      - By_PropertyType_CBSA
    """
    rows = []

    def _add(query_level, pt, cbsa, year, quarter, denom):
        rows.append({
            "Query_Level": query_level,
            "PropertyType": pt,
            "CBSAName": cbsa,
            "Year": year,
            "Quarter": quarter,
            "YYYYQ": year * 10 + quarter,
            "Denom": denom,
            "Income_Return": 0.012,
            "Capital_Return": 0.005,
            "Total_Return": 0.017,
        })

    for y, q in periods:
        # Total row
        _add("Total", "All", "All", y, q, 1_000_000_000)

        # By_PropertyType
        for pt in property_types:
            _add("By_PropertyType", pt, "All", y, q, 300_000_000)

        # By_CBSA
        for cbsa in cbsas:
            _add("By_CBSA", "All", cbsa, y, q, 250_000_000)

        # By_PropertyType_CBSA (cross-dimension)
        denom_base = 100_000_000
        for i, pt in enumerate(property_types):
            for j, cbsa in enumerate(cbsas):
                # Vary denom so the "top 5 by avg Denom" logic is testable
                d = denom_base - (i * 20_000_000) - (j * 10_000_000)
                _add("By_PropertyType_CBSA", pt, cbsa, y, q, d)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGenerateTemplateOutput:
    """Basic output shape and readability."""

    def test_returns_bytes_io_readable_as_excel(self):
        """Output is a BytesIO that openpyxl / pandas can read back."""
        df = _make_benchmark_df()
        result = generate_template(df)

        assert isinstance(result, BytesIO)
        sheets = pd.read_excel(result, sheet_name=None, engine="openpyxl")
        assert isinstance(sheets, dict)

    def test_three_sheets_with_correct_names(self):
        df = _make_benchmark_df()
        result = generate_template(df)
        sheets = pd.read_excel(result, sheet_name=None, engine="openpyxl")

        assert set(sheets.keys()) == {
            "Portfolio_Properties",
            "PropertyTypes_Reference",
            "CBSAs_Reference",
        }


class TestPortfolioPropertiesSheet:
    """Tests for the Portfolio_Properties sheet."""

    def test_columns_match_spec(self):
        df = _make_benchmark_df()
        result = generate_template(df)
        portfolio = pd.read_excel(result, sheet_name="Portfolio_Properties", engine="openpyxl")

        assert list(portfolio.columns) == PORTFOLIO_COLUMNS

    def test_uses_last_four_quarters(self):
        """Default behaviour: uses the 4 most recent quarters from the Total rows."""
        periods = ((2025, 1), (2025, 2), (2025, 3), (2025, 4), (2026, 1))
        df = _make_benchmark_df(periods=periods)
        result = generate_template(df)
        portfolio = pd.read_excel(result, sheet_name="Portfolio_Properties", engine="openpyxl")

        expected_yyyyqs = {20252, 20253, 20254, 20261}
        actual_yyyyqs = set(portfolio["YYYYQ"].unique())
        assert actual_yyyyqs == expected_yyyyqs

    def test_sample_properties_have_valid_numeric_data(self):
        df = _make_benchmark_df()
        result = generate_template(df)
        portfolio = pd.read_excel(result, sheet_name="Portfolio_Properties", engine="openpyxl")

        assert (portfolio["NOI"] > 0).all(), "All NOI values should be positive"
        assert (portfolio["Denom"] > 0).all(), "All Denom values should be positive"
        assert (portfolio["MV"] > 0).all(), "All MV values should be positive"
        assert (portfolio["MVLag1"] > 0).all(), "All MVLag1 values should be positive"
        assert (portfolio["CapEx"] >= 0).all(), "CapEx should be non-negative"

    def test_sample_segments_come_from_benchmark(self):
        """PropertyType and CBSAName in sample data should exist in the benchmark."""
        pts = ("Office", "Industrial", "Retail")
        cbsas = ("New York-Newark-Jersey City", "Los Angeles-Long Beach-Anaheim")
        df = _make_benchmark_df(property_types=pts, cbsas=cbsas)
        result = generate_template(df)
        portfolio = pd.read_excel(result, sheet_name="Portfolio_Properties", engine="openpyxl")

        assert set(portfolio["PropertyType"].unique()).issubset(set(pts))
        assert set(portfolio["CBSAName"].unique()).issubset(set(cbsas))

    def test_top_5_segments_selected_by_avg_denom(self):
        """
        The helper builds cross-dimension rows with decreasing Denom.
        With 3 property types x 3 CBSAs = 9 combos, only the top 5 by
        average Denom should be selected.
        """
        df = _make_benchmark_df()
        result = generate_template(df)
        portfolio = pd.read_excel(result, sheet_name="Portfolio_Properties", engine="openpyxl")

        unique_combos = portfolio[["PropertyType", "CBSAName"]].drop_duplicates()
        assert len(unique_combos) == 5

    def test_property_ids_are_unique_per_segment(self):
        df = _make_benchmark_df()
        result = generate_template(df)
        portfolio = pd.read_excel(result, sheet_name="Portfolio_Properties", engine="openpyxl")

        # Each PropertyID should map to exactly one (PropertyType, CBSAName)
        per_id = portfolio.groupby("PropertyID")[["PropertyType", "CBSAName"]].nunique()
        assert (per_id == 1).all().all()


class TestReferenceSheets:
    """Tests for PropertyTypes_Reference and CBSAs_Reference."""

    def test_property_types_excludes_all(self):
        df = _make_benchmark_df(property_types=("Office", "Industrial"))
        result = generate_template(df)
        pt_ref = pd.read_excel(result, sheet_name="PropertyTypes_Reference", engine="openpyxl")

        values = set(pt_ref["PropertyType"])
        assert "All" not in values
        assert values == {"Office", "Industrial"}

    def test_cbsas_excludes_all(self):
        cbsas = ("New York-Newark-Jersey City", "Chicago-Naperville-Elgin")
        df = _make_benchmark_df(cbsas=cbsas)
        result = generate_template(df)
        cbsa_ref = pd.read_excel(result, sheet_name="CBSAs_Reference", engine="openpyxl")

        values = set(cbsa_ref["CBSAName"])
        assert "All" not in values
        assert values == set(cbsas)

    def test_reference_sheets_are_sorted(self):
        pts = ("Retail", "Office", "Industrial")
        df = _make_benchmark_df(property_types=pts)
        result = generate_template(df)
        pt_ref = pd.read_excel(result, sheet_name="PropertyTypes_Reference", engine="openpyxl")

        assert list(pt_ref["PropertyType"]) == sorted(pts)


class TestPeriodsParameter:
    """Tests for the optional `periods` parameter."""

    def test_periods_limits_output(self):
        df = _make_benchmark_df()
        chosen = [(2025, 3), (2025, 4)]
        result = generate_template(df, periods=chosen)
        portfolio = pd.read_excel(result, sheet_name="Portfolio_Properties", engine="openpyxl")

        actual_yyyyqs = set(portfolio["YYYYQ"].unique())
        assert actual_yyyyqs == {20253, 20254}

    def test_single_period(self):
        df = _make_benchmark_df()
        result = generate_template(df, periods=[(2025, 4)])
        portfolio = pd.read_excel(result, sheet_name="Portfolio_Properties", engine="openpyxl")

        assert set(portfolio["YYYYQ"].unique()) == {20254}

    def test_periods_more_than_four_uses_last_four(self):
        """When >4 periods are given, only the last 4 are used for sample data."""
        all_periods = [(2025, 1), (2025, 2), (2025, 3), (2025, 4), (2026, 1)]
        df = _make_benchmark_df(periods=all_periods)
        result = generate_template(df, periods=all_periods)
        portfolio = pd.read_excel(result, sheet_name="Portfolio_Properties", engine="openpyxl")

        actual_yyyyqs = set(portfolio["YYYYQ"].unique())
        assert actual_yyyyqs == {20252, 20253, 20254, 20261}


class TestFallbackWhenCrossDimensionSparse:
    """When By_PropertyType_CBSA has < 2 segments, the function falls back
    to building combos from the property type and CBSA lists."""

    def _make_sparse_benchmark(self):
        """Build a benchmark with only 1 cross-dimension segment."""
        rows = []
        for y, q in ((2025, 3), (2025, 4)):
            rows.append({
                "Query_Level": "Total",
                "PropertyType": "All",
                "CBSAName": "All",
                "Year": y, "Quarter": q, "YYYYQ": y * 10 + q,
                "Denom": 500_000_000,
                "Income_Return": 0.01, "Capital_Return": 0.005, "Total_Return": 0.015,
            })
            for pt in ("Office", "Industrial", "Retail"):
                rows.append({
                    "Query_Level": "By_PropertyType",
                    "PropertyType": pt, "CBSAName": "All",
                    "Year": y, "Quarter": q, "YYYYQ": y * 10 + q,
                    "Denom": 200_000_000,
                    "Income_Return": 0.01, "Capital_Return": 0.005, "Total_Return": 0.015,
                })
            for cbsa in ("New York-Newark-Jersey City", "Chicago-Naperville-Elgin"):
                rows.append({
                    "Query_Level": "By_CBSA",
                    "PropertyType": "All", "CBSAName": cbsa,
                    "Year": y, "Quarter": q, "YYYYQ": y * 10 + q,
                    "Denom": 200_000_000,
                    "Income_Return": 0.01, "Capital_Return": 0.005, "Total_Return": 0.015,
                })
            # Only ONE cross-dimension segment -> triggers fallback
            rows.append({
                "Query_Level": "By_PropertyType_CBSA",
                "PropertyType": "Office", "CBSAName": "New York-Newark-Jersey City",
                "Year": y, "Quarter": q, "YYYYQ": y * 10 + q,
                "Denom": 100_000_000,
                "Income_Return": 0.01, "Capital_Return": 0.005, "Total_Return": 0.015,
            })
        return pd.DataFrame(rows)

    def test_fallback_produces_multiple_segments(self):
        df = self._make_sparse_benchmark()
        result = generate_template(df)
        portfolio = pd.read_excel(result, sheet_name="Portfolio_Properties", engine="openpyxl")

        combos = portfolio[["PropertyType", "CBSAName"]].drop_duplicates()
        assert len(combos) >= 2, "Fallback should generate at least 2 property segments"

    def test_fallback_segments_use_known_types_and_cbsas(self):
        df = self._make_sparse_benchmark()
        result = generate_template(df)
        portfolio = pd.read_excel(result, sheet_name="Portfolio_Properties", engine="openpyxl")

        known_pts = {"Office", "Industrial", "Retail"}
        known_cbsas = {"New York-Newark-Jersey City", "Chicago-Naperville-Elgin"}
        assert set(portfolio["PropertyType"].unique()).issubset(known_pts)
        assert set(portfolio["CBSAName"].unique()).issubset(known_cbsas)

    def test_fallback_caps_at_five_segments(self):
        df = self._make_sparse_benchmark()
        result = generate_template(df)
        portfolio = pd.read_excel(result, sheet_name="Portfolio_Properties", engine="openpyxl")

        combos = portfolio[["PropertyType", "CBSAName"]].drop_duplicates()
        assert len(combos) <= 5

    def test_fallback_still_produces_valid_excel(self):
        df = self._make_sparse_benchmark()
        result = generate_template(df)

        assert isinstance(result, BytesIO)
        sheets = pd.read_excel(result, sheet_name=None, engine="openpyxl")
        assert set(sheets.keys()) == {
            "Portfolio_Properties",
            "PropertyTypes_Reference",
            "CBSAs_Reference",
        }
