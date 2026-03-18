"""
Comprehensive tests for attribution.aggregator module.

Covers: is_property_level, validate_property_file, _weighted_return,
        _aggregate_group, aggregate_properties, and edge cases.
"""
import pandas as pd
import numpy as np
import pytest

from attribution.aggregator import (
    DOLLAR_COLS,
    PORTFOLIO_REQUIRED_COLUMNS,
    RETURN_COLS,
    _aggregate_group,
    _weighted_return,
    aggregate_properties,
    is_property_level,
    validate_property_file,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _make_property_row(
    year=2024, quarter=1, property_type="Office", cbsa_name="New York",
    noi=100.0, capex=10.0, mv=1000.0, mv_lag1=950.0, psales=0.0,
    denom=900.0, income_return=0.05, capital_return=0.03, total_return=0.08,
):
    """Return a dict representing a single property row."""
    return {
        "Year": year,
        "Quarter": quarter,
        "PropertyType": property_type,
        "CBSAName": cbsa_name,
        "NOI": noi,
        "CapEx": capex,
        "MV": mv,
        "MVLag1": mv_lag1,
        "PSales": psales,
        "Denom": denom,
        "Income_Return": income_return,
        "Capital_Return": capital_return,
        "Total_Return": total_return,
    }


def _make_property_df(rows=None, n=3):
    """Build a property-level DataFrame.

    If *rows* is None, create *n* default rows with differing values to make
    weighted-return tests meaningful.
    """
    if rows is not None:
        return pd.DataFrame(rows)

    default_rows = [
        _make_property_row(
            noi=100 + i * 50,
            capex=10 + i * 5,
            mv=1000 + i * 200,
            mv_lag1=950 + i * 180,
            psales=i * 10.0,
            denom=900 + i * 100,
            income_return=0.05 + i * 0.01,
            capital_return=0.03 + i * 0.005,
            total_return=0.08 + i * 0.015,
        )
        for i in range(n)
    ]
    return pd.DataFrame(default_rows)


def _two_type_two_cbsa_df():
    """Two property types x two CBSAs, single period -> 4 properties."""
    rows = [
        _make_property_row(property_type="Office", cbsa_name="New York",
                           denom=100, income_return=0.10, capital_return=0.05, total_return=0.15,
                           noi=50, capex=5, mv=500, mv_lag1=480, psales=0),
        _make_property_row(property_type="Office", cbsa_name="Chicago",
                           denom=200, income_return=0.08, capital_return=0.04, total_return=0.12,
                           noi=80, capex=8, mv=800, mv_lag1=760, psales=10),
        _make_property_row(property_type="Retail", cbsa_name="New York",
                           denom=300, income_return=0.06, capital_return=0.03, total_return=0.09,
                           noi=60, capex=6, mv=600, mv_lag1=570, psales=20),
        _make_property_row(property_type="Retail", cbsa_name="Chicago",
                           denom=400, income_return=0.04, capital_return=0.02, total_return=0.06,
                           noi=40, capex=4, mv=400, mv_lag1=380, psales=30),
    ]
    return pd.DataFrame(rows)


def _multi_period_df():
    """Two periods (2024Q1 and 2024Q2), each with 2 properties."""
    rows = [
        _make_property_row(year=2024, quarter=1, property_type="Office", cbsa_name="NY",
                           denom=100, income_return=0.10, capital_return=0.05, total_return=0.15,
                           noi=50, capex=5, mv=500, mv_lag1=480, psales=0),
        _make_property_row(year=2024, quarter=1, property_type="Retail", cbsa_name="LA",
                           denom=200, income_return=0.06, capital_return=0.03, total_return=0.09,
                           noi=60, capex=6, mv=600, mv_lag1=570, psales=10),
        _make_property_row(year=2024, quarter=2, property_type="Office", cbsa_name="NY",
                           denom=110, income_return=0.11, capital_return=0.06, total_return=0.17,
                           noi=55, capex=7, mv=520, mv_lag1=500, psales=5),
        _make_property_row(year=2024, quarter=2, property_type="Retail", cbsa_name="LA",
                           denom=210, income_return=0.07, capital_return=0.04, total_return=0.11,
                           noi=65, capex=8, mv=620, mv_lag1=590, psales=15),
    ]
    return pd.DataFrame(rows)


# ===========================================================================
# 1. is_property_level
# ===========================================================================

class TestIsPropertyLevel:
    def test_no_query_level_column(self):
        df = _make_property_df()
        assert "Query_Level" not in df.columns
        assert is_property_level(df) is True

    def test_query_level_all_nan(self):
        df = _make_property_df()
        df["Query_Level"] = np.nan
        assert is_property_level(df) is True

    def test_query_level_all_empty_string(self):
        """Empty strings are not NaN -- dropna keeps them, so len(levels) > 0."""
        df = _make_property_df()
        df["Query_Level"] = ""
        # The column has non-NaN values (empty strings), so should be False
        assert is_property_level(df) is False

    def test_query_level_mixed_nan_and_empty(self):
        df = _make_property_df(n=4)
        df["Query_Level"] = [np.nan, np.nan, np.nan, np.nan]
        assert is_property_level(df) is True

    def test_valid_aggregated_format(self):
        df = _make_property_df()
        df["Query_Level"] = "Total"
        assert is_property_level(df) is False

    def test_query_level_with_mixed_values(self):
        df = _make_property_df(n=4)
        df["Query_Level"] = ["Total", "By_PropertyType", np.nan, "By_CBSA"]
        assert is_property_level(df) is False


# ===========================================================================
# 2. validate_property_file
# ===========================================================================

class TestValidatePropertyFile:
    def test_valid_file_no_errors(self):
        df = _make_property_df()
        errors = validate_property_file(df)
        assert errors == []

    def test_missing_single_column(self):
        df = _make_property_df().drop(columns=["NOI"])
        errors = validate_property_file(df)
        assert len(errors) == 1
        assert "NOI" in errors[0]

    def test_missing_multiple_columns(self):
        df = _make_property_df().drop(columns=["NOI", "MV", "Year"])
        errors = validate_property_file(df)
        assert len(errors) == 1
        for col in ["NOI", "MV", "Year"]:
            assert col in errors[0]

    def test_extra_columns_ignored(self):
        df = _make_property_df()
        df["ExtraCol"] = 999
        errors = validate_property_file(df)
        assert errors == []

    def test_empty_dataframe_with_correct_columns(self):
        df = pd.DataFrame(columns=PORTFOLIO_REQUIRED_COLUMNS)
        errors = validate_property_file(df)
        assert errors == []

    def test_completely_empty_dataframe(self):
        df = pd.DataFrame()
        errors = validate_property_file(df)
        assert len(errors) == 1
        assert "Missing columns" in errors[0]


# ===========================================================================
# 3. _weighted_return
# ===========================================================================

class TestWeightedReturn:
    def test_normal_case(self):
        df = pd.DataFrame({
            "Income_Return": [0.10, 0.06],
            "Denom": [100, 200],
        })
        # Expected: (0.10*100 + 0.06*200) / 300 = (10+12)/300 = 22/300
        result = _weighted_return(df, "Income_Return")
        assert result == pytest.approx(22.0 / 300.0)

    def test_zero_denom(self):
        df = pd.DataFrame({
            "Income_Return": [0.10, 0.06],
            "Denom": [0, 0],
        })
        assert _weighted_return(df, "Income_Return") == 0.0

    def test_single_property(self):
        df = pd.DataFrame({
            "Total_Return": [0.08],
            "Denom": [500],
        })
        assert _weighted_return(df, "Total_Return") == pytest.approx(0.08)

    def test_custom_weight_col(self):
        df = pd.DataFrame({
            "Income_Return": [0.10, 0.20],
            "CustomWeight": [1, 3],
        })
        expected = (0.10 * 1 + 0.20 * 3) / 4.0
        assert _weighted_return(df, "Income_Return", weight_col="CustomWeight") == pytest.approx(expected)

    def test_negative_returns(self):
        df = pd.DataFrame({
            "Capital_Return": [-0.05, 0.10],
            "Denom": [200, 300],
        })
        expected = (-0.05 * 200 + 0.10 * 300) / 500.0
        assert _weighted_return(df, "Capital_Return") == pytest.approx(expected)

    def test_all_weight_on_one_property(self):
        df = pd.DataFrame({
            "Income_Return": [0.10, 0.20],
            "Denom": [0, 500],
        })
        assert _weighted_return(df, "Income_Return") == pytest.approx(0.20)


# ===========================================================================
# 4. _aggregate_group
# ===========================================================================

class TestAggregateGroup:
    def setup_method(self):
        self.df = _two_type_two_cbsa_df()

    def test_dollar_columns_summed(self):
        row = _aggregate_group(self.df, "All", "All", "Total")
        for col in DOLLAR_COLS:
            assert row[col] == pytest.approx(self.df[col].sum()), f"{col} not summed correctly"

    def test_weighted_returns(self):
        row = _aggregate_group(self.df, "All", "All", "Total")
        total_denom = self.df["Denom"].sum()  # 100+200+300+400 = 1000
        for col in RETURN_COLS:
            expected = (self.df[col] * self.df["Denom"]).sum() / total_denom
            assert row[col] == pytest.approx(expected), f"{col} weighted return incorrect"

    def test_yyyyq_calculation(self):
        row = _aggregate_group(self.df, "All", "All", "Total")
        assert row["YYYYQ"] == 2024 * 10 + 1  # 20241

    def test_prop_count(self):
        row = _aggregate_group(self.df, "All", "All", "Total")
        assert row["Prop_Count"] == 4

    def test_query_level_assignment(self):
        row = _aggregate_group(self.df, "Office", "All", "By_PropertyType")
        assert row["Query_Level"] == "By_PropertyType"
        assert row["PropertyType"] == "Office"
        assert row["CBSAName"] == "All"

    def test_year_and_quarter(self):
        row = _aggregate_group(self.df, "All", "All", "Total")
        assert row["Year"] == 2024
        assert row["Quarter"] == 1

    def test_subset_group(self):
        """Aggregate only the Office rows."""
        office = self.df[self.df["PropertyType"] == "Office"]
        row = _aggregate_group(office, "Office", "All", "By_PropertyType")
        assert row["Prop_Count"] == 2
        assert row["NOI"] == pytest.approx(50 + 80)
        assert row["Denom"] == pytest.approx(100 + 200)
        expected_ir = (0.10 * 100 + 0.08 * 200) / 300.0
        assert row["Income_Return"] == pytest.approx(expected_ir)


# ===========================================================================
# 5. aggregate_properties -- end-to-end
# ===========================================================================

class TestAggregateProperties:
    # ---- Row counts -------------------------------------------------------

    def test_row_count_single_period_2x2(self):
        """2 types x 2 CBSAs in one period ->
        1 Total + 2 By_PT + 2 By_CBSA + 4 By_PT_CBSA = 9 rows.
        """
        df = _two_type_two_cbsa_df()
        result = aggregate_properties(df)
        assert len(result) == 9

    def test_row_count_multi_period(self):
        """Each period has 2 distinct types and 2 distinct CBSAs ->
        per period: 1 + 2 + 2 + 2 = 7 (since each type-cbsa combo is unique,
        By_PT_CBSA has 2 rows).  Two periods -> 14 rows.
        """
        df = _multi_period_df()
        result = aggregate_properties(df)
        # Period: 2 types, 2 CBSAs, each combo unique -> 1+2+2+2 = 7 per period
        assert len(result) == 14

    # ---- Dollar column sums -----------------------------------------------

    def test_total_dollar_sums(self):
        df = _two_type_two_cbsa_df()
        result = aggregate_properties(df)
        total_row = result[result["Query_Level"] == "Total"].iloc[0]
        for col in DOLLAR_COLS:
            assert total_row[col] == pytest.approx(df[col].sum()), f"Total {col} mismatch"

    def test_by_property_type_dollar_sums(self):
        df = _two_type_two_cbsa_df()
        result = aggregate_properties(df)
        pt_rows = result[result["Query_Level"] == "By_PropertyType"]
        for pt in ["Office", "Retail"]:
            agg_row = pt_rows[pt_rows["PropertyType"] == pt].iloc[0]
            src = df[df["PropertyType"] == pt]
            for col in DOLLAR_COLS:
                assert agg_row[col] == pytest.approx(src[col].sum()), f"By_PT {pt} {col} mismatch"

    def test_by_cbsa_dollar_sums(self):
        df = _two_type_two_cbsa_df()
        result = aggregate_properties(df)
        cbsa_rows = result[result["Query_Level"] == "By_CBSA"]
        for cbsa in ["New York", "Chicago"]:
            agg_row = cbsa_rows[cbsa_rows["CBSAName"] == cbsa].iloc[0]
            src = df[df["CBSAName"] == cbsa]
            for col in DOLLAR_COLS:
                assert agg_row[col] == pytest.approx(src[col].sum()), f"By_CBSA {cbsa} {col} mismatch"

    def test_by_property_type_cbsa_dollar_sums(self):
        df = _two_type_two_cbsa_df()
        result = aggregate_properties(df)
        cross_rows = result[result["Query_Level"] == "By_PropertyType_CBSA"]
        for pt in ["Office", "Retail"]:
            for cbsa in ["New York", "Chicago"]:
                agg_row = cross_rows[
                    (cross_rows["PropertyType"] == pt) & (cross_rows["CBSAName"] == cbsa)
                ].iloc[0]
                src = df[(df["PropertyType"] == pt) & (df["CBSAName"] == cbsa)]
                for col in DOLLAR_COLS:
                    assert agg_row[col] == pytest.approx(src[col].sum())

    # ---- Weighted returns --------------------------------------------------

    def test_total_weighted_returns(self):
        df = _two_type_two_cbsa_df()
        result = aggregate_properties(df)
        total_row = result[result["Query_Level"] == "Total"].iloc[0]
        total_denom = df["Denom"].sum()
        for col in RETURN_COLS:
            expected = (df[col] * df["Denom"]).sum() / total_denom
            assert total_row[col] == pytest.approx(expected), f"Total {col} weighted return mismatch"

    def test_by_property_type_weighted_returns(self):
        df = _two_type_two_cbsa_df()
        result = aggregate_properties(df)
        pt_rows = result[result["Query_Level"] == "By_PropertyType"]
        for pt in ["Office", "Retail"]:
            agg_row = pt_rows[pt_rows["PropertyType"] == pt].iloc[0]
            src = df[df["PropertyType"] == pt]
            td = src["Denom"].sum()
            for col in RETURN_COLS:
                expected = (src[col] * src["Denom"]).sum() / td
                assert agg_row[col] == pytest.approx(expected)

    # ---- Multiple periods --------------------------------------------------

    def test_multiple_periods_independent(self):
        df = _multi_period_df()
        result = aggregate_properties(df)
        for (year, qtr) in [(2024, 1), (2024, 2)]:
            total = result[
                (result["Query_Level"] == "Total") &
                (result["Year"] == year) &
                (result["Quarter"] == qtr)
            ]
            assert len(total) == 1
            src = df[(df["Year"] == year) & (df["Quarter"] == qtr)]
            for col in DOLLAR_COLS:
                assert total.iloc[0][col] == pytest.approx(src[col].sum())

    def test_yyyyq_across_periods(self):
        df = _multi_period_df()
        result = aggregate_properties(df)
        yyyyqs = result[result["Query_Level"] == "Total"]["YYYYQ"].tolist()
        assert sorted(yyyyqs) == [20241, 20242]

    # ---- Column order ------------------------------------------------------

    def test_column_order(self):
        df = _two_type_two_cbsa_df()
        result = aggregate_properties(df)
        expected_cols = [
            "Query_Level", "Year", "YYYYQ", "Quarter", "PropertyType", "CBSAName",
            "HeldSold",
            "NOI", "CapEx", "MV", "MVLag1", "PSales", "Denom",
            "Income_Return", "Capital_Return", "Total_Return", "Prop_Count",
        ]
        assert list(result.columns) == expected_cols

    # ---- Query_Level labels ------------------------------------------------

    def test_query_level_labels(self):
        df = _two_type_two_cbsa_df()
        result = aggregate_properties(df)
        levels = set(result["Query_Level"].unique())
        assert levels == {"Total", "By_PropertyType", "By_CBSA", "By_PropertyType_CBSA"}

    def test_total_row_has_all_all(self):
        df = _two_type_two_cbsa_df()
        result = aggregate_properties(df)
        total = result[result["Query_Level"] == "Total"].iloc[0]
        assert total["PropertyType"] == "All"
        assert total["CBSAName"] == "All"

    def test_by_pt_has_cbsa_all(self):
        df = _two_type_two_cbsa_df()
        result = aggregate_properties(df)
        pt_rows = result[result["Query_Level"] == "By_PropertyType"]
        assert (pt_rows["CBSAName"] == "All").all()

    def test_by_cbsa_has_pt_all(self):
        df = _two_type_two_cbsa_df()
        result = aggregate_properties(df)
        cbsa_rows = result[result["Query_Level"] == "By_CBSA"]
        assert (cbsa_rows["PropertyType"] == "All").all()


# ===========================================================================
# 6. Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_single_property(self):
        """One property should produce 4 rows: Total, By_PT, By_CBSA, By_PT_CBSA."""
        df = _make_property_df(n=1)
        result = aggregate_properties(df)
        assert len(result) == 4
        # All rows should have Prop_Count == 1
        assert (result["Prop_Count"] == 1).all()
        # Dollar columns identical across all rows (same single property)
        for col in DOLLAR_COLS:
            vals = result[col].unique()
            assert len(vals) == 1

    def test_all_same_type_and_cbsa(self):
        """Three properties with identical type and CBSA.
        Expect: 1 Total + 1 By_PT + 1 By_CBSA + 1 By_PT_CBSA = 4 rows.
        """
        rows = [_make_property_row() for _ in range(3)]
        df = pd.DataFrame(rows)
        result = aggregate_properties(df)
        assert len(result) == 4
        # Total and By_PT should have same dollar sums
        total = result[result["Query_Level"] == "Total"].iloc[0]
        by_pt = result[result["Query_Level"] == "By_PropertyType"].iloc[0]
        for col in DOLLAR_COLS:
            assert total[col] == pytest.approx(by_pt[col])

    def test_numeric_coercion_of_string_values(self):
        """Dollar and return columns provided as strings should be coerced."""
        rows = [
            _make_property_row(noi=100, denom=500, income_return=0.10,
                               capital_return=0.05, total_return=0.15),
            _make_property_row(noi=200, denom=300, income_return=0.08,
                               capital_return=0.04, total_return=0.12),
        ]
        df = pd.DataFrame(rows)
        # Convert numeric columns to strings
        for col in DOLLAR_COLS + RETURN_COLS:
            df[col] = df[col].astype(str)
        result = aggregate_properties(df)
        total = result[result["Query_Level"] == "Total"].iloc[0]
        assert total["NOI"] == pytest.approx(300.0)
        assert total["Denom"] == pytest.approx(800.0)
        expected_ir = (0.10 * 500 + 0.08 * 300) / 800.0
        assert total["Income_Return"] == pytest.approx(expected_ir)

    def test_non_numeric_string_coerced_to_zero(self):
        """Non-parseable strings in dollar/return columns become 0."""
        row = _make_property_row()
        row["NOI"] = "not_a_number"
        row["Income_Return"] = "bad"
        df = pd.DataFrame([row])
        result = aggregate_properties(df)
        total = result[result["Query_Level"] == "Total"].iloc[0]
        assert total["NOI"] == pytest.approx(0.0)
        assert total["Income_Return"] == pytest.approx(0.0)

    def test_many_property_types_one_cbsa(self):
        """5 property types, 1 CBSA ->
        1 Total + 5 By_PT + 1 By_CBSA + 5 By_PT_CBSA = 12.
        """
        types = ["Office", "Retail", "Industrial", "Hotel", "Apartment"]
        rows = [
            _make_property_row(property_type=t, cbsa_name="Denver", denom=100 * (i + 1),
                               income_return=0.05 + i * 0.01,
                               capital_return=0.03,
                               total_return=0.08 + i * 0.01)
            for i, t in enumerate(types)
        ]
        df = pd.DataFrame(rows)
        result = aggregate_properties(df)
        assert len(result) == 12
        assert len(result[result["Query_Level"] == "By_PropertyType"]) == 5
        assert len(result[result["Query_Level"] == "By_CBSA"]) == 1

    def test_prop_count_at_cross_level(self):
        """Verify Prop_Count is correct at the By_PropertyType_CBSA level."""
        rows = [
            _make_property_row(property_type="Office", cbsa_name="NY"),
            _make_property_row(property_type="Office", cbsa_name="NY"),
            _make_property_row(property_type="Office", cbsa_name="LA"),
            _make_property_row(property_type="Retail", cbsa_name="NY"),
        ]
        df = pd.DataFrame(rows)
        result = aggregate_properties(df)
        cross = result[result["Query_Level"] == "By_PropertyType_CBSA"]
        office_ny = cross[(cross["PropertyType"] == "Office") & (cross["CBSAName"] == "NY")]
        assert office_ny.iloc[0]["Prop_Count"] == 2
        office_la = cross[(cross["PropertyType"] == "Office") & (cross["CBSAName"] == "LA")]
        assert office_la.iloc[0]["Prop_Count"] == 1

    def test_dollar_sums_add_up_across_subtypes(self):
        """Sum of By_PropertyType dollar columns should equal Total."""
        df = _two_type_two_cbsa_df()
        result = aggregate_properties(df)
        total = result[result["Query_Level"] == "Total"].iloc[0]
        pt_rows = result[result["Query_Level"] == "By_PropertyType"]
        for col in DOLLAR_COLS:
            assert pt_rows[col].sum() == pytest.approx(total[col])

    def test_dollar_sums_add_up_across_cbsas(self):
        """Sum of By_CBSA dollar columns should equal Total."""
        df = _two_type_two_cbsa_df()
        result = aggregate_properties(df)
        total = result[result["Query_Level"] == "Total"].iloc[0]
        cbsa_rows = result[result["Query_Level"] == "By_CBSA"]
        for col in DOLLAR_COLS:
            assert cbsa_rows[col].sum() == pytest.approx(total[col])

    def test_input_dataframe_not_mutated(self):
        """aggregate_properties should not modify the original DataFrame."""
        df = _two_type_two_cbsa_df()
        original = df.copy()
        aggregate_properties(df)
        pd.testing.assert_frame_equal(df, original)
