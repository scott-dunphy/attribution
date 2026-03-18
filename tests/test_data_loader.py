"""Comprehensive tests for attribution.data_loader module."""

import pandas as pd
import pytest

from attribution.data_loader import (
    REQUIRED_COLUMNS,
    VALID_QUERY_LEVELS,
    get_available_periods,
    get_common_periods,
    load_aggregated_file,
    load_file,
    load_property_file,
    validate_schema,
)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _make_aggregated_row(
    query_level="Total",
    year=2025,
    quarter=1,
    property_type="All",
    cbsa_name="All",
    **overrides,
):
    """Return a dict representing one valid aggregated row."""
    row = {
        "Query_Level": query_level,
        "Year": year,
        "YYYYQ": year * 10 + quarter,
        "Quarter": quarter,
        "PropertyType": property_type,
        "CBSAName": cbsa_name,
        "NOI": 100.0,
        "CapEx": 10.0,
        "MV": 1000.0,
        "MVLag1": 950.0,
        "PSales": 0.0,
        "Denom": 980.0,
        "Income_Return": 0.01,
        "Capital_Return": 0.02,
        "Total_Return": 0.03,
        "Prop_Count": 5,
    }
    row.update(overrides)
    return row


def _make_aggregated_df(rows=None):
    """Build a DataFrame with valid aggregated data.

    If *rows* is None a minimal valid dataframe with one Total row is returned.
    """
    if rows is None:
        rows = [_make_aggregated_row()]
    return pd.DataFrame(rows)


def _make_property_row(
    year=2025,
    quarter=1,
    property_type="Office",
    cbsa_name="New York",
    **overrides,
):
    """Return a dict representing one property-level row (no Query_Level)."""
    row = {
        "Year": year,
        "Quarter": quarter,
        "PropertyType": property_type,
        "CBSAName": cbsa_name,
        "NOI": 50.0,
        "CapEx": 5.0,
        "MV": 500.0,
        "MVLag1": 475.0,
        "PSales": 0.0,
        "Denom": 490.0,
        "Income_Return": 0.01,
        "Capital_Return": 0.02,
        "Total_Return": 0.03,
    }
    row.update(overrides)
    return row


def _make_property_df(rows=None):
    """Build a property-level DataFrame (no Query_Level column)."""
    if rows is None:
        rows = [_make_property_row(), _make_property_row(cbsa_name="Boston")]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# validate_schema
# ---------------------------------------------------------------------------

class TestValidateSchema:
    """Tests for validate_schema()."""

    def test_valid_data_returns_no_errors(self):
        df = _make_aggregated_df()
        assert validate_schema(df) == []

    def test_valid_data_multiple_query_levels(self):
        rows = [
            _make_aggregated_row(query_level="Total"),
            _make_aggregated_row(query_level="By_CBSA", cbsa_name="Boston"),
            _make_aggregated_row(query_level="By_PropertyType", property_type="Office"),
            _make_aggregated_row(query_level="By_PropertyType_CBSA", property_type="Office", cbsa_name="Boston"),
        ]
        df = _make_aggregated_df(rows)
        assert validate_schema(df) == []

    def test_missing_single_column(self):
        df = _make_aggregated_df()
        df = df.drop(columns=["NOI"])
        errors = validate_schema(df)
        assert len(errors) == 1
        assert "Missing columns" in errors[0]
        assert "NOI" in errors[0]

    def test_missing_multiple_columns(self):
        df = _make_aggregated_df()
        df = df.drop(columns=["NOI", "MV", "Year"])
        errors = validate_schema(df)
        assert len(errors) == 1
        assert "Missing columns" in errors[0]
        for col in ["NOI", "MV", "Year"]:
            assert col in errors[0]

    def test_missing_columns_returns_early(self):
        """When columns are missing the function returns immediately without
        checking Query_Level or Total presence."""
        df = _make_aggregated_df()
        df = df.drop(columns=["Query_Level"])
        errors = validate_schema(df)
        # Only the missing-column error, not Query_Level or Total errors
        assert len(errors) == 1
        assert "Missing columns" in errors[0]

    def test_unexpected_query_level(self):
        rows = [
            _make_aggregated_row(query_level="Total"),
            _make_aggregated_row(query_level="Invalid_Level"),
        ]
        df = _make_aggregated_df(rows)
        errors = validate_schema(df)
        assert any("Unexpected Query_Level" in e for e in errors)
        assert any("Invalid_Level" in e for e in errors)

    def test_multiple_unexpected_query_levels(self):
        rows = [
            _make_aggregated_row(query_level="Total"),
            _make_aggregated_row(query_level="Bad1"),
            _make_aggregated_row(query_level="Bad2"),
        ]
        df = _make_aggregated_df(rows)
        errors = validate_schema(df)
        unexpected_error = [e for e in errors if "Unexpected" in e][0]
        assert "Bad1" in unexpected_error
        assert "Bad2" in unexpected_error

    def test_missing_total_rows(self):
        rows = [
            _make_aggregated_row(query_level="By_CBSA", cbsa_name="Boston"),
            _make_aggregated_row(query_level="By_PropertyType", property_type="Office"),
        ]
        df = _make_aggregated_df(rows)
        errors = validate_schema(df)
        assert any("No 'Total' rows" in e for e in errors)

    def test_unexpected_level_and_missing_total(self):
        """Both errors can be reported simultaneously."""
        rows = [
            _make_aggregated_row(query_level="By_CBSA"),
            _make_aggregated_row(query_level="Bogus"),
        ]
        df = _make_aggregated_df(rows)
        errors = validate_schema(df)
        assert len(errors) == 2
        assert any("Unexpected" in e for e in errors)
        assert any("Total" in e for e in errors)

    def test_nan_query_levels_ignored(self):
        """NaN Query_Level values are dropped before checking."""
        rows = [
            _make_aggregated_row(query_level="Total"),
            _make_aggregated_row(query_level=None),
        ]
        df = _make_aggregated_df(rows)
        assert validate_schema(df) == []


# ---------------------------------------------------------------------------
# get_available_periods
# ---------------------------------------------------------------------------

class TestGetAvailablePeriods:
    """Tests for get_available_periods()."""

    def test_single_period(self):
        df = _make_aggregated_df()
        periods = get_available_periods(df)
        assert periods == [(2025, 1)]

    def test_multiple_periods_sorted(self):
        rows = [
            _make_aggregated_row(year=2025, quarter=3),
            _make_aggregated_row(year=2024, quarter=4),
            _make_aggregated_row(year=2025, quarter=1),
        ]
        df = _make_aggregated_df(rows)
        periods = get_available_periods(df)
        assert periods == [(2024, 4), (2025, 1), (2025, 3)]

    def test_only_total_rows_counted(self):
        """Non-Total rows should not contribute to the period list."""
        rows = [
            _make_aggregated_row(query_level="Total", year=2025, quarter=1),
            _make_aggregated_row(query_level="By_CBSA", year=2025, quarter=2),
        ]
        df = _make_aggregated_df(rows)
        periods = get_available_periods(df)
        assert periods == [(2025, 1)]

    def test_duplicate_periods_deduplicated(self):
        rows = [
            _make_aggregated_row(year=2025, quarter=1),
            _make_aggregated_row(year=2025, quarter=1),
        ]
        df = _make_aggregated_df(rows)
        periods = get_available_periods(df)
        assert periods == [(2025, 1)]

    def test_empty_total_rows(self):
        """If there are no Total rows the result is empty."""
        rows = [_make_aggregated_row(query_level="By_CBSA")]
        df = _make_aggregated_df(rows)
        assert get_available_periods(df) == []


# ---------------------------------------------------------------------------
# get_common_periods
# ---------------------------------------------------------------------------

class TestGetCommonPeriods:
    """Tests for get_common_periods()."""

    def test_overlapping_periods(self):
        port = _make_aggregated_df([
            _make_aggregated_row(year=2024, quarter=4),
            _make_aggregated_row(year=2025, quarter=1),
            _make_aggregated_row(year=2025, quarter=2),
        ])
        bench = _make_aggregated_df([
            _make_aggregated_row(year=2025, quarter=1),
            _make_aggregated_row(year=2025, quarter=2),
            _make_aggregated_row(year=2025, quarter=3),
        ])
        common = get_common_periods(port, bench)
        assert common == [(2025, 1), (2025, 2)]

    def test_no_overlap(self):
        port = _make_aggregated_df([
            _make_aggregated_row(year=2024, quarter=1),
        ])
        bench = _make_aggregated_df([
            _make_aggregated_row(year=2025, quarter=1),
        ])
        assert get_common_periods(port, bench) == []

    def test_identical_periods(self):
        rows = [
            _make_aggregated_row(year=2025, quarter=1),
            _make_aggregated_row(year=2025, quarter=2),
        ]
        port = _make_aggregated_df(rows)
        bench = _make_aggregated_df(rows)
        common = get_common_periods(port, bench)
        assert common == [(2025, 1), (2025, 2)]

    def test_result_is_sorted(self):
        port = _make_aggregated_df([
            _make_aggregated_row(year=2025, quarter=3),
            _make_aggregated_row(year=2024, quarter=4),
        ])
        bench = _make_aggregated_df([
            _make_aggregated_row(year=2025, quarter=3),
            _make_aggregated_row(year=2024, quarter=4),
        ])
        common = get_common_periods(port, bench)
        assert common == [(2024, 4), (2025, 3)]


# ---------------------------------------------------------------------------
# load_aggregated_file
# ---------------------------------------------------------------------------

class TestLoadAggregatedFile:
    """Tests for load_aggregated_file()."""

    def test_returns_dataframe(self):
        df = _make_aggregated_df()
        result = load_aggregated_file(df)
        assert isinstance(result, pd.DataFrame)

    def test_year_quarter_yyyyq_cast_to_int(self):
        rows = [_make_aggregated_row(year="2025", quarter="1", YYYYQ="20251")]
        df = _make_aggregated_df(rows)
        result = load_aggregated_file(df)
        assert result["Year"].dtype == int
        assert result["Quarter"].dtype == int
        assert result["YYYYQ"].dtype == int

    def test_numeric_columns_coerced(self):
        row = _make_aggregated_row(NOI="abc", MV="200")
        df = _make_aggregated_df([row])
        result = load_aggregated_file(df)
        # Non-numeric "abc" should become 0.0
        assert result["NOI"].iloc[0] == 0.0
        # Valid numeric string should parse
        assert result["MV"].iloc[0] == 200.0

    def test_prop_count_cast_to_int(self):
        row = _make_aggregated_row(Prop_Count="7")
        df = _make_aggregated_df([row])
        result = load_aggregated_file(df)
        assert result["Prop_Count"].dtype == int
        assert result["Prop_Count"].iloc[0] == 7

    def test_prop_count_nan_coerced_to_zero(self):
        row = _make_aggregated_row(Prop_Count="bad")
        df = _make_aggregated_df([row])
        result = load_aggregated_file(df)
        assert result["Prop_Count"].iloc[0] == 0

    def test_raises_on_schema_error(self):
        df = _make_aggregated_df()
        df = df.drop(columns=["NOI"])
        with pytest.raises(ValueError, match="Missing columns"):
            load_aggregated_file(df)

    def test_raises_on_missing_total(self):
        rows = [_make_aggregated_row(query_level="By_CBSA")]
        df = _make_aggregated_df(rows)
        with pytest.raises(ValueError, match="Total"):
            load_aggregated_file(df)


# ---------------------------------------------------------------------------
# load_property_file
# ---------------------------------------------------------------------------

class TestLoadPropertyFile:
    """Tests for load_property_file()."""

    def test_successful_aggregation(self):
        df = _make_property_df()
        result = load_property_file(df)
        assert isinstance(result, pd.DataFrame)
        # Should contain Total rows after aggregation
        assert "Query_Level" in result.columns
        assert "Total" in result["Query_Level"].values

    def test_aggregation_produces_all_query_levels(self):
        df = _make_property_df()
        result = load_property_file(df)
        levels = set(result["Query_Level"].unique())
        # Without a Sold column, By_HeldSold won't be produced
        expected = VALID_QUERY_LEVELS - {'By_HeldSold'}
        assert levels == expected

    def test_raises_on_missing_columns(self):
        df = _make_property_df()
        df = df.drop(columns=["NOI"])
        with pytest.raises(ValueError, match="Missing columns"):
            load_property_file(df)

    def test_numeric_coercion_before_aggregation(self):
        rows = [
            _make_property_row(NOI="bad_value", Denom=100.0),
            _make_property_row(NOI="50", Denom=100.0),
        ]
        df = _make_property_df(rows)
        result = load_property_file(df)
        # "bad_value" becomes 0.0, "50" becomes 50.0 => total NOI = 50
        total_row = result[result["Query_Level"] == "Total"]
        assert total_row["NOI"].iloc[0] == 50.0

    def test_prop_count_reflects_property_count(self):
        rows = [
            _make_property_row(cbsa_name="New York"),
            _make_property_row(cbsa_name="Boston"),
            _make_property_row(cbsa_name="Chicago"),
        ]
        df = _make_property_df(rows)
        result = load_property_file(df)
        total_row = result[result["Query_Level"] == "Total"]
        assert total_row["Prop_Count"].iloc[0] == 3


# ---------------------------------------------------------------------------
# load_file (auto-detection via Excel round-trip)
# ---------------------------------------------------------------------------

class TestLoadFile:
    """Tests for load_file() auto-detection logic using temporary .xlsx files."""

    def test_loads_aggregated_file(self, tmp_path):
        """A file with a populated Query_Level column is treated as aggregated."""
        filepath = tmp_path / "aggregated.xlsx"
        df = _make_aggregated_df()
        df.to_excel(filepath, index=False)

        result = load_file(filepath)
        assert isinstance(result, pd.DataFrame)
        assert "Query_Level" in result.columns
        assert result["Query_Level"].iloc[0] == "Total"

    def test_loads_property_file(self, tmp_path):
        """A file without Query_Level is treated as property-level."""
        filepath = tmp_path / "property.xlsx"
        df = _make_property_df()
        df.to_excel(filepath, index=False)

        result = load_file(filepath)
        assert isinstance(result, pd.DataFrame)
        # After aggregation the result should contain Query_Level
        assert "Query_Level" in result.columns
        levels = set(result["Query_Level"].unique())
        assert "Total" in levels

    def test_property_file_with_nan_query_level(self, tmp_path):
        """A file with Query_Level column but all NaN values is property-level."""
        filepath = tmp_path / "property_nan_ql.xlsx"
        df = _make_property_df()
        df["Query_Level"] = None
        df.to_excel(filepath, index=False)

        result = load_file(filepath)
        # Should be treated as property-level and aggregated
        assert "Total" in result["Query_Level"].values

    def test_aggregated_file_type_casting(self, tmp_path):
        """Verify numeric columns are properly cast after loading from Excel."""
        filepath = tmp_path / "agg_types.xlsx"
        df = _make_aggregated_df()
        df.to_excel(filepath, index=False)

        result = load_file(filepath)
        assert result["Year"].dtype == int
        assert result["Quarter"].dtype == int
        assert result["Prop_Count"].dtype == int

    def test_raises_on_invalid_aggregated_file(self, tmp_path):
        """An aggregated file missing required columns raises ValueError."""
        filepath = tmp_path / "bad_agg.xlsx"
        df = _make_aggregated_df()
        df = df.drop(columns=["NOI", "MV"])
        df.to_excel(filepath, index=False)

        with pytest.raises(ValueError, match="Missing columns"):
            load_file(filepath)

    def test_raises_on_invalid_property_file(self, tmp_path):
        """A property file missing required columns raises ValueError."""
        filepath = tmp_path / "bad_prop.xlsx"
        df = _make_property_df()
        df = df.drop(columns=["NOI"])
        df.to_excel(filepath, index=False)

        with pytest.raises(ValueError, match="Missing columns"):
            load_file(filepath)
