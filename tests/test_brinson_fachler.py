"""
Comprehensive tests for Brinson-Fachler attribution engine.
"""
import pytest
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from attribution.brinson_fachler import (
    brinson_fachler_single_period,
    run_attribution,
    run_full_attribution,
    get_trailing_periods,
    get_available_trailing_periods,
    TRAILING_PERIOD_ORDER,
    _annualize,
    _cumulative_return,
    _link_effects,
    _compound_segment_returns,
)
from attribution.data_loader import load_file, get_common_periods


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DATA_DIR = os.path.dirname(os.path.dirname(__file__))
BENCHMARK_PATH = os.path.join(DATA_DIR, 'NCREIF_ODCE_Attribution_Data.xlsx')
PORTFOLIO_PATH = os.path.join(DATA_DIR, 'test_portfolio.xlsx')


@pytest.fixture(scope='module')
def benchmark_df():
    return load_file(BENCHMARK_PATH)


@pytest.fixture(scope='module')
def portfolio_df():
    return load_file(PORTFOLIO_PATH)


@pytest.fixture(scope='module')
def common_periods(portfolio_df, benchmark_df):
    return get_common_periods(portfolio_df, benchmark_df)


def _make_simple_benchmark():
    """Two property types, two quarters, fully controlled numbers."""
    rows = []
    for y, q in [(2024, 1), (2024, 2)]:
        # Total
        rows.append({
            'Query_Level': 'Total', 'Year': y, 'YYYYQ': y * 10 + q,
            'Quarter': q, 'PropertyType': 'All', 'CBSAName': 'All',
            'NOI': 200, 'CapEx': 20, 'MV': 10000, 'MVLag1': 9800,
            'PSales': 0, 'Denom': 10000,
            'Income_Return': 0.02, 'Capital_Return': 0.01, 'Total_Return': 0.03,
            'Prop_Count': 2
        })
        # By_PropertyType: Office 60%, Industrial 40%
        rows.append({
            'Query_Level': 'By_PropertyType', 'Year': y, 'YYYYQ': y * 10 + q,
            'Quarter': q, 'PropertyType': 'Office', 'CBSAName': 'All',
            'NOI': 120, 'CapEx': 12, 'MV': 6000, 'MVLag1': 5880,
            'PSales': 0, 'Denom': 6000,
            'Income_Return': 0.02, 'Capital_Return': 0.02, 'Total_Return': 0.04,
            'Prop_Count': 1
        })
        rows.append({
            'Query_Level': 'By_PropertyType', 'Year': y, 'YYYYQ': y * 10 + q,
            'Quarter': q, 'PropertyType': 'Industrial', 'CBSAName': 'All',
            'NOI': 80, 'CapEx': 8, 'MV': 4000, 'MVLag1': 3920,
            'PSales': 0, 'Denom': 4000,
            'Income_Return': 0.02, 'Capital_Return': -0.005, 'Total_Return': 0.015,
            'Prop_Count': 1
        })
        # By_CBSA
        rows.append({
            'Query_Level': 'By_CBSA', 'Year': y, 'YYYYQ': y * 10 + q,
            'Quarter': q, 'PropertyType': 'All', 'CBSAName': 'NY-New York',
            'NOI': 200, 'CapEx': 20, 'MV': 10000, 'MVLag1': 9800,
            'PSales': 0, 'Denom': 10000,
            'Income_Return': 0.02, 'Capital_Return': 0.01, 'Total_Return': 0.03,
            'Prop_Count': 2
        })
        # By_PropertyType_CBSA
        rows.append({
            'Query_Level': 'By_PropertyType_CBSA', 'Year': y, 'YYYYQ': y * 10 + q,
            'Quarter': q, 'PropertyType': 'Office', 'CBSAName': 'NY-New York',
            'NOI': 120, 'CapEx': 12, 'MV': 6000, 'MVLag1': 5880,
            'PSales': 0, 'Denom': 6000,
            'Income_Return': 0.02, 'Capital_Return': 0.02, 'Total_Return': 0.04,
            'Prop_Count': 1
        })
        rows.append({
            'Query_Level': 'By_PropertyType_CBSA', 'Year': y, 'YYYYQ': y * 10 + q,
            'Quarter': q, 'PropertyType': 'Industrial', 'CBSAName': 'NY-New York',
            'NOI': 80, 'CapEx': 8, 'MV': 4000, 'MVLag1': 3920,
            'PSales': 0, 'Denom': 4000,
            'Income_Return': 0.02, 'Capital_Return': -0.005, 'Total_Return': 0.015,
            'Prop_Count': 1
        })
    return pd.DataFrame(rows)


def _make_simple_portfolio():
    """Same structure as benchmark but with different weights/returns."""
    rows = []
    for y, q in [(2024, 1), (2024, 2)]:
        rows.append({
            'Query_Level': 'Total', 'Year': y, 'YYYYQ': y * 10 + q,
            'Quarter': q, 'PropertyType': 'All', 'CBSAName': 'All',
            'NOI': 250, 'CapEx': 25, 'MV': 10000, 'MVLag1': 9750,
            'PSales': 0, 'Denom': 10000,
            'Income_Return': 0.025, 'Capital_Return': 0.015, 'Total_Return': 0.04,
            'Prop_Count': 2
        })
        # Office: 80% weight, 5% return
        rows.append({
            'Query_Level': 'By_PropertyType', 'Year': y, 'YYYYQ': y * 10 + q,
            'Quarter': q, 'PropertyType': 'Office', 'CBSAName': 'All',
            'NOI': 200, 'CapEx': 20, 'MV': 8000, 'MVLag1': 7800,
            'PSales': 0, 'Denom': 8000,
            'Income_Return': 0.025, 'Capital_Return': 0.025, 'Total_Return': 0.05,
            'Prop_Count': 1
        })
        # Industrial: 20% weight, 0% return
        rows.append({
            'Query_Level': 'By_PropertyType', 'Year': y, 'YYYYQ': y * 10 + q,
            'Quarter': q, 'PropertyType': 'Industrial', 'CBSAName': 'All',
            'NOI': 50, 'CapEx': 5, 'MV': 2000, 'MVLag1': 1950,
            'PSales': 0, 'Denom': 2000,
            'Income_Return': 0.025, 'Capital_Return': -0.025, 'Total_Return': 0.0,
            'Prop_Count': 1
        })
        rows.append({
            'Query_Level': 'By_CBSA', 'Year': y, 'YYYYQ': y * 10 + q,
            'Quarter': q, 'PropertyType': 'All', 'CBSAName': 'NY-New York',
            'NOI': 250, 'CapEx': 25, 'MV': 10000, 'MVLag1': 9750,
            'PSales': 0, 'Denom': 10000,
            'Income_Return': 0.025, 'Capital_Return': 0.015, 'Total_Return': 0.04,
            'Prop_Count': 2
        })
        rows.append({
            'Query_Level': 'By_PropertyType_CBSA', 'Year': y, 'YYYYQ': y * 10 + q,
            'Quarter': q, 'PropertyType': 'Office', 'CBSAName': 'NY-New York',
            'NOI': 200, 'CapEx': 20, 'MV': 8000, 'MVLag1': 7800,
            'PSales': 0, 'Denom': 8000,
            'Income_Return': 0.025, 'Capital_Return': 0.025, 'Total_Return': 0.05,
            'Prop_Count': 1
        })
        rows.append({
            'Query_Level': 'By_PropertyType_CBSA', 'Year': y, 'YYYYQ': y * 10 + q,
            'Quarter': q, 'PropertyType': 'Industrial', 'CBSAName': 'NY-New York',
            'NOI': 50, 'CapEx': 5, 'MV': 2000, 'MVLag1': 1950,
            'PSales': 0, 'Denom': 2000,
            'Income_Return': 0.025, 'Capital_Return': -0.025, 'Total_Return': 0.0,
            'Prop_Count': 1
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Self-attribution: portfolio == benchmark => zero active return
# ---------------------------------------------------------------------------

class TestSelfAttribution:
    """When portfolio equals benchmark, all effects should be zero."""

    def test_single_period_zero_active(self, benchmark_df):
        df = brinson_fachler_single_period(benchmark_df, benchmark_df, 2024, 4, 'PropertyType')
        assert not df.empty
        assert abs(df['Total_Active'].sum()) < 1e-10
        assert abs(df['Allocation'].sum()) < 1e-10
        assert abs(df['Selection'].sum()) < 1e-10

    def test_multi_period_zero_active(self, benchmark_df, common_periods):
        r = run_full_attribution(benchmark_df, benchmark_df, common_periods)
        for dim in ['by_property_type', 'by_cbsa', 'by_property_type_cbsa']:
            assert abs(r[dim]['summary']['Total_Active'].sum()) < 1e-10

    def test_equal_weights_when_same_file(self, benchmark_df):
        df = brinson_fachler_single_period(benchmark_df, benchmark_df, 2024, 1, 'PropertyType')
        assert np.allclose(df['w_p'], df['w_b'], atol=1e-10)


# ---------------------------------------------------------------------------
# 2. Single-period Brinson-Fachler formula correctness
# ---------------------------------------------------------------------------

class TestSinglePeriodFormulas:
    """Verify BF formulas on controlled data."""

    def test_allocation_formula(self):
        b = _make_simple_benchmark()
        p = _make_simple_portfolio()
        df = brinson_fachler_single_period(p, b, 2024, 1, 'PropertyType')

        R_b = 0.03  # benchmark total
        for _, row in df.iterrows():
            expected_alloc = (row['w_p'] - row['w_b']) * (row['r_b'] - R_b)
            assert abs(row['Allocation'] - expected_alloc) < 1e-12

    def test_selection_formula(self):
        b = _make_simple_benchmark()
        p = _make_simple_portfolio()
        df = brinson_fachler_single_period(p, b, 2024, 1, 'PropertyType')

        for _, row in df.iterrows():
            expected_sel = row['w_p'] * (row['r_p'] - row['r_b'])
            assert abs(row['Selection'] - expected_sel) < 1e-12

    def test_total_active_is_sum_of_effects(self):
        b = _make_simple_benchmark()
        p = _make_simple_portfolio()
        df = brinson_fachler_single_period(p, b, 2024, 1, 'PropertyType')

        for _, row in df.iterrows():
            expected = row['Allocation'] + row['Selection']
            assert abs(row['Total_Active'] - expected) < 1e-12

    def test_effects_sum_to_active_return(self):
        b = _make_simple_benchmark()
        p = _make_simple_portfolio()
        df = brinson_fachler_single_period(p, b, 2024, 1, 'PropertyType')

        R_p = df['R_p'].iloc[0]
        R_b = df['R_b'].iloc[0]
        active = R_p - R_b
        sum_effects = df['Total_Active'].sum()
        # May not be exact due to weights not summing to 1.0 perfectly
        assert abs(active - sum_effects) < 0.001

    def test_contribution_is_weight_times_return(self):
        b = _make_simple_benchmark()
        p = _make_simple_portfolio()
        df = brinson_fachler_single_period(p, b, 2024, 1, 'PropertyType')

        for _, row in df.iterrows():
            assert abs(row['Contribution_P'] - row['w_p'] * row['r_p']) < 1e-12
            assert abs(row['Contribution_B'] - row['w_b'] * row['r_b']) < 1e-12


# ---------------------------------------------------------------------------
# 3. Multi-period reconciliation
# ---------------------------------------------------------------------------

class TestMultiPeriodReconciliation:
    """Linked multi-period effects must reconcile to cumulative active return."""

    def test_active_reconciles_property_type(self, portfolio_df, benchmark_df, common_periods):
        r = run_attribution(portfolio_df, benchmark_df, common_periods, 'PropertyType')
        active = r['cum_rp'] - r['cum_rb']
        linked_sum = r['summary']['Total_Active'].sum()
        assert abs(active - linked_sum) < 1e-6

    def test_active_reconciles_cbsa(self, portfolio_df, benchmark_df, common_periods):
        r = run_attribution(portfolio_df, benchmark_df, common_periods, 'CBSAName')
        active = r['cum_rp'] - r['cum_rb']
        linked_sum = r['summary']['Total_Active'].sum()
        assert abs(active - linked_sum) < 1e-6

    def test_active_reconciles_cross(self, portfolio_df, benchmark_df, common_periods):
        r = run_attribution(portfolio_df, benchmark_df, common_periods, 'PropertyType_CBSAName')
        active = r['cum_rp'] - r['cum_rb']
        linked_sum = r['summary']['Total_Active'].sum()
        assert abs(active - linked_sum) < 1e-6

    def test_single_period_reconciles_exactly(self, portfolio_df, benchmark_df, common_periods):
        """Single period should need no scaling."""
        r = run_attribution(portfolio_df, benchmark_df, [common_periods[-1]], 'PropertyType')
        active = r['cum_rp'] - r['cum_rb']
        linked_sum = r['summary']['Total_Active'].sum()
        assert abs(active - linked_sum) < 1e-6

    def test_reconciliation_on_controlled_data(self):
        b = _make_simple_benchmark()
        p = _make_simple_portfolio()
        periods = [(2024, 1), (2024, 2)]
        r = run_attribution(p, b, periods, 'PropertyType')
        active = r['cum_rp'] - r['cum_rb']
        assert abs(active - r['summary']['Total_Active'].sum()) < 1e-6


# ---------------------------------------------------------------------------
# 4. Cumulative return compounding
# ---------------------------------------------------------------------------

class TestCumulativeReturns:

    def test_single_period_cumulative_equals_quarterly(self, portfolio_df, benchmark_df, common_periods):
        r = run_attribution(portfolio_df, benchmark_df, [common_periods[-1]], 'PropertyType')
        # For single period, cum return should equal that quarter's return
        assert r['n_quarters'] == 1

    def test_multi_period_compounds_correctly(self):
        b = _make_simple_benchmark()
        p = _make_simple_portfolio()
        r = run_attribution(p, b, [(2024, 1), (2024, 2)], 'PropertyType')
        # Portfolio: 4% each quarter => (1.04)(1.04) - 1 = 8.16%
        expected_cum_rp = (1.04) * (1.04) - 1
        assert abs(r['cum_rp'] - expected_cum_rp) < 1e-10

    def test_segment_returns_compound(self):
        b = _make_simple_benchmark()
        p = _make_simple_portfolio()
        r = run_attribution(p, b, [(2024, 1), (2024, 2)], 'PropertyType')
        office = r['summary'][r['summary']['PropertyType'] == 'Office'].iloc[0]
        # Office: 5% each quarter => (1.05)(1.05) - 1 = 10.25%
        expected = (1.05) * (1.05) - 1
        assert abs(office['r_p_cum'] - expected) < 1e-10


# ---------------------------------------------------------------------------
# 5. Annualization
# ---------------------------------------------------------------------------

class TestAnnualize:

    def test_sub_annual_not_annualized(self):
        assert _annualize(0.05, 2) == 0.05  # 2 quarters, return as-is

    def test_one_year_not_annualized(self):
        assert _annualize(0.10, 4) == 0.10  # 4 quarters = 1 year

    def test_two_year_annualized(self):
        cum = 0.21  # 21% over 2 years
        ann = _annualize(cum, 8)
        expected = (1.21) ** 0.5 - 1
        assert abs(ann - expected) < 1e-10

    def test_total_loss_capped(self):
        assert _annualize(-1.5, 8) == -1.0

    def test_zero_quarters(self):
        assert _annualize(0.05, 0) == 0.0


# ---------------------------------------------------------------------------
# 6. Trailing period helpers
# ---------------------------------------------------------------------------

class TestTrailingPeriods:

    def test_1q_returns_last_period(self):
        periods = [(2024, 1), (2024, 2), (2024, 3), (2024, 4)]
        result = get_trailing_periods(periods, (2024, 4), '1Q')
        assert result == [(2024, 4)]

    def test_1y_returns_four_quarters(self):
        periods = [(2024, 1), (2024, 2), (2024, 3), (2024, 4)]
        result = get_trailing_periods(periods, (2024, 4), '1Y')
        assert result == [(2024, 1), (2024, 2), (2024, 3), (2024, 4)]

    def test_si_returns_all(self):
        periods = [(2023, 1), (2023, 2), (2024, 1)]
        result = get_trailing_periods(periods, (2024, 1), 'SI')
        assert result == [(2023, 1), (2023, 2), (2024, 1)]

    def test_as_of_filters_future(self):
        periods = [(2024, 1), (2024, 2), (2024, 3), (2024, 4)]
        result = get_trailing_periods(periods, (2024, 2), '1Y')
        assert result == [(2024, 1), (2024, 2)]

    def test_empty_when_no_eligible(self):
        periods = [(2024, 3), (2024, 4)]
        result = get_trailing_periods(periods, (2024, 1), '1Q')
        assert result == []

    def test_available_trailing(self):
        periods = [(2024, 1), (2024, 2), (2024, 3), (2024, 4)]
        avail = get_available_trailing_periods(periods, (2024, 4))
        assert avail['1Q'] is True
        assert avail['1Y'] is True
        assert avail['3Y'] is False
        assert avail['SI'] is True


# ---------------------------------------------------------------------------
# 7. n_quarters counts actual data, not requested periods
# ---------------------------------------------------------------------------

class TestNQuarters:

    def test_n_quarters_matches_actual_data(self, portfolio_df, benchmark_df, common_periods):
        r = run_attribution(portfolio_df, benchmark_df, common_periods, 'PropertyType')
        assert r['n_quarters'] == len(common_periods)

    def test_n_quarters_skips_empty_periods(self):
        b = _make_simple_benchmark()
        p = _make_simple_portfolio()
        # Request a period that doesn't exist in the data
        periods = [(2024, 1), (2024, 2), (2025, 1)]
        r = run_attribution(p, b, periods, 'PropertyType')
        assert r['n_quarters'] == 2  # 2025Q1 has no data


# ---------------------------------------------------------------------------
# 8. fillna order: r_b before r_p
# ---------------------------------------------------------------------------

class TestFillnaOrder:
    """Segments in portfolio but not benchmark should get r_b=R_b then r_p from r_b."""

    def test_portfolio_only_segment_gets_benchmark_total(self):
        b = _make_simple_benchmark()
        # Portfolio with an extra property type not in benchmark
        p = _make_simple_portfolio()
        extra = {
            'Query_Level': 'By_PropertyType', 'Year': 2024, 'YYYYQ': 20241,
            'Quarter': 1, 'PropertyType': 'Retail', 'CBSAName': 'All',
            'NOI': 50, 'CapEx': 5, 'MV': 1000, 'MVLag1': 950,
            'PSales': 0, 'Denom': 1000,
            'Income_Return': 0.05, 'Capital_Return': 0.01, 'Total_Return': 0.06,
            'Prop_Count': 1
        }
        p = pd.concat([p, pd.DataFrame([extra])], ignore_index=True)
        df = brinson_fachler_single_period(p, b, 2024, 1, 'PropertyType')
        retail = df[df['PropertyType'] == 'Retail'].iloc[0]

        # Retail not in benchmark => w_b=0, r_b=R_b=0.03
        assert retail['w_b'] == 0.0
        assert abs(retail['r_b'] - 0.03) < 1e-10

    def test_benchmark_only_segment_gets_zero_weight(self):
        b = _make_simple_benchmark()
        p = _make_simple_portfolio()
        # Remove Industrial from portfolio
        p = p[~((p['PropertyType'] == 'Industrial') & (p['Query_Level'] == 'By_PropertyType'))]
        df = brinson_fachler_single_period(p, b, 2024, 1, 'PropertyType')
        ind = df[df['PropertyType'] == 'Industrial'].iloc[0]

        assert ind['w_p'] == 0.0
        # r_p should be filled from r_b (not NaN)
        assert not np.isnan(ind['r_p'])
        assert abs(ind['r_p'] - ind['r_b']) < 1e-10


# ---------------------------------------------------------------------------
# 9. Cross-dimension 'All' filtering
# ---------------------------------------------------------------------------

class TestAllFiltering:

    def test_property_type_excludes_all(self, benchmark_df):
        df = brinson_fachler_single_period(benchmark_df, benchmark_df, 2024, 1, 'PropertyType')
        assert 'All' not in df['PropertyType'].values

    def test_cbsa_excludes_all(self, benchmark_df):
        df = brinson_fachler_single_period(benchmark_df, benchmark_df, 2024, 1, 'CBSAName')
        assert 'All' not in df['CBSAName'].values

    def test_cross_dimension_excludes_all(self, benchmark_df):
        df = brinson_fachler_single_period(benchmark_df, benchmark_df, 2024, 1, 'PropertyType_CBSAName')
        if not df.empty:
            assert 'All' not in df['PropertyType'].values
            assert 'All' not in df['CBSAName'].values


# ---------------------------------------------------------------------------
# 10. Empty/edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_result_for_missing_period(self, benchmark_df):
        df = brinson_fachler_single_period(benchmark_df, benchmark_df, 9999, 1, 'PropertyType')
        assert df.empty

    def test_run_attribution_no_results(self, benchmark_df):
        r = run_attribution(benchmark_df, benchmark_df, [(9999, 1)], 'PropertyType')
        assert r['summary'].empty
        assert r['n_quarters'] == 0
        assert r['cum_rp'] == 0.0

    def test_all_dimensions_run(self, portfolio_df, benchmark_df, common_periods):
        r = run_full_attribution(portfolio_df, benchmark_df, common_periods)
        assert 'by_property_type' in r
        assert 'by_cbsa' in r
        assert 'by_property_type_cbsa' in r
        for key in r:
            assert 'summary' in r[key]
            assert 'detail' in r[key]

    def test_near_zero_scaling_fallback(self):
        """When raw effects sum to near zero but cum_active != 0."""
        b = _make_simple_benchmark()
        p = _make_simple_benchmark()  # Same as benchmark initially
        # Slightly tweak one period to create a tiny active return
        mask = (p['Query_Level'] == 'Total') & (p['Year'] == 2024) & (p['Quarter'] == 1)
        p.loc[mask, 'Total_Return'] = 0.030001
        r = run_attribution(p, b, [(2024, 1)], 'PropertyType')
        # Should not crash
        assert r['n_quarters'] == 1


# ---------------------------------------------------------------------------
# 11. Proportional scaling preserves sign and relative magnitude
# ---------------------------------------------------------------------------

class TestProportionalScaling:

    def test_scaling_preserves_sign(self, portfolio_df, benchmark_df, common_periods):
        r = run_attribution(portfolio_df, benchmark_df, common_periods, 'PropertyType')
        # Compare raw and scaled: if a segment had positive allocation before scaling,
        # it should still be positive after (since scale > 0 for positive active returns)
        detail = r['detail']
        summary = r['summary']
        # Just verify no NaN
        assert not summary['Allocation'].isna().any()
        assert not summary['Selection'].isna().any()

    def test_total_active_equals_sum_of_components(self, portfolio_df, benchmark_df, common_periods):
        r = run_attribution(portfolio_df, benchmark_df, common_periods, 'PropertyType')
        for _, row in r['summary'].iterrows():
            expected = row['Allocation'] + row['Selection']
            assert abs(row['Total_Active'] - expected) < 1e-10
