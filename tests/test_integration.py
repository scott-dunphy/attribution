"""
Integration tests using Test_Portfolio_Full.xlsx (20 properties, 12 quarters)
against the NCREIF ODCE benchmark. Exercises the full pipeline end-to-end.
"""
import pytest
import pandas as pd
import numpy as np
import os

from attribution.data_loader import load_file, get_common_periods
from attribution.aggregator import is_property_level, aggregate_properties
from attribution.brinson_fachler import (
    brinson_fachler_single_period,
    run_attribution,
    run_full_attribution,
    run_all_trailing_periods,
    get_trailing_periods,
    get_available_trailing_periods,
    TRAILING_PERIOD_ORDER,
    TRAILING_PERIOD_DEFS,
    _annualize,
)

DATA_DIR = os.path.dirname(os.path.dirname(__file__))
BENCHMARK_PATH = os.path.join(DATA_DIR, 'NCREIF_ODCE_Attribution_Data.xlsx')
FULL_PORTFOLIO_PATH = os.path.join(DATA_DIR, 'Test_Portfolio_Full.xlsx')


@pytest.fixture(scope='module')
def raw_portfolio():
    """Raw property-level DataFrame before aggregation."""
    return pd.read_excel(FULL_PORTFOLIO_PATH)


@pytest.fixture(scope='module')
def benchmark_df():
    return load_file(BENCHMARK_PATH)


@pytest.fixture(scope='module')
def portfolio_df():
    return load_file(FULL_PORTFOLIO_PATH)


@pytest.fixture(scope='module')
def common_periods(portfolio_df, benchmark_df):
    return get_common_periods(portfolio_df, benchmark_df)


# ---------------------------------------------------------------------------
# 1. Data loading and auto-detection
# ---------------------------------------------------------------------------

class TestDataLoading:

    def test_portfolio_detected_as_property_level(self, raw_portfolio):
        assert is_property_level(raw_portfolio)

    def test_portfolio_loads_as_aggregated(self, portfolio_df):
        assert 'Query_Level' in portfolio_df.columns
        levels = set(portfolio_df['Query_Level'].unique())
        assert levels == {'Total', 'By_PropertyType', 'By_CBSA', 'By_PropertyType_CBSA'}

    def test_benchmark_loads_as_aggregated(self, benchmark_df):
        assert 'Query_Level' in benchmark_df.columns

    def test_common_periods_are_24_quarters(self, common_periods):
        assert len(common_periods) == 24

    def test_common_periods_range(self, common_periods):
        assert common_periods[0] == (2020, 1)
        assert common_periods[-1] == (2025, 4)

    def test_common_periods_sorted(self, common_periods):
        assert common_periods == sorted(common_periods)


# ---------------------------------------------------------------------------
# 2. Aggregation correctness
# ---------------------------------------------------------------------------

class TestAggregation:

    def test_total_denom_equals_sum_of_property_denoms(self, raw_portfolio, portfolio_df):
        for (year, quarter), group in raw_portfolio.groupby(['Year', 'Quarter']):
            prop_sum = group['Denom'].sum()
            total_row = portfolio_df[
                (portfolio_df['Query_Level'] == 'Total') &
                (portfolio_df['Year'] == year) &
                (portfolio_df['Quarter'] == quarter)
            ]
            assert abs(total_row['Denom'].iloc[0] - prop_sum) < 0.01

    def test_total_mv_equals_sum_of_property_mvs(self, raw_portfolio, portfolio_df):
        for (year, quarter), group in raw_portfolio.groupby(['Year', 'Quarter']):
            prop_sum = group['MV'].sum()
            total_row = portfolio_df[
                (portfolio_df['Query_Level'] == 'Total') &
                (portfolio_df['Year'] == year) &
                (portfolio_df['Quarter'] == quarter)
            ]
            assert abs(total_row['MV'].iloc[0] - prop_sum) < 0.01

    def test_property_type_denoms_sum_to_total(self, portfolio_df):
        for (year, quarter) in portfolio_df[portfolio_df['Query_Level'] == 'Total'][['Year', 'Quarter']].values:
            total = portfolio_df[
                (portfolio_df['Query_Level'] == 'Total') &
                (portfolio_df['Year'] == year) & (portfolio_df['Quarter'] == quarter)
            ]['Denom'].iloc[0]
            pt_sum = portfolio_df[
                (portfolio_df['Query_Level'] == 'By_PropertyType') &
                (portfolio_df['Year'] == year) & (portfolio_df['Quarter'] == quarter)
            ]['Denom'].sum()
            assert abs(total - pt_sum) < 0.01

    def test_cbsa_denoms_sum_to_total(self, portfolio_df):
        for (year, quarter) in portfolio_df[portfolio_df['Query_Level'] == 'Total'][['Year', 'Quarter']].values:
            total = portfolio_df[
                (portfolio_df['Query_Level'] == 'Total') &
                (portfolio_df['Year'] == year) & (portfolio_df['Quarter'] == quarter)
            ]['Denom'].iloc[0]
            cbsa_sum = portfolio_df[
                (portfolio_df['Query_Level'] == 'By_CBSA') &
                (portfolio_df['Year'] == year) & (portfolio_df['Quarter'] == quarter)
            ]['Denom'].sum()
            assert abs(total - cbsa_sum) < 0.01

    def test_weighted_return_at_total_level(self, raw_portfolio, portfolio_df):
        """Total return should be Denom-weighted average of property returns."""
        for (year, quarter), group in raw_portfolio.groupby(['Year', 'Quarter']):
            total_denom = group['Denom'].sum()
            expected_ret = (group['Total_Return'] * group['Denom']).sum() / total_denom
            actual = portfolio_df[
                (portfolio_df['Query_Level'] == 'Total') &
                (portfolio_df['Year'] == year) & (portfolio_df['Quarter'] == quarter)
            ]['Total_Return'].iloc[0]
            assert abs(actual - expected_ret) < 1e-10

    def test_prop_count_at_total(self, raw_portfolio, portfolio_df):
        for (year, quarter), group in raw_portfolio.groupby(['Year', 'Quarter']):
            expected = len(group)
            actual = portfolio_df[
                (portfolio_df['Query_Level'] == 'Total') &
                (portfolio_df['Year'] == year) & (portfolio_df['Quarter'] == quarter)
            ]['Prop_Count'].iloc[0]
            assert actual == expected

    def test_six_property_types_in_aggregated(self, portfolio_df):
        pts = portfolio_df[portfolio_df['Query_Level'] == 'By_PropertyType']['PropertyType'].unique()
        assert set(pts) == {'Office', 'Industrial', 'Residential', 'Retail', 'Hotel', 'Self Storage'}

    def test_nine_cbsas_in_aggregated(self, portfolio_df):
        cbsas = portfolio_df[portfolio_df['Query_Level'] == 'By_CBSA']['CBSAName'].unique()
        assert len(cbsas) == 9


# ---------------------------------------------------------------------------
# 3. Single-period attribution with full portfolio
# ---------------------------------------------------------------------------

class TestSinglePeriodFull:

    def test_bf_formulas_hold_all_periods_property_type(self, portfolio_df, benchmark_df, common_periods):
        for year, quarter in common_periods:
            df = brinson_fachler_single_period(portfolio_df, benchmark_df, year, quarter, 'PropertyType')
            assert not df.empty
            R_b = df['R_b'].iloc[0]
            for _, row in df.iterrows():
                assert abs(row['Allocation'] - (row['w_p'] - row['w_b']) * (row['r_b'] - R_b)) < 1e-12
                assert abs(row['Selection'] - row['w_p'] * (row['r_p'] - row['r_b'])) < 1e-12
                assert abs(row['Total_Active'] - (row['Allocation'] + row['Selection'])) < 1e-12

    def test_portfolio_weights_sum_to_one(self, portfolio_df, benchmark_df, common_periods):
        """Portfolio weights from aggregated property data should sum to ~1.0."""
        for year, quarter in common_periods:
            df = brinson_fachler_single_period(portfolio_df, benchmark_df, year, quarter, 'PropertyType')
            w_p_sum = df['w_p'].sum()
            assert abs(w_p_sum - 1.0) < 0.01, f"Portfolio weights sum to {w_p_sum} in {year} Q{quarter}"

    def test_no_all_in_segments(self, portfolio_df, benchmark_df, common_periods):
        for year, quarter in common_periods:
            for dim in ['PropertyType', 'CBSAName', 'PropertyType_CBSAName']:
                df = brinson_fachler_single_period(portfolio_df, benchmark_df, year, quarter, dim)
                if not df.empty:
                    for col in ['PropertyType', 'CBSAName']:
                        if col in df.columns:
                            assert 'All' not in df[col].values

    def test_contributions_positive_for_positive_return_and_weight(self, portfolio_df, benchmark_df):
        df = brinson_fachler_single_period(portfolio_df, benchmark_df, 2025, 4, 'PropertyType')
        for _, row in df.iterrows():
            if row['w_p'] > 0 and row['r_p'] > 0:
                assert row['Contribution_P'] > 0


# ---------------------------------------------------------------------------
# 4. Multi-period attribution across all trailing windows
# ---------------------------------------------------------------------------

class TestTrailingPeriodsFull:

    def test_available_trailing_periods(self, common_periods):
        as_of = common_periods[-1]
        avail = get_available_trailing_periods(common_periods, as_of)
        assert avail['1Q'] is True
        assert avail['1Y'] is True
        assert avail['3Y'] is True
        assert avail['5Y'] is True
        assert avail['7Y'] is False
        assert avail['10Y'] is False
        assert avail['SI'] is True

    def test_1q_returns_single_period(self, common_periods):
        periods = get_trailing_periods(common_periods, common_periods[-1], '1Q')
        assert len(periods) == 1
        assert periods[0] == (2025, 4)

    def test_1y_returns_four_periods(self, common_periods):
        periods = get_trailing_periods(common_periods, common_periods[-1], '1Y')
        assert len(periods) == 4
        assert periods == [(2025, 1), (2025, 2), (2025, 3), (2025, 4)]

    def test_3y_returns_twelve_periods(self, common_periods):
        periods = get_trailing_periods(common_periods, common_periods[-1], '3Y')
        assert len(periods) == 12
        assert periods[0] == (2023, 1)
        assert periods[-1] == (2025, 4)

    def test_5y_returns_twenty_periods(self, common_periods):
        periods = get_trailing_periods(common_periods, common_periods[-1], '5Y')
        assert len(periods) == 20
        assert periods[0] == (2021, 1)
        assert periods[-1] == (2025, 4)

    def test_si_returns_all_24_quarters(self, common_periods):
        si = get_trailing_periods(common_periods, common_periods[-1], 'SI')
        assert len(si) == 24
        assert si == common_periods

    def test_as_of_mid_range(self, common_periods):
        as_of = (2022, 2)
        periods = get_trailing_periods(common_periods, as_of, '1Y')
        assert len(periods) == 4
        assert periods == [(2021, 3), (2021, 4), (2022, 1), (2022, 2)]


# ---------------------------------------------------------------------------
# 5. Multi-period reconciliation across all dimensions and windows
# ---------------------------------------------------------------------------

class TestReconciliationFull:

    @pytest.mark.parametrize('trailing_key', ['1Q', '1Y', '3Y', '5Y', 'SI'])
    def test_reconciliation_by_property_type(self, portfolio_df, benchmark_df, common_periods, trailing_key):
        periods = get_trailing_periods(common_periods, common_periods[-1], trailing_key)
        r = run_attribution(portfolio_df, benchmark_df, periods, 'PropertyType')
        active = r['cum_rp'] - r['cum_rb']
        linked_sum = r['summary']['Total_Active'].sum()
        assert abs(active - linked_sum) < 1e-6, \
            f"{trailing_key}: active={active:.6f}, linked_sum={linked_sum:.6f}"

    @pytest.mark.parametrize('trailing_key', ['1Q', '1Y', '3Y', '5Y', 'SI'])
    def test_reconciliation_by_cbsa(self, portfolio_df, benchmark_df, common_periods, trailing_key):
        periods = get_trailing_periods(common_periods, common_periods[-1], trailing_key)
        r = run_attribution(portfolio_df, benchmark_df, periods, 'CBSAName')
        active = r['cum_rp'] - r['cum_rb']
        linked_sum = r['summary']['Total_Active'].sum()
        assert abs(active - linked_sum) < 1e-6

    @pytest.mark.parametrize('trailing_key', ['1Q', '1Y', '3Y', '5Y', 'SI'])
    def test_reconciliation_by_cross(self, portfolio_df, benchmark_df, common_periods, trailing_key):
        periods = get_trailing_periods(common_periods, common_periods[-1], trailing_key)
        r = run_attribution(portfolio_df, benchmark_df, periods, 'PropertyType_CBSAName')
        active = r['cum_rp'] - r['cum_rb']
        linked_sum = r['summary']['Total_Active'].sum()
        assert abs(active - linked_sum) < 1e-6

    def test_component_sum_equals_total_active(self, portfolio_df, benchmark_df, common_periods):
        """Allocation + Selection = Total_Active for every segment."""
        for dim in ['PropertyType', 'CBSAName', 'PropertyType_CBSAName']:
            r = run_attribution(portfolio_df, benchmark_df, common_periods, dim)
            for _, row in r['summary'].iterrows():
                expected = row['Allocation'] + row['Selection']
                assert abs(row['Total_Active'] - expected) < 1e-10


# ---------------------------------------------------------------------------
# 6. Cumulative and annualized returns
# ---------------------------------------------------------------------------

class TestReturnsFull:

    def test_cumulative_return_compounding(self, portfolio_df, benchmark_df, common_periods):
        """Verify cumulative return is geometric compounding of period returns."""
        r = run_attribution(portfolio_df, benchmark_df, common_periods, 'PropertyType')
        # Manually compound
        total_rows = portfolio_df[portfolio_df['Query_Level'] == 'Total']
        cum = 1.0
        for year, quarter in common_periods:
            ret = total_rows[
                (total_rows['Year'] == year) & (total_rows['Quarter'] == quarter)
            ]['Total_Return'].iloc[0]
            cum *= (1 + ret)
        expected = cum - 1
        assert abs(r['cum_rp'] - expected) < 1e-10

    def test_annualized_for_si(self, portfolio_df, benchmark_df, common_periods):
        """SI (24 quarters = 6 years) should be annualized."""
        r = run_attribution(portfolio_df, benchmark_df, common_periods, 'PropertyType')
        expected_ann = (1 + r['cum_rp']) ** (1 / 6.0) - 1
        assert abs(r['ann_rp'] - expected_ann) < 1e-10

    def test_1q_not_annualized(self, portfolio_df, benchmark_df, common_periods):
        periods = get_trailing_periods(common_periods, common_periods[-1], '1Q')
        r = run_attribution(portfolio_df, benchmark_df, periods, 'PropertyType')
        assert r['cum_rp'] == r['ann_rp']

    def test_1y_not_annualized(self, portfolio_df, benchmark_df, common_periods):
        periods = get_trailing_periods(common_periods, common_periods[-1], '1Y')
        r = run_attribution(portfolio_df, benchmark_df, periods, 'PropertyType')
        assert r['cum_rp'] == r['ann_rp']

    def test_segment_cumulative_returns_present(self, portfolio_df, benchmark_df, common_periods):
        r = run_attribution(portfolio_df, benchmark_df, common_periods, 'PropertyType')
        assert 'r_p_cum' in r['summary'].columns
        assert 'r_b_cum' in r['summary'].columns
        assert not r['summary']['r_p_cum'].isna().any()
        assert not r['summary']['r_b_cum'].isna().any()

    def test_n_quarters_equals_24(self, portfolio_df, benchmark_df, common_periods):
        r = run_attribution(portfolio_df, benchmark_df, common_periods, 'PropertyType')
        assert r['n_quarters'] == 24

    def test_cum_rp_different_from_cum_rb(self, portfolio_df, benchmark_df, common_periods):
        """Portfolio and benchmark should have different returns."""
        r = run_attribution(portfolio_df, benchmark_df, common_periods, 'PropertyType')
        assert r['cum_rp'] != r['cum_rb']


# ---------------------------------------------------------------------------
# 7. run_full_attribution and run_all_trailing_periods
# ---------------------------------------------------------------------------

class TestFullAttribution:

    def test_run_full_attribution_three_dimensions(self, portfolio_df, benchmark_df, common_periods):
        r = run_full_attribution(portfolio_df, benchmark_df, common_periods)
        assert set(r.keys()) == {'by_property_type', 'by_cbsa', 'by_property_type_cbsa'}

    def test_all_dimensions_have_data(self, portfolio_df, benchmark_df, common_periods):
        r = run_full_attribution(portfolio_df, benchmark_df, common_periods)
        for dim in r:
            assert not r[dim]['summary'].empty
            assert not r[dim]['detail'].empty
            assert r[dim]['n_quarters'] == 24

    def test_same_total_returns_across_dimensions(self, portfolio_df, benchmark_df, common_periods):
        """All dimensions should report the same total portfolio/benchmark returns."""
        r = run_full_attribution(portfolio_df, benchmark_df, common_periods)
        cum_rp_pt = r['by_property_type']['cum_rp']
        cum_rb_pt = r['by_property_type']['cum_rb']
        for dim in ['by_cbsa', 'by_property_type_cbsa']:
            assert abs(r[dim]['cum_rp'] - cum_rp_pt) < 1e-10
            assert abs(r[dim]['cum_rb'] - cum_rb_pt) < 1e-10

    def test_run_all_trailing_periods(self, portfolio_df, benchmark_df, common_periods):
        as_of = common_periods[-1]
        results, available = run_all_trailing_periods(portfolio_df, benchmark_df, common_periods, as_of)
        # Should have results for 1Q, 1Y, 3Y, 5Y, SI
        assert results['1Q'] is not None
        assert results['1Y'] is not None
        assert results['3Y'] is not None
        assert results['5Y'] is not None
        assert results['SI'] is not None
        # Should be None for unavailable periods
        assert results['7Y'] is None
        assert results['10Y'] is None

    def test_all_trailing_reconcile(self, portfolio_df, benchmark_df, common_periods):
        as_of = common_periods[-1]
        results, available = run_all_trailing_periods(portfolio_df, benchmark_df, common_periods, as_of)
        for key in ['1Q', '1Y', '3Y', '5Y', 'SI']:
            for dim in ['by_property_type', 'by_cbsa', 'by_property_type_cbsa']:
                r = results[key][dim]
                active = r['cum_rp'] - r['cum_rb']
                linked = r['summary']['Total_Active'].sum()
                assert abs(active - linked) < 1e-6, \
                    f"{key}/{dim}: active={active:.6f}, linked={linked:.6f}"


# ---------------------------------------------------------------------------
# 8. Detail DataFrame correctness
# ---------------------------------------------------------------------------

class TestDetailData:

    def test_detail_has_all_periods(self, portfolio_df, benchmark_df, common_periods):
        r = run_attribution(portfolio_df, benchmark_df, common_periods, 'PropertyType')
        detail_periods = set(zip(r['detail']['Year'], r['detail']['Quarter']))
        expected = set(common_periods)
        assert detail_periods == expected

    def test_detail_has_correct_segments_per_period(self, portfolio_df, benchmark_df, common_periods):
        r = run_attribution(portfolio_df, benchmark_df, common_periods, 'PropertyType')
        for (year, quarter), group in r['detail'].groupby(['Year', 'Quarter']):
            # Should have at least 6 property types (from portfolio) and up to 8 (from benchmark)
            assert len(group) >= 6
            assert len(group) <= 8

    def test_detail_no_nan_in_effects(self, portfolio_df, benchmark_df, common_periods):
        r = run_attribution(portfolio_df, benchmark_df, common_periods, 'PropertyType')
        for col in ['Allocation', 'Selection', 'Total_Active']:
            assert not r['detail'][col].isna().any()

    def test_detail_weights_non_negative(self, portfolio_df, benchmark_df, common_periods):
        r = run_attribution(portfolio_df, benchmark_df, common_periods, 'PropertyType')
        assert (r['detail']['w_p'] >= 0).all()
        assert (r['detail']['w_b'] >= 0).all()


# ---------------------------------------------------------------------------
# 9. Summary DataFrame correctness
# ---------------------------------------------------------------------------

class TestSummaryData:

    def test_summary_sorted_by_total_active_desc(self, portfolio_df, benchmark_df, common_periods):
        r = run_attribution(portfolio_df, benchmark_df, common_periods, 'PropertyType')
        ta = r['summary']['Total_Active'].values
        assert all(ta[i] >= ta[i + 1] for i in range(len(ta) - 1))

    def test_summary_no_nan(self, portfolio_df, benchmark_df, common_periods):
        r = run_attribution(portfolio_df, benchmark_df, common_periods, 'PropertyType')
        for col in ['Allocation', 'Selection', 'Total_Active',
                     'w_p', 'w_b', 'r_p_cum', 'r_b_cum', 'Contribution_P', 'Contribution_B']:
            assert not r['summary'][col].isna().any(), f"NaN found in {col}"

    def test_summary_weights_are_averages(self, portfolio_df, benchmark_df, common_periods):
        """Summary w_p and w_b should be averages across periods."""
        r = run_attribution(portfolio_df, benchmark_df, common_periods, 'PropertyType')
        for _, row in r['summary'].iterrows():
            pt = row['PropertyType']
            detail_seg = r['detail'][r['detail']['PropertyType'] == pt]
            assert abs(row['w_p'] - detail_seg['w_p'].mean()) < 1e-10
            assert abs(row['w_b'] - detail_seg['w_b'].mean()) < 1e-10

    def test_contributions_are_weight_times_cum_return(self, portfolio_df, benchmark_df, common_periods):
        r = run_attribution(portfolio_df, benchmark_df, common_periods, 'PropertyType')
        for _, row in r['summary'].iterrows():
            assert abs(row['Contribution_P'] - row['w_p'] * row['r_p_cum']) < 1e-10
            assert abs(row['Contribution_B'] - row['w_b'] * row['r_b_cum']) < 1e-10


# ---------------------------------------------------------------------------
# 10. Cross-dimension (PropertyType x CBSA) specifics
# ---------------------------------------------------------------------------

class TestCrossDimension:

    def test_cross_segments_have_both_columns(self, portfolio_df, benchmark_df, common_periods):
        r = run_attribution(portfolio_df, benchmark_df, common_periods, 'PropertyType_CBSAName')
        assert 'PropertyType' in r['summary'].columns
        assert 'CBSAName' in r['summary'].columns

    def test_cross_has_more_segments_than_single_dim(self, portfolio_df, benchmark_df, common_periods):
        r_pt = run_attribution(portfolio_df, benchmark_df, common_periods, 'PropertyType')
        r_cbsa = run_attribution(portfolio_df, benchmark_df, common_periods, 'CBSAName')
        r_cross = run_attribution(portfolio_df, benchmark_df, common_periods, 'PropertyType_CBSAName')
        assert len(r_cross['summary']) > len(r_pt['summary'])
        assert len(r_cross['summary']) > len(r_cbsa['summary'])

    def test_cross_no_all_values(self, portfolio_df, benchmark_df, common_periods):
        r = run_attribution(portfolio_df, benchmark_df, common_periods, 'PropertyType_CBSAName')
        assert 'All' not in r['summary']['PropertyType'].values
        assert 'All' not in r['summary']['CBSAName'].values
        assert 'All' not in r['detail']['PropertyType'].values
        assert 'All' not in r['detail']['CBSAName'].values


# ---------------------------------------------------------------------------
# 11. Self-attribution with full portfolio
# ---------------------------------------------------------------------------

class TestSelfAttributionFull:

    def test_self_attribution_zero_active(self, portfolio_df, common_periods):
        r = run_full_attribution(portfolio_df, portfolio_df, common_periods)
        for dim in ['by_property_type', 'by_cbsa', 'by_property_type_cbsa']:
            assert abs(r[dim]['cum_rp'] - r[dim]['cum_rb']) < 1e-10
            assert abs(r[dim]['summary']['Total_Active'].sum()) < 1e-10

    def test_self_attribution_equal_weights(self, portfolio_df, common_periods):
        for year, quarter in common_periods[-3:]:
            df = brinson_fachler_single_period(portfolio_df, portfolio_df, year, quarter, 'PropertyType')
            assert np.allclose(df['w_p'], df['w_b'], atol=1e-10)
            assert np.allclose(df['r_p'], df['r_b'], atol=1e-10)


# ---------------------------------------------------------------------------
# 12. As-of quarter filtering
# ---------------------------------------------------------------------------

class TestAsOfFiltering:

    def test_mid_period_as_of(self, portfolio_df, benchmark_df, common_periods):
        """Run attribution as of 2024 Q2 (6th of 12 quarters)."""
        as_of = (2024, 2)
        periods = get_trailing_periods(common_periods, as_of, '1Y')
        assert len(periods) == 4
        assert all(p <= as_of for p in periods)

        r = run_attribution(portfolio_df, benchmark_df, periods, 'PropertyType')
        assert r['n_quarters'] == 4
        active = r['cum_rp'] - r['cum_rb']
        assert abs(active - r['summary']['Total_Active'].sum()) < 1e-6

    def test_earliest_as_of(self, portfolio_df, benchmark_df, common_periods):
        """Run for just the first quarter."""
        as_of = common_periods[0]
        periods = get_trailing_periods(common_periods, as_of, '1Q')
        assert periods == [(2020, 1)]

        r = run_attribution(portfolio_df, benchmark_df, periods, 'PropertyType')
        assert r['n_quarters'] == 1

    def test_si_from_mid_point(self, portfolio_df, benchmark_df, common_periods):
        """SI as of 2022 Q2 should include 2020Q1 through 2022Q2."""
        as_of = (2022, 2)
        periods = get_trailing_periods(common_periods, as_of, 'SI')
        assert len(periods) == 10
        assert periods[0] == (2020, 1)
        assert periods[-1] == (2022, 2)


# ---------------------------------------------------------------------------
# 13. Flask app integration with full portfolio
# ---------------------------------------------------------------------------

class TestFlaskIntegration:

    @pytest.fixture
    def client(self):
        from app import app
        app.config['TESTING'] = True
        with app.test_client() as c:
            yield c

    def _upload(self, client):
        with open(BENCHMARK_PATH, 'rb') as bf, open(FULL_PORTFOLIO_PATH, 'rb') as pf:
            return client.post('/upload', data={
                'benchmark_file': (bf, 'benchmark.xlsx'),
                'portfolio_file': (pf, 'portfolio.xlsx'),
            }, follow_redirects=False, content_type='multipart/form-data')

    def test_upload_redirects_to_properties(self, client):
        resp = self._upload(client)
        assert resp.status_code == 302
        assert '/properties' in resp.headers['Location']

    def test_results_page_loads(self, client):
        self._upload(client)
        resp = client.get('/results?as_of=2025-4&trailing=1Y')
        assert resp.status_code == 200

    def test_results_contains_property_types(self, client):
        self._upload(client)
        resp = client.get('/results?as_of=2025-4&trailing=1Y')
        html = resp.data.decode()
        assert 'Office' in html
        assert 'Industrial' in html

    @pytest.mark.parametrize('trailing', ['1Q', '1Y', '3Y', '5Y', 'SI'])
    def test_results_all_trailing_periods(self, client, trailing):
        self._upload(client)
        resp = client.get(f'/results?as_of=2025-4&trailing={trailing}')
        assert resp.status_code == 200

    def test_results_different_as_of(self, client):
        self._upload(client)
        resp = client.get('/results?as_of=2024-2&trailing=1Y')
        assert resp.status_code == 200

    @pytest.mark.parametrize('dim', ['property_type', 'cbsa', 'property_type_cbsa'])
    def test_export_all_dimensions(self, client, dim):
        self._upload(client)
        resp = client.get(f'/export/{dim}?as_of=2025-4&trailing=1Y')
        assert resp.status_code == 200
        assert 'spreadsheet' in resp.content_type

    def test_export_reads_as_valid_excel(self, client):
        self._upload(client)
        resp = client.get('/export/property_type?as_of=2025-4&trailing=3Y')
        from io import BytesIO
        df_summary = pd.read_excel(BytesIO(resp.data), sheet_name='Summary')
        df_detail = pd.read_excel(BytesIO(resp.data), sheet_name='Detail')
        assert len(df_summary) > 0
        assert len(df_detail) > 0
        assert 'Allocation' in df_summary.columns
        assert 'Total_Active' in df_summary.columns

    def test_download_template_from_session(self, client):
        self._upload(client)
        resp = client.post('/download-template', content_type='multipart/form-data')
        assert resp.status_code == 200
        assert 'spreadsheet' in resp.content_type
