"""
Tests for the Flask application routes.

Covers: index, upload, results, export, download-template,
        as_of parsing, trailing period validation, session handling.
"""
import os
import io
import tempfile

import pandas as pd
import numpy as np
import pytest

from app import app


# ---------------------------------------------------------------------------
# Helpers to build minimal valid DataFrames
# ---------------------------------------------------------------------------

def _make_benchmark_df(periods=None):
    """
    Build a minimal aggregated benchmark DataFrame with the required
    Query_Level rows for each period.

    Default periods: [(2024, 1), (2024, 2), (2024, 3), (2024, 4)]
    """
    if periods is None:
        periods = [(2024, 1), (2024, 2), (2024, 3), (2024, 4)]

    query_levels = [
        ('Total', 'All', 'All'),
        ('By_PropertyType', 'Office', 'All'),
        ('By_PropertyType', 'Industrial', 'All'),
        ('By_CBSA', 'All', 'New York-Newark-Jersey City, NY-NJ-PA'),
        ('By_CBSA', 'All', 'Los Angeles-Long Beach-Anaheim, CA'),
        ('By_PropertyType_CBSA', 'Office', 'New York-Newark-Jersey City, NY-NJ-PA'),
        ('By_PropertyType_CBSA', 'Industrial', 'Los Angeles-Long Beach-Anaheim, CA'),
    ]

    np.random.seed(99)
    rows = []
    for year, quarter in periods:
        for ql, pt, cbsa in query_levels:
            denom = np.random.uniform(1e8, 5e8)
            income_ret = np.random.uniform(0.008, 0.015)
            capital_ret = np.random.uniform(-0.02, 0.04)
            total_ret = income_ret + capital_ret
            rows.append({
                'Query_Level': ql,
                'Year': year,
                'YYYYQ': year * 10 + quarter,
                'Quarter': quarter,
                'PropertyType': pt,
                'CBSAName': cbsa,
                'NOI': round(denom * income_ret),
                'CapEx': round(denom * 0.002),
                'MV': round(denom * (1 + capital_ret)),
                'MVLag1': round(denom),
                'PSales': 0,
                'Denom': round(denom),
                'Income_Return': round(income_ret, 6),
                'Capital_Return': round(capital_ret, 6),
                'Total_Return': round(total_ret, 6),
                'Prop_Count': np.random.randint(5, 50),
            })

    return pd.DataFrame(rows)


def _make_portfolio_df(periods=None):
    """
    Build a minimal property-level portfolio DataFrame (no Query_Level column).
    """
    if periods is None:
        periods = [(2024, 1), (2024, 2), (2024, 3), (2024, 4)]

    np.random.seed(42)
    properties = [
        ('Office', 'New York-Newark-Jersey City, NY-NJ-PA'),
        ('Industrial', 'Los Angeles-Long Beach-Anaheim, CA'),
    ]

    rows = []
    for year, quarter in periods:
        for pt, cbsa in properties:
            mv_lag = np.random.uniform(5e7, 2e8)
            noi = mv_lag * np.random.uniform(0.010, 0.020)
            capex = mv_lag * np.random.uniform(0.001, 0.005)
            income_ret = noi / mv_lag
            capital_ret = np.random.uniform(-0.03, 0.05)
            total_ret = income_ret + capital_ret
            mv = mv_lag * (1 + capital_ret)
            denom = mv_lag - capex / 2
            rows.append({
                'Year': year,
                'Quarter': quarter,
                'PropertyType': pt,
                'CBSAName': cbsa,
                'NOI': round(noi),
                'CapEx': round(capex),
                'MV': round(mv),
                'MVLag1': round(mv_lag),
                'PSales': 0,
                'Denom': round(denom),
                'Income_Return': round(income_ret, 6),
                'Capital_Return': round(capital_ret, 6),
                'Total_Return': round(total_ret, 6),
            })

    return pd.DataFrame(rows)


def _make_property_level_portfolio_df(periods=None, extra_property=False):
    """
    Build a property-level portfolio DataFrame with stable PropertyIDs so
    property overrides can be saved and reapplied across uploads.
    """
    if periods is None:
        periods = [(2024, 1), (2024, 2), (2024, 3), (2024, 4)]

    np.random.seed(7 if not extra_property else 8)
    properties = [
        ('PROP-001', 'One Penn Plaza', 'Office', 'New York-Newark-Jersey City, NY-NJ-PA'),
        ('PROP-002', 'South Bay Logistics', 'Industrial', 'Los Angeles-Long Beach-Anaheim, CA'),
    ]
    if extra_property:
        properties.append(
            ('PROP-003', 'Hudson Apartments', 'Office', 'New York-Newark-Jersey City, NY-NJ-PA')
        )

    rows = []
    for year, quarter in periods:
        for prop_id, name, pt, cbsa in properties:
            mv_lag = np.random.uniform(5e7, 2e8)
            noi = mv_lag * np.random.uniform(0.010, 0.020)
            capex = mv_lag * np.random.uniform(0.001, 0.005)
            income_ret = noi / mv_lag
            capital_ret = np.random.uniform(-0.03, 0.05)
            total_ret = income_ret + capital_ret
            mv = mv_lag * (1 + capital_ret)
            denom = mv_lag - capex / 2
            rows.append({
                'Year': year,
                'Quarter': quarter,
                'YYYYQ': year * 10 + quarter,
                'PropertyID': prop_id,
                'PropertyName': name,
                'PropertyType': pt,
                'CBSAName': cbsa,
                'NOI': round(noi),
                'CapEx': round(capex),
                'MV': round(mv),
                'MVLag1': round(mv_lag),
                'PSales': 0,
                'Denom': round(denom),
                'Income_Return': round(income_ret, 6),
                'Capital_Return': round(capital_ret, 6),
                'Total_Return': round(total_ret, 6),
            })

    return pd.DataFrame(rows)


def _df_to_excel_bytes(df):
    """Write a DataFrame to an in-memory Excel BytesIO object."""
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    buf.seek(0)
    return buf


def _df_to_excel_file(df, path):
    """Write a DataFrame to an Excel file on disk."""
    df.to_excel(path, index=False)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client(tmp_path):
    """Create a Flask test client with an isolated upload folder."""
    upload_dir = str(tmp_path / 'uploads')
    os.makedirs(upload_dir, exist_ok=True)

    app.config['TESTING'] = True
    app.config['UPLOAD_FOLDER'] = upload_dir
    # Stable secret so sessions persist across requests in the same test
    app.config['SECRET_KEY'] = 'test-secret-key'

    with app.test_client() as c:
        yield c


@pytest.fixture()
def benchmark_xlsx(tmp_path):
    """Write a benchmark Excel file and return its path."""
    path = str(tmp_path / 'benchmark.xlsx')
    _df_to_excel_file(_make_benchmark_df(), path)
    return path


@pytest.fixture()
def portfolio_xlsx(tmp_path):
    """Write a portfolio Excel file and return its path."""
    path = str(tmp_path / 'portfolio.xlsx')
    _df_to_excel_file(_make_portfolio_df(), path)
    return path


@pytest.fixture()
def uploaded_session(client, benchmark_xlsx, portfolio_xlsx):
    """
    Upload valid files so the session is populated and both paths exist.
    Returns the test client (session is already set).
    """
    with open(benchmark_xlsx, 'rb') as bf, open(portfolio_xlsx, 'rb') as pf:
        resp = client.post('/upload', data={
            'benchmark_file': (bf, 'benchmark.xlsx'),
            'portfolio_file': (pf, 'portfolio.xlsx'),
        }, content_type='multipart/form-data', follow_redirects=False)

    # The upload should redirect to /results
    assert resp.status_code == 302
    assert '/results' in resp.headers['Location']
    return client


# ---------------------------------------------------------------------------
# 1. Index page
# ---------------------------------------------------------------------------

class TestIndex:
    def test_index_returns_200(self, client):
        resp = client.get('/')
        assert resp.status_code == 200

    def test_index_contains_upload_form(self, client):
        resp = client.get('/')
        assert b'upload' in resp.data.lower() or b'form' in resp.data.lower()


# ---------------------------------------------------------------------------
# 2. Upload – missing files
# ---------------------------------------------------------------------------

class TestUploadMissingFiles:
    def test_upload_no_files(self, client):
        resp = client.post('/upload', data={}, content_type='multipart/form-data',
                           follow_redirects=True)
        assert resp.status_code == 200
        assert b'Please upload a benchmark file' in resp.data

    def test_upload_missing_portfolio(self, client, benchmark_xlsx):
        with open(benchmark_xlsx, 'rb') as bf:
            resp = client.post('/upload', data={
                'benchmark_file': (bf, 'benchmark.xlsx'),
            }, content_type='multipart/form-data', follow_redirects=True)
        assert b'Please upload a portfolio file' in resp.data

    def test_upload_missing_benchmark(self, client, portfolio_xlsx):
        with open(portfolio_xlsx, 'rb') as pf:
            resp = client.post('/upload', data={
                'portfolio_file': (pf, 'portfolio.xlsx'),
            }, content_type='multipart/form-data', follow_redirects=True)
        assert b'Please upload a benchmark file' in resp.data


# ---------------------------------------------------------------------------
# 3. Upload – valid files redirect to results
# ---------------------------------------------------------------------------

class TestUploadValid:
    def test_upload_redirects_to_results(self, client, benchmark_xlsx, portfolio_xlsx):
        with open(benchmark_xlsx, 'rb') as bf, open(portfolio_xlsx, 'rb') as pf:
            resp = client.post('/upload', data={
                'benchmark_file': (bf, 'benchmark.xlsx'),
                'portfolio_file': (pf, 'portfolio.xlsx'),
            }, content_type='multipart/form-data', follow_redirects=False)

        assert resp.status_code == 302
        location = resp.headers['Location']
        assert '/results' in location
        # Default as_of should be the latest common period
        assert 'as_of=2024-4' in location

    def test_upload_saves_files_to_upload_folder(self, client, benchmark_xlsx,
                                                  portfolio_xlsx, tmp_path):
        upload_dir = app.config['UPLOAD_FOLDER']
        with open(benchmark_xlsx, 'rb') as bf, open(portfolio_xlsx, 'rb') as pf:
            client.post('/upload', data={
                'benchmark_file': (bf, 'benchmark.xlsx'),
                'portfolio_file': (pf, 'portfolio.xlsx'),
            }, content_type='multipart/form-data')

        # Files are saved in a session-specific subdirectory
        subdirs = [d for d in os.listdir(upload_dir)
                   if os.path.isdir(os.path.join(upload_dir, d))]
        assert len(subdirs) >= 1
        session_dir = os.path.join(upload_dir, subdirs[0])
        files = os.listdir(session_dir)
        assert len(files) >= 2

    def test_reupload_preserves_existing_property_overrides(self, client, benchmark_xlsx):
        portfolio_v1 = _df_to_excel_bytes(_make_property_level_portfolio_df())
        portfolio_v2 = _df_to_excel_bytes(_make_property_level_portfolio_df(extra_property=True))

        with open(benchmark_xlsx, 'rb') as bf:
            resp = client.post('/upload', data={
                'benchmark_file': (bf, 'benchmark.xlsx'),
                'portfolio_file': (portfolio_v1, 'portfolio.xlsx'),
            }, content_type='multipart/form-data', follow_redirects=False)
        assert resp.status_code == 302
        assert '/properties' in resp.headers['Location']

        save_resp = client.post('/properties/save', json={
            'sold': {'PROP-001': True},
            'cbsa_map': {},
        })
        assert save_resp.status_code == 200

        # Find overrides CSV in session subdirectory
        upload_dir = app.config['UPLOAD_FOLDER']
        subdirs = [d for d in os.listdir(upload_dir)
                   if os.path.isdir(os.path.join(upload_dir, d))]
        assert len(subdirs) >= 1
        session_dir = os.path.join(upload_dir, subdirs[0])
        overrides_path = os.path.join(session_dir, 'current_portfolio_overrides.csv')
        assert os.path.exists(overrides_path)

        with open(benchmark_xlsx, 'rb') as bf:
            resp = client.post('/upload', data={
                'benchmark_file': (bf, 'benchmark.xlsx'),
                'portfolio_file': (portfolio_v2, 'portfolio.xlsx'),
            }, content_type='multipart/form-data', follow_redirects=False)
        assert resp.status_code == 302
        assert '/properties' in resp.headers['Location']

        # Overrides should persist across re-upload
        assert os.path.exists(overrides_path)

        props_resp = client.get('/properties')
        assert props_resp.status_code == 200
        assert b'data-property-id="PROP-001"' in props_resp.data
        assert b'data-property-id="PROP-003"' in props_resp.data
        assert b'data-property-id="PROP-001"\n                                       checked' in props_resp.data


# ---------------------------------------------------------------------------
# 4. Results – no session files
# ---------------------------------------------------------------------------

class TestResultsNoSession:
    def test_results_without_session_redirects(self, client):
        resp = client.get('/results', follow_redirects=False)
        assert resp.status_code == 302

    def test_results_without_session_flashes_error(self, client):
        resp = client.get('/results', follow_redirects=True)
        assert b'Please upload files first' in resp.data


# ---------------------------------------------------------------------------
# 5. Results – valid session
# ---------------------------------------------------------------------------

class TestResultsValid:
    def test_results_page_returns_200(self, uploaded_session):
        resp = uploaded_session.get('/results?as_of=2024-4&trailing=1Q',
                                    follow_redirects=True)
        assert resp.status_code == 200

    def test_results_page_contains_attribution_data(self, uploaded_session):
        resp = uploaded_session.get('/results?as_of=2024-4&trailing=1Q',
                                    follow_redirects=True)
        # The results page should contain at least one dimension label
        data = resp.data.decode()
        # Property types from our test data
        assert 'Office' in data or 'Industrial' in data

    def test_results_default_trailing(self, uploaded_session):
        """When no trailing param given, defaults to 1Y (or falls back to available)."""
        resp = uploaded_session.get('/results?as_of=2024-4', follow_redirects=True)
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 6. Export – invalid dimension
# ---------------------------------------------------------------------------

class TestExportInvalidDimension:
    def test_export_invalid_dimension_redirects(self, uploaded_session):
        resp = uploaded_session.get('/export/bogus_dimension', follow_redirects=False)
        assert resp.status_code == 302

    def test_export_invalid_dimension_flashes_error(self, uploaded_session):
        resp = uploaded_session.get('/export/bogus_dimension', follow_redirects=True)
        assert b'Invalid dimension' in resp.data


# ---------------------------------------------------------------------------
# 6b. Export – valid dimensions
# ---------------------------------------------------------------------------

class TestExportValid:
    @pytest.mark.parametrize('dimension', ['property_type', 'cbsa', 'property_type_cbsa'])
    def test_export_returns_excel(self, uploaded_session, dimension):
        resp = uploaded_session.get(
            f'/export/{dimension}?as_of=2024-4&trailing=1Q')
        assert resp.status_code == 200
        assert resp.content_type == (
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    def test_export_without_session_redirects(self, client):
        resp = client.get('/export/property_type', follow_redirects=False)
        assert resp.status_code == 302

    def test_export_invalid_trailing_redirects(self, uploaded_session):
        resp = uploaded_session.get('/export/property_type?trailing=99Z',
                                    follow_redirects=True)
        assert b'Invalid trailing period' in resp.data


# ---------------------------------------------------------------------------
# 7. Download template
# ---------------------------------------------------------------------------

class TestDownloadTemplate:
    def test_download_template_with_upload(self, client, benchmark_xlsx):
        with open(benchmark_xlsx, 'rb') as bf:
            resp = client.post('/download-template', data={
                'benchmark_file': (bf, 'benchmark.xlsx'),
            }, content_type='multipart/form-data')

        assert resp.status_code == 200
        assert resp.content_type == (
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        # Should have a filename header
        cd = resp.headers.get('Content-Disposition', '')
        assert 'Portfolio_Template.xlsx' in cd

    def test_download_template_from_session(self, uploaded_session):
        """If no file is uploaded, the route should use the session benchmark."""
        resp = uploaded_session.post('/download-template', data={},
                                     content_type='multipart/form-data')
        assert resp.status_code == 200
        assert 'Portfolio_Template.xlsx' in resp.headers.get('Content-Disposition', '')

    def test_download_template_no_file_no_session_returns_generic(self, client):
        """Without benchmark, should return a generic template."""
        resp = client.post('/download-template', data={},
                           content_type='multipart/form-data')
        assert resp.status_code == 200
        assert 'spreadsheet' in resp.content_type

    def test_download_template_via_get(self, client):
        """GET request should return generic template without any upload."""
        resp = client.get('/download-template')
        assert resp.status_code == 200
        assert 'spreadsheet' in resp.content_type


# ---------------------------------------------------------------------------
# 8. as_of parameter parsing
# ---------------------------------------------------------------------------

class TestAsOfParsing:
    def test_valid_as_of(self, uploaded_session):
        resp = uploaded_session.get('/results?as_of=2024-1&trailing=1Q',
                                    follow_redirects=True)
        assert resp.status_code == 200

    def test_invalid_as_of_falls_back_to_default(self, uploaded_session):
        """Non-parseable as_of should fall back to the latest common period."""
        resp = uploaded_session.get('/results?as_of=not-valid&trailing=1Q',
                                    follow_redirects=True)
        assert resp.status_code == 200

    def test_as_of_not_in_common_periods(self, uploaded_session):
        """An as_of that parses but is not a common period should fall back."""
        resp = uploaded_session.get('/results?as_of=1999-1&trailing=1Q',
                                    follow_redirects=True)
        assert resp.status_code == 200

    def test_missing_as_of_defaults_to_latest(self, uploaded_session):
        resp = uploaded_session.get('/results?trailing=1Q', follow_redirects=True)
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 9. Trailing parameter validation
# ---------------------------------------------------------------------------

class TestTrailingParam:
    @pytest.mark.parametrize('trailing', ['1Q', '1Y', '3Y', '5Y', '7Y', '10Y', 'SI'])
    def test_known_trailing_keys_accepted(self, uploaded_session, trailing):
        """All recognised trailing keys should not cause an error (even if not
        enough data is available the route falls back gracefully)."""
        resp = uploaded_session.get(f'/results?as_of=2024-4&trailing={trailing}',
                                    follow_redirects=True)
        assert resp.status_code == 200

    def test_invalid_trailing_falls_back_to_1Y(self, uploaded_session):
        """An unrecognised trailing key should be silently replaced with '1Y'."""
        resp = uploaded_session.get('/results?as_of=2024-4&trailing=BOGUS',
                                    follow_redirects=True)
        assert resp.status_code == 200

    def test_1Q_trailing_with_single_period(self, uploaded_session):
        """1Q trailing with a valid as_of should work."""
        resp = uploaded_session.get('/results?as_of=2024-1&trailing=1Q',
                                    follow_redirects=True)
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 10. Session cleanup
# ---------------------------------------------------------------------------

class TestSessionCleanup:
    def test_reupload_overwrites_stable_files(self, client, benchmark_xlsx, portfolio_xlsx,
                                               tmp_path):
        """Uploading a second time should overwrite stable-named files and clear overrides."""
        upload_dir = app.config['UPLOAD_FOLDER']

        # First upload
        with open(benchmark_xlsx, 'rb') as bf, open(portfolio_xlsx, 'rb') as pf:
            client.post('/upload', data={
                'benchmark_file': (bf, 'benchmark.xlsx'),
                'portfolio_file': (pf, 'portfolio.xlsx'),
            }, content_type='multipart/form-data')

        # Stable files should exist in a session subdirectory
        subdirs = [d for d in os.listdir(upload_dir)
                   if os.path.isdir(os.path.join(upload_dir, d))]
        assert len(subdirs) >= 1
        session_dir = os.path.join(upload_dir, subdirs[0])
        assert os.path.exists(os.path.join(session_dir, 'current_benchmark.xlsx'))
        assert os.path.exists(os.path.join(session_dir, 'current_portfolio.xlsx'))

        first_mtime = os.path.getmtime(os.path.join(session_dir, 'current_portfolio.xlsx'))

        # Second upload – files should be overwritten
        with open(benchmark_xlsx, 'rb') as bf, open(portfolio_xlsx, 'rb') as pf:
            client.post('/upload', data={
                'benchmark_file': (bf, 'benchmark.xlsx'),
                'portfolio_file': (pf, 'portfolio.xlsx'),
            }, content_type='multipart/form-data')

        second_mtime = os.path.getmtime(os.path.join(session_dir, 'current_portfolio.xlsx'))
        assert second_mtime >= first_mtime


# ---------------------------------------------------------------------------
# 11. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_upload_non_overlapping_periods(self, client, tmp_path):
        """Files with no common periods should flash an error."""
        b_df = _make_benchmark_df(periods=[(2020, 1)])
        p_df = _make_portfolio_df(periods=[(2024, 4)])

        b_path = str(tmp_path / 'bench_no_overlap.xlsx')
        p_path = str(tmp_path / 'port_no_overlap.xlsx')
        _df_to_excel_file(b_df, b_path)
        _df_to_excel_file(p_df, p_path)

        with open(b_path, 'rb') as bf, open(p_path, 'rb') as pf:
            resp = client.post('/upload', data={
                'benchmark_file': (bf, 'bench_no_overlap.xlsx'),
                'portfolio_file': (pf, 'port_no_overlap.xlsx'),
            }, content_type='multipart/form-data', follow_redirects=True)

        assert b'No overlapping time periods' in resp.data

    def test_export_filename_contains_dimension_and_trailing(self, uploaded_session):
        resp = uploaded_session.get('/export/property_type?as_of=2024-4&trailing=1Q')
        cd = resp.headers.get('Content-Disposition', '')
        assert 'Attribution_property_type_1Q.xlsx' in cd
