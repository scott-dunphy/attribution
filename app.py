import os
import uuid
import secrets
import json
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, session, jsonify
import pandas as pd
from werkzeug.utils import secure_filename
from attribution.data_loader import load_file, load_aggregated_file, get_available_periods, get_common_periods
from attribution.aggregator import is_property_level, validate_property_file, aggregate_properties
from attribution.brinson_fachler import (
    run_full_attribution, run_attribution, run_all_trailing_periods,
    TRAILING_PERIOD_ORDER, TRAILING_PERIOD_DEFS, get_trailing_periods,
    get_available_trailing_periods, has_held_sold,
)
from attribution.template_generator import generate_template, generate_blank_template
from attribution.ncreif_api import (
    fetch_odce_data, get_cache_info, get_cached_data_path, save_to_cache,
)

app = Flask(__name__)

def _get_secret_key():
    """Load or generate a stable secret key so sessions survive restarts."""
    env_key = os.environ.get('SECRET_KEY')
    if env_key:
        return env_key
    key_file = os.path.join(os.path.dirname(__file__), '.secret_key')
    if os.path.exists(key_file):
        with open(key_file, 'r') as f:
            return f.read().strip()
    key = secrets.token_hex(32)
    with open(key_file, 'w') as f:
        f.write(key)
    return key

app.secret_key = _get_secret_key()
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB
app.config['SESSION_TYPE'] = 'cachelib'
app.config['SESSION_PERMANENT'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = 86400 * 30  # 30 days

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

from cachelib import FileSystemCache
from flask_session import Session
session_dir = os.path.join(os.path.dirname(__file__), '.flask_sessions')
app.config['SESSION_CACHELIB'] = FileSystemCache(cache_dir=session_dir, threshold=500)
Session(app)


def _get_session_dir():
    """Return a session-specific upload directory, creating it if needed."""
    sid = session.get('_upload_sid')
    if not sid:
        sid = uuid.uuid4().hex
        session['_upload_sid'] = sid
    d = os.path.join(app.config['UPLOAD_FOLDER'], sid)
    os.makedirs(d, exist_ok=True)
    return d


def _workspace_path():
    return os.path.join(_get_session_dir(), 'workspace.json')


def _save_workspace(benchmark_path, portfolio_path):
    """Persist current file paths to disk so they survive session/port changes."""
    data = {'benchmark_path': benchmark_path, 'portfolio_path': portfolio_path}
    with open(_workspace_path(), 'w') as f:
        json.dump(data, f)


def _load_workspace():
    """Load workspace from disk. Returns (benchmark_path, portfolio_path) or (None, None)."""
    wp = _workspace_path()
    if not os.path.exists(wp):
        return None, None
    try:
        with open(wp, 'r') as f:
            data = json.load(f)
        b = data.get('benchmark_path')
        p = data.get('portfolio_path')
        if b and not os.path.exists(b):
            b = None
        if p and not os.path.exists(p):
            p = None
        return b, p
    except (json.JSONDecodeError, KeyError):
        return None, None


def _restore_session_from_workspace():
    """If session is missing file paths, restore from workspace file."""
    b_path = session.get('benchmark_path')
    p_path = session.get('portfolio_path')
    if b_path and os.path.exists(b_path) and p_path and os.path.exists(p_path):
        return  # Session is fine
    wb, wp = _load_workspace()
    if wb and not (b_path and os.path.exists(b_path)):
        session['benchmark_path'] = wb
    if wp and not (p_path and os.path.exists(p_path)):
        session['portfolio_path'] = wp


def _save_upload(file_storage, stable_name=None):
    """Save an uploaded file and return the path.
    If stable_name is given, use it (overwriting any existing file).
    Files are stored in a session-specific directory to avoid collisions."""
    if stable_name:
        filename = stable_name
    else:
        safe_name = secure_filename(file_storage.filename) or 'upload.xlsx'
        filename = f"{uuid.uuid4().hex}_{safe_name}"
    path = os.path.join(_get_session_dir(), filename)
    file_storage.save(path)
    return path


def _cleanup_old_files():
    """Clear active session file references before a new upload."""
    session.pop('benchmark_path', None)
    session.pop('portfolio_path', None)


# ── Property overrides (sold flags + CBSA remaps) via CSV ─────────────

def _get_overrides_path():
    """Return the path to the property overrides CSV, or None."""
    p_path = session.get('portfolio_path')
    if not p_path:
        return None
    return os.path.splitext(p_path)[0] + '_overrides.csv'


def _get_cbsa_map_path():
    """Return the path to the CBSA remapping CSV, or None."""
    p_path = session.get('portfolio_path')
    if not p_path:
        return None
    return os.path.splitext(p_path)[0] + '_cbsa_map.csv'


def _load_overrides():
    """Load property sold flags from CSV. Returns dict of {PropertyID: {sold}}."""
    path = _get_overrides_path()
    if not path or not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path, dtype=str).fillna('')
        overrides = {}
        for _, row in df.iterrows():
            overrides[row['PropertyID']] = {
                'sold': row.get('Sold', '0') == '1',
            }
        return overrides
    except Exception:
        return {}


def _save_overrides(overrides):
    """Save property sold flags to CSV."""
    path = _get_overrides_path()
    if not path:
        return
    rows = []
    for prop_id, vals in overrides.items():
        if vals.get('sold'):
            rows.append({'PropertyID': prop_id, 'Sold': '1'})
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)
    elif os.path.exists(path):
        os.remove(path)


def _load_cbsa_map():
    """Load CBSA remappings from CSV. Returns dict of {original_cbsa: new_cbsa}."""
    path = _get_cbsa_map_path()
    if not path or not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path, dtype=str).fillna('')
        return dict(zip(df['OriginalCBSA'], df['NewCBSA']))
    except Exception:
        return {}


def _save_cbsa_map(cbsa_map):
    """Save CBSA remappings to CSV."""
    path = _get_cbsa_map_path()
    if not path:
        return
    rows = [{'OriginalCBSA': k, 'NewCBSA': v} for k, v in cbsa_map.items() if v]
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)
    elif os.path.exists(path):
        os.remove(path)


def _get_sold_properties():
    """Return list of PropertyIDs marked as sold."""
    overrides = _load_overrides()
    return [pid for pid, v in overrides.items() if v.get('sold')]


def _get_cbsa_remaps():
    """Return dict of {original_cbsa: new_cbsa} for CBSA overrides."""
    return _load_cbsa_map()


def _load_portfolio_with_overrides(p_path):
    """Load portfolio file, apply sold flags and CBSA remaps from CSV, and aggregate."""
    raw_df = pd.read_excel(p_path)
    if is_property_level(raw_df):
        errors = validate_property_file(raw_df)
        if errors:
            raise ValueError('; '.join(errors))
        for col in ['NOI', 'CapEx', 'MV', 'MVLag1', 'PSales', 'Denom',
                     'Income_Return', 'Capital_Return', 'Total_Return']:
            if col in raw_df.columns:
                raw_df[col] = pd.to_numeric(raw_df[col], errors='coerce').fillna(0.0)
        # Apply CBSA remaps (CBSA-to-CBSA, applies to all properties in that CBSA)
        cbsa_map = _get_cbsa_remaps()
        if cbsa_map and 'CBSAName' in raw_df.columns:
            raw_df['CBSAName'] = raw_df['CBSAName'].replace(cbsa_map)
        # Apply sold flags
        sold_props = _get_sold_properties()
        if sold_props and 'PropertyID' in raw_df.columns:
            raw_df['Sold'] = raw_df['PropertyID'].astype(str).isin(
                [str(p) for p in sold_props]
            ).astype(int)
        return aggregate_properties(raw_df)
    else:
        return load_aggregated_file(raw_df)


def _get_benchmark_cbsas(b_path):
    """Extract sorted list of unique CBSA names from the benchmark file."""
    b_df = load_file(b_path)
    cbsas = b_df[b_df['CBSAName'] != 'All']['CBSAName'].unique()
    return sorted(cbsas)


def _get_property_list(p_path):
    """Extract unique property list with attributes from raw portfolio file."""
    raw_df = pd.read_excel(p_path)
    if not is_property_level(raw_df) or 'PropertyID' not in raw_df.columns:
        return None

    # Apply CBSA remaps so the list reflects current state
    cbsa_map = _get_cbsa_remaps()

    props = []
    for pid, group in raw_df.groupby('PropertyID'):
        first = group.iloc[0]
        last = group.sort_values(['Year', 'Quarter']).iloc[-1]
        original_cbsa = first.get('CBSAName', '')
        # Use remapped CBSA if one exists for this CBSA
        display_cbsa = cbsa_map.get(original_cbsa, original_cbsa)
        props.append({
            'PropertyID': pid,
            'PropertyName': first.get('PropertyName', ''),
            'PropertyType': first.get('PropertyType', ''),
            'CBSAName': display_cbsa,
            'OriginalCBSAName': original_cbsa,
            'Quarters': len(group),
            'FirstPeriod': f"{int(first['Year'])} Q{int(first['Quarter'])}",
            'LastPeriod': f"{int(last['Year'])} Q{int(last['Quarter'])}",
            'LatestMV': float(last.get('MV', 0)),
        })
    return sorted(props, key=lambda p: p['PropertyName'])


@app.before_request
def before_request():
    """Restore session from workspace file if session is empty (e.g. port change)."""
    _restore_session_from_workspace()


@app.route('/')
def index():
    cache_info = get_cache_info(app.config['UPLOAD_FOLDER'])
    return render_template('index.html', ncreif_cache=cache_info)


@app.route('/upload', methods=['POST'])
def upload():
    benchmark_file = request.files.get('benchmark_file')
    portfolio_file = request.files.get('portfolio_file')
    use_ncreif = request.form.get('use_ncreif') == '1'

    if not use_ncreif and (not benchmark_file or not benchmark_file.filename):
        flash('Please upload a benchmark file or use the NCREIF API option.', 'error')
        return redirect(url_for('index'))
    if not portfolio_file or not portfolio_file.filename:
        flash('Please upload a portfolio file.', 'error')
        return redirect(url_for('index'))

    try:
        # Save new files to temp names first, validate before committing
        if use_ncreif:
            b_path = get_cached_data_path(app.config['UPLOAD_FOLDER'])
            if not b_path:
                flash('No cached NCREIF data. Fetch it first or upload a file.', 'error')
                return redirect(url_for('index'))
        else:
            b_path = _save_upload(benchmark_file, stable_name='_pending_benchmark.xlsx')

        p_path = _save_upload(portfolio_file, stable_name='_pending_portfolio.xlsx')

        b_df = load_file(b_path)
        p_df = load_file(p_path)

        common = get_common_periods(p_df, b_df)
        if not common:
            flash('No overlapping time periods found between the two files.', 'error')
            return redirect(url_for('index'))

        # Validation passed — promote pending files to stable names
        _cleanup_old_files()
        session_dir = _get_session_dir()
        stable_b = b_path  # NCREIF cache path stays as-is
        if not use_ncreif:
            stable_b = os.path.join(session_dir, 'current_benchmark.xlsx')
            os.replace(b_path, stable_b)
        stable_p = os.path.join(session_dir, 'current_portfolio.xlsx')
        os.replace(p_path, stable_p)
        b_path, p_path = stable_b, stable_p

        # Check for CBSA / PropertyType mismatches
        b_cbsas = set(b_df[b_df['CBSAName'] != 'All']['CBSAName'].unique())
        p_cbsas = set(p_df[p_df['CBSAName'] != 'All']['CBSAName'].unique())
        unmatched_cbsas = sorted(p_cbsas - b_cbsas)

        b_pts = set(b_df[b_df['PropertyType'] != 'All']['PropertyType'].unique())
        p_pts = set(p_df[p_df['PropertyType'] != 'All']['PropertyType'].unique())
        unmatched_pts = sorted(p_pts - b_pts)

        if unmatched_cbsas:
            flash(
                f"Warning: {len(unmatched_cbsas)} CBSA name(s) in your portfolio "
                f"do not match the benchmark and will be treated as portfolio-only "
                f"segments (zero benchmark weight, benchmark total return as proxy): "
                f"{', '.join(unmatched_cbsas)}",
                'warning'
            )
        if unmatched_pts:
            flash(
                f"Warning: {len(unmatched_pts)} Property Type(s) in your portfolio "
                f"do not match the benchmark and will be treated as portfolio-only "
                f"segments (zero benchmark weight, benchmark total return as proxy): "
                f"{', '.join(unmatched_pts)}",
                'warning'
            )

        session['benchmark_path'] = b_path
        session['portfolio_path'] = p_path
        _save_workspace(b_path, p_path)

        # Check if portfolio has properties we can list
        prop_list = _get_property_list(p_path)
        if prop_list:
            return redirect(url_for('properties'))

        # Default as_of = most recent common period
        as_of = common[-1]
        return redirect(url_for('results', as_of=f"{as_of[0]}-{as_of[1]}"))

    except ValueError as e:
        flash(f'Validation error: {str(e)}', 'error')
        return redirect(url_for('index'))
    except Exception as e:
        flash(f'Error processing files: {str(e)}', 'error')
        return redirect(url_for('index'))


@app.route('/properties')
def properties():
    p_path = session.get('portfolio_path')
    b_path = session.get('benchmark_path')
    if not p_path or not os.path.exists(p_path) or not b_path or not os.path.exists(b_path):
        flash('Please upload files first.', 'error')
        return redirect(url_for('index'))

    prop_list = _get_property_list(p_path)
    if not prop_list:
        flash('Portfolio file does not contain property-level data.', 'error')
        return redirect(url_for('index'))

    sold_properties = _get_sold_properties()
    cbsa_remaps = _get_cbsa_remaps()

    # Get benchmark CBSAs for mismatch detection
    benchmark_cbsas = _get_benchmark_cbsas(b_path) if b_path and os.path.exists(b_path) else []

    # Get common periods for the "Continue" link
    b_df = load_file(b_path)
    p_df = _load_portfolio_with_overrides(p_path)
    common = get_common_periods(p_df, b_df)
    if not common:
        flash('No overlapping time periods found between the two files.', 'error')
        return redirect(url_for('index'))
    as_of = common[-1]

    return render_template('properties.html',
                           properties=prop_list,
                           sold_properties=sold_properties,
                           cbsa_remaps=cbsa_remaps,
                           benchmark_cbsas=benchmark_cbsas,
                           as_of=as_of)


@app.route('/properties/save', methods=['POST'])
def save_overrides_route():
    """Save all property overrides (sold flags + CBSA remaps) in one shot."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Missing data'}), 400

    # Save sold flags (per property)
    sold_overrides = {}
    for prop_id, vals in data.get('sold', {}).items():
        if vals:
            sold_overrides[str(prop_id)] = {'sold': True}
    _save_overrides(sold_overrides)

    # Save CBSA remaps (per CBSA, applies to all properties in that CBSA)
    cbsa_map = data.get('cbsa_map', {})
    _save_cbsa_map(cbsa_map)

    return jsonify({'ok': True})


@app.route('/fetch-ncreif', methods=['POST'])
def fetch_ncreif():
    email = request.form.get('ncreif_email', '').strip()
    password = request.form.get('ncreif_password', '').strip()

    if not email or not password:
        flash('Please provide your NCREIF email and password.', 'error')
        return redirect(url_for('index'))

    try:
        df = fetch_odce_data(email, password)
        save_to_cache(df, app.config['UPLOAD_FOLDER'])

        max_yyyyq = int(df['YYYYQ'].max())
        max_year = max_yyyyq // 10
        max_quarter = max_yyyyq % 10
        flash(
            f"NCREIF ODCE data fetched and cached: {len(df):,} rows "
            f"through {max_year} Q{max_quarter}.",
            'success',
        )
    except Exception as e:
        flash(f'Error fetching NCREIF data: {str(e)}', 'error')

    return redirect(url_for('index'))


@app.route('/results')
def results():
    b_path = session.get('benchmark_path')
    p_path = session.get('portfolio_path')

    if not b_path or not p_path or not os.path.exists(b_path) or not os.path.exists(p_path):
        flash('Please upload files first.', 'error')
        return redirect(url_for('index'))

    try:
        b_df = load_file(b_path)
        p_df = _load_portfolio_with_overrides(p_path)

        common_periods = get_common_periods(p_df, b_df)
        if not common_periods:
            flash('No overlapping time periods found between the two files.', 'error')
            return redirect(url_for('index'))

        # Parse as_of quarter
        as_of = common_periods[-1]
        as_of_str = request.args.get('as_of', '')
        if as_of_str:
            try:
                parts = as_of_str.split('-')
                candidate = (int(parts[0]), int(parts[1]))
                if candidate in common_periods:
                    as_of = candidate
            except (IndexError, ValueError):
                pass  # Fall through to default

        # Selected trailing period (default 1Y, fallback to 1Q)
        trailing_key = request.args.get('trailing', '1Y')
        if trailing_key not in TRAILING_PERIOD_ORDER:
            trailing_key = '1Y'

        # Determine available trailing periods
        available_trailing = get_available_trailing_periods(common_periods, as_of)
        if not available_trailing.get(trailing_key, False):
            # Fall back to longest available
            for k in reversed(TRAILING_PERIOD_ORDER):
                if available_trailing.get(k, False):
                    trailing_key = k
                    break

        # Get the periods for this trailing window
        periods = get_trailing_periods(common_periods, as_of, trailing_key)

        # Reconciliation method: 'scale' (proportional) or 'residual' (raw + residual row)
        recon_method = request.args.get('recon', 'scale')
        if recon_method not in ('scale', 'residual'):
            recon_method = 'scale'

        # Run attribution for selected trailing period
        results_data = run_full_attribution(p_df, b_df, periods, method=recon_method)

        # Annualize effects for periods > 4 quarters so everything is
        # on the same basis as the annualized returns.
        effect_cols = ['Allocation', 'Selection', 'Total_Active']

        def _annualize_effects(result):
            """Scale cumulative attribution effects to an annualized basis."""
            if result['n_quarters'] <= 4:
                return  # Sub-annual / 1-year: no annualization needed
            cum_active = result['cum_rp'] - result['cum_rb']
            ann_active = result['ann_rp'] - result['ann_rb']
            if abs(cum_active) < 1e-12:
                return  # No excess return to rescale
            scale = ann_active / cum_active
            summary = result['summary']
            for col in effect_cols:
                if col in summary.columns:
                    summary[col] = summary[col] * scale
            # Annualize segment-level cumulative returns
            n_years = result['n_quarters'] / 4.0
            for col in ['r_p_cum', 'r_b_cum']:
                if col in summary.columns:
                    summary[col] = summary[col].apply(
                        lambda r: (1 + r) ** (1 / n_years) - 1 if r > -1 else -1.0
                    )

        def _blank_zero_weight_returns(result):
            """Set segment returns to None where the portfolio or benchmark has no exposure."""
            summary = result['summary']
            if summary.empty:
                return
            import numpy as np
            if 'r_p_cum' in summary.columns:
                summary.loc[summary['w_p'].abs() < 1e-10, 'r_p_cum'] = np.nan
            if 'r_b_cum' in summary.columns:
                summary.loc[summary['w_b'].abs() < 1e-10, 'r_b_cum'] = np.nan

        for dim_result in results_data.values():
            _annualize_effects(dim_result)
            _blank_zero_weight_returns(dim_result)

        # Compute totals for each dimension
        def compute_totals(result):
            summary = result['summary']
            if summary.empty:
                return {}
            return {
                'Allocation': summary['Allocation'].sum(),
                'Selection': summary['Selection'].sum(),
                'Total_Active': summary['Total_Active'].sum(),
                'Contribution_P': summary['Contribution_P'].sum(),
                'Contribution_B': summary['Contribution_B'].sum(),
                'w_p': summary['w_p'].sum(),
                'w_b': summary['w_b'].sum(),
                'cum_rp': result['cum_rp'],
                'cum_rb': result['cum_rb'],
                'ann_rp': result['ann_rp'],
                'ann_rb': result['ann_rb'],
                'n_quarters': result['n_quarters'],
            }

        show_held_sold = 'by_held_sold' in results_data

        totals = {
            'by_property_type': compute_totals(results_data['by_property_type']),
            'by_cbsa': compute_totals(results_data['by_cbsa']),
            'by_property_type_cbsa': compute_totals(results_data['by_property_type_cbsa']),
        }

        def to_records(summary_df):
            if summary_df.empty:
                return []
            return summary_df.to_dict('records')

        summaries = {
            'by_property_type': to_records(results_data['by_property_type']['summary']),
            'by_cbsa': to_records(results_data['by_cbsa']['summary']),
            'by_property_type_cbsa': to_records(results_data['by_property_type_cbsa']['summary']),
        }

        def detail_to_records(detail_df):
            if detail_df.empty:
                return []
            return detail_df.to_dict('records')

        details = {
            'by_property_type': detail_to_records(results_data['by_property_type']['detail']),
            'by_cbsa': detail_to_records(results_data['by_cbsa']['detail']),
            'by_property_type_cbsa': detail_to_records(results_data['by_property_type_cbsa']['detail']),
        }

        if show_held_sold:
            totals['by_held_sold'] = compute_totals(results_data['by_held_sold'])
            summaries['by_held_sold'] = to_records(results_data['by_held_sold']['summary'])
            details['by_held_sold'] = detail_to_records(results_data['by_held_sold']['detail'])

        # Build heatmap data: PropertyType (columns) x CBSA (rows), cell = w_p - w_b
        heatmap = {'property_types': [], 'cbsas': [], 'cells': {}}
        cross_summary = results_data['by_property_type_cbsa']['summary']
        if not cross_summary.empty:
            pts = sorted(cross_summary['PropertyType'].dropna().unique())
            pts = [p for p in pts if p != 'Residual/Other']
            cbsas = sorted(cross_summary['CBSAName'].dropna().unique())
            cbsas = [c for c in cbsas if c != 'Residual/Other']
            heatmap['property_types'] = pts
            heatmap['cbsas'] = cbsas
            for _, row in cross_summary.iterrows():
                pt = row.get('PropertyType')
                cbsa = row.get('CBSAName')
                if pt in pts and cbsa in cbsas:
                    diff = row['w_p'] - row['w_b']
                    heatmap['cells'][(pt, cbsa)] = round(diff * 100, 2)
            # Row totals (per CBSA) and column totals (per PropertyType)
            row_totals = {}
            col_totals = {}
            for cbsa in cbsas:
                row_totals[cbsa] = round(sum(heatmap['cells'].get((pt, cbsa), 0) for pt in pts), 2)
            for pt in pts:
                col_totals[pt] = round(sum(heatmap['cells'].get((pt, cbsa), 0) for cbsa in cbsas), 2)
            heatmap['row_totals'] = row_totals
            heatmap['col_totals'] = col_totals
            heatmap['grand_total'] = round(sum(row_totals.values()), 2)

        # Build single-dimension heatmap data for PropertyType and CBSA
        def _build_1d_heatmap(summary_df, key_col):
            if summary_df.empty:
                return []
            rows = []
            for _, row in summary_df.iterrows():
                name = row.get(key_col)
                if name == 'Residual/Other':
                    continue
                rows.append({
                    'name': name,
                    'w_p': round(row['w_p'] * 100, 2),
                    'w_b': round(row['w_b'] * 100, 2),
                    'diff': round((row['w_p'] - row['w_b']) * 100, 2),
                })
            return sorted(rows, key=lambda r: r['diff'])

        heatmap_pt = _build_1d_heatmap(results_data['by_property_type']['summary'], 'PropertyType')
        heatmap_cbsa = _build_1d_heatmap(results_data['by_cbsa']['summary'], 'CBSAName')

        # Trailing period label
        n_q = TRAILING_PERIOD_DEFS.get(trailing_key)
        if n_q is None:
            trailing_label = 'Since Inception'
        elif n_q == 1:
            trailing_label = 'Trailing Quarter'
        else:
            trailing_label = f"Trailing {trailing_key}"

        return render_template('results.html',
                               summaries=summaries,
                               totals=totals,
                               details=details,
                               heatmap=heatmap,
                               heatmap_pt=heatmap_pt,
                               heatmap_cbsa=heatmap_cbsa,
                               common_periods=common_periods,
                               as_of=as_of,
                               trailing_key=trailing_key,
                               trailing_label=trailing_label,
                               available_trailing=available_trailing,
                               trailing_period_order=TRAILING_PERIOD_ORDER,
                               selected_periods=periods,
                               show_held_sold=show_held_sold,
                               recon_method=recon_method)

    except Exception as e:
        flash(f'Error computing attribution: {str(e)}', 'error')
        import traceback
        traceback.print_exc()
        return redirect(url_for('index'))


@app.route('/download-template', methods=['GET', 'POST'])
def download_template():
    # POST with benchmark file: generate customized template
    if request.method == 'POST':
        benchmark_file = request.files.get('benchmark_file')
        if benchmark_file and benchmark_file.filename:
            b_path = _save_upload(benchmark_file)
            try:
                b_df = load_file(b_path)
                output = generate_template(b_df)
                return send_file(
                    output,
                    mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    as_attachment=True,
                    download_name='Portfolio_Template.xlsx'
                )
            except Exception as e:
                flash(f'Error generating template: {str(e)}', 'error')
                return redirect(url_for('index'))
            finally:
                if os.path.exists(b_path):
                    try:
                        os.remove(b_path)
                    except OSError:
                        pass

    # GET or POST without file: use session benchmark, then NCREIF cache, then generic
    for candidate_path in [
        session.get('benchmark_path'),
        get_cached_data_path(app.config['UPLOAD_FOLDER']),
    ]:
        if candidate_path and os.path.exists(candidate_path):
            try:
                b_df = load_file(candidate_path)
                output = generate_template(b_df)
                return send_file(
                    output,
                    mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    as_attachment=True,
                    download_name='Portfolio_Template.xlsx'
                )
            except Exception:
                continue  # Try next source

    # Generic template — no benchmark needed
    output = generate_blank_template()
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name='Portfolio_Template.xlsx'
    )


@app.route('/export/<dimension>')
def export(dimension):
    """Export attribution results to Excel."""
    b_path = session.get('benchmark_path')
    p_path = session.get('portfolio_path')
    if not b_path or not p_path or not os.path.exists(b_path) or not os.path.exists(p_path):
        flash('Please upload files first.', 'error')
        return redirect(url_for('index'))

    try:
        b_df = load_file(b_path)
        p_df = _load_portfolio_with_overrides(p_path)
        common_periods = get_common_periods(p_df, b_df)
        if not common_periods:
            flash('No overlapping time periods.', 'error')
            return redirect(url_for('index'))

        trailing_key = request.args.get('trailing', '1Y')
        if trailing_key not in TRAILING_PERIOD_ORDER:
            flash('Invalid trailing period.', 'error')
            return redirect(url_for('results'))

        as_of = common_periods[-1]
        as_of_str = request.args.get('as_of', '')
        if as_of_str:
            try:
                parts = as_of_str.split('-')
                candidate = (int(parts[0]), int(parts[1]))
                if candidate in common_periods:
                    as_of = candidate
            except (IndexError, ValueError):
                pass

        periods = get_trailing_periods(common_periods, as_of, trailing_key)

        recon_method = request.args.get('recon', 'scale')
        if recon_method not in ('scale', 'residual'):
            recon_method = 'scale'

        dim_map = {
            'property_type': 'PropertyType',
            'cbsa': 'CBSAName',
            'property_type_cbsa': 'PropertyType_CBSAName',
            'held_sold': 'HeldSold',
        }
        if dimension not in dim_map:
            flash('Invalid dimension.', 'error')
            return redirect(url_for('results'))

        result = run_attribution(p_df, b_df, periods, dim_map[dimension], method=recon_method)

        from io import BytesIO
        import pandas as pd
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            if not result['summary'].empty:
                result['summary'].to_excel(writer, index=False, sheet_name='Summary')
            if not result['detail'].empty:
                result['detail'].to_excel(writer, index=False, sheet_name='Detail')
        output.seek(0)

        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=f'Attribution_{dimension}_{trailing_key}.xlsx'
        )
    except Exception as e:
        flash(f'Export error: {str(e)}', 'error')
        return redirect(url_for('results'))


if __name__ == '__main__':
    app.run(debug=True, port=5000)
