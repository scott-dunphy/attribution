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


def _workspace_path():
    return os.path.join(app.config['UPLOAD_FOLDER'], 'workspace.json')


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
    If stable_name is given, use it (overwriting any existing file)."""
    if stable_name:
        filename = stable_name
    else:
        safe_name = secure_filename(file_storage.filename) or 'upload.xlsx'
        filename = f"{uuid.uuid4().hex}_{safe_name}"
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
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
    # Overrides CSV sits next to the portfolio file, same name with _overrides.csv
    return os.path.splitext(p_path)[0] + '_overrides.csv'


def _load_overrides():
    """Load property overrides from CSV. Returns dict of {PropertyID: {sold, cbsa_override}}."""
    path = _get_overrides_path()
    if not path or not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path, dtype=str).fillna('')
        overrides = {}
        for _, row in df.iterrows():
            overrides[row['PropertyID']] = {
                'sold': row.get('Sold', '0') == '1',
                'cbsa_override': row.get('CBSAOverride', ''),
            }
        return overrides
    except Exception:
        return {}


def _save_overrides(overrides):
    """Save property overrides dict to CSV."""
    path = _get_overrides_path()
    if not path:
        return
    rows = []
    for prop_id, vals in overrides.items():
        rows.append({
            'PropertyID': prop_id,
            'Sold': '1' if vals.get('sold') else '0',
            'CBSAOverride': vals.get('cbsa_override', ''),
        })
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)
    elif os.path.exists(path):
        os.remove(path)


def _get_sold_properties():
    """Return list of PropertyIDs marked as sold."""
    overrides = _load_overrides()
    return [pid for pid, v in overrides.items() if v.get('sold')]


def _get_cbsa_remaps():
    """Return dict of {PropertyID: new_cbsa} for properties with CBSA overrides."""
    overrides = _load_overrides()
    return {pid: v['cbsa_override'] for pid, v in overrides.items()
            if v.get('cbsa_override')}


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
        # Apply CBSA remaps from overrides CSV
        cbsa_remaps = _get_cbsa_remaps()
        if cbsa_remaps and 'PropertyID' in raw_df.columns:
            for prop_id, new_cbsa in cbsa_remaps.items():
                mask = raw_df['PropertyID'].astype(str) == str(prop_id)
                raw_df.loc[mask, 'CBSAName'] = new_cbsa
        # Apply sold flags from overrides CSV
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

    # Apply CBSA remaps from overrides CSV so the list reflects current state
    cbsa_remaps = _get_cbsa_remaps()

    props = []
    for pid, group in raw_df.groupby('PropertyID'):
        first = group.iloc[0]
        last = group.sort_values(['Year', 'Quarter']).iloc[-1]
        original_cbsa = first.get('CBSAName', '')
        # Use remapped CBSA if one exists for this property
        display_cbsa = cbsa_remaps.get(str(pid), original_cbsa)
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
        _cleanup_old_files()

        # Resolve benchmark: NCREIF cache or uploaded file
        if use_ncreif:
            b_path = get_cached_data_path(app.config['UPLOAD_FOLDER'])
            if not b_path:
                flash('No cached NCREIF data. Fetch it first or upload a file.', 'error')
                return redirect(url_for('index'))
        else:
            b_path = _save_upload(benchmark_file, stable_name='current_benchmark.xlsx')

        p_path = _save_upload(portfolio_file, stable_name='current_portfolio.xlsx')

        b_df = load_file(b_path)
        p_df = load_file(p_path)

        common = get_common_periods(p_df, b_df)
        if not common:
            flash('No overlapping time periods found between the two files.', 'error')
            return redirect(url_for('index'))

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
    if not p_path or not os.path.exists(p_path):
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
    as_of = common[-1] if common else (2024, 4)

    return render_template('properties.html',
                           properties=prop_list,
                           sold_properties=sold_properties,
                           cbsa_remaps=cbsa_remaps,
                           benchmark_cbsas=benchmark_cbsas,
                           as_of=as_of)


@app.route('/properties/save', methods=['POST'])
def save_overrides():
    """Save all property overrides (sold flags + CBSA remaps) in one shot."""
    data = request.get_json()
    if not data or 'overrides' not in data:
        return jsonify({'error': 'Missing overrides'}), 400

    overrides = {}
    for prop_id, vals in data['overrides'].items():
        entry = {}
        if vals.get('sold'):
            entry['sold'] = True
        if vals.get('cbsa_override'):
            entry['cbsa_override'] = vals['cbsa_override']
        if entry:
            overrides[str(prop_id)] = {
                'sold': entry.get('sold', False),
                'cbsa_override': entry.get('cbsa_override', ''),
            }

    _save_overrides(overrides)
    return jsonify({'ok': True, 'count': len(overrides)})


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

        # Run attribution for selected trailing period
        results_data = run_full_attribution(p_df, b_df, periods)

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
                               common_periods=common_periods,
                               as_of=as_of,
                               trailing_key=trailing_key,
                               trailing_label=trailing_label,
                               available_trailing=available_trailing,
                               trailing_period_order=TRAILING_PERIOD_ORDER,
                               selected_periods=periods,
                               show_held_sold=show_held_sold)

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
    if not b_path or not p_path:
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

        dim_map = {
            'property_type': 'PropertyType',
            'cbsa': 'CBSAName',
            'property_type_cbsa': 'PropertyType_CBSAName',
            'held_sold': 'HeldSold',
        }
        if dimension not in dim_map:
            flash('Invalid dimension.', 'error')
            return redirect(url_for('results'))

        result = run_attribution(p_df, b_df, periods, dim_map[dimension])

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
