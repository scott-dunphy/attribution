"""
NCREIF ODCE API client with file-based caching.

Pulls four levels of ODCE return data (Total, By Property Type, By CBSA,
By Property Type x CBSA) and caches the result as an Excel file.
"""

import os
import time
import json
import defusedxml.ElementTree as ET
from datetime import datetime, timedelta

import requests
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────
BASE_URL = "https://qt-api.ncreif.org"
RATE_LIMIT_SECONDS = 8

SELECT = (
    "SUM(NOI) AS NOI, SUM(CapEx) AS CapEx, SUM(MV) AS MV, SUM(MVLag1) AS MVLag1, "
    "SUM(PSales) AS PSales, SUM(Denom) AS Denom, "
    "SUM(NOI)/SUM(Denom) AS 'Income_Return', "
    "(SUM(MV)-SUM(MVLag1)-SUM(CapEx)+SUM(PSales))/SUM(Denom) AS 'Capital_Return', "
    "(SUM(NOI)+SUM(MV)-SUM(MVLag1)-SUM(CapEx)+SUM(PSales))/SUM(Denom) AS 'Total_Return', "
    "COUNT(MV) AS 'Prop_Count'"
)

WHERE = "NPI_Plus=1 AND FundType = 'D'"

QUERIES = [
    {"name": "Total",                "group_by": "[Year],[YYYYQ]"},
    {"name": "By_PropertyType",      "group_by": "[Year],[YYYYQ],[PropertyType]"},
    {"name": "By_CBSA",             "group_by": "[Year],[YYYYQ],[CBSAName]"},
    {"name": "By_PropertyType_CBSA", "group_by": "[Year],[YYYYQ],[PropertyType],[CBSAName]"},
]

COLUMN_ORDER = [
    "Query_Level", "Year", "YYYYQ", "Quarter", "PropertyType", "CBSAName",
    "NOI", "CapEx", "MV", "MVLag1", "PSales", "Denom",
    "Income_Return", "Capital_Return", "Total_Return", "Prop_Count"
]


def _authenticate(email, password):
    """Authenticate and return a bearer token."""
    resp = requests.post(
        f"{BASE_URL}/Login/Login",
        json={"Email": email, "Password": password},
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    token_data = json.loads(resp.text)
    if isinstance(token_data, dict) and "message" in token_data:
        return token_data["message"]
    return resp.text.strip('"')


def _parse_xml_response(xml_text):
    """Parse NCREIF XML response into list of dicts."""
    root = ET.fromstring(xml_text)
    records = []
    for result in root:
        record = {}
        for field in result:
            val = field.text
            if val is not None:
                try:
                    val = float(val) if '.' in val else int(val)
                except ValueError:
                    pass
            record[field.tag] = val
        records.append(record)
    return records


def _run_query(headers, query_def):
    """Execute a single NCREIF query and return a DataFrame."""
    payload = {
        "p_DataTypeId": 3,
        "p_SelectQuery": SELECT,
        "p_WhereClause": WHERE,
        "p_GroupbyClause": query_def["group_by"],
        "p_QueryData": 0,
        "kpi": "",
    }
    resp = requests.post(
        f"{BASE_URL}/QT/ExecuteQuery",
        json=payload,
        headers=headers,
        timeout=120,
    )
    resp.raise_for_status()
    records = _parse_xml_response(resp.text)
    df = pd.DataFrame(records)
    df["Query_Level"] = query_def["name"]
    return df


def _build_dataframe(frames):
    """Combine query results into a single formatted DataFrame."""
    combined = pd.concat(frames, ignore_index=True)
    combined["PropertyType"] = combined["PropertyType"].fillna("All")
    combined["CBSAName"] = combined["CBSAName"].fillna("All")
    combined["Quarter"] = combined["YYYYQ"].astype(str).str[-1].astype(int)
    combined = combined[COLUMN_ORDER]
    combined = combined.sort_values(
        ["Query_Level", "PropertyType", "CBSAName", "YYYYQ"],
        na_position="first",
    ).reset_index(drop=True)
    return combined


def fetch_odce_data(email, password, progress_callback=None):
    """
    Fetch all four levels of ODCE data from the NCREIF API.

    Args:
        email: NCREIF account email
        password: NCREIF account password
        progress_callback: optional callable(step, total, message)

    Returns:
        pandas DataFrame with COLUMN_ORDER columns

    Raises:
        requests.RequestException on network/auth errors
        ValueError on empty or invalid data
    """
    if progress_callback:
        progress_callback(0, len(QUERIES) + 1, "Authenticating with NCREIF...")

    token = _authenticate(email, password)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    frames = []
    for i, q in enumerate(QUERIES):
        if progress_callback:
            progress_callback(i + 1, len(QUERIES) + 1, f"Pulling {q['name']}...")
        df = _run_query(headers, q)
        if df.empty:
            raise ValueError(f"No data returned for query: {q['name']}")
        frames.append(df)
        if i < len(QUERIES) - 1:
            time.sleep(RATE_LIMIT_SECONDS)

    combined = _build_dataframe(frames)
    if combined.empty:
        raise ValueError("No data returned from NCREIF API")
    return combined


# ── Cache Management ───────────────────────────────────────────────────────

CACHE_METADATA_FILE = "ncreif_cache_meta.json"


def _get_cache_dir(app_upload_folder):
    """Return the cache directory path, creating it if needed."""
    cache_dir = os.path.join(app_upload_folder, 'ncreif_cache')
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _get_cache_paths(cache_dir):
    """Return (data_path, meta_path) for the cached NCREIF data."""
    return (
        os.path.join(cache_dir, 'ncreif_odce_cache.xlsx'),
        os.path.join(cache_dir, CACHE_METADATA_FILE),
    )


def get_cache_info(app_upload_folder):
    """
    Return cache metadata if a valid cache exists, else None.

    Returns dict with keys: fetched_at, row_count, max_period, age_description
    """
    cache_dir = _get_cache_dir(app_upload_folder)
    data_path, meta_path = _get_cache_paths(cache_dir)

    if not os.path.exists(data_path) or not os.path.exists(meta_path):
        return None

    try:
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        fetched_at = datetime.fromisoformat(meta['fetched_at'])
        age = datetime.now() - fetched_at
        if age.days > 0:
            age_desc = f"{age.days} day{'s' if age.days != 1 else ''} ago"
        elif age.seconds >= 3600:
            hours = age.seconds // 3600
            age_desc = f"{hours} hour{'s' if hours != 1 else ''} ago"
        else:
            minutes = max(1, age.seconds // 60)
            age_desc = f"{minutes} minute{'s' if minutes != 1 else ''} ago"

        meta['age_description'] = age_desc
        return meta
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def get_cached_data_path(app_upload_folder):
    """Return the path to cached NCREIF data if it exists, else None."""
    cache_dir = _get_cache_dir(app_upload_folder)
    data_path, meta_path = _get_cache_paths(cache_dir)
    if os.path.exists(data_path) and os.path.exists(meta_path):
        return data_path
    return None


def save_to_cache(df, app_upload_folder):
    """Save a DataFrame to the cache with metadata."""
    cache_dir = _get_cache_dir(app_upload_folder)
    data_path, meta_path = _get_cache_paths(cache_dir)

    df.to_excel(data_path, index=False, sheet_name='ODCE_Returns')

    max_yyyyq = int(df['YYYYQ'].max())
    max_year = max_yyyyq // 10
    max_quarter = max_yyyyq % 10

    meta = {
        'fetched_at': datetime.now().isoformat(),
        'row_count': len(df),
        'max_period': f"{max_year} Q{max_quarter}",
        'max_yyyyq': max_yyyyq,
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f)

    return data_path
