import pandas as pd
import numpy as np
from io import BytesIO


PORTFOLIO_COLUMNS = [
    'Year', 'Quarter', 'YYYYQ',
    'PropertyID', 'PropertyName', 'PropertyType', 'CBSAName',
    'NOI', 'CapEx', 'MV', 'MVLag1', 'PSales', 'Denom',
    'Income_Return', 'Capital_Return', 'Total_Return',
]


def generate_blank_template():
    """
    Generate a generic portfolio template with sample data and column headers.
    No benchmark required — uses common NCREIF property types and example CBSAs.
    """
    sample_segments = [
        ('Office', 'NY-NJ-PA-New York-Jersey City-White Plains'),
        ('Industrial', 'CA-Riverside-San Bernardino-Ontario'),
        ('Residential', 'DC-VA-MD-WV-Washington-Arlington-Alexandria'),
        ('Retail', 'IL-IN-WI-Chicago-Naperville-Arlington Heights'),
        ('Hotel', 'FL-Miami-Miami Beach-Kendall'),
    ]
    sample_periods = [(2024, 1), (2024, 2), (2024, 3), (2024, 4)]

    np.random.seed(42)
    rows = []
    for i, (pt, cbsa) in enumerate(sample_segments):
        pid = f'PROP-{i+1:03d}'
        pname = f'Sample {pt} Property'
        for year, quarter in sample_periods:
            mv_lag = np.random.uniform(50_000_000, 200_000_000)
            noi = mv_lag * np.random.uniform(0.010, 0.020)
            capex = mv_lag * np.random.uniform(0.001, 0.005)
            income_ret = noi / mv_lag
            capital_ret = np.random.uniform(-0.03, 0.05)
            total_ret = income_ret + capital_ret
            mv = mv_lag * (1 + capital_ret)
            denom = mv_lag - capex / 2
            rows.append({
                'Year': year, 'Quarter': quarter,
                'YYYYQ': year * 10 + quarter,
                'PropertyID': pid, 'PropertyName': pname,
                'PropertyType': pt, 'CBSAName': cbsa,
                'NOI': round(noi), 'CapEx': round(capex),
                'MV': round(mv), 'MVLag1': round(mv_lag),
                'PSales': 0, 'Denom': round(denom),
                'Income_Return': round(income_ret, 6),
                'Capital_Return': round(capital_ret, 6),
                'Total_Return': round(total_ret, 6),
            })

    template_df = pd.DataFrame(rows, columns=PORTFOLIO_COLUMNS)

    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        template_df.to_excel(writer, index=False, sheet_name='Portfolio_Properties')
    output.seek(0)
    return output


def generate_template(benchmark_df, periods=None):
    """
    Generate a property-level portfolio template pre-filled with sample data
    drawn from the benchmark's actual property types and CBSAs.
    """
    # Extract the benchmark's segment universe
    property_types = sorted(
        benchmark_df[benchmark_df['PropertyType'] != 'All']['PropertyType'].unique()
    )
    cbsa_names = sorted(
        benchmark_df[benchmark_df['CBSAName'] != 'All']['CBSAName'].unique()
    )

    if periods:
        period_set = set(periods)
    else:
        total = benchmark_df[benchmark_df['Query_Level'] == 'Total']
        period_set = set(zip(total['Year'].astype(int), total['Quarter'].astype(int)))

    periods_sorted = sorted(period_set)

    # Use the 4 most recent quarters for sample data
    sample_periods = periods_sorted[-4:] if len(periods_sorted) >= 4 else periods_sorted

    # Build sample properties from the benchmark's actual segments.
    # Pick up to 5 (PropertyType, CBSA) pairs that exist in the cross-dimension.
    cross = benchmark_df[benchmark_df['Query_Level'] == 'By_PropertyType_CBSA']
    cross_pairs = cross[['PropertyType', 'CBSAName']].drop_duplicates()

    # Pick the largest segments by average Denom so the sample is realistic
    cross_avg = benchmark_df[
        benchmark_df['Query_Level'] == 'By_PropertyType_CBSA'
    ].groupby(['PropertyType', 'CBSAName'])['Denom'].mean().reset_index()
    cross_avg = cross_avg.sort_values('Denom', ascending=False)
    sample_segments = cross_avg.head(5)[['PropertyType', 'CBSAName']].values.tolist()

    # Fallback if cross-dimension data is sparse
    if len(sample_segments) < 2:
        for pt in property_types[:3]:
            for cbsa in cbsa_names[:2]:
                sample_segments.append([pt, cbsa])
                if len(sample_segments) >= 5:
                    break
            if len(sample_segments) >= 5:
                break

    # Generate sample data
    np.random.seed(42)
    rows = []
    for i, (pt, cbsa) in enumerate(sample_segments):
        pid = f'PROP-{i+1:03d}'
        pname = f'Sample {pt} in {cbsa.split("-")[-1]}'
        for year, quarter in sample_periods:
            mv_lag = np.random.uniform(50_000_000, 200_000_000)
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
                'PropertyID': pid,
                'PropertyName': pname,
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

    template_df = pd.DataFrame(rows, columns=PORTFOLIO_COLUMNS)

    pt_ref = pd.DataFrame({'PropertyType': property_types})
    cbsa_ref = pd.DataFrame({'CBSAName': cbsa_names})

    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        template_df.to_excel(writer, index=False, sheet_name='Portfolio_Properties')
        pt_ref.to_excel(writer, index=False, sheet_name='PropertyTypes_Reference')
        cbsa_ref.to_excel(writer, index=False, sheet_name='CBSAs_Reference')
    output.seek(0)
    return output
