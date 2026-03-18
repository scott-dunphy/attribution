# Portfolio Attribution Analysis

A web application for running Brinson-Hood-Beebower (MSCI) attribution analysis on real estate portfolios against the NCREIF ODCE benchmark.

## Overview

Upload property-level portfolio data, review and adjust property attributes, then run multi-period attribution analysis decomposing active returns into **allocation** and **selection** effects across multiple dimensions:

- **By Property Type** (Office, Industrial, Retail, Residential, etc.)
- **By CBSA** (metro area)
- **By Property Type x CBSA** (cross-sectional)
- **By Held vs. Sold** (optional, if properties are marked as sold)

Supports trailing period analysis (1Q, 1Y, 3Y, 5Y, 7Y, 10Y, Since Inception) with geometric compounding and proportional linking for multi-period reconciliation.

## Quick Start

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Open http://localhost:5000 in your browser.

## Workflow

1. **Benchmark Data** -- Either fetch directly from the NCREIF API (requires NCREIF Query Tool credentials) or upload a pre-formatted benchmark Excel file. NCREIF data is cached locally after the first fetch.

2. **Upload Portfolio** -- Upload an Excel file with one row per property per quarter. The app aggregates property-level data into the groupings needed for attribution.

3. **Review Properties** -- Review the property list, mark any properties as sold, and fix CBSA mismatches against the benchmark using dropdowns. Changes are saved to a CSV and persist across sessions.

4. **View Results** -- Attribution results by dimension with trailing period selection. Export any dimension to Excel.

## Portfolio File Format

| Column | Description | Example |
|--------|-------------|---------|
| `Year` | Year | 2024 |
| `Quarter` | Quarter number | 1 |
| `YYYYQ` | Year-quarter code | 20241 |
| `PropertyID` | Unique property identifier | PROP-001 |
| `PropertyName` | Property name | Oakwood Industrial Park |
| `PropertyType` | Property type | Industrial |
| `CBSAName` | Metro area (should match benchmark) | CA-Los Angeles-Long Beach-Glendale |
| `NOI` | Net Operating Income ($) | 1446527 |
| `CapEx` | Capital Expenditures ($) | 835810 |
| `MV` | Market Value, end of quarter ($) | 130600000 |
| `MVLag1` | Market Value, beginning of quarter ($) | 111516100 |
| `PSales` | Partial Sales ($) | 0 |
| `Denom` | Return denominator (adjusted BOQ MV) | 111451829 |
| `Income_Return` | Quarterly income return (decimal) | 0.0130 |
| `Capital_Return` | Quarterly capital return (decimal) | 0.1637 |
| `Total_Return` | Quarterly total return (decimal) | 0.1767 |

Download a template from the app's upload page (with or without a benchmark for customized CBSAs/property types).

## Attribution Method

Uses Brinson-Hood-Beebower (MSCI) decomposition:

- **Allocation** = (w_p - w_b) x (r_b - R_b) -- Did overweighting/underweighting a segment help or hurt?
- **Selection** = w_p x (r_p - r_b) -- Did the portfolio's assets in a segment outperform or underperform?
- **Total Active** = Allocation + Selection

Where:
- w_p, w_b = portfolio and benchmark segment weights
- r_p, r_b = portfolio and benchmark segment returns
- R_b = total benchmark return

Multi-period linking uses proportional scaling to reconcile summed single-period effects to the cumulative geometric active return.

## Project Structure

```
Attribution/
├── app.py                          # Flask application
├── requirements.txt                # Python dependencies
├── attribution/                    # Core analysis engine
│   ├── brinson_fachler.py         # BHB attribution (single & multi-period)
│   ├── aggregator.py              # Property-level aggregation
│   ├── data_loader.py             # File loading & validation
│   ├── template_generator.py      # Portfolio template generation
│   └── ncreif_api.py             # NCREIF ODCE API client with caching
├── templates/                      # Jinja2 templates
│   ├── base.html                  # Base layout (Bootstrap 5, DataTables)
│   ├── index.html                 # Upload page + NCREIF fetch
│   ├── properties.html            # Property review & overrides
│   └── results.html               # Attribution results & export
├── static/
│   ├── css/style.css
│   └── js/app.js
├── tests/                          # Test suite (267 tests)
│   ├── test_brinson_fachler.py    # Attribution engine tests
│   ├── test_aggregator.py         # Aggregation tests
│   ├── test_data_loader.py        # Data loading tests
│   ├── test_template_generator.py # Template tests
│   ├── test_app.py                # Flask route tests
│   └── test_integration.py        # End-to-end integration tests
└── uploads/                        # File storage (gitignored)
    ├── workspace.json             # Current session file paths
    ├── current_portfolio.xlsx     # Active portfolio file
    ├── current_benchmark.xlsx     # Active benchmark file
    ├── *_overrides.csv            # Property sold/CBSA overrides
    └── ncreif_cache/              # Cached NCREIF API data
```

## Persistence

All state persists across server restarts and port changes:

| Data | Storage | Location |
|------|---------|----------|
| NCREIF benchmark | Excel + JSON metadata | `uploads/ncreif_cache/` |
| Uploaded files | Stable-named Excel files | `uploads/current_*.xlsx` |
| File path references | JSON | `uploads/workspace.json` |
| Sold flags & CBSA remaps | CSV | `uploads/current_portfolio_overrides.csv` |
| Session data | Server-side file cache | `.flask_sessions/` |

## Running Tests

```bash
# All tests
python -m pytest tests/

# Specific module
python -m pytest tests/test_brinson_fachler.py

# With verbose output
python -m pytest tests/ -v
```

## Key Concepts

- **Segments with no benchmark match**: Portfolio CBSAs or property types not in the benchmark get w_b=0 and r_b=R_b. Allocation is zero; selection captures the full active contribution.
- **Segments with no portfolio exposure**: If the benchmark has a segment the portfolio doesn't, w_p=0. This correctly shows positive allocation for avoiding underperforming segments.
- **Held vs. Sold**: A portfolio-only dimension. Both segments get w_b=0 and r_b=R_b, making it a pure selection analysis showing whether sold properties were under- or over-performers.
- **Property classification**: If ANY quarter for a property has Sold=1, ALL quarters for that property are classified as Sold.
- **Annualization**: Cumulative returns are annualized for periods longer than 1 year. Sub-annual periods show cumulative returns without annualization.
