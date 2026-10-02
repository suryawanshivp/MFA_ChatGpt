# Mutual Fund Research Tool

A lightweight, personal, single-page Streamlit application for Indian mutual-fund comparison.

## Why Streamlit instead of a pure HTML file?

MFapi is a public API, but a browser-only app can be affected by browser CORS/network policy and would make robust error handling and category calculations harder. This app keeps the UI to one page while using a tiny Python backend. No database, scheduler, API key, or paid service is required.

## Data sources

1. MFapi.in — scheme list, metadata, latest NAV and complete NAV history.
2. AMFI — scheme-level AUM/AAUM is needed for the user's eligibility rule. Because MFapi does not expose this field, the app accepts an AMFI-derived CSV:
   `scheme_code,month,aum_cr`

The app never pretends a missing AUM value passes the ₹5 crore rule.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Put `app.py` and `requirements.txt` in a GitHub repository.
2. Go to Streamlit Community Cloud.
3. Create an app and select the repository, branch and `app.py`.
4. Deploy.

No secrets are required for MFapi.

## AUM CSV

Prepare a CSV with at least the latest six monthly observations per scheme:

```csv
scheme_code,month,aum_cr
119551,2026-04,12345.6
119551,2026-05,12400.2
119551,2026-06,12501.7
119551,2026-07,12610.3
119551,2026-08,12750.1
119551,2026-09,12810.8
```

The app averages the latest six available months and checks it against ₹5 crore.

## Caching / refresh

MFapi calls are cached for 24 hours. Nothing runs on a schedule. Press **Refresh data** to clear the cache and fetch fresh data.

## Known scope

- Equity and Hybrid only.
- Direct Growth active-fund universe is inferred from scheme names.
- MFapi's category wording is used as the subcategory taxonomy.
- Category benchmark is the equal-weighted monthly return index of the loaded peer set.
- For a true full-category leaderboard, load a sufficiently broad peer set. The default fetch is intentionally capped to keep the tool lightweight.
- AUM remains an explicit external eligibility input because it is not provided by MFapi.

## Recommended future modules

1. AMFI AUM ingestion/parser.
2. Full-category peer universe cache.
3. Expense ratio / exit load.
4. Benchmark-index comparison in addition to category-relative alpha/beta.
5. Portfolio overlap and concentration using monthly holdings data.
6. Tax-aware return analysis and SIP/XIRR.
