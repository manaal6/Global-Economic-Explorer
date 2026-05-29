# IMF World Economic Outlook Economic Analytics Dashboard

A professional Streamlit analytics product built for the IMF World Economic Outlook (WEO 9.0.0) dataset. The dashboard converts the original wide IMF file into an analysis-ready long time-series model and delivers executive KPIs, economic rankings, country comparisons, anomaly detection, correlation analysis, and historical-vs-forecast views.

## Overview

The uploaded WEO file contains 8,200 rows and 117 columns. The first 65 columns are metadata fields such as country, indicator, topic, scale, unit, series name, and source information. The remaining 52 columns are annual values from 1980 through 2031.

The data pipeline automatically detects annual columns, cleans metadata, parses numeric values, and uses `pandas.melt()` to create:

```text
Country | Indicator | Year | Value
```

IMF metadata is preserved so every chart can remain filterable, explainable, and auditable.

## Features

- Executive KPI cards for countries, indicators, records, selected year, GDP leader, inflation leader, average growth, and key indicator count.
- Global filter system with country selection, multi-country comparison, indicator selector, year range, selected year, topic, scale, search, aggregate toggle, and reset control.
- Required course charts adapted to economics:
  - Pie chart for indicator category distribution
  - Histogram for selected indicator distribution
  - Line chart for cross-country economic trends
  - Bar chart for country rankings
  - Scatter plot for GDP vs inflation, GDP vs population, and debt vs growth
  - Box plot by economic topic
  - Correlation heatmap
  - Area chart for cumulative trends
  - Count plot for indicator frequency
  - Violin plot for historical vs forecast distributions
- Advanced analytics:
  - Global country ranking table
  - Multi-country trend comparison
  - Robust anomaly detector
  - Correlation explorer
  - Forecast vs historical separation
  - Dynamic insight panel
- Professional Streamlit UI with custom CSS, responsive columns, tabs, metric cards, and clean chart aesthetics.

## Folder Structure

```text
dashboard_project/
|-- data/
|   |-- dataset_2026-05-29T09_26_32.090555303Z_DEFAULT_INTEGRATION_IMF.RES_WEO_9.0.0.csv
|-- notebooks/
|   |-- analysis.ipynb
|-- app.py
|-- charts.py
|-- filters.py
|-- requirements.txt
|-- README.md
```

## Requirements

- Python 3.9+
- pandas
- numpy
- matplotlib
- seaborn
- streamlit
- openpyxl
- jupyter
- ipykernel

Install dependencies:

```bash
pip install -r requirements.txt
```

## How To Run

From the project folder:

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Dashboard Walkthrough

### Executive View

Start here for the flagship workflow. Use the sidebar to choose countries, topics, scale, indicator, and year range. The KPI cards update instantly, followed by a country comparison trend, a sortable global ranking table, and a dynamic insight panel.

### Course Charts

This tab contains all required visualization types, implemented with economic context and consistent styling. The scatter plot can switch between GDP/inflation, GDP/population, and debt/growth relationships.

### Advanced Analytics

This section highlights premium data science capabilities: anomaly detection, correlation exploration, multi-country trend comparison, and forecast-vs-historical analysis.

### Data Quality

This tab documents the engineering profile: rows, columns, year coverage, duplicate checks, numeric parsing status, missingness, and a preview of the filtered long-format dataset.

## Key Dataset Findings

- Shape: 8,200 rows by 117 columns.
- Annual coverage: 1980 through 2031.
- Annual time-series columns: 52.
- Metadata columns: 65.
- Countries/groups: 210.
- Indicators: 145.
- Topics: 15.
- Full duplicate rows: 0.
- Numeric parse failures in annual columns: 0.
- Valid long-format numeric observations: 361,733.
- Missing annual value rate: approximately 15.17%.

## Presentation Strategy

1. Open with the data engineering story: wide IMF annual columns were converted into a reusable long-format analytical model with `pandas.melt()`.
2. Show the executive KPIs and explain that all cards and charts are controlled by one global filter system.
3. Demonstrate a GDP comparison for major economies, then switch to inflation to show volatility and anomaly detection.
4. Use the scatter selector to compare GDP vs inflation, GDP vs population, and debt vs growth.
5. Highlight the forecast shading and historical-vs-forecast split so the audience understands projected values are separated from realized data.
6. End with the Data Quality tab to show that the product is not just visual but engineered and auditable.

## Notes

The dashboard treats IMF aggregate regions such as World, G7, Euro Area, and Advanced Economies separately from countries. By default, country rankings exclude these aggregates; turn on "Include regional aggregates" in the sidebar to include them.
