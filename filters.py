from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st


NULL_TOKENS = ["", "NA", "N/A", "NULL", "null", "None", "none", "...", "--", "-"]

AGGREGATE_LABELS = {
    "World",
    "Advanced Economies",
    "Advanced economies",
    "Emerging Market and Developing Economies",
    "Emerging market and developing economies",
    "Emerging and Developing Asia",
    "Emerging and Developing Europe",
    "Euro Area (EA)",
    "Euro area",
    "European Union (EU)",
    "European Union",
    "G7",
    "Major advanced economies (G7)",
    "Middle East and Central Asia",
    "Other Advanced Economies (Advanced Economies excluding G7 and Euro Area countries)",
    "Other advanced economies",
    "Sub-Saharan Africa",
    "Sub-Saharan Africa (SSA)",
    "Latin America and the Caribbean (LAC)",
    "Western Hemisphere",
}


@dataclass(frozen=True)
class DataBundle:
    path: Path
    raw: pd.DataFrame
    wide: pd.DataFrame
    long: pd.DataFrame
    metadata_columns: list[str]
    year_columns: list[str]


def find_dataset_file(data_dir: str | Path = "data") -> Path:
    """Return the first supported dataset file from the dashboard data directory."""
    data_path = Path(data_dir)
    candidates = sorted(
        [*data_path.glob("*.csv"), *data_path.glob("*.xlsx"), *data_path.glob("*.xls")]
    )
    if not candidates:
        raise FileNotFoundError(
            f"No CSV or Excel dataset found in {data_path.resolve()}. "
            "Place the original IMF WEO file in dashboard_project/data/."
        )
    return candidates[0]


def detect_year_columns(columns: Iterable[str], start: int = 1900, end: int = 2100) -> list[str]:
    """Identify annual time-series columns such as 1980, 1981, ... 2031."""
    year_columns: list[str] = []
    for column in columns:
        label = str(column).strip()
        if label.isdigit() and start <= int(label) <= end:
            year_columns.append(column)
    return sorted(year_columns, key=lambda col: int(str(col).strip()))


def read_dataset(path: str | Path) -> pd.DataFrame:
    """Read CSV or Excel input while preserving raw schema as strings for safe cleaning."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=True, na_values=NULL_TOKENS)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, dtype=str, keep_default_na=True, na_values=NULL_TOKENS)
    raise ValueError(f"Unsupported dataset type: {path.suffix}")


def clean_wide_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize headers, whitespace, and common missing-value markers."""
    cleaned = df.copy()
    cleaned.columns = [str(column).strip() for column in cleaned.columns]

    year_columns = set(detect_year_columns(cleaned.columns))
    text_columns = [
        column
        for column in cleaned.select_dtypes(include=["object"]).columns
        if column not in year_columns
    ]
    for column in text_columns:
        cleaned[column] = (
            cleaned[column]
            .astype("string")
            .str.replace("\u00a0", " ", regex=False)
            .str.strip()
            .replace(NULL_TOKENS, pd.NA)
        )

    if "KEY_INDICATOR" in cleaned.columns:
        cleaned["KEY_INDICATOR_BOOL"] = (
            cleaned["KEY_INDICATOR"]
            .astype("string")
            .str.lower()
            .isin(["true", "1", "yes", "y"])
        )

    if "LATEST_ACTUAL_ANNUAL_DATA" in cleaned.columns:
        cleaned["LATEST_ACTUAL_YEAR"] = pd.to_numeric(
            cleaned["LATEST_ACTUAL_ANNUAL_DATA"], errors="coerce"
        )

    return cleaned


def parse_numeric(series: pd.Series) -> pd.Series:
    """Convert IMF numeric strings to floats without failing on mixed types."""
    normalized = (
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.replace("\u00a0", "", regex=False)
        .str.strip()
        .replace(NULL_TOKENS, pd.NA)
    )
    return pd.to_numeric(normalized, errors="coerce")


def build_long_dataframe(df: pd.DataFrame, year_columns: list[str] | None = None) -> pd.DataFrame:
    """Transform WEO wide annual columns into an analysis-ready long dataframe."""
    detected_years = year_columns or detect_year_columns(df.columns)
    metadata_columns = [column for column in df.columns if column not in detected_years]

    long_df = df.melt(
        id_vars=metadata_columns,
        value_vars=detected_years,
        var_name="Year",
        value_name="Raw_Value",
    )
    long_df["Year"] = pd.to_numeric(long_df["Year"], errors="coerce").astype("Int64")
    long_df["Value"] = parse_numeric(long_df["Raw_Value"])

    alias_map = {
        "COUNTRY": "Country",
        "INDICATOR": "Indicator",
        "TOPIC": "Topic",
        "SCALE": "Scale",
        "UNIT": "Unit",
        "SERIES_CODE": "Series Code",
        "SERIES_NAME": "Series Name",
        "PRIMARY_DOMESTIC_CURRENCY": "Domestic Currency",
    }
    for source, alias in alias_map.items():
        if source in long_df.columns and alias not in long_df.columns:
            long_df[alias] = long_df[source]

    if "LATEST_ACTUAL_YEAR" in long_df.columns:
        latest_actual = long_df["LATEST_ACTUAL_YEAR"]
    elif "LATEST_ACTUAL_ANNUAL_DATA" in long_df.columns:
        latest_actual = pd.to_numeric(long_df["LATEST_ACTUAL_ANNUAL_DATA"], errors="coerce")
    else:
        latest_actual = pd.Series(np.nan, index=long_df.index)

    fallback_latest = min(2025, int(long_df["Year"].max(skipna=True)))
    latest_actual = latest_actual.fillna(fallback_latest)
    long_df["Year Type"] = np.where(long_df["Year"] > latest_actual, "Forecast", "Historical")
    long_df["Is Valid Value"] = long_df["Value"].notna()
    long_df["Is Aggregate"] = long_df["Country"].isin(AGGREGATE_LABELS) if "Country" in long_df else False

    category_columns = [
        "Country",
        "Indicator",
        "Topic",
        "Scale",
        "Unit",
        "Series Code",
        "Year Type",
    ]
    for column in category_columns:
        if column in long_df.columns:
            long_df[column] = long_df[column].astype("category")
    long_df["Year"] = long_df["Year"].astype("Int16")

    return long_df


@st.cache_resource(show_spinner="Loading and transforming IMF WEO data...")
def load_data(data_dir: str = "data") -> DataBundle:
    dataset_path = find_dataset_file(data_dir)
    raw = read_dataset(dataset_path)
    wide = clean_wide_dataframe(raw)
    year_columns = detect_year_columns(wide.columns)
    metadata_columns = [column for column in wide.columns if column not in year_columns]
    long_df = build_long_dataframe(wide, year_columns)
    return DataBundle(
        path=dataset_path,
        raw=raw,
        wide=wide,
        long=long_df,
        metadata_columns=metadata_columns,
        year_columns=year_columns,
    )


def profile_dataset(bundle: DataBundle) -> dict[str, object]:
    """Generate reusable quality diagnostics for the app and notebook."""
    total_year_cells = int(bundle.raw.shape[0] * len(bundle.year_columns))
    valid_observations = int(bundle.long["Value"].notna().sum())
    non_missing_year_values = int(bundle.long["Raw_Value"].notna().sum())
    return {
        "file_name": bundle.path.name,
        "rows": int(bundle.raw.shape[0]),
        "columns": int(bundle.raw.shape[1]),
        "metadata_columns": len(bundle.metadata_columns),
        "year_columns": len(bundle.year_columns),
        "year_min": int(min(bundle.year_columns, key=int)),
        "year_max": int(max(bundle.year_columns, key=int)),
        "countries": int(bundle.wide["COUNTRY"].nunique(dropna=True)),
        "indicators": int(bundle.wide["INDICATOR"].nunique(dropna=True)),
        "topics": int(bundle.wide["TOPIC"].nunique(dropna=True)),
        "full_duplicates": int(bundle.wide.duplicated().sum()),
        "valid_observations": valid_observations,
        "missing_year_value_pct": float((1 - valid_observations / total_year_cells) * 100),
        "numeric_parse_failures": int(non_missing_year_values - valid_observations),
    }


def _safe_options(series: pd.Series, limit: int | None = None) -> list[str]:
    values = sorted(series.dropna().astype(str).unique().tolist())
    return values[:limit] if limit else values


def reset_filter_state() -> None:
    for key in [
        "countries",
        "indicator",
        "topics",
        "scales",
        "search",
        "year_range",
        "selected_year",
        "include_aggregates",
        "top_n",
    ]:
        st.session_state.pop(key, None)


def render_sidebar_filters(long_df: pd.DataFrame) -> dict[str, object]:
    """Render global Streamlit controls and return the selected filter state."""
    valid = long_df[long_df["Value"].notna()]
    available_years = sorted(valid["Year"].dropna().astype(int).unique().tolist())
    min_year, max_year = min(available_years), max(available_years)
    default_start = max(min_year, 2000)
    default_end = min(max_year, 2031)
    default_selected = min(2025, default_end)

    with st.sidebar:
        st.markdown("### Global Economic Explorer")
        if st.button("Reset filters", width="stretch"):
            reset_filter_state()
            st.rerun()

        search = st.text_input(
            "Search country, indicator, topic, or series",
            key="search",
            placeholder="e.g. India, inflation, debt",
        )

        topics = st.multiselect(
            "Topic",
            options=_safe_options(valid["Topic"]),
            default=[],
            key="topics",
        )

        topic_filtered = valid
        if topics:
            topic_filtered = topic_filtered[topic_filtered["Topic"].isin(topics)]

        scales = st.multiselect(
            "Scale",
            options=_safe_options(topic_filtered["Scale"]),
            default=[],
            key="scales",
        )

        scale_filtered = topic_filtered
        if scales:
            scale_filtered = scale_filtered[scale_filtered["Scale"].isin(scales)]

        indicator_options = _safe_options(scale_filtered["Indicator"])
        if st.session_state.get("indicator") not in indicator_options:
            st.session_state.pop("indicator", None)
        default_indicator = choose_indicator(
            indicator_options,
            ["Gross domestic product (GDP), Current prices, US dollar"],
            fallback_index=0,
        )
        indicator = st.selectbox(
            "Indicator",
            options=indicator_options,
            index=indicator_options.index(default_indicator) if default_indicator in indicator_options else 0,
            key="indicator",
        )

        country_options = _safe_options(scale_filtered["Country"])
        if any(country not in country_options for country in st.session_state.get("countries", [])):
            st.session_state.pop("countries", None)
        preferred = [country for country in ["United States", "China", "India", "Germany", "Japan"] if country in country_options]
        countries = st.multiselect(
            "Countries / groups",
            options=country_options,
            default=preferred[:5],
            key="countries",
        )

        year_range = st.slider(
            "Year range",
            min_value=min_year,
            max_value=max_year,
            value=(default_start, default_end),
            step=1,
            key="year_range",
        )

        if "selected_year" in st.session_state:
            current_year = int(st.session_state["selected_year"])
            if current_year < year_range[0] or current_year > year_range[1]:
                st.session_state["selected_year"] = min(max(current_year, year_range[0]), year_range[1])

        selected_year = st.slider(
            "Selected year",
            min_value=year_range[0],
            max_value=year_range[1],
            value=min(max(default_selected, year_range[0]), year_range[1]),
            step=1,
            key="selected_year",
        )

        include_aggregates = st.toggle(
            "Include regional aggregates",
            value=False,
            key="include_aggregates",
        )

        top_n = st.slider("Top N rankings", min_value=5, max_value=30, value=12, step=1, key="top_n")

    return {
        "search": search,
        "topics": topics,
        "scales": scales,
        "indicator": indicator,
        "countries": countries,
        "year_range": year_range,
        "selected_year": selected_year,
        "include_aggregates": include_aggregates,
        "top_n": top_n,
    }


def apply_global_filters(long_df: pd.DataFrame, filters: dict[str, object]) -> pd.DataFrame:
    year_start, year_end = filters["year_range"]
    mask = long_df["Year"].between(year_start, year_end) & long_df["Value"].notna()

    if filters.get("topics"):
        mask &= long_df["Topic"].isin(filters["topics"])

    if filters.get("scales"):
        mask &= long_df["Scale"].isin(filters["scales"])

    if filters.get("indicator"):
        mask &= long_df["Indicator"].eq(filters["indicator"])

    if filters.get("countries"):
        mask &= long_df["Country"].isin(filters["countries"])

    if not filters.get("include_aggregates", False):
        mask &= ~long_df["Is Aggregate"]

    search = str(filters.get("search", "")).strip()
    if search:
        searchable_columns = [
            column
            for column in ["Country", "Indicator", "Topic", "Series Name", "Series Code", "FULL_DESCRIPTION"]
            if column in long_df.columns
        ]
        pattern = re.escape(search)
        search_mask = pd.Series(False, index=long_df.index)
        for column in searchable_columns:
            search_mask |= long_df[column].astype("string").str.contains(pattern, case=False, na=False)
        mask &= search_mask

    return long_df.loc[mask]


def apply_context_filters(
    long_df: pd.DataFrame,
    filters: dict[str, object],
    keep_indicator: bool = False,
    keep_countries: bool = False,
) -> pd.DataFrame:
    """Apply global filters while allowing specific charts to compare beyond the selected indicator/countries."""
    adjusted = dict(filters)
    if keep_indicator:
        adjusted["indicator"] = None
    if keep_countries:
        adjusted["countries"] = []
    return apply_global_filters(long_df, adjusted)


def choose_indicator(
    indicators: Iterable[str],
    preferred_patterns: list[str],
    fallback_index: int = 0,
) -> str:
    """Pick the most relevant indicator from text patterns."""
    options = list(indicators)
    if not options:
        return ""
    lowered = [(indicator, indicator.lower()) for indicator in options]
    for pattern in preferred_patterns:
        needle = pattern.lower()
        for indicator, lower in lowered:
            if needle in lower:
                return indicator
    return options[min(fallback_index, len(options) - 1)]


def find_indicator_options(long_df: pd.DataFrame, patterns: list[str], limit: int = 25) -> list[str]:
    unique_indicators = _safe_options(long_df["Indicator"])
    matches = [
        indicator
        for indicator in unique_indicators
        if any(re.search(pattern, indicator, flags=re.IGNORECASE) for pattern in patterns)
    ]
    return matches[:limit]


def compact_number(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    value = float(value)
    abs_value = abs(value)
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:,.2f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:,.2f}K"
    if abs_value >= 100:
        return f"{value:,.1f}"
    return f"{value:,.2f}"
