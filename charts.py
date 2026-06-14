from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Iterable

_MPLCONFIGDIR = Path(__file__).resolve().parent / ".matplotlib"
_MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st


PALETTE = [
    "#0E7490",
    "#2563EB",
    "#7C3AED",
    "#DB2777",
    "#EA580C",
    "#16A34A",
    "#64748B",
    "#B45309",
]
ACCENT = "#0F766E"
GRID = "#D8DEE9"
TEXT = "#172033"


def set_chart_theme() -> None:
    sns.set_theme(
        context="notebook",
        style="whitegrid",
        palette=PALETTE,
        rc={
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "font.family": "DejaVu Sans",
        },
    )


def _finalize(ax: plt.Axes, title: str, xlabel: str = "", ylabel: str = "") -> plt.Figure:
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=30)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig = ax.figure
    try:
        fig.tight_layout()
    except Exception:
        pass
    return fig


def empty_figure(message: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12, color="#475569")
    ax.axis("off")
    try:
        fig.tight_layout()
    except Exception:
        pass
    return fig


def _top_categories(df: pd.DataFrame, column: str, n: int = 12) -> list[str]:
    return df[column].dropna().value_counts().head(n).index.tolist()


@st.cache_data(show_spinner=False)
def _series_count_frame(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    series_col = "Series Code" if "Series Code" in df.columns else "Indicator"
    return (
        df.drop_duplicates(subset=[series_col])
        .groupby(group_col, observed=True, dropna=False)[series_col]
        .count()
        .reset_index(name="Count")
        .sort_values("Count", ascending=False)
    )


def indicator_category_pie(df: pd.DataFrame) -> plt.Figure:
    if df.empty or "Topic" not in df.columns:
        return empty_figure("No topic data available for the selected filters.")
    counts = _series_count_frame(df, "Topic").head(8)
    if counts.empty:
        return empty_figure("No indicator categories available.")

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.pie(
        counts["Count"],
        labels=counts["Topic"],
        autopct=lambda pct: f"{pct:.1f}%" if pct >= 4 else "",
        startangle=120,
        colors=sns.color_palette(PALETTE, n_colors=len(counts)),
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
        textprops={"fontsize": 9, "color": TEXT},
    )
    ax.set_title("Indicator Category Distribution", loc="left", fontsize=13, fontweight="bold", pad=10)
    try:
        fig.tight_layout()
    except Exception:
        pass
    return fig


def indicator_histogram(df: pd.DataFrame, indicator: str) -> plt.Figure:
    data = df[df["Indicator"].eq(indicator)].dropna(subset=["Value"])
    if data.empty:
        return empty_figure("No values available for the selected indicator.")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.histplot(data=data, x="Value", hue="Year Type", bins=35, kde=True, ax=ax)
    return _finalize(ax, f"Distribution of {indicator}", "Value", "Observation count")


@st.cache_data(show_spinner=False)
def _trend_grouped(df: pd.DataFrame, indicator: str) -> pd.DataFrame:
    data = df[df["Indicator"].eq(indicator)].dropna(subset=["Value", "Year"])
    if data.empty:
        return data
    return (
        data.groupby(["Year", "Country", "Year Type"], observed=True, as_index=False)["Value"]
        .mean()
        .sort_values("Year")
    )


def trend_line(df: pd.DataFrame, indicator: str) -> plt.Figure:
    grouped = _trend_grouped(df, indicator)
    if grouped.empty:
        return empty_figure("No time-series data available for the selected trend.")

    fig, ax = plt.subplots(figsize=(10, 5.2))
    sns.lineplot(
        data=grouped,
        x="Year",
        y="Value",
        hue="Country",
        style="Year Type",
        markers=False,
        linewidth=2.2,
        ax=ax,
    )
    ax.axvspan(2026, grouped["Year"].max(), color="#F1F5F9", alpha=0.7, label="Projection window")
    ax.legend(loc="best", fontsize=8, frameon=True)
    return _finalize(ax, f"Economic Trend Comparison: {indicator}", "Year", "Value")


@st.cache_data(show_spinner=False)
def ranking_table(
    df: pd.DataFrame,
    indicator: str,
    selected_year: int,
    top_n: int,
    include_aggregates: bool,
) -> pd.DataFrame:
    data = df[
        df["Indicator"].eq(indicator)
        & df["Year"].eq(selected_year)
        & df["Value"].notna()
    ].copy()
    if not include_aggregates and "Is Aggregate" in data.columns:
        data = data[~data["Is Aggregate"]]
    if data.empty:
        return pd.DataFrame(columns=["Rank", "Country", "Indicator", "Year", "Value", "Unit", "Scale", "Year Type"])
    data = data.sort_values("Value", ascending=False).head(top_n)
    data.insert(0, "Rank", range(1, len(data) + 1))
    columns = [column for column in ["Rank", "Country", "Indicator", "Year", "Value", "Unit", "Scale", "Year Type"] if column in data.columns]
    return data[columns].reset_index(drop=True)


def ranking_bar(
    df: pd.DataFrame,
    indicator: str,
    selected_year: int,
    top_n: int,
    include_aggregates: bool,
) -> plt.Figure:
    table = ranking_table(df, indicator, selected_year, top_n, include_aggregates)
    if table.empty:
        return empty_figure("No ranking values available for this indicator and year.")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.barplot(data=table.sort_values("Value"), x="Value", y="Country", hue="Year Type", dodge=False, ax=ax)
    ax.legend(loc="lower right", frameon=True, fontsize=8)
    return _finalize(ax, f"Top {len(table)} Countries: {indicator} ({selected_year})", "Value", "")


@st.cache_data(show_spinner=False)
def _scatter_pivot(
    df: pd.DataFrame,
    x_indicator: str,
    y_indicator: str,
    selected_year: int,
    include_aggregates: bool,
) -> pd.DataFrame:
    data = df[
        df["Year"].eq(selected_year)
        & df["Indicator"].isin([x_indicator, y_indicator])
        & df["Value"].notna()
    ].copy()
    if not include_aggregates and "Is Aggregate" in data.columns:
        data = data[~data["Is Aggregate"]]
    if data.empty:
        return data

    pivot = data.pivot_table(index="Country", columns="Indicator", values="Value", aggfunc="mean", observed=True)
    return pivot.dropna(subset=[x_indicator, y_indicator]).reset_index()


def scatter_economics(
    df: pd.DataFrame,
    x_indicator: str,
    y_indicator: str,
    selected_year: int,
    include_aggregates: bool,
) -> plt.Figure:
    if not x_indicator or not y_indicator or x_indicator == y_indicator:
        return empty_figure("Choose two different economic indicators for the scatter plot.")

    pivot = _scatter_pivot(df, x_indicator, y_indicator, selected_year, include_aggregates)
    if pivot.empty:
        return empty_figure("No paired values available for the selected year.")

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    sns.scatterplot(data=pivot, x=x_indicator, y=y_indicator, s=70, color=ACCENT, edgecolor="white", ax=ax)
    if len(pivot) >= 3:
        sns.regplot(data=pivot, x=x_indicator, y=y_indicator, scatter=False, ci=None, color="#334155", ax=ax)

    for _, row in pivot.nlargest(min(5, len(pivot)), y_indicator).iterrows():
        ax.annotate(row["Country"], (row[x_indicator], row[y_indicator]), fontsize=8, xytext=(4, 4), textcoords="offset points")

    return _finalize(ax, f"{x_indicator} vs {y_indicator} ({selected_year})", x_indicator, y_indicator)


def box_plot_by_topic(df: pd.DataFrame) -> plt.Figure:
    data = df.dropna(subset=["Value", "Topic"]).copy()
    if data.empty:
        return empty_figure("No topic distribution data available.")
    top_topics = _top_categories(data, "Topic", n=8)
    data = data[data["Topic"].isin(top_topics)]

    fig, ax = plt.subplots(figsize=(10, 5.4))
    sns.boxplot(data=data, x="Topic", y="Value", hue="Year Type", showfliers=False, ax=ax)
    ax.legend(loc="best", fontsize=8, frameon=True)
    return _finalize(ax, "Economic Value Distribution by Topic", "Topic", "Value")


@st.cache_data(show_spinner=False)
def _correlation_matrix(df: pd.DataFrame, selected_indicators: tuple[str, ...] | None) -> pd.DataFrame:
    data = df.dropna(subset=["Value", "Country", "Indicator", "Year"])
    if data.empty:
        return pd.DataFrame()

    if selected_indicators:
        data = data[data["Indicator"].isin(list(selected_indicators))]

    if data["Indicator"].nunique() > 10:
        top = data["Indicator"].value_counts().head(10).index
        data = data[data["Indicator"].isin(top)]

    pivot = data.pivot_table(index=["Country", "Year"], columns="Indicator", values="Value", aggfunc="mean", observed=True)
    pivot = pivot.dropna(axis=1, thresh=max(5, int(len(pivot) * 0.08)))
    if pivot.shape[1] < 2:
        return pd.DataFrame()

    return pivot.corr(numeric_only=True)


def correlation_heatmap(df: pd.DataFrame, selected_indicators: Iterable[str] | None = None) -> plt.Figure:
    indicators_key = tuple(sorted(selected_indicators)) if selected_indicators else None
    corr = _correlation_matrix(df, indicators_key)
    if corr.empty:
        return empty_figure("Select at least two indicators with overlapping country-year values.")

    fig, ax = plt.subplots(figsize=(9.5, 7.0))
    sns.heatmap(
        corr,
        cmap="vlag",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        linecolor="#E2E8F0",
        cbar_kws={"shrink": 0.75},
        ax=ax,
    )
    ax.set_title("Country-Indicator Correlation Matrix", loc="left", fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(axis="x", labelrotation=45)
    ax.tick_params(axis="y", labelrotation=0)
    try:
        fig.tight_layout()
    except Exception:
        pass
    return fig


@st.cache_data(show_spinner=False)
def _area_pivot(df: pd.DataFrame, indicator: str) -> pd.DataFrame:
    data = df[df["Indicator"].eq(indicator)].dropna(subset=["Value", "Year", "Country"])
    if data.empty:
        return data

    latest_year = int(data["Year"].max())
    top_countries = (
        data[data["Year"].eq(latest_year)]
        .sort_values("Value", ascending=False)
        .head(6)["Country"]
        .tolist()
    )
    data = data[data["Country"].isin(top_countries)]
    return data.pivot_table(index="Year", columns="Country", values="Value", aggfunc="mean", observed=True).sort_index()


def area_chart(df: pd.DataFrame, indicator: str) -> plt.Figure:
    pivot = _area_pivot(df, indicator)
    if pivot.empty:
        return empty_figure("No cumulative trend data available.")

    fig, ax = plt.subplots(figsize=(10, 5.2))
    has_negatives = (pivot < 0).any().any()
    pivot.plot.area(
        ax=ax,
        color=sns.color_palette(PALETTE, n_colors=pivot.shape[1]),
        alpha=0.82,
        linewidth=0.5,
        stacked=not has_negatives,
    )
    ax.legend(loc="best", fontsize=8, frameon=True)
    title = f"Economic Trend: {indicator}" if has_negatives else f"Cumulative Economic Trend: {indicator}"
    return _finalize(ax, title, "Year", "Value")


def count_plot(df: pd.DataFrame) -> plt.Figure:
    if df.empty:
        return empty_figure("No indicators available for count plot.")
    counts = _series_count_frame(df, "Indicator").head(15)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.barplot(data=counts.sort_values("Count"), x="Count", y="Indicator", color=PALETTE[0], ax=ax)
    return _finalize(ax, "Indicator Frequency", "Series count", "")


def violin_plot(df: pd.DataFrame, indicator: str) -> plt.Figure:
    data = df[df["Indicator"].eq(indicator)].dropna(subset=["Value", "Year Type"]).copy()
    if data.empty:
        return empty_figure("No values available for violin distribution.")

    fig, ax = plt.subplots(figsize=(8, 5.2))
    sns.violinplot(data=data, x="Year Type", y="Value", hue="Year Type", palette=PALETTE[:2], inner="quartile", legend=False, ax=ax)
    return _finalize(ax, f"Historical vs Forecast Distribution: {indicator}", "Observation type", "Value")


@st.cache_data(show_spinner=False)
def anomaly_detector(df: pd.DataFrame, indicator: str, limit: int = 15) -> pd.DataFrame:
    data = df[df["Indicator"].eq(indicator)].dropna(subset=["Value"]).copy()
    if data.empty:
        return pd.DataFrame()

    median = data.groupby("Year", observed=True)["Value"].transform("median")
    mad = data["Value"].sub(median).abs().groupby(data["Year"]).transform("median")
    robust_z = 0.6745 * (data["Value"] - median) / mad.replace(0, np.nan)
    data["Robust Z-Score"] = robust_z.replace([np.inf, -np.inf], np.nan)
    anomalies = data[data["Robust Z-Score"].abs() >= 3.5].copy()
    if anomalies.empty:
        anomalies = data.assign(**{"Robust Z-Score": robust_z}).reindex(
            data["Value"].sub(median).abs().sort_values(ascending=False).index
        )
    columns = [column for column in ["Country", "Indicator", "Year", "Value", "Robust Z-Score", "Year Type", "Topic", "Scale", "Unit"] if column in anomalies.columns]
    return (
        anomalies.sort_values("Robust Z-Score", key=lambda series: series.abs(), ascending=False)
        .head(limit)[columns]
        .reset_index(drop=True)
    )


@st.cache_data(show_spinner=False)
def calculate_kpis(
    visible_df: pd.DataFrame,
    context_df: pd.DataFrame,
    selected_year: int,
    selected_indicator: str,
    include_aggregates: bool,
) -> dict[str, str]:
    year_context = context_df[context_df["Year"].eq(selected_year)].dropna(subset=["Value"]).copy()
    if not include_aggregates and "Is Aggregate" in year_context.columns:
        year_context = year_context[~year_context["Is Aggregate"]]

    def leader(patterns: list[str]) -> str:
        matches = [
            indicator
            for indicator in year_context["Indicator"].dropna().unique()
            if any(pattern.lower() in indicator.lower() for pattern in patterns)
        ]
        if not matches:
            return "n/a"
        ranked = ranking_table(year_context, matches[0], selected_year, 1, include_aggregates)
        return ranked.loc[0, "Country"] if not ranked.empty else "n/a"

    growth_candidates = year_context[
        year_context["Indicator"].str.contains("constant prices, percent change", case=False, na=False)
    ]
    avg_growth = growth_candidates["Value"].mean() if not growth_candidates.empty else np.nan

    key_count = (
        visible_df.drop_duplicates("Series Code")["KEY_INDICATOR_BOOL"].sum()
        if "KEY_INDICATOR_BOOL" in visible_df.columns and "Series Code" in visible_df.columns
        else np.nan
    )

    return {
        "Total Countries": f"{visible_df['Country'].nunique():,}",
        "Total Indicators": f"{visible_df['Indicator'].nunique():,}",
        "Total Records": f"{len(visible_df):,}",
        "Selected Year": str(selected_year),
        "Highest GDP Country": leader(["gross domestic product (gdp), current prices, us dollar"]),
        "Highest Inflation Country": leader(["consumer price index", "percent change"]),
        "Average Economic Growth": "n/a" if pd.isna(avg_growth) else f"{avg_growth:,.2f}%",
        "Key Indicator Count": "n/a" if pd.isna(key_count) else f"{int(key_count):,}",
        "Selected Indicator": selected_indicator,
    }


@st.cache_data(show_spinner=False)
def generate_insights(
    df: pd.DataFrame,
    indicator: str,
    selected_year: int,
    top_n: int,
    include_aggregates: bool,
) -> list[str]:
    insights: list[str] = []
    data = df[df["Indicator"].eq(indicator)].dropna(subset=["Value", "Year", "Country"]).copy()
    if not include_aggregates and "Is Aggregate" in data.columns:
        data = data[~data["Is Aggregate"]]
    if data.empty:
        return ["No valid values are available for the current filter combination."]

    latest = data[data["Year"].eq(selected_year)].sort_values("Value", ascending=False)
    if not latest.empty:
        leader = latest.iloc[0]
        insights.append(
            f"{leader['Country']} leads the selected view for {indicator} in {selected_year}, "
            f"with a value of {leader['Value']:,.2f}."
        )

    trend = data.pivot_table(index="Country", columns="Year", values="Value", aggfunc="mean", observed=True)
    if trend.shape[1] >= 2:
        first_year = int(trend.columns.min())
        last_year = int(trend.columns.max())
        changes = trend[last_year] - trend[first_year]
        changes = changes.dropna().sort_values(ascending=False)
        if not changes.empty:
            country = changes.index[0]
            insights.append(
                f"{country} shows the strongest absolute increase from {first_year} to {last_year} "
                f"within the current selection."
            )

    forecast_share = data["Year Type"].eq("Forecast").mean()
    if forecast_share > 0:
        insights.append(
            f"Forecast observations represent {forecast_share:.0%} of the selected indicator view, "
            "so projection years should be interpreted separately from realized history."
        )

    anomalies = anomaly_detector(data, indicator, limit=1)
    if not anomalies.empty:
        row = anomalies.iloc[0]
        score = row.get("Robust Z-Score", np.nan)
        score_text = "high" if pd.isna(score) else f"{abs(score):.1f}"
        insights.append(
            f"{row['Country']} in {int(row['Year'])} is the most unusual observation detected "
            f"for this indicator, with a robust z-score of {score_text}."
        )

    if len(insights) < 3:
        countries = data["Country"].nunique()
        years = int(data["Year"].max() - data["Year"].min() + 1)
        insights.append(
            f"The current slice covers {countries:,} countries/groups across {years:,} annual periods."
        )

    return insights[:5]


def choose_scatter_pairs(all_indicators: Iterable[str]) -> dict[str, tuple[str, str]]:
    indicators = list(all_indicators)

    def find(patterns: list[str]) -> str:
        for pattern in patterns:
            for indicator in indicators:
                if pattern.lower() in indicator.lower():
                    return indicator
        return indicators[0] if indicators else ""

    gdp = find(["Gross domestic product (GDP), Current prices, US dollar"])
    inflation = find(["Consumer price index (CPI), Period average, percent change"])
    population = find(["Population, Persons"])
    debt = find(["Gross debt, General government, Percent of GDP"])
    growth = find(["Gross domestic product (GDP), Constant prices, Percent change"])

    return {
        "GDP vs Inflation": (gdp, inflation),
        "GDP vs Population": (gdp, population),
        "Debt vs Growth": (debt, growth),
    }


def dataframe_download_name(label: str) -> str:
    safe = "".join(character.lower() if character.isalnum() else "_" for character in label)
    safe = "_".join(part for part in safe.split("_") if part)
    return f"{safe or 'weo_export'}.csv"