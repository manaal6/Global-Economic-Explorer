from __future__ import annotations

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from charts import (
    area_chart,
    anomaly_detector,
    box_plot_by_topic,
    calculate_kpis,
    choose_scatter_pairs,
    correlation_heatmap,
    count_plot,
    dataframe_download_name,
    generate_insights,
    indicator_category_pie,
    indicator_histogram,
    ranking_bar,
    ranking_table,
    scatter_economics,
    set_chart_theme,
    trend_line,
    violin_plot,
)
from filters import (
    apply_context_filters,
    apply_global_filters,
    compact_number,
    load_data,
    profile_dataset,
    render_sidebar_filters,
)


st.set_page_config(
    page_title="IMF WEO Economic Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
            :root {
                --bg: #f6f8fb;
                --panel: #ffffff;
                --ink: #172033;
                --muted: #64748b;
                --line: #d8dee9;
                --blue: #0e7490;
                --green: #0f766e;
                --amber: #b45309;
            }

            .stApp {
                background: var(--bg);
                color: var(--ink);
            }

            [data-testid="stSidebar"] {
                background: #111827;
            }

            [data-testid="stSidebar"] * {
                color: #eef2ff;
            }

            [data-testid="stSidebar"] input,
            [data-testid="stSidebar"] textarea,
            [data-testid="stSidebar"] [data-baseweb="select"] * {
                color: #111827;
            }

            [data-testid="stSidebar"] button {
                color: #111827;
                background: #f8fafc;
                border-color: #cbd5e1;
            }

            [data-testid="stSidebar"] button p {
                color: #111827;
            }

            .main .block-container {
                padding-top: 1.5rem;
                padding-bottom: 2rem;
                max-width: 1480px;
            }

            .hero {
                padding: 1.1rem 1.25rem;
                background: linear-gradient(90deg, #0f172a 0%, #164e63 52%, #365314 100%);
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 8px;
                color: white;
                margin-bottom: 1rem;
            }

            .hero h1 {
                margin: 0;
                font-size: 2rem;
                line-height: 1.1;
                letter-spacing: 0;
                color: white;
            }

            .hero p {
                color: #dbeafe;
                margin: 0.45rem 0 0 0;
                max-width: 980px;
                font-size: 0.98rem;
            }

            .badge-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.5rem;
                margin-top: 0.8rem;
            }

            .badge {
                border: 1px solid rgba(255,255,255,0.25);
                background: rgba(255,255,255,0.10);
                color: #f8fafc;
                border-radius: 999px;
                padding: 0.25rem 0.62rem;
                font-size: 0.78rem;
                max-width: 100%;
                overflow-wrap: anywhere;
                word-break: break-word;
            }

            .kpi-card {
                background: var(--panel);
                border: 1px solid var(--line);
                border-left: 4px solid var(--blue);
                border-radius: 8px;
                padding: 0.9rem 0.95rem;
                min-height: 104px;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
            }

            .kpi-card .label {
                color: var(--muted);
                font-size: 0.78rem;
                text-transform: uppercase;
                letter-spacing: 0.04em;
                margin-bottom: 0.35rem;
            }

            .kpi-card .value {
                color: var(--ink);
                font-size: 1.35rem;
                line-height: 1.15;
                font-weight: 760;
                overflow-wrap: anywhere;
            }

            .section-note {
                color: var(--muted);
                font-size: 0.9rem;
                margin-top: -0.35rem;
                margin-bottom: 0.75rem;
            }

            div[data-testid="stDataFrame"] {
                border: 1px solid var(--line);
                border-radius: 8px;
                overflow: hidden;
            }

            /* ---- Mobile / narrow viewport tuning ---- */
            @media (max-width: 768px) {
                .main .block-container {
                    padding-top: 0.85rem;
                    padding-bottom: 1.25rem;
                    padding-left: 0.75rem;
                    padding-right: 0.75rem;
                }

                .hero {
                    padding: 0.85rem 0.9rem;
                    margin-bottom: 0.75rem;
                }

                .hero h1 {
                    font-size: 1.35rem;
                    line-height: 1.25;
                }

                .hero p {
                    font-size: 0.85rem;
                    margin-top: 0.35rem;
                }

                .badge {
                    font-size: 0.7rem;
                    padding: 0.2rem 0.5rem;
                }

                .kpi-card {
                    min-height: auto;
                    padding: 0.7rem 0.8rem;
                }

                .kpi-card .label {
                    font-size: 0.7rem;
                }

                .kpi-card .value {
                    font-size: 1.05rem;
                }

                .section-note {
                    font-size: 0.82rem;
                }

                /* Let the view switcher wrap into a tidy grid instead of one
                   overflowing horizontal line */
                div[role="radiogroup"] {
                    flex-wrap: wrap;
                    gap: 0.4rem;
                }

                div[role="radiogroup"] label {
                    font-size: 0.82rem;
                    padding: 0.25rem 0.5rem;
                }

                /* Matplotlib figures: never overflow the viewport width */
                div[data-testid="stImage"] img {
                    max-width: 100%;
                    height: auto;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(profile: dict[str, object]) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <h1>IMF World Economic Outlook Analytics</h1>
            <p>
                A professional macroeconomic intelligence dashboard built from IMF WEO 9.0.0,
                with adaptive schema detection, long-format transformation, historical and forecast
                separation, anomaly screening, rankings, and cross-country comparison.
            </p>
            <div class="badge-row">
                <span class="badge">{profile['countries']:,} countries/groups</span>
                <span class="badge">{profile['indicators']:,} indicators</span>
                <span class="badge">{profile['valid_observations']:,} valid observations</span>
                <span class="badge">{profile['year_min']} to {profile['year_max']}</span>
                <span class="badge">{profile['file_name']}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_cards(kpis: dict[str, str]) -> None:
    ordered = [
        "Total Countries",
        "Total Indicators",
        "Total Records",
        "Selected Year",
        "Highest GDP Country",
        "Highest Inflation Country",
        "Average Economic Growth",
        "Key Indicator Count",
    ]
    rows = [ordered[:4], ordered[4:]]
    for row in rows:
        columns = st.columns(4)
        for column, key in zip(columns, row):
            with column:
                st.markdown(
                    f"""
                    <div class="kpi-card">
                        <div class="label">{key}</div>
                        <div class="value">{kpis.get(key, "n/a")}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def format_ranking_table(table: pd.DataFrame) -> pd.io.formats.style.Styler | pd.DataFrame:
    if table.empty:
        return table
    formats = {}
    if "Value" in table.columns:
        formats["Value"] = "{:,.2f}"
    if "Robust Z-Score" in table.columns:
        formats["Robust Z-Score"] = "{:,.2f}"
    return table.style.format(formats)


def forecast_year_type_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["Year", "Year Type"], observed=True, as_index=False)["Value"]
        .mean()
        .pivot(index="Year", columns="Year Type", values="Value")
        .reset_index()
    )
    summary.columns.name = None
    return summary


def main() -> None:
    inject_css()
    set_chart_theme()

    bundle = load_data("data")
    profile = profile_dataset(bundle)
    long_df = bundle.long

    filter_state = render_sidebar_filters(long_df)
    selected_indicator = str(filter_state["indicator"])
    selected_year = int(filter_state["selected_year"])
    include_aggregates = bool(filter_state["include_aggregates"])
    top_n = int(filter_state["top_n"])

    filtered_df = apply_global_filters(long_df, filter_state)
    context_df = apply_context_filters(long_df, filter_state, keep_indicator=True, keep_countries=True)
    country_context_df = apply_context_filters(long_df, filter_state, keep_indicator=True, keep_countries=False)
    indicator_context_df = apply_context_filters(long_df, filter_state, keep_indicator=False, keep_countries=True)

    render_hero(profile)

    if filtered_df.empty:
        st.warning("No data matches the current filters. Use Reset filters or broaden the selection.")
        return

    st.subheader("Executive Overview")
    st.markdown(
        '<div class="section-note">All cards react to the active filters and selected year.</div>',
        unsafe_allow_html=True,
    )
    kpis = calculate_kpis(
        filtered_df,
        context_df,
        selected_year,
        selected_indicator,
        include_aggregates,
    )
    render_kpi_cards(kpis)

    active_view = st.radio(
        "Dashboard view",
        ["Executive View", "Course Charts", "Advanced Analytics", "Data Quality"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if active_view == "Executive View":
        primary_trend_fig = trend_line(filtered_df, selected_indicator)
        insights = generate_insights(
            indicator_context_df,
            selected_indicator,
            selected_year,
            top_n,
            include_aggregates,
        )
        st.subheader("Global Economic Explorer")
        st.markdown(
            '<div class="section-note">The comparison trend and ranking table use the active country, topic, scale, year, and search filters.</div>',
            unsafe_allow_html=True,
        )
        left, right = st.columns([1.45, 1])
        with left:
            st.pyplot(primary_trend_fig, width="stretch")
        with right:
            ranking = ranking_table(context_df, selected_indicator, selected_year, top_n, include_aggregates)
            st.markdown("#### Global Country Ranking Table")
            st.dataframe(
                format_ranking_table(ranking),
                width="stretch",
                hide_index=True,
                height=420,
            )
            st.download_button(
                "Download ranking CSV",
                data=ranking.to_csv(index=False).encode("utf-8"),
                file_name=dataframe_download_name(f"ranking_{selected_indicator}_{selected_year}"),
                mime="text/csv",
                width="stretch",
            )

        st.markdown("#### Insight Panel")
        for insight in insights:
            st.info(insight)

    if active_view == "Course Charts":
        primary_trend_fig = trend_line(filtered_df, selected_indicator)
        st.subheader("Required Course Charts")
        st.markdown(
            '<div class="section-note">Each chart is adapted to macroeconomic data and shares the global filter context.</div>',
            unsafe_allow_html=True,
        )
        chart_a, chart_b = st.columns(2)
        with chart_a:
            st.pyplot(indicator_category_pie(context_df), width="stretch")
        with chart_b:
            st.pyplot(indicator_histogram(filtered_df, selected_indicator), width="stretch")

        chart_c, chart_d = st.columns(2)
        with chart_c:
            st.pyplot(primary_trend_fig, width="stretch")
        with chart_d:
            st.pyplot(ranking_bar(context_df, selected_indicator, selected_year, top_n, include_aggregates), width="stretch")

        scatter_pairs = choose_scatter_pairs(context_df["Indicator"].dropna().unique())
        scatter_choice = st.radio(
            "Scatter relationship",
            options=list(scatter_pairs.keys()),
            index=0,
            horizontal=True,
        )
        x_indicator, y_indicator = scatter_pairs[scatter_choice]
        st.pyplot(
            scatter_economics(context_df, x_indicator, y_indicator, selected_year, include_aggregates),
            width="stretch",
        )

        chart_e, chart_f = st.columns(2)
        with chart_e:
            st.pyplot(box_plot_by_topic(context_df), width="stretch")
        with chart_f:
            st.pyplot(correlation_heatmap(context_df), width="stretch")

        chart_g, chart_h = st.columns(2)
        with chart_g:
            st.pyplot(area_chart(filtered_df, selected_indicator), width="stretch")
        with chart_h:
            st.pyplot(count_plot(context_df), width="stretch")

        st.pyplot(violin_plot(filtered_df, selected_indicator), width="stretch")

    if active_view == "Advanced Analytics":
        primary_trend_fig = trend_line(filtered_df, selected_indicator)
        st.subheader("Advanced Data Scientist Features")
        st.markdown(
            '<div class="section-note">Premium analytics for ranking, trend comparison, anomalies, correlations, and forecast interpretation.</div>',
            unsafe_allow_html=True,
        )
        adv_left, adv_right = st.columns([1.1, 1])
        with adv_left:
            st.markdown("#### Forecast vs Historical View")
            year_type_summary = forecast_year_type_summary(filtered_df)
            y_cols = [c for c in ["Historical", "Forecast"] if c in year_type_summary.columns]
            if y_cols:
                fig_fc, ax_fc = plt.subplots(figsize=(8, 3.5))
                for col in y_cols:
                    ax_fc.plot(year_type_summary["Year"], year_type_summary[col], label=col, marker="o", markersize=3)
                ax_fc.set_xlabel("Year")
                ax_fc.set_ylabel("Average Value")
                ax_fc.legend()
                ax_fc.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig_fc, width="stretch")
                plt.close(fig_fc)
            else:
                st.info("No historical/forecast data available for current selection.")

            st.markdown("#### Economic Trend Comparison")
            st.pyplot(primary_trend_fig, width="stretch")

        with adv_right:
            st.markdown("#### Anomaly Detector")
            anomalies = anomaly_detector(indicator_context_df, selected_indicator, limit=15)
            st.dataframe(
                format_ranking_table(anomalies),
                width="stretch",
                hide_index=True,
                height=400,
            )

            st.markdown("#### Correlation Explorer")
            all_indicators = context_df["Indicator"].dropna().value_counts().head(20).index.tolist()
            selected_corr = st.multiselect(
                "Indicators in correlation matrix",
                options=all_indicators,
                default=all_indicators[: min(6, len(all_indicators))],
            )
            st.pyplot(correlation_heatmap(context_df, selected_corr), width="stretch")

    if active_view == "Data Quality":
        st.subheader("Dataset Engineering Profile")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Rows", f"{profile['rows']:,}")
        p2.metric("Columns", f"{profile['columns']:,}")
        p3.metric("Year columns", f"{profile['year_columns']:,}")
        p4.metric("Missing year values", f"{profile['missing_year_value_pct']:.2f}%")

        st.markdown("#### Transformation")
        st.write(
            "The raw WEO file is loaded as strings, metadata whitespace is normalized, annual columns "
            "are detected automatically, and `pandas.melt()` converts the data into a long analytical "
            "table while preserving metadata."
        )

        quality = pd.DataFrame(
            [
                {"Check": "Full duplicate rows", "Result": profile["full_duplicates"]},
                {"Check": "Numeric parse failures in year columns", "Result": profile["numeric_parse_failures"]},
                {"Check": "Long-format valid observations", "Result": profile["valid_observations"]},
                {"Check": "Metadata columns preserved", "Result": profile["metadata_columns"]},
                {"Check": "Time-series coverage", "Result": f"{profile['year_min']} to {profile['year_max']}"},
            ]
        )
        st.dataframe(quality, width="stretch", hide_index=True)

        st.markdown("#### Filtered Long Format Preview")
        preview_cols = [
            column
            for column in ["Country", "Indicator", "Year", "Value", "Topic", "Scale", "Unit", "Year Type", "Series Code"]
            if column in filtered_df.columns
        ]
        st.dataframe(
            filtered_df[preview_cols].sort_values(["Country", "Indicator", "Year"]).head(1000),
            width="stretch",
            hide_index=True,
            height=460,
        )

        st.download_button(
            "Download filtered long-format CSV",
            data=filtered_df[preview_cols].to_csv(index=False).encode("utf-8"),
            file_name=dataframe_download_name("filtered_weo_long_format"),
            mime="text/csv",
            width="stretch",
        )


if __name__ == "__main__":
    main()
