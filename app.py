"""Run with: python -m streamlit run app.py"""

from importlib.metadata import version
import json

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from utils import (ROOT, RAW_DATA, NUMERIC, CATEGORICAL, OPTIONAL, TARGET,
                   file_hash, load_data, predict_revenue, read_json)

st.set_page_config(page_title="Content Monetization Modeler", page_icon="▶",
                   layout="wide", initial_sidebar_state="auto")
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = ["#137F76", "#D28A53", "#6486A0", "#A69A62"]
LABELS = {"views": "Views", "likes": "Likes", "comments": "Comments",
          "watch_time_minutes": "Total watch time (minutes)",
          "video_length_minutes": "Video length (minutes)", "subscribers": "Subscribers",
          TARGET: "Ad revenue (USD)", "category": "Category", "country": "Country",
          "device": "Device"}


@st.cache_data
def get_data(modified_time):
    """The timestamp makes Streamlit refresh its cache when the CSV changes."""
    return load_data()


@st.cache_resource
def get_model(model_time, metadata_time):
    """Load only the project's own model, checking its hash and package versions."""
    metadata = read_json(ROOT / "models" / "metadata.json")
    model_path = ROOT / "models" / "revenue_pipeline.joblib"
    if file_hash(model_path) != metadata["model_sha256"]:
        raise ValueError("Model and metadata do not match. Run python train_model.py.")
    for package, expected in metadata["versions"].items():
        if version(package) != expected:
            raise ValueError(f"Install requirements.txt: {package} must be {expected}.")
    return joblib.load(model_path), metadata


def model_assets():
    """Turn missing or incompatible model files into a helpful recovery message."""
    try:
        model_path = ROOT / "models" / "revenue_pipeline.joblib"
        metadata_path = ROOT / "models" / "metadata.json"
        return get_model(model_path.stat().st_mtime_ns, metadata_path.stat().st_mtime_ns)
    except Exception as error:
        st.error(f"The saved model is unavailable: {error}")
        st.info("From the project folder, install requirements.txt and run "
                "python train_model.py. Then refresh this page.")
        st.stop()


def plot(figure):
    figure.update_layout(margin=dict(l=15, r=15, t=50, b=20), height=370,
                          paper_bgcolor="rgba(0,0,0,0)", font=dict(size=13))
    st.plotly_chart(figure, use_container_width=True)


def overview(data, quality):
    st.caption("SOCIAL MEDIA ANALYTICS / PYTHON + MACHINE LEARNING")
    st.title("YouTube revenue, explained")
    st.write("Explore video performance and estimate advertising revenue with a "
             "model trained on the supplied learning dataset.")
    columns = st.columns(4)
    columns[0].metric("Observations", f"{len(data):,}")
    columns[1].metric("Video identifiers", f"{data.video_id.nunique():,}")
    columns[2].metric("Mean revenue / observation", f"${data[TARGET].mean():,.2f}")
    columns[3].metric("Exact duplicates removed", f"{quality['duplicates_removed']:,}")
    st.divider()
    left, right = st.columns([2, 1])
    with left:
        sample = data.sample(min(2000, len(data)), random_state=42)
        plot(px.scatter(sample, x="watch_time_minutes", y=TARGET, color="category",
                        opacity=0.45, labels=LABELS,
                        title="Watch time and revenue in the supplied sample"))
        st.caption("2,000 sampled rows, after exact duplicate removal. "
                   "This is a descriptive relationship, not a causal result.")
    with right:
        st.subheader("Start with the data")
        st.write("**Explore Data** — filter observations and inspect distributions.")
        st.write("**Predict Revenue** — enter video metrics to estimate USD revenue.")
        st.write("**Model Results** — understand validation scores and test errors.")
        st.info("Synthetic learning data. Estimates apply to one reporting observation, "
                "not a video's lifetime earnings.")
    with st.expander("Understand the dataset before making predictions"):
        st.write("Views in the source range from 9,521 to 10,468. The prediction form "
                 "accepts the numeric ranges used to fit the model. It does not "
                 "extrapolate to very small or very large videos.")
        st.write("Video identifiers repeat with inconsistent categories and lengths. "
                 "We preserve distinct observations and keep each ID in one data split.")
        st.write("Likes, comments and watch time each have 5% missing values after "
                 "deduplication. The model uses fitted medians when these are unknown.")


def explore(data):
    st.title("Explore the dataset")
    st.write("Descriptive charts use all deduplicated observations. Missing values "
             "remain missing here; they are not replaced by model estimates.")
    controls = st.columns(3)
    selected = {}
    for column, control in zip(CATEGORICAL, controls):
        selected[column] = control.multiselect(LABELS[column],
                                               sorted(data[column].dropna().unique()))
    dates = st.date_input("Reporting dates", value=(data.date.min().date(), data.date.max().date()),
                          min_value=data.date.min().date(), max_value=data.date.max().date())
    if len(dates) != 2:
        st.info("Choose both a start date and an end date.")
        return
    filtered = data[(data.date.dt.date >= dates[0]) & (data.date.dt.date <= dates[1])]
    for column in CATEGORICAL:
        if selected[column]:
            filtered = filtered[filtered[column].isin(selected[column])]
    if filtered.empty:
        st.info("No observations match these filters. Change a date or category filter.")
        return
    metrics = st.columns(3)
    metrics[0].metric("Matching observations", f"{len(filtered):,}")
    metrics[1].metric("Mean revenue (USD)", f"${filtered[TARGET].mean():,.2f}")
    metrics[2].metric("Median revenue (USD)", f"${filtered[TARGET].median():,.2f}")
    first, second, third = st.tabs(["Revenue patterns", "Distributions & quality", "Data preview"])
    with first:
        left, right = st.columns(2)
        with left:
            group = st.selectbox("Compare by", CATEGORICAL, format_func=lambda x: LABELS[x])
            means = filtered.groupby(group, as_index=False)[TARGET].mean()
            plot(px.bar(means, x=group, y=TARGET, labels=LABELS, title="Mean revenue by group"))
        with right:
            sample = filtered.sample(min(2500, len(filtered)), random_state=42)
            feature = st.selectbox("Compare revenue with", NUMERIC, format_func=lambda x: LABELS[x])
            plot(px.scatter(sample, x=feature, y=TARGET, opacity=0.3, labels=LABELS,
                            title="Revenue relationship (up to 2,500 rows)"))
        monthly = filtered.set_index("date")[TARGET].resample("MS").mean().reset_index()
        plot(px.line(monthly, x="date", y=TARGET, markers=True, labels=LABELS,
                     title="Mean revenue per observation by month"))
        st.caption("The date chart describes this sample; it does not establish "
                   "growth or seasonality for a real channel.")
    with second:
        field = st.selectbox("Inspect a numeric field", [TARGET] + NUMERIC,
                             format_func=lambda x: LABELS[x])
        left, right = st.columns(2)
        with left:
            plot(px.histogram(filtered, x=field, nbins=35, labels=LABELS, title="Distribution"))
        with right:
            plot(px.box(filtered, y=field, points=False, labels=LABELS, title="Spread and quartiles"))
        missing = filtered.isna().sum().rename("Missing rows").to_frame()
        st.dataframe(missing, width="stretch")
        correlation = filtered[NUMERIC + [TARGET]].corr()
        plot(px.imshow(correlation, text_auto=".2f", zmin=-1, zmax=1,
                       color_continuous_scale="BrBG", title="Numeric correlations"))
    with third:
        st.caption("Preview shows the first 200 matching rows. Download includes all matching rows.")
        st.dataframe(filtered.head(200), hide_index=True, width="stretch")
        st.download_button("Download filtered CSV", filtered.to_csv(index=False).encode("utf-8"),
                           "filtered_youtube_data.csv", "text/csv")


def predict():
    st.title("Estimate advertising revenue")
    st.write("Enter metrics for one reporting observation. Use the same reporting "
             "period for views, interactions and total watch time.")
    pipeline, metadata = model_assets()
    st.caption(f"Model: {metadata['model']} · Revenue currency: USD")
    unknown = st.multiselect("Mark inputs you do not know", OPTIONAL,
                             format_func=lambda x: LABELS[x])
    values = {}
    with st.form("revenue_form"):
        left, right = st.columns(2)
        for index, column in enumerate(NUMERIC):
            panel = left if index < 3 else right
            low, high = metadata["ranges"][column]
            integer = column in ["views", "likes", "comments", "subscribers"]
            default = metadata["defaults"][column]
            value = panel.number_input(LABELS[column], min_value=0 if integer else 0.0,
                                        value=int(default) if integer else round(default, 2),
                                        step=1 if integer else 0.1, disabled=column in unknown,
                                        help=f"Fitted range: {low:,.2f} to {high:,.2f}",
                                        key="input_" + column)
            values[column] = None if column in unknown else value
        columns = st.columns(3)
        for column, panel in zip(CATEGORICAL, columns):
            values[column] = panel.selectbox(LABELS[column], metadata["categories"][column])
        submitted = st.form_submit_button("Predict revenue", type="primary", width="stretch")
    if submitted:
        try:
            revenue, notes = predict_revenue(pipeline, values, metadata)
        except ValueError as error:
            st.error(str(error))
            return
        with st.container(border=True):
            st.metric("Estimated ad revenue (USD)", f"${revenue:,.2f}")
            st.caption("Estimate for this reporting observation. It is not a guaranteed payment.")
        for note in notes:
            st.warning(note)
        scores = read_json(ROOT / "reports" / "test_metrics.json")
        group = "missing" if "watch_time_minutes" in unknown else "observed"
        error = scores["watch_time_groups"][group]["mae"]
        st.info(f"In held-out test rows with {group} watch time, mean absolute error "
                f"was ${error:.2f}. This is historical average error, not a confidence interval.")
        payload = {"inputs": values, "estimated_ad_revenue_usd": revenue, "model": metadata["model"]}
        st.download_button("Download this estimate", json.dumps(payload, indent=2),
                           "revenue_estimate.json", "application/json")
    with st.expander("Supported ranges and missing inputs"):
        ranges = pd.DataFrame(metadata["ranges"], index=["Minimum", "Maximum"]).T
        st.dataframe(ranges, width="stretch")
        st.write("Numeric inputs outside these ranges are rejected. Inputs within the "
                 "individual ranges can still form an unusual combination. The synthetic "
                 "training data does not validate estimates for real channels.")
        st.write("Unknown values use the medians saved during training. In particular, "
                 "missing watch time makes estimates much less informative.")


def model_results():
    st.title("Model comparison & results")
    pipeline, metadata = model_assets()
    scores = read_json(ROOT / "reports" / "test_metrics.json")
    comparison = pd.read_csv(ROOT / "reports" / "model_comparison.csv")
    st.write(f"**Selected model: {metadata['model']}.** {metadata['selection_rule']}.")
    st.caption("These headline metrics use the reserved test set after the selected "
               "model was refitted on training + validation data.")
    columns = st.columns(3)
    columns[0].metric("Test MAE (USD)", f"${scores['mae']:.2f}")
    columns[1].metric("Test RMSE (USD)", f"${scores['rmse']:.2f}")
    columns[2].metric("Test R²", f"{scores['r2']:.4f}")
    with st.expander("What do these scores mean?"):
        st.write("MAE is the average size of a prediction error in dollars. RMSE gives "
                 "larger errors extra weight. Lower is better for both. R² compares "
                 "the model with a mean prediction; it is not an accuracy percentage.")
        st.write(f"A simple mean-only reference has test MAE ${scores['mean_baseline']['mae']:.2f}.")
    st.subheader("Validation comparison")
    plot(px.bar(comparison.sort_values("validation_rmse"), x="model", y="validation_rmse",
                 labels={"model": "Model", "validation_rmse": "Validation RMSE (USD)"},
                 title="All five models used the same training and validation partitions"))
    st.dataframe(comparison.drop(columns="fit_and_score_seconds").round(4),
                 hide_index=True, width="stretch")
    st.subheader("Why missing watch time matters")
    groups = pd.DataFrame(scores["watch_time_groups"]).T
    st.dataframe(groups.round(4), width="stretch")
    st.caption("Rows are grouped by whether watch time was originally observed. "
               "Imputation cannot recover information that was never recorded.")
    st.subheader("Prediction errors")
    report_image("09_prediction_errors.png")
    st.subheader("Which inputs help the model?")
    report_image("10_feature_importance.png")
    st.caption("Importance was measured on validation rows before refitting. It measures "
               "predictive reliance, not causation; correlated features can share information.")
    st.subheader("Data separation")
    st.dataframe(pd.DataFrame(metadata["split"]).T, width="stretch")
    st.write("Each video ID belongs to exactly one partition. The target, date and video ID "
             "are excluded from model inputs. Test rows did not choose the model or medians.")


def report_image(filename):
    path = ROOT / "reports" / "figures" / filename
    if path.exists():
        st.image(str(path), width="stretch")
    else:
        st.warning(f"Missing chart: {filename}. Run python train_model.py to rebuild it.")


def main():
    st.sidebar.title("Content Monetization")
    st.sidebar.caption("YouTube revenue modeler")
    page = st.sidebar.radio("Navigation", ["Overview", "Explore Data", "Predict Revenue", "Model Results"])
    st.sidebar.divider()
    st.sidebar.caption("Python · Pandas · Scikit-learn · Streamlit")
    st.sidebar.info("Learning project using synthetic data. All revenue values are in USD.")
    try:
        data, quality = get_data(RAW_DATA.stat().st_mtime_ns)
        if page == "Overview":
            overview(data, quality)
        elif page == "Explore Data":
            explore(data)
        elif page == "Predict Revenue":
            predict()
        else:
            model_results()
    except (OSError, ValueError, KeyError) as error:
        st.error(f"A project file needs attention: {error}")
        st.info("Keep the data, models and reports folders together. Restore the supplied "
                "CSV or run python train_model.py to regenerate the reports.")


if __name__ == "__main__":
    main()
