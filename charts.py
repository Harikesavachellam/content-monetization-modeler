"""Create readable figures for the notebook, reports and explanation PDF."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from utils import ROOT, NUMERIC, TARGET


def create_figures(train, quality, comparison, predictions, importance):
    """EDA uses training rows. Error plots use the final, reserved test rows."""
    folder = ROOT / "reports" / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="deep", font_scale=1.05)
    paths = []

    def finish(figure, name):
        figure.tight_layout()
        path = folder / f"{name}.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)

    figure, axis = plt.subplots(figsize=(9, 4))
    missing = quality["missing_after_dedup"]
    missing = {key: value for key, value in missing.items() if value > 0}
    axis.bar(list(missing), list(missing.values()), color="#18857d")
    axis.set(title="Missing values after duplicate removal (all rows)", ylabel="Rows")
    finish(figure, "01_missing_values")

    figure, axes = plt.subplots(2, 2, figsize=(11, 7))
    for axis, column in zip(axes.flat, [TARGET, "views", "watch_time_minutes", "likes"]):
        axis.hist(train[column].dropna(), bins=35, color="#18857d", edgecolor="white")
        axis.set(title=column.replace("_", " ").title(), ylabel="Training rows")
    finish(figure, "02_distributions")

    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, column in zip(axes, ["views", "watch_time_minutes", TARGET]):
        sns.boxplot(y=train[column], ax=axis, color="#86c8ba")
        axis.set(title="Training distribution", ylabel=column.replace("_", " "))
    finish(figure, "03_outliers")

    sample = train.sample(min(2500, len(train)), random_state=42)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for axis, column in zip(axes, ["views", "watch_time_minutes"]):
        axis.scatter(sample[column], sample[TARGET], s=9, alpha=0.25, color="#18857d")
        axis.set(xlabel=column.replace("_", " "), ylabel="Ad revenue (USD)",
                 title="2,500 sampled training observations")
    finish(figure, "04_revenue_relationships")

    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, column in zip(axes, ["category", "country", "device"]):
        means = train.groupby(column)[TARGET].mean().sort_values()
        axis.barh(means.index, means.values, color="#18857d")
        axis.set(xlabel="Mean ad revenue (USD)", title=f"Training sample by {column}")
        axis.set_xlim(left=0)
    finish(figure, "05_group_comparisons")

    figure, axis = plt.subplots(figsize=(9, 7))
    sns.heatmap(train[NUMERIC + [TARGET]].corr(), annot=True, fmt=".2f",
                cmap="BrBG", center=0, vmin=-1, vmax=1, ax=axis)
    axis.set_title("Numeric correlations in training data")
    finish(figure, "06_correlations")

    figure, axis = plt.subplots(figsize=(10, 4))
    monthly = train.set_index("date")[TARGET].resample("MS").mean()
    axis.plot(monthly.index, monthly.values, marker="o", color="#18857d")
    axis.set(title="Mean revenue per observation by month (training sample)",
             ylabel="Mean revenue (USD)", xlabel="Reporting month")
    finish(figure, "07_monthly_sample")

    figure, axis = plt.subplots(figsize=(9, 4))
    ordered = comparison.sort_values("validation_rmse")
    axis.barh(ordered["model"], ordered["validation_rmse"], color="#18857d")
    axis.invert_yaxis()
    axis.set(xlabel="Validation RMSE (USD), lower is better", title="Five-model comparison")
    finish(figure, "08_model_comparison")

    sample = predictions.sample(min(3000, len(predictions)), random_state=42)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(sample["actual"], sample["predicted"], s=8, alpha=0.3)
    low = min(sample["actual"].min(), sample["predicted"].min())
    high = max(sample["actual"].max(), sample["predicted"].max())
    axes[0].plot([low, high], [low, high], "--", color="#bd5c3c")
    axes[0].set(xlabel="Actual revenue (USD)", ylabel="Predicted revenue (USD)",
                title="Reserved test sample: ideal predictions lie on the line")
    axes[1].scatter(sample["predicted"], sample["actual"] - sample["predicted"],
                    s=8, alpha=0.3, color="#18857d")
    axes[1].axhline(0, linestyle="--", color="#bd5c3c")
    axes[1].set(xlabel="Predicted revenue (USD)", ylabel="Actual minus predicted (USD)",
                title="Reserved test residuals")
    finish(figure, "09_prediction_errors")

    figure, axis = plt.subplots(figsize=(9, 5))
    ordered = importance.sort_values("mae_increase_usd")
    axis.barh(ordered["feature"], ordered["mae_increase_usd"], color="#18857d")
    axis.set(xlabel="Increase in validation MAE after shuffling (USD)",
             title="Feature importance before final refit; correlated inputs share effects")
    finish(figure, "10_feature_importance")
    return paths
