"""Run this file to train all five models and regenerate the model and reports."""

import platform
import time
from importlib.metadata import version

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.tree import DecisionTreeRegressor

from charts import create_figures
from utils import (ROOT, RAW_DATA, TARGET, SEED, INPUTS, NUMERIC, CATEGORICAL, DERIVED,
                   add_features, build_pipeline, file_hash, load_data,
                   make_input_metadata, save_json, split_data)


def regression_metrics(actual, predicted):
    return {"mae": float(mean_absolute_error(actual, predicted)),
            "rmse": float(root_mean_squared_error(actual, predicted)),
            "r2": float(r2_score(actual, predicted))}


def get_models():
    """Fixed, modest settings keep the comparison reproducible and easy to explain."""
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=10.0),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, min_samples_leaf=20,
                                                random_state=SEED),
        "Random Forest": RandomForestRegressor(n_estimators=80, max_depth=12,
                                                min_samples_leaf=8, n_jobs=2,
                                                random_state=SEED),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=2,
                                                        min_samples_leaf=20,
                                                        random_state=SEED),
    }


def export_clean_data(data, pipeline, partitions):
    """Export unscaled, readable values using medians learned from train + validation."""
    cleaned = data.copy()
    features = add_features(data)
    number_imputer = pipeline["preparation"].named_transformers_["numbers"]["imputer"]
    filled = number_imputer.transform(features[NUMERIC + DERIVED])
    cleaned[NUMERIC + DERIVED] = filled[:, :len(NUMERIC + DERIVED)]
    for column in NUMERIC + DERIVED:
        cleaned[column + "_was_missing"] = features[column].isna()
    category_imputer = pipeline["preparation"].named_transformers_["categories"]["imputer"]
    cleaned[CATEGORICAL] = category_imputer.transform(features[CATEGORICAL])
    cleaned["partition"] = cleaned["video_id"].map(partitions)
    cleaned.to_csv(ROOT / "data" / "cleaned_youtube_data.csv", index=False)


def run_training():
    """Compare on validation, refit the selected model, then evaluate test data once."""
    for name in ["models", "reports", "reports/figures"]:
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    data, quality = load_data()
    train, validation, test = split_data(data)
    models = get_models()
    fitted = {}
    rows = []
    for name, estimator in models.items():
        print(f"Training {name} ...", flush=True)
        started = time.perf_counter()
        pipeline = build_pipeline(estimator)
        pipeline.fit(train[INPUTS], train[TARGET])
        row = {"model": name}
        for label, frame in [("train", train), ("validation", validation)]:
            scores = regression_metrics(frame[TARGET], pipeline.predict(frame[INPUTS]))
            row.update({label + "_" + key: value for key, value in scores.items()})
        row["fit_and_score_seconds"] = round(time.perf_counter() - started, 2)
        rows.append(row)
        fitted[name] = pipeline
    comparison = pd.DataFrame(rows)
    # Within five cents of the lowest RMSE, prefer the earlier, simpler model.
    limit = comparison["validation_rmse"].min() + 0.05
    chosen = comparison.loc[comparison["validation_rmse"] <= limit, "model"].iloc[0]
    comparison["selected"] = comparison["model"] == chosen
    comparison.to_csv(ROOT / "reports" / "model_comparison.csv", index=False)

    # Measure original input importance on validation BEFORE refitting on it.
    sample = validation.sample(min(1500, len(validation)), random_state=SEED)
    result = permutation_importance(fitted[chosen], sample[INPUTS], sample[TARGET],
                                    scoring="neg_mean_absolute_error", n_repeats=3,
                                    random_state=SEED, n_jobs=1)
    importance = pd.DataFrame({"feature": INPUTS, "mae_increase_usd": result.importances_mean,
                               "std_usd": result.importances_std})
    importance.to_csv(ROOT / "reports" / "feature_importance.csv", index=False)

    development = pd.concat([train, validation])
    final_pipeline = build_pipeline(clone(models[chosen]))
    final_pipeline.fit(development[INPUTS], development[TARGET])
    predicted = final_pipeline.predict(test[INPUTS])
    scores = regression_metrics(test[TARGET], predicted)
    baseline = np.full(len(test), development[TARGET].mean())
    scores["mean_baseline"] = regression_metrics(test[TARGET], baseline)
    scores["model"] = chosen
    scores["test_rows"] = len(test)
    scores["watch_time_groups"] = {}
    for missing in [False, True]:
        mask = test["watch_time_minutes"].isna().to_numpy() == missing
        label = "missing" if missing else "observed"
        group_scores = regression_metrics(test.loc[mask, TARGET], predicted[mask])
        scores["watch_time_groups"][label] = {"rows": int(mask.sum()), **group_scores}
    save_json(scores, ROOT / "reports" / "test_metrics.json")

    predictions = pd.DataFrame({"video_id": test["video_id"], "actual": test[TARGET],
                                "predicted": predicted,
                                "watch_time_missing": test["watch_time_minutes"].isna()})
    predictions.to_csv(ROOT / "reports" / "test_predictions.csv", index=False)
    partitions = {}
    split_summary = {}
    for label, frame in [("train", train), ("validation", validation), ("test", test)]:
        partitions.update(dict.fromkeys(frame["video_id"].unique(), label))
        split_summary[label] = {"rows": len(frame), "video_ids": int(frame.video_id.nunique())}
    pd.DataFrame(list(partitions.items()), columns=["video_id", "partition"]).to_csv(
        ROOT / "reports" / "data_partitions.csv", index=False)
    save_json(quality, ROOT / "reports" / "data_quality.json")
    model_path = ROOT / "models" / "revenue_pipeline.joblib"
    joblib.dump(final_pipeline, model_path, compress=3)
    # A round-trip check catches serialization or feature-order mistakes immediately.
    restored = joblib.load(model_path)
    np.testing.assert_allclose(restored.predict(test[INPUTS].head(20)), predicted[:20])
    metadata = make_input_metadata(development)
    metadata.update({"model": chosen, "seed": SEED, "split": split_summary,
                     "model_sha256": file_hash(model_path), "data_sha256": file_hash(RAW_DATA),
                     "python_version": platform.python_version(),
                     "versions": {name: version(name) for name in
                                  ["scikit-learn", "numpy", "pandas", "joblib"]},
                     "selection_rule": "Lowest validation RMSE; simpler model within $0.05",
                     "fitted_on": "Training and validation IDs; test IDs excluded"})
    save_json(metadata, ROOT / "models" / "metadata.json")
    export_clean_data(data, final_pipeline, partitions)
    create_figures(train, quality, comparison, predictions, importance)
    print(f"Selected {chosen}. Test MAE=${scores['mae']:.2f}, "
          f"RMSE=${scores['rmse']:.2f}, R2={scores['r2']:.4f}", flush=True)
    return comparison, scores


if __name__ == "__main__":
    run_training()
