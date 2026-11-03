"""Data, leakage, saved-model, invalid-input and Streamlit regression checks."""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import utils
from train_model import regression_metrics
from utils import (ROOT, INPUTS, TARGET, NUMERIC, DERIVED, add_features, build_pipeline,
                   file_hash, load_data, predict_revenue, read_json, split_data)


@pytest.fixture(scope="module")
def data():
    return load_data()[0]


@pytest.fixture(scope="module")
def model():
    return joblib.load(ROOT / "models" / "revenue_pipeline.joblib")


@pytest.fixture(scope="module")
def metadata():
    return read_json(ROOT / "models" / "metadata.json")


@pytest.fixture
def valid_input(metadata):
    values = metadata["defaults"].copy()
    for column, choices in metadata["categories"].items():
        values[column] = choices[0]
    return values


def test_source_quality_and_exact_duplicates(data):
    frame, quality = load_data()
    assert quality["raw_rows"] == 122400
    assert quality["duplicates_removed"] == 2400
    assert len(frame) == 120000 and not frame.duplicated().any()
    assert data["likes"].isna().sum() == 6000
    assert data.duplicated(["video_id", "date"]).sum() == 3849


def test_group_split_has_no_overlap_and_is_reproducible(data):
    train, validation, test = split_data(data)
    ids = [set(part.video_id) for part in [train, validation, test]]
    assert not ids[0] & ids[1] and not ids[0] & ids[2] and not ids[1] & ids[2]
    assert len(train) + len(validation) + len(test) == len(data)
    for first, second in zip([train, validation, test], split_data(data)):
        pd.testing.assert_frame_equal(first, second)


@pytest.mark.parametrize("column,value", [("views", -1), ("likes", np.inf),
    ("comments", 1.5), ("views", 0), ("video_length_minutes", 0),
    (TARGET, np.nan), ("date", "not-a-date"), ("video_id", " ")])
def test_invalid_source_rejected(data, tmp_path, column, value):
    frame = data.head(20).copy()
    if column == "date":
        frame[column] = frame[column].astype(object)
    frame.loc[0, column] = value
    path = tmp_path / "bad.csv"
    frame.to_csv(path, index=False)
    with pytest.raises((ValueError, TypeError)):
        load_data(path)


def test_missing_schema_and_empty_file(data, tmp_path):
    path = tmp_path / "bad.csv"
    data.head(2).drop(columns=[TARGET]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing columns"):
        load_data(path)
    data.head(0).to_csv(path, index=False)
    with pytest.raises(ValueError, match="no rows"):
        load_data(path)


def test_feature_math_no_target_leak_and_zero_division(data):
    row = data.head(1).copy()
    row.loc[0, ["views", "likes", "comments", "watch_time_minutes"]] = [100, 10, 5, 250]
    features = add_features(row)
    assert features.loc[0, "engagement_rate"] == 0.15
    assert features.loc[0, "average_watch_minutes"] == 2.5
    assert set(features.columns) == set(INPUTS + DERIVED)
    row.loc[0, TARGET] = 999999
    pd.testing.assert_frame_equal(features, add_features(row))
    row.loc[0, "views"] = 0
    assert add_features(row)[DERIVED].isna().all().all()


def test_imputer_learns_only_fitted_rows(data):
    train = data.head(100).copy()
    pipeline = build_pipeline(LinearRegression()).fit(train[INPUTS], train[TARGET])
    imputer = pipeline["preparation"].named_transformers_["numbers"]["imputer"]
    expected = add_features(train)[NUMERIC + DERIVED].median().to_numpy()
    np.testing.assert_allclose(imputer.statistics_, expected)
    before = imputer.statistics_.copy()
    other = data.iloc[100:120][INPUTS].copy()
    other["likes"] = 1e9
    pipeline.predict(other)
    np.testing.assert_array_equal(imputer.statistics_, before)


def test_final_imputer_and_artifact_hashes(data, model, metadata):
    train, validation, test = split_data(data)
    development = pd.concat([train, validation])
    expected = add_features(development)[NUMERIC + DERIVED].median().to_numpy()
    imputer = model["preparation"].named_transformers_["numbers"]["imputer"]
    np.testing.assert_allclose(imputer.statistics_, expected)
    assert file_hash(utils.RAW_DATA) == metadata["data_sha256"]
    assert file_hash(ROOT / "models" / "revenue_pipeline.joblib") == metadata["model_sha256"]


def test_predictions_match_saved_test_results(data, model):
    test = split_data(data)[2]
    saved = pd.read_csv(ROOT / "reports" / "test_predictions.csv")
    predicted = model.predict(test[INPUTS])
    np.testing.assert_allclose(predicted, saved["predicted"], rtol=1e-10)
    scores = regression_metrics(test[TARGET], predicted)
    recorded = read_json(ROOT / "reports" / "test_metrics.json")
    for key in scores:
        assert scores[key] == pytest.approx(recorded[key])
    assert recorded["mae"] < recorded["mean_baseline"]["mae"]


def test_model_reload_and_column_order(data, model, tmp_path):
    path = tmp_path / "copy.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)
    sample = data.head(100)[INPUTS]
    np.testing.assert_allclose(model.predict(sample), restored.predict(sample[INPUTS[::-1]]))


def test_unknown_category_safe_at_pipeline_level(data, model):
    sample = data.head(3)[INPUTS].copy()
    sample["category"] = "Unseen category"
    assert np.isfinite(model.predict(sample)).all()


@pytest.mark.parametrize("column,value", [("views", 0), ("views", 1e6),
    ("likes", -1), ("comments", 48.5), ("watch_time_minutes", np.inf),
    ("views", None), ("video_length_minutes", "invalid"), ("country", "XX")])
def test_bad_prediction_inputs(model, metadata, valid_input, column, value):
    valid_input[column] = value
    with pytest.raises(ValueError):
        predict_revenue(model, valid_input, metadata)


@pytest.mark.parametrize("unknown", [[], ["likes"], ["comments"], ["watch_time_minutes"],
                                     ["likes", "comments", "watch_time_minutes"]])
def test_valid_and_missing_input_predictions(model, metadata, valid_input, unknown):
    for column in unknown:
        valid_input[column] = None
    prediction, notes = predict_revenue(model, valid_input, metadata)
    assert np.isfinite(prediction) and prediction >= 0
    assert len(notes) >= len(unknown)


def test_clean_export_is_complete_and_auditable():
    cleaned = pd.read_csv(ROOT / "data" / "cleaned_youtube_data.csv")
    assert len(cleaned) == 120000 and not cleaned.isna().any().any()
    assert cleaned["likes_was_missing"].sum() == 6000
    assert set(cleaned["partition"]) == {"train", "validation", "test"}


@pytest.mark.parametrize("column", NUMERIC)
def test_displayed_range_endpoints_are_accepted(model, metadata, valid_input, column):
    for value in metadata["ranges"][column]:
        valid_input[column] = value
        prediction, notes = predict_revenue(model, valid_input, metadata)
        assert np.isfinite(prediction)


@pytest.mark.parametrize("page", ["Overview", "Explore Data", "Predict Revenue", "Model Results"])
def test_all_app_pages(page):
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=40).run()
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception and not app.error


def test_app_valid_invalid_and_missing_predictions():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=40).run()
    app.sidebar.radio[0].set_value("Predict Revenue").run()
    app.button[0].click().run()
    assert app.metric[0].value.startswith("$") and not app.exception
    app.number_input(key="input_views").set_value(1000000)
    app.button[0].click().run()
    assert app.error and "outside" in app.error[0].value and not app.exception
    app.number_input(key="input_views").set_value(10000)
    app.multiselect[0].set_value(["watch_time_minutes"]).run()
    app.button[0].click().run()
    assert app.warning and app.metric and not app.exception


def test_app_filters_and_partial_date():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=40).run()
    app.sidebar.radio[0].set_value("Explore Data").run()
    app.multiselect[0].set_value(["Gaming"]).run()
    assert int(app.metric[0].value.replace(",", "")) < 120000
    app.date_input[0].set_value((pd.Timestamp("2024-06-09").date(),)).run()
    assert app.info and not app.exception


def test_missing_model_has_recovery_message(monkeypatch, tmp_path):
    monkeypatch.setattr(utils, "ROOT", tmp_path)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=40).run()
    app.sidebar.radio[0].set_value("Predict Revenue").run()
    assert app.error and "unavailable" in app.error[0].value and not app.exception
