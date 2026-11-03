"""Small shared functions for data preparation, training and prediction."""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
RAW_DATA = ROOT / "data" / "youtube_ad_revenue_dataset.csv"
TARGET = "ad_revenue_usd"
SEED = 42
NUMERIC = ["views", "likes", "comments", "watch_time_minutes",
           "video_length_minutes", "subscribers"]
CATEGORICAL = ["category", "device", "country"]
INPUTS = NUMERIC + CATEGORICAL
DERIVED = ["engagement_rate", "average_watch_minutes"]
OPTIONAL = ["likes", "comments", "watch_time_minutes"]


def file_hash(path):
    """Identify a file so reports can be matched to their data and model."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(data, path):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_data(path=RAW_DATA):
    """Validate the source and remove exact duplicates without changing the file."""
    data = pd.read_csv(path)
    required = ["video_id", "date"] + INPUTS + [TARGET]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    if data.empty:
        raise ValueError("The dataset has no rows.")
    data = data[required].copy()
    for column in NUMERIC + [TARGET]:
        data[column] = pd.to_numeric(data[column], errors="raise")
        observed = data[column].dropna()
        if not np.isfinite(observed).all() or (observed < 0).any():
            raise ValueError(f"{column} must contain finite, non-negative numbers.")
        if column in ["views", "likes", "comments", "subscribers"]:
            if (observed % 1 != 0).any():
                raise ValueError(f"{column} must contain whole-number counts.")
    essential = ["video_id", "date", "views", "video_length_minutes", "subscribers", TARGET]
    if data[essential].isna().any().any():
        raise ValueError("An essential identifier, date or numeric field is missing.")
    if (data["views"] <= 0).any() or (data["video_length_minutes"] <= 0).any():
        raise ValueError("Views and video length must be greater than zero.")
    data["date"] = pd.to_datetime(data["date"], format="ISO8601", errors="raise")
    for column in ["video_id"] + CATEGORICAL:
        data[column] = data[column].str.strip().replace("", np.nan)
    if data["video_id"].isna().any():
        raise ValueError("Video IDs cannot be blank.")
    quality = {"raw_rows": len(data), "duplicates_removed": int(data.duplicated().sum())}
    data = data.drop_duplicates().reset_index(drop=True)
    quality["clean_rows"] = len(data)
    quality["missing_after_dedup"] = data.isna().sum().astype(int).to_dict()
    quality["unique_video_ids"] = int(data["video_id"].nunique())
    quality["repeated_video_date_rows"] = int(data.duplicated(["video_id", "date"]).sum())
    return data, quality


def split_data(data):
    """Allocate 60/20/20 percent of video IDs to train/validation/test."""
    if data["video_id"].nunique() < 10:
        raise ValueError("At least 10 distinct video IDs are needed for this split.")
    splitter = GroupShuffleSplit(n_splits=1, train_size=0.6, random_state=SEED)
    train_index, other_index = next(splitter.split(data, groups=data["video_id"]))
    train = data.iloc[train_index].copy()
    other = data.iloc[other_index].copy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
    val_index, test_index = next(splitter.split(other, groups=other["video_id"]))
    return train, other.iloc[val_index].copy(), other.iloc[test_index].copy()


def add_features(data):
    """Create two ratios; unknown inputs remain unknown until imputation."""
    result = data[INPUTS].copy()
    safe_views = result["views"].replace(0, np.nan)
    result["engagement_rate"] = (result["likes"] + result["comments"]) / safe_views
    result["average_watch_minutes"] = result["watch_time_minutes"] / safe_views
    return result


def build_pipeline(estimator):
    """Save feature creation, imputation, encoding, scaling and the model together."""
    numbers = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    categories = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preparation = ColumnTransformer([
        ("numbers", numbers, NUMERIC + DERIVED),
        ("categories", categories, CATEGORICAL),
    ])
    return Pipeline([
        ("features", FunctionTransformer(add_features, validate=False)),
        ("preparation", preparation),
        ("model", estimator),
    ])


def make_input_metadata(data):
    """Describe only the data used to fit the final model, not the test data."""
    metadata = {"ranges": {}, "defaults": {}, "categories": {}}
    for column in NUMERIC:
        low, high = float(data[column].min()), float(data[column].max())
        # Round inward so the two-decimal limits shown in the app are actually valid.
        if column in ["watch_time_minutes", "video_length_minutes"]:
            low, high = float(np.ceil(low * 100) / 100), float(np.floor(high * 100) / 100)
        metadata["ranges"][column] = [low, high]
        metadata["defaults"][column] = float(data[column].median())
    for column in CATEGORICAL:
        metadata["categories"][column] = sorted(data[column].dropna().unique().tolist())
    return metadata


def predict_revenue(pipeline, values, metadata):
    """Validate one input. Refuse unsupported ranges instead of hiding extrapolation."""
    row = {}
    notes = []
    for column in NUMERIC:
        value = values.get(column)
        if value is None or pd.isna(value):
            if column not in OPTIONAL:
                raise ValueError(f"Please provide {column.replace('_', ' ')}.")
            row[column] = np.nan
            notes.append(f"{column.replace('_', ' ')} is unknown; saved medians are used.")
            continue
        try:
            value = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{column} must be a number.") from error
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{column} must be finite and non-negative.")
        if column in ["views", "likes", "comments", "subscribers"] and not value.is_integer():
            raise ValueError(f"{column} must be a whole number.")
        low, high = metadata["ranges"][column]
        if not low <= value <= high:
            raise ValueError(f"{column.replace('_', ' ')} is outside the fitted range "
                             f"({low:,.2f} to {high:,.2f}). Use a value in this range.")
        row[column] = value
    for column in CATEGORICAL:
        value = values.get(column)
        if value not in metadata["categories"][column]:
            raise ValueError(f"Choose a supported {column}.")
        row[column] = value
    if row["watch_time_minutes"] / row["views"] > row["video_length_minutes"]:
        notes.append("Average watch time exceeds video length. Check the reporting "
                     "definitions; similar observations occur in the synthetic data.")
    prediction = float(pipeline.predict(pd.DataFrame([row]))[0])
    if not np.isfinite(prediction) or prediction < 0:
        raise ValueError("The model could not return a valid non-negative estimate.")
    return prediction, notes
