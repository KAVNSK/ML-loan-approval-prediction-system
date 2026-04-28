"""Общие функции инференса для CLI и GUI."""

from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd

import config

ID_COL = "Loan_ID"
TARGET = "Loan_Status"


def artifact_paths() -> dict:
    return {
        "preprocessor": config.MODELS_DIR / "preprocessor.joblib",
        "model": config.MODELS_DIR / "best_model.joblib",
        "feature_columns": config.MODELS_DIR / "feature_columns.joblib",
    }


def missing_artifacts() -> list[Path]:
    return [path for path in artifact_paths().values() if not path.exists()]


def artifacts_ready() -> bool:
    return not missing_artifacts()


def load_artifacts():
    paths = artifact_paths()
    preprocessor = joblib.load(paths["preprocessor"])
    model = joblib.load(paths["model"])
    feature_cols = joblib.load(paths["feature_columns"])
    return preprocessor, model, feature_cols


def feature_order(feature_cols: dict) -> list[str]:
    return feature_cols["numeric"] + feature_cols["categorical"]


def build_row_df(parts: Iterable[str], feature_cols: dict) -> pd.DataFrame:
    values = [p.strip() for p in parts]
    numeric = feature_cols["numeric"]
    cols = feature_order(feature_cols)
    numeric_set = set(numeric)

    if len(values) != len(cols):
        raise ValueError(f"Неверное число значений: ожидается {len(cols)}, получено {len(values)}")

    row_dict = {}
    for i, col in enumerate(cols):
        value = values[i]
        if col in numeric_set:
            if value == "":
                row_dict[col] = [None]
            else:
                try:
                    row_dict[col] = [float(value)]
                except ValueError as exc:
                    raise ValueError(f"Поле '{col}' должно быть числом, получено: '{value}'") from exc
        else:
            row_dict[col] = [value if value else None]
    return pd.DataFrame(row_dict)


def normalize_features(df: pd.DataFrame, feature_cols: dict) -> pd.DataFrame:
    expected = feature_order(feature_cols)
    drop_cols = [c for c in (ID_COL, TARGET) if c in df.columns]
    source = df.drop(columns=drop_cols) if drop_cols else df
    missing = [c for c in expected if c not in source.columns]
    if missing:
        raise ValueError(f"Отсутствуют обязательные колонки: {', '.join(missing)}")
    return source[expected]


def predict_dataframe(preprocessor, model, feature_cols: dict, df: pd.DataFrame) -> pd.DataFrame:
    features = normalize_features(df, feature_cols)
    transformed = preprocessor.transform(features)
    preds = model.predict(transformed)
    labels = ["Y" if p == 1 else "N" for p in preds]

    result = pd.DataFrame({"Loan_Status_Pred": labels}, index=features.index)
    if hasattr(model, "predict_proba"):
        result["Probability_Approved"] = model.predict_proba(transformed)[:, 1]
    return result


def predict_one(preprocessor, model, feature_cols: dict, row: pd.DataFrame):
    result = predict_dataframe(preprocessor, model, feature_cols, row).iloc[0]
    label = result["Loan_Status_Pred"]
    proba = result.get("Probability_Approved")
    return label, float(proba) if proba is not None else None
