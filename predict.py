"""
Console service: predict loan approval (Y/N) from applicant data.
Usage:
  python predict.py                          # interactive prompts
  python predict.py --csv path/to/rows.csv   # batch from CSV (same columns as train, no Loan_Status)
  python predict.py --row "Male,Yes,0,Graduate,No,5000,0,120,360,1,Urban"  # single row (no headers)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import joblib

import config

ID_COL = "Loan_ID"
TARGET = "Loan_Status"


def load_artifacts():
    preprocessor = joblib.load(config.MODELS_DIR / "preprocessor.joblib")
    model = joblib.load(config.MODELS_DIR / "best_model.joblib")
    feature_cols = joblib.load(config.MODELS_DIR / "feature_columns.joblib")
    return preprocessor, model, feature_cols


def predict_one(preprocessor, model, row: pd.DataFrame):
    """row: DataFrame with same feature columns as training (no Loan_ID, no Loan_Status)."""
    X = preprocessor.transform(row)
    proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
    pred = model.predict(X)[0]
    label = "Y" if pred == 1 else "N"
    return label, float(proba[0]) if proba is not None else None


def run_interactive(preprocessor, model, feature_cols):
    numeric = feature_cols["numeric"]
    categorical = feature_cols["categorical"]
    cols = numeric + categorical
    print("Введите данные заявителя (через запятую). Признаки:", cols)
    print("Пример (сначала числа, потом категории): 5000,0,120,360,1,Male,Yes,0,Graduate,No,Urban")
    print("Порядок: ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Gender, Married, Dependents, Education, Self_Employed, Property_Area")
    line = input("> ").strip()
    if not line:
        return
    parts = [p.strip() for p in line.split(",")]
    n = len(numeric) + len(categorical)
    if len(parts) != n:
        print(f"Неверное количество значений. Ожидается: {n}")
        return
    col_order = cols
    row_dict = {}
    for i, col in enumerate(col_order):
        if col in numeric:
            try:
                row_dict[col] = [float(parts[i]) if parts[i] else None]
            except ValueError:
                row_dict[col] = [None]
        else:
            row_dict[col] = [parts[i] if parts[i] else None]
    row_df = pd.DataFrame(row_dict)
    label, proba = predict_one(preprocessor, model, row_df)
    status = "Одобрено" if label == "Y" else "Не одобрено"
    print("Прогноз:", label, f"({status})")
    if proba is not None:
        print("Вероятность одобрения:", f"{proba:.2%}")


def run_csv(preprocessor, model, feature_cols, csv_path: str):
    df = pd.read_csv(csv_path)
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])
    if TARGET in df.columns:
        df = df.drop(columns=[TARGET])
    numeric = feature_cols["numeric"]
    categorical = feature_cols["categorical"]
    for c in numeric + categorical:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    df = df[numeric + categorical]
    X = preprocessor.transform(df)
    preds = model.predict(X)
    labels = ["Y" if p == 1 else "N" for p in preds]
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X)[:, 1]
        result = pd.DataFrame({"Loan_Status_Pred": labels, "Probability_Approved": probas})
    else:
        result = pd.DataFrame({"Loan_Status_Pred": labels})
    out_path = Path(csv_path).parent / (Path(csv_path).stem + "_predictions.csv")
    result.to_csv(out_path, index=False)
    print("Прогнозы сохранены в", out_path)
    print(result.head(10).to_string())


def run_single_row(preprocessor, model, feature_cols, row_str: str):
    numeric = feature_cols["numeric"]
    categorical = feature_cols["categorical"]
    parts = [p.strip() for p in row_str.split(",")]
    n = len(numeric) + len(categorical)
    if len(parts) != n:
        print(f"Неверное количество значений. Ожидается: {n}, введено: {len(parts)}")
        return
    col_order = numeric + categorical
    row_dict = {}
    for i, col in enumerate(col_order):
        if col in numeric:
            try:
                row_dict[col] = [float(parts[i]) if parts[i] else None]
            except ValueError:
                row_dict[col] = [None]
        else:
            row_dict[col] = [parts[i] if parts[i] else None]
    row_df = pd.DataFrame(row_dict)
    label, proba = predict_one(preprocessor, model, row_df)
    print("Прогноз:", label)
    if proba is not None:
        print("Вероятность одобрения:", f"{proba:.2%}")


def main():
    if not (config.MODELS_DIR / "preprocessor.joblib").exists():
        print("Модели не найдены. Запустите: python pipeline.py")
        sys.exit(1)
    preprocessor, model, feature_cols = load_artifacts()

    parser = argparse.ArgumentParser(description="Прогноз одобрения займа по данным заявителя")
    parser.add_argument("--csv", type=str, help="Путь к CSV с заявителями (без Loan_Status)")
    parser.add_argument("--row", type=str, help="Одна строка: значения через запятую (без заголовков)")
    args = parser.parse_args()

    if args.csv:
        run_csv(preprocessor, model, feature_cols, args.csv)
    elif args.row:
        run_single_row(preprocessor, model, feature_cols, args.row)
    else:
        run_interactive(preprocessor, model, feature_cols)


if __name__ == "__main__":
    main()
