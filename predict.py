"""
  python predict.py
  python predict.py --csv path/to/rows.csv
  python predict.py --row "5000,0,120,360,1,Male,Yes,0,Graduate,No,Urban"
"""
import argparse
import sys
from pathlib import Path

import inference


def predict_from_row_text(preprocessor, model, feature_cols, row_str: str):
    row_df = inference.build_row_df(row_str.split(","), feature_cols)
    return inference.predict_one(preprocessor, model, feature_cols, row_df)


def run_interactive(preprocessor, model, feature_cols):
    cols = inference.feature_order(feature_cols)
    print("Введите данные заявителя через запятую.")
    print("Порядок признаков:", ", ".join(cols))
    print("Пример: 5000,0,120,360,1,Male,Yes,0,Graduate,No,Urban")
    line = input("> ").strip()
    if not line:
        return
    try:
        label, proba = predict_from_row_text(preprocessor, model, feature_cols, line)
    except ValueError as exc:
        print(exc)
        return

    status = "Одобрено" if label == "Y" else "Не одобрено"
    print("Прогноз:", label, f"({status})")
    if proba is not None:
        print("Вероятность одобрения:", f"{proba:.2%}")


def run_csv(preprocessor, model, feature_cols, csv_path: str):
    df = pd.read_csv(csv_path)
    result = inference.predict_dataframe(preprocessor, model, feature_cols, df)
    out_path = Path(csv_path).parent / (Path(csv_path).stem + "_predictions.csv")
    result.to_csv(out_path, index=False)
    print("Прогнозы сохранены в", out_path)
    print(result.head(10).to_string())


def run_single_row(preprocessor, model, feature_cols, row_str: str):
    try:
        label, proba = predict_from_row_text(preprocessor, model, feature_cols, row_str)
    except ValueError as exc:
        print(exc)
        return

    print("Прогноз:", label)
    if proba is not None:
        print("Вероятность одобрения:", f"{proba:.2%}")


def main():
    if not inference.artifacts_ready():
        print("Модели не найдены. Запустите: python pipeline.py")
        sys.exit(1)
    preprocessor, model, feature_cols = inference.load_artifacts()

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
