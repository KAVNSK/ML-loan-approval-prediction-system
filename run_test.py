"""Quick test: after pipeline.py has been run, load model and predict one row."""
import sys
from pathlib import Path

import config

def main():
    if not (config.MODELS_DIR / "best_model.joblib").exists():
        print("Сначала запустите pipeline.py для обучения и сохранения модели.")
        sys.exit(1)
    import predict
    preprocessor, model, feature_cols = predict.load_artifacts()
    row_str = "5000,0,120,360,1,Male,Yes,0,Graduate,No,Urban"
    predict.run_single_row(preprocessor, model, feature_cols, row_str)
    print("Тест пройден.")

if __name__ == "__main__":
    main()
