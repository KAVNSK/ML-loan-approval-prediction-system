"""
ML-based Loan Approval Prediction: full pipeline.
EDA, preprocessing, modeling (5 models + GridSearch top 2), visualizations, save artifacts.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

import config

# --- 1. Load data ---
def load_data():
    df = pd.read_csv(config.TRAIN_CSV)
    df = df.drop(columns=[config.ID_COL], errors="ignore")
    return df

# --- 2. EDA ---
def run_eda(df: pd.DataFrame):
    print("=" * 60)
    print("Разведка данных: распределения, пропуски, выбросы")
    print("=" * 60)
    print(df.info())
    print("\n--- Пропуски ---")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print(missing)
    print("\n--- Распределение целевой переменной ---")
    print(df[config.TARGET].value_counts())
    print("\n--- Числовые признаки (описание) ---")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if config.TARGET in numeric_cols:
        numeric_cols.remove(config.TARGET)
    print(df[numeric_cols].describe())
    for col in numeric_cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        low, high = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        n_out = ((df[col] < low) | (df[col] > high)).sum()
        if n_out > 0:
            print(f"  {col}: {n_out} выбросов (IQR)")
    return df

# --- 3. EDA visualizations ---
def save_eda_plots(df: pd.DataFrame):
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != config.TARGET]
    categorical_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != config.TARGET]

    # Histograms (numeric)
    n_num = len(numeric_cols)
    if n_num:
        fig, axes = plt.subplots((n_num + 2) // 3, min(3, n_num), figsize=(12, 4 * ((n_num + 2) // 3)))
        if n_num == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        for i, col in enumerate(numeric_cols):
            axes[i].hist(df[col].dropna(), bins=30, edgecolor="black", alpha=0.7)
            axes[i].set_title(col)
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout()
        plt.savefig(config.OUTPUT_DIR / "eda_histograms.png", dpi=120, bbox_inches="tight")
        plt.close()

    # Boxplots (numeric)
    if n_num:
        fig, axes = plt.subplots((n_num + 2) // 3, min(3, n_num), figsize=(12, 4 * ((n_num + 2) // 3)))
        if n_num == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        for i, col in enumerate(numeric_cols):
            axes[i].boxplot(df[col].dropna())
            axes[i].set_title(col)
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout()
        plt.savefig(config.OUTPUT_DIR / "eda_boxplots.png", dpi=120, bbox_inches="tight")
        plt.close()

    # Correlation heatmap (encode target for correlation)
    df_corr = df.copy()
    df_corr[config.TARGET] = (df_corr[config.TARGET] == "Y").astype(int)
    corr_cols = [c for c in df_corr.columns if df_corr[c].dtype in [np.int64, np.float64]]
    if len(corr_cols) > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df_corr[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
        plt.title("Корреляционная тепловая карта")
        plt.tight_layout()
        plt.savefig(config.OUTPUT_DIR / "eda_correlation_heatmap.png", dpi=120, bbox_inches="tight")
        plt.close()
    print("Графики EDA сохранены в output/")

# --- 4. Preprocessing pipeline ---
def get_feature_columns(df: pd.DataFrame):
    exclude = [config.TARGET]
    numeric = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    categorical = [c for c in df.select_dtypes(include=["object"]).columns if c not in exclude]
    return numeric, categorical

def build_preprocessor(numeric_cols, categorical_cols):
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])
    return preprocessor

def preprocess_data(df: pd.DataFrame, fit_preprocessor=True, preprocessor=None):
    numeric_cols, categorical_cols = get_feature_columns(df)
    if preprocessor is None:
        preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    X = df.drop(columns=[config.TARGET])
    y = (df[config.TARGET] == "Y").astype(int)
    if fit_preprocessor:
        X_processed = preprocessor.fit_transform(X)
    else:
        X_processed = preprocessor.transform(X)
    return X_processed, y, preprocessor, numeric_cols, categorical_cols

# --- 5. Modeling ---
def get_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_STATE),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=config.RANDOM_STATE),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=config.RANDOM_STATE),
        "LightGBM": lgb.LGBMClassifier(verbose=-1, random_state=config.RANDOM_STATE),
    }

def evaluate_models(X, y, use_smote=False):
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    models = get_models()
    results = {}
    confusion_matrices = {}
    fitted_models = {}
    roc_curves_data = {}
    # Use RandomOverSampler instead of SMOTE to avoid threadpoolctl bug (SMOTE KNN triggers get_config().split() on None on Windows/Anaconda)

    for name, model in models.items():
        if use_smote:
            pipe = ImbPipeline([("resampler", RandomOverSampler(random_state=config.RANDOM_STATE)), ("clf", model)])
        else:
            pipe = model
        scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring, return_estimator=True)
        results[name] = {
            "Accuracy": scores["test_accuracy"].mean(),
            "Precision": scores["test_precision"].mean(),
            "Recall": scores["test_recall"].mean(),
            "F1": scores["test_f1"].mean(),
            "ROC-AUC": scores["test_roc_auc"].mean(),
        }
        # Fit on full data for confusion matrix and ROC
        pipe_full = ImbPipeline([("resampler", RandomOverSampler(random_state=config.RANDOM_STATE)), ("clf", model)]) if use_smote else model
        pipe_full.fit(X, y)
        y_pred = pipe_full.predict(X)
        y_proba = pipe_full.predict_proba(X)[:, 1] if hasattr(pipe_full, "predict_proba") else pipe_full.predict(X)
        confusion_matrices[name] = confusion_matrix(y, y_pred)
        fitted_models[name] = pipe_full
        roc_curves_data[name] = roc_curve(y, y_proba)
    return results, confusion_matrices, fitted_models, roc_curves_data

# --- 6. Visualizations: bar chart, ROC curves ---
def save_metrics_barchart(results: dict):
    metrics_df = pd.DataFrame(results).T
    metrics_df.plot(kind="bar", figsize=(10, 6), width=0.8)
    plt.title("Сравнение моделей: Accuracy, Precision, Recall, F1, ROC-AUC")
    plt.ylabel("Оценка")
    plt.xticks(rotation=15)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / "model_comparison_barchart.png", dpi=120, bbox_inches="tight")
    plt.close()

def save_roc_curves(roc_curves_data: dict):
    plt.figure(figsize=(8, 6))
    for name, (fpr, tpr, _) in roc_curves_data.items():
        plt.plot(fpr, tpr, label=name)
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("Доля ложноположительных")
    plt.ylabel("Доля истинно положительных")
    plt.title("ROC-кривые (все модели)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / "roc_curves_all.png", dpi=120, bbox_inches="tight")
    plt.close()

def save_confusion_matrices(confusion_matrices: dict):
    n = len(confusion_matrices)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()
    for i, (name, cm) in enumerate(confusion_matrices.items()):
        sns.heatmap(cm, annot=True, fmt="d", ax=axes[i], cmap="Blues")
        axes[i].set_title(name)
        axes[i].set_xlabel("Предсказано")
        axes[i].set_ylabel("Фактически")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / "confusion_matrices.png", dpi=120, bbox_inches="tight")
    plt.close()

def save_feature_importance(best_model, feature_names: list, filepath: Path):
    if hasattr(best_model, "feature_importances_"):
        imp = best_model.feature_importances_
    elif hasattr(best_model, "coef_"):
        imp = np.abs(best_model.coef_).flatten()
    else:
        return
    idx = np.argsort(imp)[::-1][:min(20, len(imp))]
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(idx)), imp[idx])
    plt.yticks(range(len(idx)), [feature_names[i] for i in idx])
    plt.gca().invert_yaxis()
    plt.title("Важность признаков (топ)")
    plt.tight_layout()
    plt.savefig(filepath, dpi=120, bbox_inches="tight")
    plt.close()

# --- 7. GridSearch for top 2 ---
def gridsearch_top2(X, y, results, use_smote=False):
    sorted_models = sorted(results.keys(), key=lambda m: results[m]["ROC-AUC"], reverse=True)
    top2_names = sorted_models[:2]
    param_grids = {
        "LogisticRegression": {"clf__C": [0.1, 1, 10], "clf__solver": ["lbfgs", "liblinear"]},
        "RandomForest": {"clf__n_estimators": [100, 200], "clf__max_depth": [10, 20, None]},
        "GradientBoosting": {"clf__n_estimators": [100, 200], "clf__max_depth": [3, 5]},
        "XGBoost": {"clf__n_estimators": [100, 200], "clf__max_depth": [3, 6]},
        "LightGBM": {"clf__n_estimators": [100, 200], "clf__max_depth": [3, 6]},
    }
    best_estimator = None
    best_score = -1
    for name in top2_names:
        model = get_models()[name]
        params = param_grids.get(name, {}).copy()
        if use_smote:
            pipe = ImbPipeline([
                ("resampler", RandomOverSampler(random_state=config.RANDOM_STATE)),
                ("clf", model)
            ])
        else:
            pipe = model
            params = {k.replace("clf__", ""): v for k, v in params.items()}
        if not params:
            continue
        gs = GridSearchCV(pipe, params, cv=config.CV_FOLDS, scoring="roc_auc", n_jobs=-1)
        gs.fit(X, y)
        if gs.best_score_ > best_score:
            best_score = gs.best_score_
            best_estimator = gs.best_estimator_
        print(f"  {name} лучшие параметры: {gs.best_params_}, ROC-AUC: {gs.best_score_:.4f}")
    return best_estimator

# --- Main ---
def main():
    plt.rcParams["font.family"] = "DejaVu Sans"
    df = load_data()
    run_eda(df)
    save_eda_plots(df)

    X, y, preprocessor, numeric_cols, categorical_cols = preprocess_data(df)
    feature_names = (
        numeric_cols
        + preprocessor.named_transformers_["cat"].named_steps["encoder"].get_feature_names_out(categorical_cols).tolist()
    )

    neg, pos = np.bincount(y)
    use_smote = min(neg, pos) < len(y) * 0.4
    if use_smote:
        print("\nПрименяем передискретизацию для дисбаланса классов.")
    else:
        print("\nБаланс классов в норме; передискретизация опциональна.")

    print("\n--- Результаты кросс-валидации (k=5) ---")
    results, confusion_matrices, fitted_models, roc_curves_data = evaluate_models(X, y, use_smote=use_smote)
    for name, res in results.items():
        print(f"  {name}: Acc={res['Accuracy']:.4f}, Prec={res['Precision']:.4f}, Rec={res['Recall']:.4f}, F1={res['F1']:.4f}, AUC={res['ROC-AUC']:.4f}")

    save_metrics_barchart(results)
    save_roc_curves(roc_curves_data)
    save_confusion_matrices(confusion_matrices)

    print("\n--- GridSearchCV для двух лучших моделей ---")
    best_estimator = gridsearch_top2(X, y, results, use_smote=use_smote)

    # Finalize best: either GridSearch best or highest ROC-AUC from CV
    if best_estimator is not None:
        final_model = best_estimator
    else:
        best_name = max(results.keys(), key=lambda m: results[m]["ROC-AUC"])
        final_model = fitted_models[best_name]
    # If final_model is pipeline with smote+clf, get the classifier for feature importance
    if hasattr(final_model, "named_steps") and "clf" in final_model.named_steps:
        clf_for_importance = final_model.named_steps["clf"]
    else:
        clf_for_importance = final_model
    save_feature_importance(clf_for_importance, feature_names, config.OUTPUT_DIR / "feature_importance.png")

    # Save classifier only for inference (no SMOTE at predict time)
    if hasattr(final_model, "named_steps") and "clf" in final_model.named_steps:
        classifier_for_save = final_model.named_steps["clf"]
    else:
        classifier_for_save = final_model
    joblib.dump(preprocessor, config.MODELS_DIR / "preprocessor.joblib")
    joblib.dump(classifier_for_save, config.MODELS_DIR / "best_model.joblib")
    joblib.dump({"numeric": numeric_cols, "categorical": categorical_cols}, config.MODELS_DIR / "feature_columns.joblib")
    print("\nПрепроцессор и лучшая модель сохранены в models/")
    print("Готово.")

if __name__ == "__main__":
    main()
