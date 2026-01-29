# -*- coding: utf-8 -*-
"""Конфигурация: пути и константы."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "archive"
TRAIN_CSV = DATA_DIR / "loan_sanction_train.csv"
TEST_CSV = DATA_DIR / "loan_sanction_test.csv"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = PROJECT_ROOT / "models"

TARGET = "Loan_Status"
ID_COL = "Loan_ID"
RANDOM_STATE = 42
CV_FOLDS = 5

OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
