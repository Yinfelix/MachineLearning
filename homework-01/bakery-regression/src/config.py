"""读取项目 JSON 配置，并以项目目录解析相对路径。"""

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETTINGS = json.loads((PROJECT_ROOT / "config.json").read_text(encoding="utf-8"))

DEFAULT_DATA_PATH = (PROJECT_ROOT / SETTINGS["paths"]["data"]).resolve()
AUDIT_OUTPUT_PATH = PROJECT_ROOT / SETTINGS["paths"]["audit"]
FIGURE_DIRECTORY = PROJECT_ROOT / SETTINGS["paths"]["figures"]
TARGET_COLUMN = SETTINGS["target"]
DATE_FORMAT = SETTINGS["formats"]["date"]
TIME_FORMAT = SETTINGS["formats"]["time"]
NUMERIC_FEATURES = SETTINGS["features"]["numeric"]
CATEGORICAL_FEATURES = SETTINGS["features"]["categorical"]
BINARY_FEATURES = SETTINGS["features"]["binary"]
SPECIAL_CATEGORICAL_FEATURES = SETTINGS["features"]["special_categorical"]
FEATURE_COLUMNS = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
    + BINARY_FEATURES
    + SPECIAL_CATEGORICAL_FEATURES
)
BINARY_VALUES = SETTINGS["binary_values"]
MISSING_VALUES = SETTINGS["missing_values"]
EQUALITY_TOLERANCE = SETTINGS["audit"]["equality_tolerance"]
EXCLUDED_FIELDS = SETTINGS["excluded_fields"]
SPLIT_PERIODS = SETTINGS["split_periods"]
