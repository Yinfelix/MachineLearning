"""Project-wide configuration: paths, constants, feature groups, schemes."""
from __future__ import annotations

from pathlib import Path


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "PhiUSIIL_Phishing_URL_Dataset.csv"
SPLITS_DIR = ROOT / "splits"
CONFIG_DIR = ROOT / "configs"
FIG_DIR = ROOT / "outputs" / "figures"
METRIC_DIR = ROOT / "outputs" / "metrics"
MODEL_DIR = ROOT / "outputs" / "models"

for _d in (SPLITS_DIR, CONFIG_DIR, FIG_DIR, METRIC_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_INNER_FOLDS = 5
TEST_FRACTION = 0.2
SVC_SUBSAMPLE = 20_000

# Positive class = phishing.  Raw label 1 = legitimate, 0 = phishing.
POSITIVE_LABEL = "phishing"

# --------------------------------------------------------------------------- #
# Feature groups (54 raw features; URL/Domain/Title/FILENAME are text)
# --------------------------------------------------------------------------- #
URL_NUMERIC = [
    "URLLength",
    "DomainLength",
    "IsDomainIP",
    "URLSimilarityIndex",
    "CharContinuationRate",
    "TLDLegitimateProb",
    "URLCharProb",
    "TLDLength",
    "NoOfSubDomain",
    "HasObfuscation",
    "NoOfObfuscatedChar",
    "ObfuscationRatio",
    "NoOfLettersInURL",
    "LetterRatioInURL",
    "NoOfDegitsInURL",
    "DegitRatioInURL",
    "NoOfEqualsInURL",
    "NoOfQMarkInURL",
    "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL",
    "SpacialCharRatioInURL",
    "IsHTTPS",
]

PAGE_NUMERIC = [
    "LineOfCode",
    "LargestLineLength",
    "HasTitle",
    "DomainTitleMatchScore",
    "URLTitleMatchScore",
    "HasFavicon",
    "Robots",
    "IsResponsive",
    "NoOfURLRedirect",
    "NoOfSelfRedirect",
    "HasDescription",
    "NoOfPopup",
    "NoOfiFrame",
    "HasExternalFormSubmit",
    "HasSocialNet",
    "HasSubmitButton",
    "HasHiddenFields",
    "HasPasswordField",
    "Bank",
    "Pay",
    "Crypto",
    "HasCopyrightInfo",
    "NoOfImage",
    "NoOfCSS",
    "NoOfJS",
    "NoOfSelfRef",
    "NoOfEmptyRef",
    "NoOfExternalRef",
]

CATEGORICAL = ["TLD"]

# Features described in the source paper as engineered / template-derived.
# URLSimilarityIndex is additionally a known label shortcut: every legitimate
# sample in this release has the value 100.
SHORTCUT_FEATURES = [
    "URLSimilarityIndex",
    "CharContinuationRate",
    "TLDLegitimateProb",
    "URLCharProb",
    "DomainTitleMatchScore",
    "URLTitleMatchScore",
]

# Page-scale features are a second, undeclared shortcut: their univariate AUC
# exceeds 0.95 on the training split (see outputs/metrics/univariate_auc.csv).
PAGE_SIZE_SHORTCUTS = [
    "LineOfCode",
    "NoOfExternalRef",
    "NoOfImage",
    "NoOfSelfRef",
    "NoOfJS",
    "NoOfCSS",
]

SCHEMES = {
    "A": {
        "label": "A 全特征",
        "numeric": URL_NUMERIC + PAGE_NUMERIC,
        "categorical": CATEGORICAL,
    },
    "B": {
        "label": "B 去官方捷径",
        "numeric": [c for c in URL_NUMERIC + PAGE_NUMERIC if c not in SHORTCUT_FEATURES],
        "categorical": CATEGORICAL,
    },
    "C": {
        "label": "C 仅 URL",
        "numeric": URL_NUMERIC,
        "categorical": CATEGORICAL,
    },
    "D": {
        "label": "D 仅页面",
        "numeric": PAGE_NUMERIC,
        "categorical": [],
    },
    # Supplementary ablations added after the EDA revealed page-scale shortcuts.
    "S": {
        "label": "S 严格去捷径",
        "numeric": [
            c
            for c in URL_NUMERIC + PAGE_NUMERIC
            if c not in SHORTCUT_FEATURES and c not in PAGE_SIZE_SHORTCUTS
        ],
        "categorical": CATEGORICAL,
    },
    "U": {
        "label": "U 仅 URL 基础词法",
        "numeric": [c for c in URL_NUMERIC if c not in SHORTCUT_FEATURES],
        "categorical": CATEGORICAL,
    },
    # Strictest variants: IsHTTPS is also a one-sided label indicator here
    # (all legitimate samples use HTTPS, only ~49% of phishing do).
    "T": {
        "label": "T 去除全部单向捷径",
        "numeric": [
            c
            for c in URL_NUMERIC + PAGE_NUMERIC
            if c not in SHORTCUT_FEATURES and c not in PAGE_SIZE_SHORTCUTS and c != "IsHTTPS"
        ],
        "categorical": CATEGORICAL,
    },
    "V": {
        "label": "V 仅 URL 基础词法（去 HTTPS）",
        "numeric": [c for c in URL_NUMERIC if c not in SHORTCUT_FEATURES and c != "IsHTTPS"],
        "categorical": CATEGORICAL,
    },
}

# The evaluated pipelines use counts on their original scale.
LOG_FEATURES: list[str] = []

MODEL_ORDER = ["Logistic Regression", "Gaussian NB", "Decision Tree", "Random Forest", "SVM (RBF)"]
SCHEME_ORDER = ["A", "B", "C", "D"]
SUPPLEMENTARY_SCHEME_ORDER = ["S", "U", "T", "V"]

# Palette validated by the project's data-viz guideline (light surface).
PALETTE = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "magenta": "#e87ba4",
    "green": "#008300",
    "violet": "#6250d6",
    "red": "#e34948",
}
MODEL_COLORS = dict(
    zip(
        MODEL_ORDER,
        [PALETTE["blue"], PALETTE["orange"], PALETTE["aqua"], PALETTE["violet"], PALETTE["green"]],
    )
)
SCHEME_COLORS = dict(zip(SCHEME_ORDER, [PALETTE["blue"], PALETTE["orange"], PALETTE["aqua"], PALETTE["violet"]]))

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def scheme_columns(scheme: str) -> tuple[list[str], list[str]]:
    cfg = SCHEMES[scheme]
    return list(cfg["numeric"]), list(cfg["categorical"])


def all_feature_columns() -> list[str]:
    return URL_NUMERIC + PAGE_NUMERIC + CATEGORICAL
