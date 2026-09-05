import os
import pickle
import pandas as pd
import xgboost as xgb


# ============================================================
# PATHS
# ============================================================
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL2_PATH = os.path.join(BASE_PATH, "models", "Model2")

# Use the stable JSON model instead of the incompatible pickle
MODEL2_FILE = os.path.join(MODEL2_PATH, "model2_xgb.json")
FEATURES_FILE = os.path.join(MODEL2_PATH, "model2_features.pkl")
CATEGORICAL_FILE = os.path.join(MODEL2_PATH, "model2_categorical_cols.pkl")


# ============================================================
# LOAD MODEL 2
# ============================================================
model2 = xgb.XGBClassifier(enable_categorical=True)
model2.load_model(MODEL2_FILE)

print("Model 2 loaded successfully")
print("Model 2 features:", model2.get_booster().num_features())


# ============================================================
# LOAD FEATURE METADATA
# ============================================================
with open(FEATURES_FILE, "rb") as f:
    MODEL2_FEATURES = list(pickle.load(f))

with open(CATEGORICAL_FILE, "rb") as f:
    MODEL2_CATEGORICAL_COLS = list(pickle.load(f))


# ============================================================
# VALIDATE ARTIFACT CONTRACT
# ============================================================
if len(MODEL2_FEATURES) != 432:
    raise RuntimeError(
        f"Expected 432 Model 2 features, got {len(MODEL2_FEATURES)}"
    )

if model2.get_booster().num_features() != 432:
    raise RuntimeError(
        f"Expected Model 2 to contain 432 features, "
        f"got {model2.get_booster().num_features()}"
    )

if not hasattr(model2, "predict_proba"):
    raise RuntimeError(
        "Model 2 artifact does not provide predict_proba()"
    )


# ============================================================
# LIVE FEATURE MAPPING
# ============================================================
def build_model2_features(
    transaction_dt: int,
    transaction_amt: float,
    email: str | None = None,
    contact: str | None = None,
    method: str | None = None,
) -> pd.DataFrame:

    """Build the 432-column Model 2 serving schema.

    No old encoder is used.

    Only fields with defensible live mappings are populated.
    Unavailable training features remain NaN.
    """

    if transaction_dt is None:
        raise ValueError("transaction_dt is required")

    if transaction_amt is None:
        raise ValueError("transaction_amt is required")

    if transaction_amt < 0:
        raise ValueError(
            "transaction_amt must be non-negative"
        )

    # Start with all 432 features as missing
    data = {
        feature: float("nan")
        for feature in MODEL2_FEATURES
    }

    # Direct mappings
    data["TransactionDT"] = int(transaction_dt)
    data["TransactionAmt"] = float(transaction_amt)

    # Email domain is a defensible direct mapping
    if email and "@" in email:
        domain = (
            email
            .strip()
            .lower()
            .split("@", 1)[1]
            .strip()
        )

        if domain and "P_emaildomain" in data:
            data["P_emaildomain"] = domain

    features = pd.DataFrame(
        [[data[column] for column in MODEL2_FEATURES]],
        columns=MODEL2_FEATURES,
    )

    # Convert categorical metadata columns to numeric/missing
    for column in MODEL2_CATEGORICAL_COLS:
        features[column] = pd.to_numeric(
            features[column],
            errors="coerce",
        )

    # Convert all remaining columns to numeric
    numerical_columns = [
        column
        for column in MODEL2_FEATURES
        if column not in MODEL2_CATEGORICAL_COLS
    ]

    features[numerical_columns] = features[
        numerical_columns
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Validate feature order
    if list(features.columns) != MODEL2_FEATURES:
        raise RuntimeError(
            "Model 2 feature order mismatch"
        )

    # Validate shape
    if features.shape != (1, 432):
        raise RuntimeError(
            f"Expected Model 2 shape (1, 432), "
            f"got {features.shape}"
        )

    return features


# ============================================================
# MODEL 2 PREDICTION
# ============================================================
def predict_model2(
    transaction_dt: int | dict,
    transaction_amt: float | None = None,
    email: str | None = None,
    contact: str | None = None,
    method: str | None = None,
) -> float:

    """Predict Model 2 fraud risk.

    Supports both:
      1. keyword arguments
      2. transaction dictionary
    """

    # Backward compatibility
    if isinstance(transaction_dt, dict):

        transaction = transaction_dt

        transaction_dt = transaction.get(
            "transaction_dt"
        )

        transaction_amt = transaction.get(
            "transaction_amt"
        )

        email = transaction.get("email")
        contact = transaction.get("contact")
        method = transaction.get("method")

    features = build_model2_features(
        transaction_dt=int(transaction_dt),
        transaction_amt=float(transaction_amt),
        email=email,
        contact=contact,
        method=method,
    )

    probability = float(
        model2.predict_proba(features)[0, 1]
    )

    if not 0.0 <= probability <= 1.0:
        raise RuntimeError(
            f"Invalid Model 2 probability: {probability}"
        )

    return probability


# ============================================================
# STARTUP INFORMATION
# ============================================================
print("===================================")
print("RiskGuard AI Model 2 Adapter")
print("===================================")
print(
    "Model 2 loaded:",
    os.path.basename(MODEL2_FILE)
)
print(
    "Features:",
    len(MODEL2_FEATURES)
)
print(
    "Categorical metadata:",
    len(MODEL2_CATEGORICAL_COLS)
)
print("Encoder: NOT USED")
print("===================================")