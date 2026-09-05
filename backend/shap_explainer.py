import os
import pickle

import shap
import xgboost as xgb


# ============================================================
# PATHS
# ============================================================

BASE_PATH = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL3_DIR = os.path.join(
    BASE_PATH,
    "models",
    "Fusion_v1.0"
)


MODEL3_FILE = os.path.join(
    MODEL3_DIR,
    "model3_v2.json"
)

MODEL3_FEATURE_FILE = os.path.join(
    MODEL3_DIR,
    "model3_v2_features.pkl"
)


# ============================================================
# LOAD MODEL
# ============================================================

model3 = xgb.XGBClassifier()

model3.load_model(
    MODEL3_FILE
)


# ============================================================
# LOAD FEATURE ORDER
# ============================================================

with open(
    MODEL3_FEATURE_FILE,
    "rb"
) as f:

    MODEL3_FEATURES = pickle.load(f)


# ============================================================
# SHAP EXPLAINER
# ============================================================

explainer = shap.TreeExplainer(
    model3
)


# ============================================================
# EXPLAIN MODEL 3
# ============================================================

def explain_model3(features):

    if features is None:
        raise ValueError(
            "Features cannot be None"
        )

    if features.shape != (1, len(MODEL3_FEATURES)):
        raise ValueError(
            f"Invalid Model 3 feature shape: "
            f"{features.shape}"
        )

    if list(features.columns) != list(
        MODEL3_FEATURES
    ):
        raise ValueError(
            "SHAP feature order mismatch"
        )

    # --------------------------------------------------------
    # Calculate SHAP values
    # --------------------------------------------------------

    shap_values = explainer.shap_values(
        features
    )

    # XGBoost binary classifier normally
    # returns shape (1, number_of_features)

    if hasattr(shap_values, "values"):
        shap_values = shap_values.values

    if len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]

    shap_row = shap_values[0]

    # --------------------------------------------------------
    # Create explanation
    # --------------------------------------------------------

    explanations = []

    for feature_name, feature_value, shap_value in zip(
        MODEL3_FEATURES,
        features.iloc[0].values,
        shap_row
    ):

        contribution = float(shap_value)

        explanations.append({

            "feature":
                feature_name,

            "value":
                float(feature_value),

            "shap_value":
                round(
                    contribution,
                    6
                ),

            "impact":
                (
                    "increases_risk"
                    if contribution > 0
                    else "decreases_risk"
                ),

        })

    # --------------------------------------------------------
    # Sort by absolute SHAP impact
    # --------------------------------------------------------

    explanations.sort(
        key=lambda item:
            abs(item["shap_value"]),
        reverse=True
    )

    return explanations


# ============================================================
# TOP SHAP REASONS
# ============================================================

def get_top_shap_reasons(
    features,
    top_n=5
):

    explanations = explain_model3(
        features
    )

    return explanations[:top_n]


print(
    "RiskGuard AI SHAP Explainer"
)

print(
    "Model 3 SHAP: READY"
)

print(
    "Features:",
    len(MODEL3_FEATURES)
)