from backend.rag_explainer import build_rag_explanation
import os
import json
import pickle
from typing import Optional

import xgboost as xgb

from backend.model2_adapter import predict_model2
from backend.risk_policy import get_active_policy


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


# ============================================================
# FILE PATHS
# ============================================================

MODEL3_FILE = os.path.join(
    MODEL3_DIR,
    "model3_v2.json"
)

MODEL3_FEATURE_FILE = os.path.join(
    MODEL3_DIR,
    "model3_v2_features.pkl"
)

FUSION_CONFIG_FILE = os.path.join(
    MODEL3_DIR,
    "fusion_config.json"
)


# ============================================================
# LOAD MODEL 3
# ============================================================

model3 = xgb.XGBClassifier()

model3.load_model(
    MODEL3_FILE
)


# ============================================================
# LOAD MODEL 3 FEATURES
# ============================================================

with open(
    MODEL3_FEATURE_FILE,
    "rb"
) as f:

    model3_features = pickle.load(f)


model3_features = list(
    model3_features
)


# ============================================================
# LOAD FUSION CONFIG
# ============================================================

with open(
    FUSION_CONFIG_FILE,
    "r",
    encoding="utf-8"
) as f:

    fusion_config = json.load(f)


# ============================================================
# PRODUCTION CONFIGURATION
# ============================================================

MODEL2_WEIGHT = float(
    fusion_config["fusion"]["model2_weight"]
)

MODEL3_WEIGHT = float(
    fusion_config["fusion"]["model3_weight"]
)

PRODUCTION_THRESHOLD = float(
    fusion_config["production"]["threshold"]
)


# ============================================================
# VALIDATE FUSION WEIGHTS
# ============================================================

if abs(
    (MODEL2_WEIGHT + MODEL3_WEIGHT) - 1.0
) > 1e-6:

    raise RuntimeError(
        "Fusion weights must sum to 1.0"
    )


# ============================================================
# VALIDATE MODEL 3 FEATURES
# ============================================================

if len(model3_features) != 18:

    raise RuntimeError(
        f"Model 3 must have 18 features, "
        f"found {len(model3_features)}"
    )


if model3.get_booster().num_features() != 18:

    raise RuntimeError(
        "Model 3 booster feature count mismatch"
    )


# ============================================================
# MODEL 3 PREDICTION
# ============================================================

def predict_model3(
    model3_input
) -> float:

    if model3_input is None:

        raise ValueError(
            "model3_input cannot be None"
        )

    # --------------------------------------------------------
    # Exactly one transaction
    # --------------------------------------------------------

    if model3_input.shape[0] != 1:

        raise ValueError(
            "Model 3 expects exactly one transaction"
        )

    # --------------------------------------------------------
    # Feature order
    # --------------------------------------------------------

    if list(model3_input.columns) != model3_features:

        raise ValueError(
            "Model 3 feature order mismatch"
        )

    # --------------------------------------------------------
    # Feature count
    # --------------------------------------------------------

    if model3_input.shape[1] != len(
        model3_features
    ):

        raise ValueError(
            f"Expected {len(model3_features)} "
            f"Model 3 features, got "
            f"{model3_input.shape[1]}"
        )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    if model3_input.isna().sum().sum() != 0:

        raise ValueError(
            "Model 3 input contains missing values"
        )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    probability = float(
        model3.predict_proba(
            model3_input
        )[0][1]
    )

    return probability


# ============================================================
# MODEL 3 EXPLANATION
# ============================================================

def explain_model3(
    model3_input
) -> list:

    """
    Generate feature-level explanations for Model 3.

    Uses XGBoost contribution values (SHAP values)
    directly from the trained booster.

    Positive contribution:
        pushes transaction toward fraud/risk.

    Negative contribution:
        pushes transaction toward legitimate/low risk.
    """

    if model3_input is None:

        raise ValueError(
            "model3_input cannot be None"
        )

    if model3_input.shape != (1, 18):

        raise ValueError(
            "Model 3 explanation expects "
            "one transaction with 18 features"
        )

    if list(model3_input.columns) != model3_features:

        raise ValueError(
            "Model 3 explanation feature order mismatch"
        )

    # --------------------------------------------------------
    # Booster
    # --------------------------------------------------------

    booster = model3.get_booster()

    # --------------------------------------------------------
    # DMatrix
    # --------------------------------------------------------

    dmatrix = xgb.DMatrix(
        model3_input,
        feature_names=model3_features
    )

    # --------------------------------------------------------
    # SHAP contribution values
    #
    # pred_contribs=True returns:
    #
    # feature contribution values
    # +
    # bias/base value
    # --------------------------------------------------------

    shap_values = booster.predict(
        dmatrix,
        pred_contribs=True
    )

    if shap_values.shape[0] != 1:

        raise RuntimeError(
            "Invalid SHAP output shape"
        )

    contributions = shap_values[0]

    # Last value is the bias/base contribution
    feature_contributions = contributions[:-1]

    explanations = []

    # --------------------------------------------------------
    # Build explanation objects
    # --------------------------------------------------------

    for index, feature_name in enumerate(
        model3_features
    ):

        value = model3_input.iloc[
            0,
            index
        ]

        contribution = float(
            feature_contributions[index]
        )

        if contribution > 0:

            direction = "INCREASES_RISK"

        elif contribution < 0:

            direction = "REDUCES_RISK"

        else:

            direction = "NEUTRAL"

        explanations.append({

            "feature":
                feature_name,

            "value":
                float(value),

            "shap_value":
                round(
                    contribution,
                    6
                ),

            "direction":
                direction,
        })

    # --------------------------------------------------------
    # Most important features first
    # --------------------------------------------------------

    explanations.sort(
        key=lambda item:
            abs(item["shap_value"]),
        reverse=True
    )

    return explanations


# ============================================================
# TOP RISK FACTORS
# ============================================================

def get_top_risk_factors(
    explanations: list,
    limit: int = 5
) -> list:

    """
    Return the strongest features pushing the
    transaction toward higher risk.
    """

    risk_factors = [

        item

        for item in explanations

        if item["shap_value"] > 0

    ]

    risk_factors.sort(
        key=lambda item:
            item["shap_value"],
        reverse=True
    )

    return risk_factors[:limit]


# ============================================================
# TOP SAFE FACTORS
# ============================================================

def get_top_safe_factors(
    explanations: list,
    limit: int = 5
) -> list:

    """
    Return the strongest features reducing risk.
    """

    safe_factors = [

        item

        for item in explanations

        if item["shap_value"] < 0

    ]

    safe_factors.sort(
        key=lambda item:
            abs(item["shap_value"]),
        reverse=True
    )

    return safe_factors[:limit]


# ============================================================
# HUMAN READABLE EXPLANATION
# ============================================================

def build_risk_explanation(
    explanations: list
) -> list:

    """
    Convert SHAP feature contributions into
    simple messages suitable for the dashboard.
    """

    messages = []

    for item in explanations[:5]:

        feature = item["feature"]

        value = item["value"]

        shap_value = item["shap_value"]

        if shap_value > 0:

            messages.append({

                "feature":
                    feature,

                "message":
                    (
                        f"{feature} increased "
                        f"the transaction risk."
                    ),

                "impact":
                    "HIGHER_RISK",

                "value":
                    value,

                "shap_value":
                    shap_value,
            })

        elif shap_value < 0:

            messages.append({

                "feature":
                    feature,

                "message":
                    (
                        f"{feature} reduced "
                        f"the transaction risk."
                    ),

                "impact":
                    "LOWER_RISK",

                "value":
                    value,

                "shap_value":
                    shap_value,
            })

    return messages


# ============================================================
# FUSION
# ============================================================

def fuse_predictions(
    model2_probability: float,
    model3_probability: float
) -> float:

    policy = get_active_policy()

    return policy.fuse(
        model2_probability,
        model3_probability
    )


# ============================================================
# RISK BAND
# ============================================================

def get_risk_band(
    risk_score: float
) -> str:

    risk_score = float(risk_score)

    if risk_score < 0.30:
        return "LOW"

    elif risk_score < 0.50:
        return "MEDIUM"

    elif risk_score < 0.70:
        return "HIGH"

    else:
        return "CRITICAL"


# ============================================================
# FINAL BUSINESS DECISION
# ============================================================

def get_decision(
    risk_score: float
) -> str:

    policy = get_active_policy()

    return policy.decide(
        risk_score
    )

# ============================================================
# COMPLETE RISK CALCULATION
# ============================================================

def calculate_risk(
    model2_probability: float,
    model3_probability: float,
    model3_input=None
) -> dict:

    # --------------------------------------------------------
    # Fusion
    # --------------------------------------------------------

    combined_risk = fuse_predictions(

        model2_probability,

        model3_probability
    )

    # --------------------------------------------------------
    # Risk band
    # --------------------------------------------------------

    risk_band = get_risk_band(
        combined_risk
    )

    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------

    decision = get_decision(
        combined_risk
    )

    # --------------------------------------------------------
    # Fraud prediction
    # --------------------------------------------------------

    policy = get_active_policy()
    fraud_prediction = int(
        combined_risk >= policy.block_threshold
    )

    # --------------------------------------------------------
    # SHAP explanation
    # --------------------------------------------------------

    shap_explanations = []

    top_risk_factors = []

    top_safe_factors = []

    risk_explanation = []

    rag_explanation = None

    if model3_input is not None:

        try:

            shap_explanations = explain_model3(
                model3_input
            )

            top_risk_factors = get_top_risk_factors(
                shap_explanations
            )

            top_safe_factors = get_top_safe_factors(
                shap_explanations
            )

            rag_explanation = build_rag_explanation(
                shap_explanations=shap_explanations,
                risk_score=combined_risk,
                risk_band=risk_band,
                decision=decision,
                threshold=policy.block_threshold,
            )
            risk_explanation = build_risk_explanation(
                shap_explanations
            )

        except Exception as exc:

            print(
                "SHAP explanation failed:",
                repr(exc)
            )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    return {

        # ================================================
        # MODEL SCORES
        # ================================================

        "model2_score":
            round(
                float(model2_probability),
                6
            ),

        "model3_score":
            round(
                float(model3_probability),
                6
            ),

        # ================================================
        # FUSION RESULT
        # ================================================

        "combined_risk":
            round(
                float(combined_risk),
                6
            ),

        "risk_score":
            round(
                float(combined_risk),
                6
            ),

        "risk_score_percent":
            round(
                float(combined_risk) * 100,
                2
            ),

        # ================================================
        # RISK CLASSIFICATION
        # ================================================

        "risk_band":
            risk_band,

        "fraud_prediction":
            fraud_prediction,

        "decision":
            decision,

        # ================================================
        # CONFIG
        # ================================================

        "threshold":
            get_active_policy().block_threshold,

        "model2_weight":
            get_active_policy().model2_weight,

        "model3_weight":
            get_active_policy().model3_weight,

        # ================================================
        # EXPLAINABILITY
        # ================================================

        "shap_explanations":
            shap_explanations,

        "top_risk_factors":
            top_risk_factors,

        "top_safe_factors":
            top_safe_factors,

        "rag_explanation":
            rag_explanation,
        "risk_explanation":
            risk_explanation,
    }


# ============================================================
# COMPLETE FUSION PREDICTION
# ============================================================

def calculate_fusion_risk(
    model2_input,
    model3_input
) -> dict:

    # ========================================================
    # MODEL 2
    # ========================================================

    model2_probability = float(

        predict_model2(

            transaction_dt=int(
                model2_input[
                    "transaction_dt"
                ]
            ),

            transaction_amt=float(
                model2_input[
                    "transaction_amt"
                ]
            ),

            email=model2_input.get(
                "email"
            ),

            contact=model2_input.get(
                "contact"
            ),

            method=model2_input.get(
                "method"
            ),
        )
    )

    # ========================================================
    # MODEL 3
    # ========================================================

    model3_probability = predict_model3(
        model3_input
    )

    # ========================================================
    # FUSION + EXPLANATION
    # ========================================================

    result = calculate_risk(

        model2_probability,

        model3_probability,

        model3_input
    )

    return result


# ============================================================
# STARTUP VERIFICATION
# ============================================================

print(
    "==================================="
)

print(
    "RiskGuard AI Fusion v1.0"
)

print(
    "==================================="
)

print(
    "Model 2: external adapter"
)

print(
    "Model 3:",
    os.path.basename(
        MODEL3_FILE
    )
)

print(
    "Model 3 features:",
    len(
        model3_features
    )
)

print(
    "Model 3 booster features:",
    model3.get_booster().num_features()
)

print(
    "Model 2 weight:",
    MODEL2_WEIGHT
)

print(
    "Model 3 weight:",
    MODEL3_WEIGHT
)

print(
    "Production threshold:",
    PRODUCTION_THRESHOLD
)

print(
    "Fusion method:",
    fusion_config[
        "fusion"
    ][
        "method"
    ]
)

print(
    "SHAP explanations: ENABLED"
)

print(
    "STATUS: PRODUCTION FUSION READY"
)

print(
    "==================================="
)







