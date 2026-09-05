# ============================================
# RiskGuard AI — Decision Engine
# ============================================
#
# Production decision logic is centralized in:
#     backend/risk_policy.py
#
# This module is kept as a compatibility wrapper.
# It does NOT define independent thresholds.
# ============================================

from backend.risk_policy import RiskPolicy


def make_decision(
    risk_score: float,
    policy: RiskPolicy | None = None,
) -> dict:
    """
    Convert a fused ML risk score into the production
    RiskGuard AI business decision.

    The actual decision thresholds are controlled by
    RiskPolicy. This prevents duplicate or conflicting
    decision logic.
    """

    if risk_score is None:
        raise ValueError("risk_score cannot be None")

    risk_score = float(risk_score)

    if not 0.0 <= risk_score <= 1.0:
        raise ValueError(
            f"risk_score must be between 0 and 1, got {risk_score}"
        )

    if policy is None:
        policy = RiskPolicy()

    policy.validate()

    action = policy.decide(risk_score)

    # Risk bands are descriptive only.
    # The actual business decision comes from RiskPolicy.
    if risk_score < policy.review_threshold:
        risk_band = "LOW"

    elif risk_score < policy.block_threshold:
        risk_band = "HIGH"

    else:
        risk_band = "CRITICAL"

    fraud_prediction = int(
        risk_score >= policy.block_threshold
    )

    return {
        "risk_score": risk_score,
        "risk_band": risk_band,
        "fraud_prediction": fraud_prediction,
        "action": action,
    }


# ============================================
# TEST
# ============================================

if __name__ == "__main__":

    policy = RiskPolicy(
        model2_weight=0.70,
        model3_weight=0.30,
        review_threshold=0.50,
        block_threshold=0.70,
    )

    test_scores = [
        0.10,
        0.35,
        0.55,
        0.80,
    ]

    print("===================================")
    print("RiskGuard AI Decision Engine")
    print("Production RiskPolicy")
    print("===================================")

    for score in test_scores:

        result = make_decision(
            score,
            policy=policy,
        )

        print(
            f"Score: {score:.2f} | "
            f"Band: {result['risk_band']} | "
            f"Action: {result['action']}"
        )

    print("===================================")
