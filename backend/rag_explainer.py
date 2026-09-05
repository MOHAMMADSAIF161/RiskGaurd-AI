"""
RiskGuard AI - Grounded RAG Explanation Layer

This module does NOT make fraud decisions.
It retrieves trusted feature explanations from a local knowledge base
and generates merchant-friendly text using only the supplied SHAP evidence.
"""

FEATURE_KNOWLEDGE = {
    "TransactionHour": {
        "title": "Transaction time",
        "risk": "The payment occurred at a time that the risk model considers unusual.",
        "safe": "The payment occurred at a time that did not add meaningful risk.",
    },
    "TransactionDay": {
        "title": "Transaction date",
        "risk": "The transaction date contributed to a higher-risk pattern identified by the model.",
        "safe": "The transaction date did not materially increase the risk signal.",
    },
    "TransactionAmt": {
        "title": "Transaction amount",
        "risk": "The transaction amount contributed to a higher-risk pattern.",
        "safe": "The transaction amount reduced the risk signal.",
    },
    "previous_transactions": {
        "title": "Customer transaction history",
        "risk": "The available customer transaction history increased the risk signal.",
        "safe": "The available customer transaction history reduced the risk signal.",
    },
    "previous_avg_amount": {
        "title": "Customer's previous average payment",
        "risk": "The customer's previous payment history increased the risk signal.",
        "safe": "The customer's previous payment history reduced the risk signal.",
    },
    "amount_deviation": {
        "title": "Difference from usual payment amount",
        "risk": "The payment amount differed from the customer's previous payment pattern.",
        "safe": "The payment amount did not add risk based on the available payment history.",
    },
    "amount_last_24h": {
        "title": "Amount spent in the last 24 hours",
        "risk": "Recent spending volume increased the risk signal.",
        "safe": "Recent spending volume did not increase the risk signal.",
    },
    "transactions_last_24h": {
        "title": "Transactions in the last 24 hours",
        "risk": "The number of recent transactions increased the risk signal.",
        "safe": "No elevated transaction activity was detected in the last 24 hours.",
    },
    "transactions_last_5min": {
        "title": "Recent transaction activity",
        "risk": "Multiple transactions in a short period increased the risk signal.",
        "safe": "No elevated short-term transaction activity was detected.",
    },
    "transactions_last_10min": {
        "title": "Recent transaction activity",
        "risk": "Transaction activity during the recent time window increased the risk signal.",
        "safe": "Recent transaction activity did not increase the risk signal.",
    },
    "transactions_last_30min": {
        "title": "Recent transaction activity",
        "risk": "Transaction activity during the recent time window increased the risk signal.",
        "safe": "Recent transaction activity did not increase the risk signal.",
    },
    "transactions_last_1h": {
        "title": "Recent transaction activity",
        "risk": "Transaction activity during the last hour increased the risk signal.",
        "safe": "Transaction activity during the last hour did not increase the risk signal.",
    },
    "amount_last_10min": {
        "title": "Amount spent recently",
        "risk": "Recent payment volume increased the risk signal.",
        "safe": "Recent payment volume did not increase the risk signal.",
    },
    "amount_last_1h": {
        "title": "Amount spent in the last hour",
        "risk": "Recent payment volume increased the risk signal.",
        "safe": "Recent payment volume did not increase the risk signal.",
    },
    "rapid_transaction_flag": {
        "title": "Rapid transaction activity",
        "risk": "The model detected a rapid-transaction pattern that increased risk.",
        "safe": "No rapid-transaction pattern was detected.",
    },
    "high_velocity_flag": {
        "title": "High transaction activity",
        "risk": "The model detected unusually high transaction activity.",
        "safe": "No high-velocity transaction pattern was detected.",
    },
    "large_amount_deviation_flag": {
        "title": "Large payment deviation",
        "risk": "The payment showed a large deviation from the available customer payment pattern.",
        "safe": "No large payment deviation was detected.",
    },
    "time_since_previous": {
        "title": "Time since previous transaction",
        "risk": "The time since the previous transaction contributed to the risk signal.",
        "safe": "The time since the previous transaction did not materially increase risk.",
    },
}


def _knowledge(feature):
    return FEATURE_KNOWLEDGE.get(
        feature,
        {
            "title": feature.replace("_", " ").replace("Transaction", "Transaction "),
            "risk": "This model feature increased the transaction risk signal.",
            "safe": "This model feature reduced the transaction risk signal.",
        },
    )


def _format_value(value):
    if value is None:
        return "not available"

    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.2f}"

    return str(value)


def _retrieve_factors(shap_explanations, positive=True, limit=3):
    """
    Retrieval step:
    select the strongest SHAP-supported factors and retrieve their
    explanation from the trusted local knowledge base.
    """
    factors = []

    for item in shap_explanations or []:
        shap_value = float(item.get("shap_value", 0))

        if positive and shap_value <= 0:
            continue

        if not positive and shap_value >= 0:
            continue

        knowledge = _knowledge(item.get("feature", "Feature"))

        factors.append(
            {
                "feature": item.get("feature"),
                "title": knowledge["title"],
                "value": item.get("value"),
                "shap_value": shap_value,
                "explanation": (
                    knowledge["risk"]
                    if positive
                    else knowledge["safe"]
                ),
            }
        )

    factors.sort(
        key=lambda item: abs(item["shap_value"]),
        reverse=True,
    )

    return factors[:limit]


def build_rag_explanation(
    shap_explanations,
    risk_score=None,
    risk_band=None,
    decision=None,
    amount=None,
    threshold=None,
):
    """
    Generation step:
    creates a merchant-friendly explanation using ONLY retrieved
    knowledge and actual SHAP evidence.
    """

    risk_factors = _retrieve_factors(
        shap_explanations,
        positive=True,
        limit=3,
    )

    safe_factors = _retrieve_factors(
        shap_explanations,
        positive=False,
        limit=3,
    )

    decision_text = {
        "ALLOW": "Payment allowed",
        "MANUAL_REVIEW": "Manual review recommended",
        "BLOCK": "Payment blocked",
    }.get(
        decision,
        "Risk decision generated",
    )

    if risk_score is not None:
        score_text = f"{float(risk_score) * 100:.2f}%"
    else:
        score_text = "not available"

    summary = f"{decision_text}. Risk score: {score_text}"

    if risk_band:
        summary += f" ({risk_band} risk)."

    if threshold is not None:
        summary += (
            f" The active block threshold is "
            f"{float(threshold):.2f}."
        )

    return {
        "type": "grounded_rag",
        "source": "RiskGuard AI Risk Explanation Knowledge Base",
        "summary": summary,
        "risk_factors": risk_factors,
        "safe_factors": safe_factors,
        "evidence_count": len(
            risk_factors
        ) + len(
            safe_factors
        ),
        "grounding": (
            "Explanation generated only from the transaction's "
            "SHAP evidence and the local feature knowledge base."
        ),
    }
