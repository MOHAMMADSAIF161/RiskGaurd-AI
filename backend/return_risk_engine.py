from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

try:
    import shap
except Exception:
    shap = None


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models" / "ReturnRisk"

MODEL_PATH = MODEL_DIR / "return_risk_rf.joblib"
CONFIG_PATH = MODEL_DIR / "return_risk_config.json"


class ReturnRiskEngine:
    """
    RiskGuard AI return-risk scoring engine.

    This module is independent from the existing fraud models.
    It predicts return probability and converts it into an
    expected-return-loss priority score.
    """

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        config_path: Path = CONFIG_PATH,
    ):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Return-risk model not found: {model_path}"
            )

        if not config_path.exists():
            raise FileNotFoundError(
                f"Return-risk config not found: {config_path}"
            )

        self.model = joblib.load(model_path)

        with open(config_path, "r", encoding="utf-8") as file:
            self.config = json.load(file)

        self.expected_loss_cutoff = float(
            self.config["expected_loss_cutoff"]
        )

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            if value is None:
                return default
            return int(value)
        except (TypeError, ValueError):
            return default

    def _build_shap_explanation(self, model_input: pd.DataFrame) -> dict[str, Any]:
        """Return an order-level SHAP explanation for the saved RF pipeline."""
        if shap is None:
            return {"available": False, "reason": "SHAP package unavailable"}

        try:
            preprocessor = self.model.named_steps["preprocessor"]
            classifier = self.model.named_steps["model"]
            transformed = preprocessor.transform(model_input)

            # SHAP TreeExplainer works more reliably with dense
            # numeric arrays for this Random Forest pipeline.
            if hasattr(transformed, "toarray"):
                transformed_for_shap = transformed.toarray()
            else:
                transformed_for_shap = transformed

            feature_names = list(
                preprocessor.get_feature_names_out()
            )

            explainer = shap.TreeExplainer(classifier)
            values = explainer.shap_values(
                transformed_for_shap
            )

            # Handle SHAP versions returning either:
            # [class_0, class_1] or a 3-D ndarray.
            if isinstance(values, list):

                if len(values) > 1:
                    shap_values = values[1][0]
                else:
                    shap_values = values[0][0]

            else:

                arr = getattr(
                    values,
                    "values",
                    values
                )

                if getattr(arr, "ndim", 0) == 3:
                    shap_values = arr[0, :, 1]

                elif getattr(arr, "ndim", 0) == 2:
                    shap_values = arr[0]

                else:
                    shap_values = arr

            original_features = [
                "price", "discount", "quantity", "total_amount",
                "shipping_cost", "profit_margin", "delivery_time_days",
                "customer_age", "previous_orders", "previous_returns",
                "previous_return_rate", "previous_avg_order_value",
                "days_since_previous_order", "category", "payment_method",
                "region", "customer_gender",
            ]

            grouped = {name: 0.0 for name in original_features}
            display_values = {name: model_input.iloc[0].get(name) for name in original_features}

            for name, value in zip(feature_names, shap_values):
                clean = name
                if clean.startswith("num__"):
                    clean = clean[len("num__"):]
                    if clean in grouped:
                        grouped[clean] += float(value)
                    continue
                if clean.startswith("cat__"):
                    clean = clean[len("cat__"):]
                    for original in ("category", "payment_method", "region", "customer_gender"):
                        if clean == original or clean.startswith(original + "_"):
                            grouped[original] += float(value)
                            break

            ranked = sorted(grouped.items(), key=lambda item: abs(item[1]), reverse=True)
            risk_factors = []
            safe_factors = []
            for feature, impact in ranked:
                display_value = display_values.get(feature)
                if hasattr(display_value, "item"):
                    display_value = display_value.item()

                item = {
                    "feature": feature,
                    "value": display_value,
                    "shap_value": round(float(impact), 6),
                }
                if impact > 0:
                    risk_factors.append(item)
                elif impact < 0:
                    safe_factors.append(item)

            return {
                "available": True,
                "risk_factors": risk_factors[:5],
                "safe_factors": safe_factors[:5],
            }
        except Exception as exc:
            return {"available": False, "reason": f"SHAP explanation unavailable: {exc}"}

    def score(self, order: dict[str, Any]) -> dict[str, Any]:
        """
        Score one order.

        Required model inputs:
        price
        discount
        quantity
        total_amount
        shipping_cost
        profit_margin
        delivery_time_days
        customer_age
        previous_orders
        previous_returns
        previous_return_rate
        previous_avg_order_value
        days_since_previous_order
        category
        payment_method
        region
        customer_gender
        """

        model_input = pd.DataFrame(
            [
                {
                    "price": self._safe_float(order.get("price")),
                    "discount": self._safe_float(order.get("discount")),
                    "quantity": self._safe_int(order.get("quantity")),
                    "total_amount": self._safe_float(
                        order.get("total_amount")
                    ),
                    "shipping_cost": self._safe_float(
                        order.get("shipping_cost")
                    ),
                    "profit_margin": self._safe_float(
                        order.get("profit_margin")
                    ),
                    "delivery_time_days": self._safe_int(
                        order.get("delivery_time_days")
                    ),
                    "customer_age": self._safe_int(
                        order.get("customer_age")
                    ),
                    "previous_orders": self._safe_int(
                        order.get("previous_orders")
                    ),
                    "previous_returns": self._safe_int(
                        order.get("previous_returns")
                    ),
                    "previous_return_rate": self._safe_float(
                        order.get("previous_return_rate")
                    ),
                    "previous_avg_order_value": self._safe_float(
                        order.get("previous_avg_order_value")
                    ),
                    "days_since_previous_order": self._safe_float(
                        order.get("days_since_previous_order")
                    ),
                    "category": order.get("category"),
                    "payment_method": order.get("payment_method"),
                    "region": order.get("region"),
                    "customer_gender": order.get("customer_gender"),
                }
            ]
        )

        return_probability = float(
            self.model.predict_proba(model_input)[0, 1]
        )

        profit_margin = max(
            self._safe_float(order.get("profit_margin")),
            0.0,
        )

        shipping_cost = self._safe_float(
            order.get("shipping_cost")
        )

        loss_if_returned = profit_margin + shipping_cost

        expected_return_loss = (
            return_probability * loss_if_returned
        )

        flagged = (
            expected_return_loss >= self.expected_loss_cutoff
        )

        # ----------------------------------------------------
        # Automatic merchant decision
        # ----------------------------------------------------
        # The validated cutoff defines the high-risk review
        # boundary. Half of that cutoff is the contact boundary.
        #
        # HIGH   -> REVIEW ORDER
        # MEDIUM -> MAKE CONTACT
        # LOW    -> SAFE
        # ----------------------------------------------------

        contact_cutoff = (
            self.expected_loss_cutoff * 0.5
        )

        if expected_return_loss >= self.expected_loss_cutoff:

            priority = "HIGH"
            merchant_decision = "REVIEW_ORDER"

        elif expected_return_loss >= contact_cutoff:

            priority = "MEDIUM"
            merchant_decision = "MAKE_CONTACT"

        else:

            priority = "LOW"
            merchant_decision = "SAFE"

        shap_result = self._build_shap_explanation(
            model_input
        )

        return {
            "return_probability": round(
                return_probability,
                6,
            ),
            "return_probability_percent": round(
                return_probability * 100,
                2,
            ),
            "loss_if_returned": round(
                loss_if_returned,
                2,
            ),
            "expected_return_loss": round(
                expected_return_loss,
                2,
            ),
            "expected_loss_cutoff": round(
                self.expected_loss_cutoff,
                6,
            ),
            "priority": priority,
            "flagged": flagged,
            "merchant_decision": merchant_decision,
            "contact_cutoff": round(
                contact_cutoff,
                6,
            ),
            "shap_available": shap_result.get("available", False),
            "shap_reason": shap_result.get("reason"),
            "shap_risk_factors": shap_result.get("risk_factors", []),
            "shap_safe_factors": shap_result.get("safe_factors", []),
            "model_name": self.config.get(
                "model_name",
                "RiskGuard Return Risk Model",
            ),
        }


# Load once when the backend imports the module.
return_risk_engine = ReturnRiskEngine()


if __name__ == "__main__":
    example_order = {
        "price": 100,
        "discount": 10,
        "quantity": 1,
        "total_amount": 90,
        "shipping_cost": 6,
        "profit_margin": 20,
        "delivery_time_days": 3,
        "customer_age": 25,
        "previous_orders": 2,
        "previous_returns": 1,
        "previous_return_rate": 0.5,
        "previous_avg_order_value": 85,
        "days_since_previous_order": 30,
        "category": "Fashion",
        "payment_method": "UPI",
        "region": "South",
        "customer_gender": "Male",
    }

    result = return_risk_engine.score(example_order)

    print("=== RETURN RISK ENGINE TEST ===")

    for key, value in result.items():
        print(f"{key}: {value}")

