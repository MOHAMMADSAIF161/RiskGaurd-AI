from dataclasses import dataclass

from backend.database import SessionLocal, RiskPolicyConfig


DEFAULT_MERCHANT_ID = "default"


@dataclass
class RiskPolicy:
    model2_weight: float = 0.70
    model3_weight: float = 0.30
    review_threshold: float = 0.50
    block_threshold: float = 0.70

    def validate(self):
        values = [
            self.model2_weight,
            self.model3_weight,
            self.review_threshold,
            self.block_threshold,
        ]

        if not all(0.0 <= float(v) <= 1.0 for v in values):
            raise ValueError(
                "All policy values must be between 0 and 1."
            )

        if abs(
            (self.model2_weight + self.model3_weight) - 1.0
        ) > 1e-6:
            raise ValueError(
                "Model 2 and Model 3 weights must sum to 1.0."
            )

        if self.review_threshold >= self.block_threshold:
            raise ValueError(
                "Review threshold must be lower than block threshold."
            )

        return True

    def fuse(
        self,
        model2_probability: float,
        model3_probability: float,
    ) -> float:

        model2_probability = float(model2_probability)
        model3_probability = float(model3_probability)

        if not 0.0 <= model2_probability <= 1.0:
            raise ValueError(
                "Model 2 probability must be between 0 and 1."
            )

        if not 0.0 <= model3_probability <= 1.0:
            raise ValueError(
                "Model 3 probability must be between 0 and 1."
            )

        self.validate()

        return (
            self.model2_weight * model2_probability
            + self.model3_weight * model3_probability
        )

    def decide(self, risk_score: float) -> str:

        risk_score = float(risk_score)

        if risk_score >= self.block_threshold:
            return "BLOCK"

        if risk_score >= self.review_threshold:
            return "MANUAL_REVIEW"

        return "ALLOW"

    def to_dict(self):
        return {
            "model2_weight": self.model2_weight,
            "model3_weight": self.model3_weight,
            "review_threshold": self.review_threshold,
            "block_threshold": self.block_threshold,
        }


POLICY_PRESETS = {
    "high_protection": RiskPolicy(
        model2_weight=0.70,
        model3_weight=0.30,
        review_threshold=0.45,
        block_threshold=0.60,
    ),

    "balanced": RiskPolicy(
        model2_weight=0.70,
        model3_weight=0.30,
        review_threshold=0.50,
        block_threshold=0.70,
    ),

    "customer_friendly": RiskPolicy(
        model2_weight=0.70,
        model3_weight=0.30,
        review_threshold=0.55,
        block_threshold=0.80,
    ),
}


def _row_to_policy(row):
    return RiskPolicy(
        model2_weight=row.model2_weight,
        model3_weight=row.model3_weight,
        review_threshold=row.review_threshold,
        block_threshold=row.block_threshold,
    )


def get_active_policy(
    merchant_id: str = DEFAULT_MERCHANT_ID,
) -> RiskPolicy:

    db = SessionLocal()

    try:
        row = (
            db.query(RiskPolicyConfig)
            .filter(
                RiskPolicyConfig.merchant_id == merchant_id
            )
            .first()
        )

        if row is None:
            policy = RiskPolicy()
            policy.validate()

            row = RiskPolicyConfig(
                merchant_id=merchant_id,
                model2_weight=policy.model2_weight,
                model3_weight=policy.model3_weight,
                review_threshold=policy.review_threshold,
                block_threshold=policy.block_threshold,
                profile="balanced",
            )

            db.add(row)
            db.commit()
            db.refresh(row)

        policy = _row_to_policy(row)
        policy.validate()

        return policy

    finally:
        db.close()


def set_active_policy(
    model2_weight: float,
    model3_weight: float,
    review_threshold: float,
    block_threshold: float,
    profile: str = "custom",
    merchant_id: str = DEFAULT_MERCHANT_ID,
):

    new_policy = RiskPolicy(
        model2_weight=float(model2_weight),
        model3_weight=float(model3_weight),
        review_threshold=float(review_threshold),
        block_threshold=float(block_threshold),
    )

    new_policy.validate()

    db = SessionLocal()

    try:
        row = (
            db.query(RiskPolicyConfig)
            .filter(
                RiskPolicyConfig.merchant_id == merchant_id
            )
            .first()
        )

        if row is None:
            row = RiskPolicyConfig(
                merchant_id=merchant_id,
            )
            db.add(row)

        row.model2_weight = new_policy.model2_weight
        row.model3_weight = new_policy.model3_weight
        row.review_threshold = new_policy.review_threshold
        row.block_threshold = new_policy.block_threshold
        row.profile = profile

        db.commit()
        db.refresh(row)

        return _row_to_policy(row)

    finally:
        db.close()


def get_policy_preset(name: str):

    if name not in POLICY_PRESETS:
        raise ValueError(
            f"Unknown policy preset: {name}"
        )

    policy = POLICY_PRESETS[name]

    policy.validate()

    return policy
