from datetime import datetime
import os

from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Boolean,
    DateTime,
    Float,
    JSON,
)
from sqlalchemy.orm import declarative_base, sessionmaker


# ============================================================
# DATABASE CONFIG
# ============================================================

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./riskguard.db"
)

if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace(
        "postgresql://",
        "postgresql+psycopg://",
        1
    )

if DATABASE_URL.startswith("postgresql+psycopg://"):
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
    )
else:
    engine = create_engine(
        DATABASE_URL,
        connect_args={
            "check_same_thread": False
        },
    )




SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


Base = declarative_base()


# ============================================================
# TRANSACTION MODEL
# ============================================================

class Transaction(Base):

    __tablename__ = "transactions"

    # ========================================================
    # TRANSACTION IDENTIFICATION
    # ========================================================

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    payment_id = Column(
        String,
        unique=True,
        index=True,
        nullable=False,
    )

    order_id = Column(
        String,
        index=True,
        nullable=False,
    )

    # ========================================================
    # PAYMENT DETAILS
    # ========================================================

    amount = Column(
        Integer,
        nullable=False,
    )

    currency = Column(
        String,
        nullable=False,
    )

    status = Column(
        String,
        nullable=False,
    )

    method = Column(
        String,
        nullable=True,
    )

    # ========================================================
    # CUSTOMER INFORMATION
    # ========================================================

    email = Column(
        String,
        nullable=True,
    )

    contact = Column(
        String,
        nullable=True,
    )

    customer_key = Column(
        String,
        index=True,
        nullable=True,
    )

    # ========================================================
    # TRANSACTION TIMING
    # ========================================================

    transaction_dt = Column(
        Integer,
        index=True,
        nullable=True,
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
    )

    # ========================================================
    # PAYMENT STATUS
    # ========================================================

    captured = Column(
        Boolean,
        default=False,
    )

    event_type = Column(
        String,
        nullable=False,
    )

    # ========================================================
    # RISKGUARD AI SCORES
    # ========================================================

    model2_score = Column(
        Float,
        nullable=True,
    )

    model3_score = Column(
        Float,
        nullable=True,
    )

    risk_score = Column(
        Float,
        nullable=True,
    )

    risk_band = Column(
        String,
        nullable=True,
    )

    fraud_prediction = Column(
        Boolean,
        nullable=True,
    )

    action = Column(
        String,
        nullable=True,
    )

    # ========================================================
    # SHAP EXPLAINABILITY
    # ========================================================

    shap_explanations = Column(
        JSON,
        nullable=True,
    )

    top_risk_factors = Column(
        JSON,
        nullable=True,
    )

    top_safe_factors = Column(
        JSON,
        nullable=True,
    )

    risk_explanation = Column(
        JSON,
        nullable=True,
    )


# ============================================================
# MERCHANT ORDER MODEL
# ============================================================

class MerchantOrder(Base):

    __tablename__ = "merchant_orders"

    id = Column(Integer, primary_key=True, index=True)
    order_id = Column(String, unique=True, index=True, nullable=False)

    customer_name = Column(String, nullable=True)
    email = Column(String, nullable=True)
    contact = Column(String, nullable=True)

    price = Column(Float, nullable=False)
    discount = Column(Float, nullable=False, default=0.0)
    quantity = Column(Integer, nullable=False, default=1)
    total_amount = Column(Float, nullable=False)
    shipping_cost = Column(Float, nullable=False, default=0.0)
    profit_margin = Column(Float, nullable=False, default=0.0)
    delivery_time_days = Column(Integer, nullable=False, default=0)
    customer_age = Column(Integer, nullable=False, default=0)

    previous_orders = Column(Integer, nullable=False, default=0)
    previous_returns = Column(Integer, nullable=False, default=0)
    previous_return_rate = Column(Float, nullable=False, default=0.0)
    previous_avg_order_value = Column(Float, nullable=False, default=0.0)
    days_since_previous_order = Column(Float, nullable=False, default=0.0)

    category = Column(String, nullable=True)
    payment_method = Column(String, nullable=True)
    region = Column(String, nullable=True)
    customer_gender = Column(String, nullable=True)

    return_probability = Column(Float, nullable=True)
    loss_if_returned = Column(Float, nullable=True)
    expected_return_loss = Column(Float, nullable=True)
    expected_loss_cutoff = Column(Float, nullable=True)
    return_priority = Column(String, nullable=True)
    return_flagged = Column(Boolean, nullable=True)

    merchant_action = Column(String, nullable=False, default="PENDING")
    payment_status = Column(String, nullable=False, default="PENDING")
    razorpay_order_id = Column(String, nullable=True, index=True)
    payment_id = Column(String, nullable=True, index=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ============================================================
# RISK POLICY CONFIGURATION
# ============================================================

class RiskPolicyConfig(Base):

    __tablename__ = "risk_policies"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    merchant_id = Column(
        String,
        unique=True,
        index=True,
        nullable=False,
        default="default",
    )

    model2_weight = Column(
        Float,
        nullable=False,
        default=0.70,
    )

    model3_weight = Column(
        Float,
        nullable=False,
        default=0.30,
    )

    review_threshold = Column(
        Float,
        nullable=False,
        default=0.50,
    )

    block_threshold = Column(
        Float,
        nullable=False,
        default=0.70,
    )

    profile = Column(
        String,
        nullable=False,
        default="balanced",
    )

    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )



# ============================================================
# AUDIT LOG
# ============================================================

class AuditLog(Base):

    __tablename__ = "audit_logs"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    transaction_id = Column(
        Integer,
        index=True,
        nullable=False,
    )

    payment_id = Column(
        String,
        index=True,
        nullable=False,
    )

    event_type = Column(
        String,
        nullable=False,
    )

    amount = Column(
        Integer,
        nullable=True,
    )

    currency = Column(
        String,
        nullable=True,
    )

    model2_score = Column(
        Float,
        nullable=True,
    )

    model3_score = Column(
        Float,
        nullable=True,
    )

    risk_score = Column(
        Float,
        nullable=True,
    )

    risk_band = Column(
        String,
        nullable=True,
    )

    fraud_prediction = Column(
        Boolean,
        nullable=True,
    )

    decision = Column(
        String,
        nullable=True,
    )

    review_threshold = Column(
        Float,
        nullable=True,
    )

    block_threshold = Column(
        Float,
        nullable=True,
    )

    model2_weight = Column(
        Float,
        nullable=True,
    )

    model3_weight = Column(
        Float,
        nullable=True,
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
    )

# ============================================================
# CREATE TABLES
# ============================================================

Base.metadata.create_all(
    bind=engine
)
