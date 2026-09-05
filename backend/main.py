from datetime import datetime, timedelta
import os
import hmac
import hashlib
import time
from uuid import uuid4
from pathlib import Path

import razorpay
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import func

from backend.database import (
    Base,
    engine,
    SessionLocal,
    Transaction,
    AuditLog,
    MerchantOrder,
)

from backend.feature_engine import (
    build_model3_features,
)

from backend.risk_engine import (
    calculate_fusion_risk,
)
from backend.risk_policy import (
    RiskPolicy,
    get_active_policy,
    set_active_policy,
    get_policy_preset,
    POLICY_PRESETS,
)
from backend.rag_explainer import build_rag_explanation

from backend.return_risk_engine import (
    return_risk_engine,
)

# ============================================================
# RISK POLICY API MODELS
# ============================================================

class BatchEvaluationItem(BaseModel):
    amount: float
    email: str = "batch@riskguard.ai"
    contact: str = "9999999999"
    method: str = "upi"
    transaction_dt: int | None = None


class BatchEvaluationRequest(BaseModel):
    transactions: list[BatchEvaluationItem]

class RiskPolicyRequest(BaseModel):
    model2_weight: float
    model3_weight: float
    review_threshold: float
    block_threshold: float
    profile: str = "custom"


class ReturnRiskRequest(BaseModel):
    price: float
    discount: float = 0.0
    quantity: int = 1
    total_amount: float
    shipping_cost: float = 0.0
    profit_margin: float = 0.0
    delivery_time_days: int = 0
    customer_age: int = 0

    previous_orders: int = 0
    previous_returns: int = 0
    previous_return_rate: float = 0.0
    previous_avg_order_value: float = 0.0
    days_since_previous_order: float = 0.0

    category: str = "Unknown"
    payment_method: str = "Unknown"
    region: str = "Unknown"
    customer_gender: str = "Unknown"

class CreateMerchantOrderRequest(ReturnRiskRequest):
    customer_name: str = ""
    email: str = ""
    contact: str = ""


class MerchantActionRequest(BaseModel):
    action: str


class RazorpayOrderRequest(BaseModel):
    amount: float
    internal_order_id: str | None = None

# ============================================================
# CONFIG
# ============================================================

load_dotenv()

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

TEMPLATES_DIR = (
    BASE_DIR / "templates"
)

STATIC_DIR = (
    BASE_DIR / "static"
)


app = FastAPI(
    title="RiskGuard AI",
    description="Real-Time AI Fraud Risk Manager",
    version="1.0.0",
)


# ============================================================
# STATIC FILES
# ============================================================

app.mount(
    "/static",
    StaticFiles(
        directory=STATIC_DIR
    ),
    name="static",
)


# ============================================================
# DATABASE
# ============================================================

Base.metadata.create_all(
    bind=engine
)


# ============================================================
# RAZORPAY CONFIGURATION
# ============================================================

RAZORPAY_KEY_ID = os.getenv(
    "RAZORPAY_KEY_ID"
)

RAZORPAY_KEY_SECRET = os.getenv(
    "RAZORPAY_KEY_SECRET"
)

RAZORPAY_WEBHOOK_SECRET = os.getenv(
    "RAZORPAY_WEBHOOK_SECRET"
)


if not RAZORPAY_KEY_ID:

    raise RuntimeError(
        "RAZORPAY_KEY_ID is not configured"
    )


if not RAZORPAY_KEY_SECRET:

    raise RuntimeError(
        "RAZORPAY_KEY_SECRET is not configured"
    )


if not RAZORPAY_WEBHOOK_SECRET:

    raise RuntimeError(
        "RAZORPAY_WEBHOOK_SECRET is not configured"
    )


razorpay_client = razorpay.Client(
    auth=(
        RAZORPAY_KEY_ID,
        RAZORPAY_KEY_SECRET,
    )
)


# ============================================================
# HTML PAGE LOADER
# ============================================================

def load_page(
    filename: str
) -> str:

    file = (
        TEMPLATES_DIR / filename
    )

    if not file.exists():

        raise HTTPException(
            status_code=404,
            detail=f"{filename} not found",
        )

    return file.read_text(
        encoding="utf-8"
    )



# ============================================================
# RISK POLICY API
# ============================================================

@app.post("/api/risk-policy")
def update_risk_policy(request: RiskPolicyRequest):

    try:
        policy = set_active_policy(
            model2_weight=request.model2_weight,
            model3_weight=request.model3_weight,
            review_threshold=request.review_threshold,
            block_threshold=request.block_threshold,
            profile=request.profile,
        )

        return {
            "status": "success",
            "message": "Risk policy updated successfully.",
            "policy": policy.to_dict(),
            "profile": request.profile,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

@app.post("/api/risk-policy/preview")
def preview_risk_policy(request: RiskPolicyRequest):

    try:
        proposed_policy = RiskPolicy(
            model2_weight=request.model2_weight,
            model3_weight=request.model3_weight,
            review_threshold=request.review_threshold,
            block_threshold=request.block_threshold,
        )

        proposed_policy.validate()

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    db = SessionLocal()

    try:
        transactions = (
            db.query(Transaction)
            .filter(
                Transaction.model2_score.isnot(None),
                Transaction.model3_score.isnot(None),
            )
            .all()
        )

        counts = {
            "ALLOW": 0,
            "MANUAL_REVIEW": 0,
            "BLOCK": 0,
        }

        for transaction in transactions:

            proposed_risk = proposed_policy.fuse(
                transaction.model2_score,
                transaction.model3_score,
            )

            decision = proposed_policy.decide(
                proposed_risk
            )

            counts[decision] += 1

        return {
            "status": "success",
            "transactions_evaluated": len(transactions),
            "proposed_policy": proposed_policy.to_dict(),
            "decision_distribution": counts,
        }

    finally:
        db.close()

@app.get("/api/risk-policy")
def get_risk_policy():
    policy = get_active_policy()

    return {
        "status": "success",
        "policy": policy.to_dict(),
    }
@app.get("/api/return-risk/config")
def get_return_risk_config():
    return {
        "status": "success",
        "model": return_risk_engine.config,
        "expected_loss_cutoff": return_risk_engine.expected_loss_cutoff,
    }


@app.post("/api/return-risk/score")
def score_return_risk(request: ReturnRiskRequest):
    """
    Score an order for merchant return-risk and
    expected return-loss prioritization.
    """

    try:
        result = return_risk_engine.score(
            request.model_dump()
        )

        return {
            "success": True,
            "result": result,
        }

    except Exception as exc:
        print("Return-risk scoring failed:", repr(exc))
        raise HTTPException(
            status_code=500,
            detail="Unable to calculate return risk",
        )
@app.post("/api/orders")
def create_merchant_order(request: CreateMerchantOrderRequest):
    """Create a merchant order, score return risk, and persist the decision."""
    if request.price <= 0 or request.total_amount <= 0 or request.quantity <= 0:
        raise HTTPException(status_code=400, detail="Price, total amount and quantity must be positive.")

    order_id = f"RG-{time.strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:6].upper()}"
    data = request.model_dump()
    customer_name = data.pop("customer_name", "")
    email = data.pop("email", "")
    contact = data.pop("contact", "")

    # ============================================================
    # AUTOMATIC CUSTOMER BEHAVIOR FROM DATABASE
    # Never trust history values submitted by the browser.
    # ============================================================
    history_db = SessionLocal()
    try:
        previous_query = history_db.query(MerchantOrder).filter(
            MerchantOrder.order_id != order_id
        )

        if email:
            previous_query = previous_query.filter(
                MerchantOrder.email == email
            )
        elif contact:
            previous_query = previous_query.filter(
                MerchantOrder.contact == contact
            )
        else:
            previous_query = previous_query.filter(
                MerchantOrder.id == -1
            )

        previous_customer_orders = (
            previous_query
            .order_by(MerchantOrder.created_at.asc())
            .all()
        )

        previous_orders = len(previous_customer_orders)

        if previous_orders:
            previous_avg_order_value = sum(
                float(o.total_amount or 0)
                for o in previous_customer_orders
            ) / previous_orders

            latest_previous = previous_customer_orders[-1]

            if latest_previous.created_at:
                days_since_previous_order = max(
                    0.0,
                    (datetime.utcnow() - latest_previous.created_at).total_seconds()
                    / 86400.0
                )
            else:
                days_since_previous_order = 0.0
        else:
            previous_avg_order_value = 0.0
            days_since_previous_order = 0.0

        # The current DB has no verified return-outcome column.
        # Therefore we do NOT invent previous returns.
        previous_returns = 0
        previous_return_rate = 0.0

        data["previous_orders"] = previous_orders
        data["previous_returns"] = previous_returns
        data["previous_return_rate"] = previous_return_rate
        data["previous_avg_order_value"] = previous_avg_order_value
        data["days_since_previous_order"] = days_since_previous_order

    finally:
        history_db.close()

    try:
        risk = return_risk_engine.score(data)

        # ========================================================
        # AUTOMATIC MERCHANT DECISION
        # Based on model expected-return-loss output.
        # ========================================================
        expected_loss = float(risk.get("expected_return_loss", 0) or 0)
        cutoff = float(risk.get("expected_loss_cutoff", 0) or 0)

        if cutoff > 0 and expected_loss >= cutoff:
            merchant_decision = "REVIEW_ORDER"
        elif cutoff > 0 and expected_loss >= (cutoff * 0.50):
            merchant_decision = "MAKE_CONTACT"
        else:
            merchant_decision = "SAFE"

        risk["merchant_decision"] = merchant_decision
    except Exception as exc:
        print("Merchant order scoring failed:", repr(exc))
        raise HTTPException(status_code=500, detail="Unable to score merchant order")

    db = SessionLocal()
    try:
        order = MerchantOrder(
            order_id=order_id,
            customer_name=customer_name,
            email=email,
            contact=contact,
            **{key: data[key] for key in [
                "price", "discount", "quantity", "total_amount", "shipping_cost",
                "profit_margin", "delivery_time_days", "customer_age", "previous_orders",
                "previous_returns", "previous_return_rate", "previous_avg_order_value",
                "days_since_previous_order", "category", "payment_method", "region", "customer_gender"
            ]},
            return_probability=risk["return_probability"],
            loss_if_returned=risk["loss_if_returned"],
            expected_return_loss=risk["expected_return_loss"],
            expected_loss_cutoff=risk["expected_loss_cutoff"],
            return_priority=risk["priority"],
            return_flagged=risk["flagged"],
            merchant_action=risk.get(
                "merchant_decision",
                "SAFE",
            ),
            payment_status="PENDING",
        )
        db.add(order)
        db.commit()
        db.refresh(order)
        return serialize_merchant_order(order)
    finally:
        db.close()


@app.get("/api/orders")
def list_merchant_orders(limit: int = 200):
    db = SessionLocal()
    try:
        orders = db.query(MerchantOrder).order_by(MerchantOrder.created_at.desc()).limit(min(max(limit, 1), 500)).all()
        return {"orders": [serialize_merchant_order(o) for o in orders], "count": len(orders)}
    finally:
        db.close()


@app.get("/api/orders/{order_id}")
def get_merchant_order(order_id: str):
    db = SessionLocal()
    try:
        order = db.query(MerchantOrder).filter(MerchantOrder.order_id == order_id).first()
        if not order:
            raise HTTPException(status_code=404, detail="Order not found")
        return serialize_merchant_order(order)
    finally:
        db.close()


@app.post("/api/orders/{order_id}/action")
def update_merchant_order_action(order_id: str, request: MerchantActionRequest):
    allowed = {"REVIEW_ORDER", "MAKE_CONTACT", "SAFE", "PENDING"}
    action = request.action.strip().upper()
    if action not in allowed:
        raise HTTPException(status_code=400, detail="Invalid merchant action")

    db = SessionLocal()
    try:
        order = db.query(MerchantOrder).filter(MerchantOrder.order_id == order_id).first()
        if not order:
            raise HTTPException(status_code=404, detail="Order not found")
        order.merchant_action = action
        db.commit()
        db.refresh(order)
        return {"success": True, "order": serialize_merchant_order(order)}
    finally:
        db.close()


@app.get("/api/orders/{order_id}/shap")
def get_merchant_order_shap(order_id: str):
    db = SessionLocal()
    try:
        order = db.query(MerchantOrder).filter(MerchantOrder.order_id == order_id).first()
        if not order:
            raise HTTPException(status_code=404, detail="Order not found")
        model_data = {
            "price": order.price, "discount": order.discount, "quantity": order.quantity,
            "total_amount": order.total_amount, "shipping_cost": order.shipping_cost,
            "profit_margin": order.profit_margin, "delivery_time_days": order.delivery_time_days,
            "customer_age": order.customer_age, "previous_orders": order.previous_orders,
            "previous_returns": order.previous_returns, "previous_return_rate": order.previous_return_rate,
            "previous_avg_order_value": order.previous_avg_order_value,
            "days_since_previous_order": order.days_since_previous_order,
            "category": order.category, "payment_method": order.payment_method,
            "region": order.region, "customer_gender": order.customer_gender,
        }
        result = return_risk_engine.score(model_data)
        return {
            "order_id": order.order_id,
            "return_probability_percent": result["return_probability_percent"],
            "expected_return_loss": result["expected_return_loss"],
            "shap_available": result.get(
                "shap_available",
                False,
            ),
            "shap_reason": result.get(
                "shap_reason"
            ),
            "risk_factors": result.get(
                "shap_risk_factors",
                [],
            ),
            "safe_factors": result.get(
                "shap_safe_factors",
                [],
            ),
        }
    finally:
        db.close()


def serialize_merchant_order(order: MerchantOrder):
    return {
        "order_id": order.order_id,
        "customer_name": order.customer_name,
        "email": order.email,
        "contact": order.contact,
        "price": order.price,
        "discount": order.discount,
        "quantity": order.quantity,
        "total_amount": order.total_amount,
        "shipping_cost": order.shipping_cost,
        "profit_margin": order.profit_margin,
        "delivery_time_days": order.delivery_time_days,
        "customer_age": order.customer_age,
        "previous_orders": order.previous_orders,
        "previous_returns": order.previous_returns,
        "previous_return_rate": order.previous_return_rate,
        "previous_avg_order_value": order.previous_avg_order_value,
        "days_since_previous_order": order.days_since_previous_order,
        "category": order.category,
        "payment_method": order.payment_method,
        "region": order.region,
        "customer_gender": order.customer_gender,
        "return_probability": order.return_probability,
        "return_probability_percent": round((order.return_probability or 0) * 100, 2) if order.return_probability is not None else None,
        "loss_if_returned": order.loss_if_returned,
        "expected_return_loss": order.expected_return_loss,
        "expected_loss_cutoff": order.expected_loss_cutoff,
        "return_priority": order.return_priority,
        "return_flagged": order.return_flagged,
        "merchant_action": order.merchant_action,
        "payment_status": order.payment_status,
        "razorpay_order_id": order.razorpay_order_id,
        "payment_id": order.payment_id,
        "created_at": ((order.created_at + timedelta(hours=5, minutes=30)).isoformat() if order.created_at else None),
    }


# ============================================================
# HTML ROUTES
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
def root():

    return load_page(
        "index.html"
    )


@app.get(
    "/index",
    response_class=HTMLResponse
)
def index_page():

    return load_page(
        "index.html"
    )


@app.get(
    "/payment",
    response_class=HTMLResponse
)
def payment_page():

    return load_page(
        "payment.html"
    )

@app.get(
    "/create-order-page",
    response_class=HTMLResponse
)
def create_order_page():

    return load_page(
        "create-order.html"
    )


@app.get(
    "/dashboard",
    response_class=HTMLResponse
)
def dashboard_page():

    return load_page(
        "dashboard.html"
    )

@app.get(
    "/transactions",
    response_class=HTMLResponse
)
def transactions_page():

    return load_page(
        "transactions.html"
    )


@app.get(
    "/transaction-details",
    response_class=HTMLResponse
)
def transaction_detail_page():

    return load_page(
        "transaction-details.html"
    )
@app.get(
    "/model-performance",
    response_class=HTMLResponse
)
def model_performance_page():

    return load_page(
        "model-performance.html"
    )
@app.get(
    "/batch-evaluation",
    response_class=HTMLResponse
)
def batch_evaluation_page():

    return load_page(
        "batch-evaluation.html"
    )

@app.get(
    "/return-risk",
    response_class=HTMLResponse
)
def return_risk_page():

    return load_page(
        "return-risk.html"
    )


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================================
# TRANSACTION SERIALIZER
# ============================================================

def serialize_transaction(
    transaction: Transaction
):

    # STEP 5: Customer history / cold-start status
    previous_transaction_count = None

    if transaction.shap_explanations:
        for item in transaction.shap_explanations:
            if item.get("feature") == "previous_transactions":
                previous_transaction_count = item.get("value")
                break

    if previous_transaction_count is not None:
        previous_transaction_count = int(previous_transaction_count)
        cold_start = previous_transaction_count == 0
        history_available = previous_transaction_count > 0
    else:
        cold_start = None
        history_available = None
    rag_explanation = None

    if transaction.shap_explanations:
        rag_explanation = build_rag_explanation(
            shap_explanations=transaction.shap_explanations,
            risk_score=transaction.risk_score,
            risk_band=transaction.risk_band,
            decision=transaction.action,
        )

    return {

        # ----------------------------------------------------
        # IDENTIFICATION
        # ----------------------------------------------------

        "id":
            transaction.id,

        "payment_id":
            transaction.payment_id,

        "order_id":
            transaction.order_id,


        # ----------------------------------------------------
        # PAYMENT
        # ----------------------------------------------------

        "amount":
            round(
                transaction.amount / 100,
                2,
            ),

        "currency":
            transaction.currency,

        "status":
            transaction.status,

        "method":
            transaction.method,


        # ----------------------------------------------------
        # CUSTOMER
        # ----------------------------------------------------

        "email":
            transaction.email,

        "contact":
            transaction.contact,

        "customer_key":
            transaction.customer_key,


        # ----------------------------------------------------
        # TIMING
        # ----------------------------------------------------

        "transaction_dt":
            transaction.transaction_dt,

        "created_at":
            (
                transaction.created_at.isoformat()
                if transaction.created_at
                else None
            ),


        # ----------------------------------------------------
        # PAYMENT STATUS
        # ----------------------------------------------------

        "captured":
            transaction.captured,

        "event_type":
            transaction.event_type,


        # ----------------------------------------------------
        # AI SCORES
        # ----------------------------------------------------

        "model2_score":
            transaction.model2_score,

        "model3_score":
            transaction.model3_score,

        "risk_score":
            transaction.risk_score,

        "risk_score_percent":
            (
                round(
                    transaction.risk_score * 100,
                    2,
                )
                if transaction.risk_score is not None
                else None
            ),

        "risk_band":
            transaction.risk_band,

        "fraud_prediction":
            transaction.fraud_prediction,

        "action":
            transaction.action,


        # ----------------------------------------------------
        # SHAP
        # ----------------------------------------------------

        "shap_explanations":
            transaction.shap_explanations,

        "top_risk_factors":
            transaction.top_risk_factors,

        "top_safe_factors":
            transaction.top_safe_factors,

        "risk_explanation":
            transaction.risk_explanation,

        "rag_explanation":
            rag_explanation,

    "created_date":
(
    transaction.created_at.strftime("%d %b %Y %H:%M")
    if transaction.created_at
    else None
),

        "previous_transaction_count":
            previous_transaction_count,

        "cold_start":
            cold_start,

        "history_available":
            history_available,

"fraud_status":
(
    "Fraud"
    if transaction.fraud_prediction
    else "Legitimate"
),
    }


# ============================================================
# COMPLETE DASHBOARD
#
# SINGLE ENDPOINT
#
# /api/dashboard
#
# Provides:
# - Statistics
# - Risk distribution
# - Decision distribution
# - Payment methods
# - Fraud counts
# - Latest transaction
# - Recent transactions
#
# ============================================================

@app.get(
    "/api/dashboard"
)
def dashboard():

    db = SessionLocal()

    try:

        # ====================================================
        # BASIC STATISTICS
        # ====================================================

        total_transactions = (
            db.query(
                Transaction
            ).count()
        )


        total_amount_minor = (
            db.query(
                func.coalesce(
                    func.sum(
                        Transaction.amount
                    ),
                    0,
                )
            ).scalar()
        )


        high_risk_transactions = (
            db.query(
                Transaction
            )
            .filter(
                Transaction.risk_band.in_(
                    [
                        "HIGH",
                        "CRITICAL",
                    ]
                )
            )
            .count()
        )


        blocked_transactions = (
            db.query(
                Transaction
            )
            .filter(
                Transaction.action
                == "BLOCK"
            )
            .count()
        )


        review_transactions = (
            db.query(
                Transaction
            )
            .filter(
                Transaction.action
                == "MANUAL_REVIEW"
            )
            .count()
        )


        allowed_transactions = (
            db.query(
                Transaction
            )
            .filter(
                Transaction.action
                == "ALLOW"
            )
            .count()
        )


        # ====================================================
        # FRAUD STATISTICS
        # ====================================================

        fraud_transactions = (
            db.query(
                Transaction
            )
            .filter(
                Transaction.fraud_prediction
                == True
            )
            .count()
        )


        legitimate_transactions = (
            db.query(
                Transaction
            )
            .filter(
                Transaction.fraud_prediction
                == False
            )
            .count()
        )


        # ====================================================
        # AVERAGE RISK
        # ====================================================

        average_risk = (
            db.query(
                func.avg(
                    Transaction.risk_score
                )
            )
            .filter(
                Transaction.risk_score.isnot(None)
            )
            .scalar()
        )


        average_risk = float(
            average_risk or 0
        )


        # ====================================================
        # RISK DISTRIBUTION
        # ====================================================

        risk_bands = (
            db.query(
                Transaction.risk_band,
                func.count(
                    Transaction.id
                ),
            )
            .filter(
                Transaction.risk_band.isnot(None)
            )
            .group_by(
                Transaction.risk_band
            )
            .all()
        )


        risk_distribution = {

            "LOW":
                0,

            "MEDIUM":
                0,

            "HIGH":
                0,

            "CRITICAL":
                0,
        }


        for risk_band, count in risk_bands:

            if risk_band in risk_distribution:

                risk_distribution[
                    risk_band
                ] = count


        # ====================================================
        # DECISION DISTRIBUTION
        # ====================================================

        decisions = (
            db.query(
                Transaction.action,
                func.count(
                    Transaction.id
                ),
            )
            .filter(
                Transaction.action.isnot(None)
            )
            .group_by(
                Transaction.action
            )
            .all()
        )


        decision_distribution = {

            "ALLOW":
                0,

            "MANUAL_REVIEW":
                0,

            "BLOCK":
                0,
        }


        for action, count in decisions:

            if action in decision_distribution:

                decision_distribution[
                    action
                ] = count


        # ====================================================
        # PAYMENT METHODS
        # ====================================================

        methods = (
            db.query(
                Transaction.method,
                func.count(
                    Transaction.id
                ),
            )
            .filter(
                Transaction.method.isnot(None)
            )
            .group_by(
                Transaction.method
            )
            .all()
        )


        payment_methods = dict(
            methods
        )


        # ====================================================
        # RECENT TRANSACTIONS
        # ====================================================

        transactions = (
            db.query(
                Transaction
            )
            .order_by(
                Transaction.created_at.desc()
            )
            .limit(100)
            .all()
        )


        transaction_list = [

            serialize_transaction(
                transaction
            )

            for transaction
            in transactions
        ]


        # ====================================================
        # MERCHANT ORDER ANALYTICS
        # ====================================================

        merchant_orders = (
            db.query(MerchantOrder)
            .order_by(MerchantOrder.created_at.desc())
            .limit(500)
            .all()
        )

        merchant_risk_distribution = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
        merchant_action_distribution = {"REVIEW_ORDER": 0, "MAKE_CONTACT": 0, "SAFE": 0, "PENDING": 0}
        total_expected_return_loss = 0.0
        paid_orders = 0
        high_priority = 0

        for order in merchant_orders:
            if order.return_priority in merchant_risk_distribution:
                merchant_risk_distribution[order.return_priority] += 1
            action = order.merchant_action or "PENDING"
            if action in merchant_action_distribution:
                merchant_action_distribution[action] += 1
            total_expected_return_loss += float(order.expected_return_loss or 0.0)
            if str(order.payment_status or "").upper() in {"CAPTURED", "PAID"}:
                paid_orders += 1
            if order.return_priority == "HIGH":
                high_priority += 1

        merchant_order_statistics = {
            "total_orders": len(merchant_orders),
            "paid_orders": paid_orders,
            "high_priority": high_priority,
            "total_expected_return_loss": round(total_expected_return_loss, 2),
        }

        # ====================================================
        # RESPONSE
        # ====================================================

        return {

            "merchant_order_statistics": merchant_order_statistics,
            "merchant_order_risk_distribution": merchant_risk_distribution,
            "merchant_order_action_distribution": merchant_action_distribution,
            "merchant_orders": [serialize_merchant_order(o) for o in merchant_orders],

            # ------------------------------------------------
            # STATISTICS
            # ------------------------------------------------

            "statistics": {

                "total_transactions":
                    total_transactions,

                "total_amount":
                    round(
                        float(
                            total_amount_minor
                        ) / 100,
                        2,
                    ),

                "high_risk_transactions":
                    high_risk_transactions,

                "blocked_transactions":
                    blocked_transactions,

                "review_transactions":
                    review_transactions,

                "allowed_transactions":
                    allowed_transactions,

                "fraud_transactions":
                    fraud_transactions,

                "legitimate_transactions":
                    legitimate_transactions,

                "average_risk":
                    round(
                        average_risk,
                        4,
                    ),

                "average_risk_percent":
                    round(
                        average_risk * 100,
                        2,
                    ),
            },


            # ------------------------------------------------
            # RISK
            # ------------------------------------------------

            "risk_distribution":
                risk_distribution,


            # ------------------------------------------------
            # DECISIONS
            # ------------------------------------------------

            "decision_distribution":
                decision_distribution,


            # ------------------------------------------------
            # PAYMENT METHODS
            # ------------------------------------------------

            "payment_methods":
                payment_methods,


            # ------------------------------------------------
            # FRAUD
            # ------------------------------------------------

            "fraud_count":
                fraud_transactions,

            "legitimate_count":
                legitimate_transactions,


            # ------------------------------------------------
            # LATEST TRANSACTION
            # ------------------------------------------------

            "latest_transaction":
                (
                    transaction_list[0]
                    if transaction_list
                    else None
                ),


            # ------------------------------------------------
            # ALL RECENT TRANSACTIONS
            # ------------------------------------------------

            "transactions":
                transaction_list,
        }


    finally:

        db.close()

# ============================================================
# TRANSACTION LIST
# ============================================================

@app.get("/api/transactions")
def list_transactions(limit: int = 100):

    if limit < 1 or limit > 500:
        raise HTTPException(
            status_code=400,
            detail="limit must be between 1 and 500"
        )

    db = SessionLocal()

    try:
        transactions = (
            db.query(Transaction)
            .order_by(Transaction.created_at.desc())
            .limit(limit)
            .all()
        )

        return {
            "transactions": [
                serialize_transaction(t)
                for t in transactions
            ],
            "count": len(transactions),
            "limit": limit,
        }

    finally:
        db.close()


# ============================================================
# SINGLE TRANSACTION
# ============================================================


# ============================================================
# AUDIT TRAIL
# ============================================================

@app.get("/api/audit-log")
def get_audit_log(
    limit: int = 100
):
    db = SessionLocal()

    try:

        safe_limit = max(
            1,
            min(limit, 500)
        )

        logs = (
            db.query(AuditLog)
            .order_by(
                AuditLog.created_at.desc()
            )
            .limit(safe_limit)
            .all()
        )

        return {
            "audit_logs": [
                {
                    "id": log.id,
                    "transaction_id": log.transaction_id,
                    "payment_id": log.payment_id,
                    "event_type": log.event_type,
                    "amount": (
                        round(
                            float(log.amount) / 100,
                            2
                        )
                        if log.amount is not None
                        else None
                    ),
                    "currency": log.currency,
                    "model2_score": log.model2_score,
                    "model3_score": log.model3_score,
                    "risk_score": log.risk_score,
                    "risk_band": log.risk_band,
                    "fraud_prediction": log.fraud_prediction,
                    "decision": log.decision,
                    "review_threshold": log.review_threshold,
                    "block_threshold": log.block_threshold,
                    "model2_weight": log.model2_weight,
                    "model3_weight": log.model3_weight,
                    "created_at": (
                        log.created_at.isoformat()
                        if log.created_at
                        else None
                    ),
                }
                for log in logs
            ],
            "count": len(logs),
            "limit": safe_limit,
        }

    finally:
        db.close()

@app.get("/api/transactions/{transaction_id}")
def get_transaction(transaction_id: int):

    db = SessionLocal()

    try:

        transaction = (
            db.query(Transaction)
            .filter(Transaction.id == transaction_id)
            .first()
        )

        if not transaction:

            raise HTTPException(
                status_code=404,
                detail="Transaction not found"
            )

        return serialize_transaction(transaction)

    finally:

        db.close()
# ============================================================
# CREATE RAZORPAY ORDER
# ============================================================

class OrderRequest(
    BaseModel
):

    internal_order_id: str


@app.post(
    "/create-order"
)
def create_order(
    request: OrderRequest
):

    if not request.internal_order_id:

        raise HTTPException(
            status_code=400,
            detail="Order ID is required",
        )

    # --------------------------------------------------------
    # Fetch merchant order
    # --------------------------------------------------------

    db = SessionLocal()

    try:

        merchant_order = (
            db.query(MerchantOrder)
            .filter(
                MerchantOrder.order_id ==
                request.internal_order_id
            )
            .first()
        )

        if not merchant_order:

            raise HTTPException(
                status_code=404,
                detail="Merchant order not found",
            )

        # IMPORTANT:
        # Payment amount comes ONLY from the saved order.
        amount_rupees = float(
            merchant_order.total_amount
        )

    finally:

        db.close()

    if amount_rupees <= 0:

        raise HTTPException(
            status_code=400,
            detail="Order amount must be greater than 0",
        )

    amount_minor = int(
        round(
            amount_rupees * 100
        )
    )

    # --------------------------------------------------------
    # Create Razorpay order
    # --------------------------------------------------------

    order_data = {

        "amount":
            amount_minor,

        "currency":
            "INR",

        "receipt":
            f"riskguard_{int(time.time())}",

        "notes": {

            "project":
                "RiskGuard AI",

            "purpose":
                "merchant_return_risk",

            "internal_order_id":
                request.internal_order_id,
        },
    }

    try:

        order = (
            razorpay_client
            .order
            .create(
                data=order_data
            )
        )

    except Exception as exc:

        print(
            "Razorpay order creation failed:",
            repr(exc),
        )

        raise HTTPException(
            status_code=502,
            detail="Unable to create Razorpay order",
        )

    # --------------------------------------------------------
    # Link Razorpay order to merchant order
    # --------------------------------------------------------

    db = SessionLocal()

    try:

        merchant_order = (
            db.query(MerchantOrder)
            .filter(
                MerchantOrder.order_id ==
                request.internal_order_id
            )
            .first()
        )

        if merchant_order:

            merchant_order.razorpay_order_id =                 order.get("id")

            merchant_order.payment_status =                 "ORDER_CREATED"

            db.commit()

    finally:

        db.close()

    return {

        **order,

        "key_id":
            RAZORPAY_KEY_ID,

        "amount_rupees":
            amount_rupees,

        "internal_order_id":
            request.internal_order_id,
    }


# ============================================================
# ============================================================
# BATCH EVALUATION - DRY RUN
# ============================================================

@app.post("/api/batch-evaluate")
def batch_evaluate(request: BatchEvaluationRequest):

    if not request.transactions:
        raise HTTPException(
            status_code=400,
            detail="At least one transaction is required."
        )

    if len(request.transactions) > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum batch size is 100 transactions."
        )

    db = SessionLocal()
    results = []

    try:
        for index, item in enumerate(request.transactions, start=1):

            if item.amount <= 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"Transaction {index}: amount must be greater than 0."
                )

            transaction_dt = (
                item.transaction_dt
                if item.transaction_dt is not None
                else int(time.time())
            )

            customer_key = (
                item.email.strip().lower()
                if item.email
                else f"phone:{item.contact}"
            )

            model3_input = build_model3_features(
                db=db,
                customer_key=customer_key,
                transaction_amt=item.amount,
                transaction_dt=transaction_dt,
            )

            model2_input = {
                "transaction_dt": transaction_dt,
                "transaction_amt": item.amount,
                "email": item.email,
                "contact": item.contact,
                "method": item.method,
            }

            risk_result = calculate_fusion_risk(
                model2_input,
                model3_input,
            )

            results.append({
                "batch_index": index,
                "amount": item.amount,
                "currency": "INR",
                "customer": customer_key,
                "model2_score": risk_result["model2_score"],
                "model3_score": risk_result["model3_score"],
                "combined_risk": risk_result["combined_risk"],
                "risk_score_percent": risk_result["risk_score_percent"],
                "risk_band": risk_result["risk_band"],
                "fraud_prediction": risk_result["fraud_prediction"],
                "decision": risk_result["decision"],
                "threshold": risk_result["threshold"],
                "model2_weight": risk_result["model2_weight"],
                "model3_weight": risk_result["model3_weight"],
            })

        summary = {
            "total": len(results),
            "allow": sum(
                1 for r in results
                if r["decision"] == "ALLOW"
            ),
            "manual_review": sum(
                1 for r in results
                if r["decision"] == "MANUAL_REVIEW"
            ),
            "block": sum(
                1 for r in results
                if r["decision"] == "BLOCK"
            ),
            "low": sum(
                1 for r in results
                if r["risk_band"] == "LOW"
            ),
            "medium": sum(
                1 for r in results
                if r["risk_band"] == "MEDIUM"
            ),
            "high": sum(
                1 for r in results
                if r["risk_band"] == "HIGH"
            ),
        }

        return {
            "batch_evaluation": True,
            "dry_run": True,
            "message": (
                "Batch evaluation completed. "
                "No transactions or audit records were created."
            ),
            "summary": summary,
            "results": results,
        }

    finally:
        db.close()
# SIMULATE PAYMENT
# ============================================================

@app.post(
    "/simulate-payment"
)
def simulate_payment(

    amount: float,

    email: str =
        "simulation@riskguard.ai",

    contact: str =
        "9999999999",

    method: str =
        "upi",
):

    if amount <= 0:

        raise HTTPException(
            status_code=400,
            detail="Amount must be greater than 0",
        )


    transaction_dt = int(
        time.time()
    )


    customer_key = (

        email.strip().lower()

        if email

        else
        f"phone:{contact}"
    )


    db = SessionLocal()

    try:

        model3_input = (
            build_model3_features(

                db=db,

                customer_key=
                    customer_key,

                transaction_amt=
                    amount,

                transaction_dt=
                    transaction_dt,
            )
        )


        model2_input = {

            "transaction_dt":
                transaction_dt,

            "transaction_amt":
                amount,

            "email":
                email,

            "contact":
                contact,

            "method":
                method,
        }


        risk_result = (
            calculate_fusion_risk(

                model2_input,

                model3_input,
            )
        )


        return {

            "simulation":
                True,

            "message":
                (
                    "Simulation only. "
                    "No real payment was created."
                ),

            "customer":
                customer_key,

            "amount":
                amount,

            "currency":
                "INR",


            "model2_score":
                risk_result[
                    "model2_score"
                ],

            "model3_score":
                risk_result[
                    "model3_score"
                ],

            "combined_risk":
                risk_result[
                    "combined_risk"
                ],

            "risk_score":
                risk_result[
                    "risk_score"
                ],

            "risk_score_percent":
                risk_result[
                    "risk_score_percent"
                ],

            "risk_band":
                risk_result[
                    "risk_band"
                ],

            "decision":
                risk_result[
                    "decision"
                ],

            "threshold":
                risk_result[
                    "threshold"
                ],

            "model2_weight":
                risk_result[
                    "model2_weight"
                ],

            "model3_weight":
                risk_result[
                    "model3_weight"
                ],


            # SHAP

            "shap_explanations":
                risk_result.get(
                    "shap_explanations"
                ),

            "top_risk_factors":
                risk_result.get(
                    "top_risk_factors"
                ),

            "top_safe_factors":
                risk_result.get(
                    "top_safe_factors"
                ),

            "rag_explanation":
                risk_result.get(
                    "rag_explanation"
                ),
            "risk_explanation":
                risk_result.get(
                    "risk_explanation"
                ),
        }

    except Exception as exc:

        print(
            "Simulation failed:",
            repr(exc),
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Simulation risk "
                "evaluation failed"
            ),
        )

    finally:

        db.close()


# ============================================================
# RAZORPAY WEBHOOK
# ============================================================

@app.post(
    "/webhooks/razorpay"
)
async def razorpay_webhook(
    request: Request
):

    # ========================================================
    # READ BODY
    # ========================================================

    body = await request.body()


    if not body:

        raise HTTPException(
            status_code=400,
            detail="Empty webhook body",
        )


    # ========================================================
    # VERIFY SIGNATURE
    # ========================================================

    received_signature = (
        request.headers.get(
            "X-Razorpay-Signature"
        )
    )


    if not received_signature:

        raise HTTPException(
            status_code=400,
            detail="Missing Razorpay signature",
        )


    expected_signature = hmac.new(

        RAZORPAY_WEBHOOK_SECRET.encode(
            "utf-8"
        ),

        body,

        hashlib.sha256,
    ).hexdigest()


    if not hmac.compare_digest(

        expected_signature,

        received_signature,
    ):

        print(
            "SECURITY ALERT: "
            "Invalid Razorpay signature"
        )

        raise HTTPException(
            status_code=400,
            detail="Invalid Razorpay signature",
        )


    # ========================================================
    # PARSE JSON
    # ========================================================

    try:

        payload = await request.json()

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Invalid JSON payload",
        )


    # ========================================================
    # EVENT
    # ========================================================

    event_type = payload.get(
        "event",
        "unknown",
    )


    # ========================================================
    # PAYMENT ENTITY
    # ========================================================

    payment = (

        payload

        .get(
            "payload",
            {}
        )

        .get(
            "payment",
            {}
        )

        .get(
            "entity",
            {}
        )
    )


    if not payment:

        return {

            "status":
                "success",

            "message":
                (
                    "Webhook verified but "
                    "no payment entity found"
                ),
        }


    # ========================================================
    # PAYMENT DATA
    # ========================================================

    payment_id = payment.get(
        "id"
    )

    order_id = payment.get(
        "order_id"
    )

    amount_minor = payment.get(
        "amount"
    )

    currency = payment.get(
        "currency",
        "INR"
    )

    status = payment.get(
        "status",
        "unknown"
    )

    method = payment.get(
        "method"
    )

    email = payment.get(
        "email"
    )

    contact = payment.get(
        "contact"
    )

    captured = payment.get(
        "captured",
        False
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    if not payment_id:

        raise HTTPException(
            status_code=400,
            detail="Missing payment ID",
        )


    if not order_id:

        raise HTTPException(
            status_code=400,
            detail="Missing order ID",
        )


    if amount_minor is None:

        raise HTTPException(
            status_code=400,
            detail="Missing payment amount",
        )


    transaction_amount = (
        float(amount_minor) / 100
    )


    transaction_dt = int(

        payment.get(

            "created_at",

            time.time(),
        )
    )


    # ========================================================
    # MERCHANT ORDER LINK
    # ========================================================

    merchant_order = None

    db = SessionLocal()
    try:
        merchant_order = (
            db.query(MerchantOrder)
            .filter(MerchantOrder.razorpay_order_id == order_id)
            .first()
        )
    finally:
        db.close()


    # ========================================================
    # CUSTOMER KEY
    # ========================================================

    if email:

        customer_key = (
            email.strip().lower()
        )

    elif contact:

        customer_key = (
            f"phone:{contact}"
        )

    else:

        customer_key = (
            f"anonymous:{payment_id}"
        )


    db = SessionLocal()

    try:

        # ====================================================
        # IDEMPOTENCY
        # ====================================================

        existing = (

            db.query(
                Transaction
            )

            .filter(
                Transaction.payment_id
                == payment_id
            )

            .first()
        )


        if existing:

            return {

                "status":
                    "success",

                "message":
                    (
                        "Transaction "
                        "already processed"
                    ),

                "payment_id":
                    payment_id,
            }


        # ====================================================
        # DEFAULT RISK VALUES
        # ====================================================

        model2_score = None

        model3_score = None

        risk_score = None

        risk_band = (
            "UNAVAILABLE"
        )

        fraud_prediction = None

        action = (
            "MANUAL_REVIEW"
        )


        # ====================================================
        # SHAP DEFAULT VALUES
        # ====================================================

        shap_explanations = None

        top_risk_factors = None

        top_safe_factors = None

        risk_explanation = None


        # ====================================================
        # RISK ENGINE
        # ====================================================

        try:

            # ------------------------------------------------
            # MODEL 2 INPUT
            # ------------------------------------------------

            model2_input = {

                "transaction_dt":
                    transaction_dt,

                "transaction_amt":
                    transaction_amount,

                "email":
                    email,

                "contact":
                    contact,

                "method":
                    method,
            }


            # ------------------------------------------------
            # MODEL 3 FEATURES
            # ------------------------------------------------

            model3_input = (
                build_model3_features(

                    db=db,

                    customer_key=
                        customer_key,

                    transaction_amt=
                        transaction_amount,

                    transaction_dt=
                        transaction_dt,
                )
            )


            # ------------------------------------------------
            # FUSION RISK
            # ------------------------------------------------

            risk_result = (
                calculate_fusion_risk(

                    model2_input,

                    model3_input,
                )
            )


            # ------------------------------------------------
            # MODEL SCORES
            # ------------------------------------------------

            model2_score = float(

                risk_result[
                    "model2_score"
                ]
            )


            model3_score = float(

                risk_result[
                    "model3_score"
                ]
            )


            risk_score = float(

                risk_result[
                    "risk_score"
                ]
            )


            # ------------------------------------------------
            # CLASSIFICATION
            # ------------------------------------------------

            risk_band = (
                risk_result[
                    "risk_band"
                ]
            )


            fraud_prediction = bool(

                risk_result[
                    "fraud_prediction"
                ]
            )


            action = (
                risk_result[
                    "decision"
                ]
            )


            # =================================================
            # SHAP DATA
            # =================================================

            shap_explanations = (
                risk_result.get(
                    "shap_explanations"
                )
            )


            top_risk_factors = (
                risk_result.get(
                    "top_risk_factors"
                )
            )


            top_safe_factors = (
                risk_result.get(
                    "top_safe_factors"
                )
            )


            risk_explanation = (
                risk_result.get(
                    "risk_explanation"
                )
            )


            print(
                "SHAP explanation generated "
                f"for {payment_id}"
            )


        except Exception as exc:

            print(
                "Risk model failed:",
                repr(exc),
            )


        # ====================================================
        # SAVE TRANSACTION
        # ====================================================

        transaction = Transaction(

            # ------------------------------------------------
            # IDENTIFICATION
            # ------------------------------------------------

            payment_id=
                payment_id,

            order_id=
                order_id,


            # ------------------------------------------------
            # PAYMENT
            # ------------------------------------------------

            amount=
                int(amount_minor),

            currency=
                currency,

            status=
                status,

            method=
                method,


            # ------------------------------------------------
            # CUSTOMER
            # ------------------------------------------------

            email=
                email,

            contact=
                contact,

            customer_key=
                customer_key,


            # ------------------------------------------------
            # TIMING
            # ------------------------------------------------

            transaction_dt=
                transaction_dt,


            # ------------------------------------------------
            # PAYMENT STATUS
            # ------------------------------------------------

            captured=
                captured,

            event_type=
                event_type,


            # ------------------------------------------------
            # AI SCORES
            # ------------------------------------------------

            model2_score=
                model2_score,

            model3_score=
                model3_score,

            risk_score=
                risk_score,

            risk_band=
                risk_band,

            fraud_prediction=
                fraud_prediction,

            action=
                action,


            # ------------------------------------------------
            # SHAP
            # ------------------------------------------------

            shap_explanations=
                shap_explanations,

            top_risk_factors=
                top_risk_factors,

            top_safe_factors=
                top_safe_factors,

            risk_explanation=
                risk_explanation,
        )


        # ====================================================
        # AUDIT TRAIL
        # ====================================================

        db.add(
            transaction
        )

        # Get the database-generated transaction ID
        # before committing the transaction and audit record.
        db.flush()

        AuditLogEntry = AuditLog(
            transaction_id=transaction.id,
            payment_id=payment_id,
            event_type=event_type,
            amount=int(amount_minor),
            currency=currency,
            model2_score=model2_score,
            model3_score=model3_score,
            risk_score=risk_score,
            risk_band=risk_band,
            fraud_prediction=fraud_prediction,
            decision=action,
            review_threshold=(float(risk_result.get("review_threshold")) if risk_result.get("review_threshold") is not None else None),
            block_threshold=(
                float(risk_result.get("threshold"))
                if 'risk_result' in locals()
                and risk_result.get("threshold") is not None
                else None
            ),
            model2_weight=(
                float(risk_result.get("model2_weight"))
                if 'risk_result' in locals()
                and risk_result.get("model2_weight") is not None
                else None
            ),
            model3_weight=(
                float(risk_result.get("model3_weight"))
                if 'risk_result' in locals()
                and risk_result.get("model3_weight") is not None
                else None
            ),
        )

        db.add(
            AuditLogEntry
        )

        # Transaction + audit record commit together.
        db.commit()

        db.refresh(
            transaction
        )

        # Link the paid transaction to the merchant order created before checkout.
        linked_order = (
            db.query(MerchantOrder)
            .filter(MerchantOrder.razorpay_order_id == order_id)
            .first()
        )
        if linked_order:
            linked_order.payment_id = payment_id
            linked_order.payment_status = status.upper() if status else event_type.upper()
            db.commit()


        # ====================================================
        # RESPONSE
        # ====================================================

        return {

            "status":
                "success",

            "payment_id":
                payment_id,

            "transaction_id":
                transaction.id,


            # AI

            "model2_score":
                model2_score,

            "model3_score":
                model3_score,

            "risk_score":
                risk_score,

            "risk_band":
                risk_band,

            "fraud_prediction":
                fraud_prediction,

            "action":
                action,


            # SHAP

            "shap_explanations":
                shap_explanations,

            "top_risk_factors":
                top_risk_factors,

            "top_safe_factors":
                top_safe_factors,

            "risk_explanation":
                risk_explanation,
        }


    except Exception as exc:

        db.rollback()

        print(
            "Transaction processing failed:",
            repr(exc),
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Transaction processing failed"
            ),
        )


    finally:

        db.close()


# ============================================================
# STARTUP
# ============================================================

print(
    "==================================="
)

print(
    "RiskGuard AI"
)

print(
    "Real-Time Fraud Risk Manager"
)

print(
    "==================================="
)

print(
    "SHAP transaction explanations: ENABLED"
)

print(
    "Single dashboard endpoint: ENABLED"
)

print(
    "Transaction detail data: INCLUDED IN DASHBOARD"
)

print(
    "STATUS: READY"
)

print(
    "==================================="
)
print("\nREGISTERED ROUTES")
for route in app.routes:
    print(route.path)





















