import pandas as pd
from sqlalchemy.orm import Session
from backend.database import Transaction

MODEL3_FEATURES = [
    "TransactionAmt", "previous_transactions", "previous_avg_amount",
    "amount_deviation", "transactions_last_5min", "transactions_last_10min",
    "transactions_last_30min", "transactions_last_1h", "transactions_last_24h",
    "amount_last_10min", "amount_last_1h", "amount_last_24h",
    "time_since_previous", "rapid_transaction_flag", "high_velocity_flag",
    "large_amount_deviation_flag", "TransactionHour", "TransactionDay",
]


def build_model3_features(db: Session, customer_key: str, transaction_amt: float, transaction_dt: int) -> pd.DataFrame:
    transaction_amt = float(transaction_amt)
    transaction_dt = int(transaction_dt)
    if transaction_amt < 0:
        raise ValueError("transaction_amt must be non-negative")

    previous = (
        db.query(Transaction)
        .filter(
            Transaction.customer_key == customer_key,
            Transaction.transaction_dt.isnot(None),
            Transaction.transaction_dt < transaction_dt,
        )
        .order_by(Transaction.transaction_dt.asc())
        .all()
    )

    amounts = [float(t.amount) / 100.0 for t in previous if t.amount is not None]
    count = len(amounts)
    avg = sum(amounts) / count if count else 0.0

    if previous:
        last_dt = int(previous[-1].transaction_dt)
        time_since = max(transaction_dt - last_dt, 0)
    else:
        # Cold-start: there is no previous event, so don't pretend it happened
        # 0 seconds ago. This avoids falsely setting rapid_transaction_flag.
        time_since = 86400.0

    # No history means there is no valid baseline for amount deviation.
    # Use a neutral value rather than the previous 100x artificial deviation.
    amount_deviation = (transaction_amt / avg) if avg > 0 else 1.0
    amount_deviation = min(max(amount_deviation, 0.0), 100.0)

    def tx_count(seconds):
        cutoff = transaction_dt - seconds
        return sum(1 for t in previous if t.transaction_dt >= cutoff)

    def amount_sum(seconds):
        cutoff = transaction_dt - seconds
        return sum(float(t.amount) / 100.0 for t in previous
                   if t.amount is not None and t.transaction_dt >= cutoff)

    data = {
        "TransactionAmt": transaction_amt,
        "previous_transactions": count,
        "previous_avg_amount": avg,
        "amount_deviation": amount_deviation,
        "transactions_last_5min": tx_count(300),
        "transactions_last_10min": tx_count(600),
        "transactions_last_30min": tx_count(1800),
        "transactions_last_1h": tx_count(3600),
        "transactions_last_24h": tx_count(86400),
        "amount_last_10min": amount_sum(600),
        "amount_last_1h": amount_sum(3600),
        "amount_last_24h": amount_sum(86400),
        "time_since_previous": time_since,
        "rapid_transaction_flag": int(time_since <= 60),
        "high_velocity_flag": int(tx_count(300) >= 3),
        "large_amount_deviation_flag": int(count > 0 and amount_deviation >= 3),
        "TransactionHour": (transaction_dt // 3600) % 24,
        "TransactionDay": transaction_dt // 86400,
    }

    features = pd.DataFrame([[data[c] for c in MODEL3_FEATURES]], columns=MODEL3_FEATURES)
    if features.shape != (1, 18) or features.isna().any().any():
        raise RuntimeError("Invalid Model 3 feature frame")
    return features
