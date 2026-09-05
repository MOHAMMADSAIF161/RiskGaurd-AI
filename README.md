# RiskGuard AI 🛡️

## Merchant Profit Protection Engine

RiskGuard AI is an AI-powered merchant decision system that helps businesses identify **orders with a higher probability of being returned** and, more importantly, prioritize the orders that could create the **largest expected financial loss**.

Instead of treating every potentially returned order equally, RiskGuard AI combines:

- Return probability
- Potential loss if the order is returned
- Customer historical behavior
- Order and product attributes
- Expected return loss
- SHAP-based model explanations
- Merchant-facing priority and action recommendations

The result is a practical workflow:

> **Predict → Estimate Financial Impact → Prioritize → Explain → Act**

---

## 🎯 Problem Statement

Product returns can create significant costs for merchants.

A merchant may know that some orders are more likely to be returned, but return probability alone is not enough.

For example:

- Order A has a 30% return probability but very little potential loss.
- Order B has a 20% return probability but a much higher potential loss.

If merchant review capacity is limited, Order B may deserve more attention.

RiskGuard AI therefore focuses on:

> **Expected Return Loss = Return Probability × Loss If Returned**

This converts an ML prediction into a business-oriented decision signal.

---

# 💡 Solution

RiskGuard AI uses a **Random Forest return-risk model** to estimate the probability that an order may be returned.

It then calculates the financial impact of that possible return.

### Processing pipeline

```text
Order + Customer Information
            │
            ▼
     Feature Engineering
            │
            ▼
   Return Risk Random Forest
            │
            ▼
     Return Probability
            │
            ▼
      Loss If Returned
            │
            ▼
    Expected Return Loss
            │
            ▼
    Priority Classification
            │
            ▼
 Merchant Recommendation
```

The system is designed so that the model predicts risk, while the business layer converts that prediction into an actionable merchant workflow.

---

# 🧠 Machine Learning Model

## Model

The deployed Return Risk model is:

**RandomForestClassifier**

Configuration:

```text
n_estimators      = 400
max_depth         = 8
min_samples_leaf  = 10
class_weight      = balanced
random_state      = 42
n_jobs             = -1
```

The trained model is stored in:

```text
models/ReturnRisk/return_risk_rf.joblib
```

The model configuration and evaluation metadata are stored in:

```text
models/ReturnRisk/return_risk_config.json
```

---

# 📊 Model Features

The model uses **17 logical features**.

### Numeric features

```text
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
```

### Categorical features

```text
category
payment_method
region
customer_gender
```

Categorical values are handled with one-hot encoding, while missing numeric and categorical values are handled through imputation.

Unknown categorical values are supported using:

```text
handle_unknown = "ignore"
```

---

# 👥 Customer Behavioral Features

RiskGuard AI does not only look at the current order.

When customer history is available, the system derives behavioral information from previous orders.

Examples include:

- Number of previous orders
- Previous returns
- Historical return rate
- Previous average order value
- Time since previous order
- Previous spending behavior

The historical features are generated from **past orders only** to avoid using future information during prediction.

This helps the system distinguish between:

```text
New / Unknown Customer
        vs.
Customer With Historical Behavior
```

The production application does not invent previous returns when verified return history is unavailable.

---

# 💰 Expected Return Loss

The most important business calculation is expected return loss.

### 1. Loss if returned

```text
Loss If Returned =
max(Profit Margin, 0) + Shipping Cost
```

The profit-margin component is prevented from becoming negative.

### 2. Expected return loss

```text
Expected Return Loss =
Return Probability × Loss If Returned
```

### Example

Suppose:

```text
Return Probability = 25%
Loss If Returned   = ₹500
```

Then:

```text
Expected Return Loss
= 0.25 × ₹500
= ₹125
```

This allows the merchant to compare orders using a common financial-impact measure.

---

# 🚦 Merchant Decision System

RiskGuard AI converts expected return loss into a simple merchant-facing recommendation.

The current decision flow is:

```text
Expected Loss >= Cutoff
        │
        ▼
   REVIEW_ORDER
```

If the expected loss is below the cutoff but still significant:

```text
0.5 × Cutoff <= Expected Loss < Cutoff
        │
        ▼
   MAKE_CONTACT
```

Otherwise:

```text
Expected Loss < 0.5 × Cutoff
        │
        ▼
       SAFE
```

### Merchant actions

| Priority / Action | Meaning |
|---|---|
| 🟢 LOW / SAFE | Lower expected return loss; process normally |
| 🟡 MEDIUM / MAKE CONTACT | Moderate expected loss; consider contacting the customer |
| 🔴 HIGH / REVIEW ORDER | Higher expected loss; give the order additional attention |

These recommendations are intended to **prioritize merchant attention**, not automatically reject customers.

---

# 📈 Verified Model Evaluation

The Return Risk model was evaluated using a chronological evaluation strategy.

The evaluation uses a time-based split so that later orders are evaluated using information that would have been available before those orders occurred.

### Verified test results

| Metric | Result |
|---|---:|
| Test Orders | 5,277 |
| ROC-AUC | 0.5809 |
| PR-AUC | 0.0673 |
| Precision | 5.79% |
| Recall | 10.20% |
| F1 Score | 7.39% |
| Loss Capture Rate | 42.48% |
| Loss Capture Lift vs Random | 4.24× |
| Test Review Workload | 10.14% |
| Expected-Loss Cutoff | ₹42.824297 |

### Business interpretation

The most important business metric is the **loss capture lift**.

At approximately a 10% review workload, the prioritization strategy captured about **4.24× the loss of a random 10% selection strategy** on the evaluated test set.

The frozen expected-loss cutoff is:

```text
₹42.824297
```

The model therefore focuses limited merchant review capacity on orders with higher expected financial impact.

---

# 🧪 Model Development & Evaluation

The complete model-development workflow is documented in:

```text
Model_Developemet&Evalution.ipynb
```

The notebook covers:

1. Dataset loading
2. Data preparation
3. Return-label creation
4. Customer history feature generation
5. Chronological splitting
6. Feature engineering
7. Model training
8. Model evaluation
9. Expected-loss calculation
10. Review prioritization
11. Business metric evaluation
12. Final model/configuration generation

The notebook is included to make the model-development process transparent and reproducible.

---

# 🔍 Explainability

RiskGuard AI uses **SHAP (SHapley Additive exPlanations)** to provide feature-level evidence for the model output.

For an individual order, the system can show:

- Which features increased predicted return risk
- Which features reduced predicted return risk
- Feature values
- SHAP contribution values

This makes the prediction more useful to a merchant because the system does not only say:

> "High Risk"

It can also provide evidence about **why the model reached that result**.

---

# 🖥️ Application Pages

The application provides a merchant-oriented web interface.

### Home

Introduces the RiskGuard AI product and explains the merchant protection workflow.

### Create Order

Allows merchant/order information to be entered and evaluated.

### Order Queue

Displays created merchant orders and helps prioritize them using:

- Return risk
- Expected return loss
- Payment status
- Merchant action

### Transaction Details

Provides detailed information for an individual order, including risk evidence and SHAP explanations.

### Dashboard

Provides an operational overview of stored application/payment activity.

### Model Performance

Displays verified Return Risk model performance and business metrics.

### Batch Evaluation

Provides a dry-run evaluation workflow for testing multiple inputs without creating normal transaction records.

---

# 💳 Razorpay Integration

RiskGuard AI also integrates with Razorpay for the payment lifecycle.

The payment workflow is:

```text
Merchant / Customer
       │
       ▼
RiskGuard AI
       │
       ▼
Create Razorpay Order
       │
       ▼
Razorpay Checkout
       │
       ▼
Customer Payment
       │
       ▼
Razorpay Signed Webhook
       │
       ▼
RiskGuard Backend
       │
       ▼
Validate Webhook
       │
       ▼
Persist Payment / Risk Information
```

The backend requires:

```text
RAZORPAY_KEY_ID
RAZORPAY_KEY_SECRET
RAZORPAY_WEBHOOK_SECRET
```

Webhook signature verification is performed server-side.

**Important:** Razorpay credentials and webhook secrets must never be committed to GitHub.

---

# 🗄️ Database

RiskGuard AI uses SQLAlchemy for persistence.

Production deployment can use PostgreSQL through:

```text
DATABASE_URL
```

The application also supports a local SQLite fallback when a production database URL is not configured.

Important application data includes:

- Merchant orders
- Transactions
- Risk results
- Merchant actions
- Risk policies
- Audit records

The database allows the application to maintain customer/order history and preserve decision context.

---

# 🔌 API Overview

The FastAPI backend exposes the application and API layer.

## Health

```http
GET /health
```

Checks whether the service is running.

## Return Risk

```http
GET /api/return-risk/config
```

Returns Return Risk configuration and expected-loss cutoff.

```http
POST /api/return-risk/score
```

Scores an order for return probability and expected return loss.

## Merchant Orders

```http
POST /api/orders
```

Creates and scores a merchant order.

```http
GET /api/orders
```

Returns merchant orders.

```http
GET /api/orders/{order_id}
```

Returns an individual order.

```http
POST /api/orders/{order_id}/action
```

Updates the merchant action.

```http
GET /api/orders/{order_id}/shap
```

Returns SHAP explanation data for an order.

## Payment

```http
POST /create-order
```

Creates a Razorpay order.

```http
POST /webhooks/razorpay
```

Receives signed Razorpay webhook events.

## Dashboard / Transactions

```http
GET /api/dashboard
GET /api/transactions
GET /api/transactions/{transaction_id}
GET /api/audit-log
```

## Evaluation

```http
POST /api/batch-evaluate
POST /simulate-payment
```

These endpoints support controlled evaluation and simulation.

---

# 🏗️ Technology Stack

### Backend

- Python
- FastAPI
- Uvicorn
- Pydantic
- SQLAlchemy

### Machine Learning

- scikit-learn
- Random Forest
- pandas
- NumPy
- joblib

### Explainability

- SHAP

### Payment

- Razorpay API
- Razorpay Webhooks

### Database

- PostgreSQL / Neon
- SQLite fallback
- SQLAlchemy ORM

### Frontend

- HTML
- CSS
- JavaScript

### Deployment

- GitHub
- Render
- Neon PostgreSQL

---

# 📁 Project Structure

```text
RiskGaurd-AI/
│
├── backend/
│   ├── database.py
│   ├── decision_engine.py
│   ├── feature_engine.py
│   ├── main.py
│   ├── model2_adapter.py
│   ├── rag_explainer.py
│   ├── razorpay_test.py
│   ├── return_risk_engine.py
│   ├── risk_engine.py
│   ├── risk_policy.py
│   └── shap_explainer.py
│
├── models/
│   ├── Fusion_v1.0/
│   ├── Model2/
│   └── ReturnRisk/
│       ├── return_risk_config.json
│       └── return_risk_rf.joblib
│
├── static/
│   ├── style.css
│   └── model2_evaluation_verified.json
│
├── templates/
│   ├── index.html
│   ├── create-order.html
│   ├── return-risk.html
│   ├── transactions.html
│   ├── transaction-details.html
│   ├── dashboard.html
│   ├── model-performance.html
│   ├── batch-evaluation.html
│   └── payment.html
│
├── Model_Developemet&Evalution.ipynb
├── migrate_db.py
├── requirements.txt
├── .gitignore
└── README.md
```

### Important files

**`backend/main.py`**

Main FastAPI application, routes, payment integration, merchant-order APIs and application orchestration.

**`backend/return_risk_engine.py`**

Loads the deployed Return Risk model, creates predictions, calculates expected loss and generates SHAP evidence.

**`backend/feature_engine.py`**

Builds customer-history/behavioral features used by the application.

**`backend/database.py`**

Defines database models and database connection behavior.

**`models/ReturnRisk/return_risk_rf.joblib`**

Trained Return Risk Random Forest model.

**`models/ReturnRisk/return_risk_config.json`**

Frozen model configuration, cutoff and evaluation metadata.

**`Model_Developemet&Evalution.ipynb`**

Model training and evaluation evidence.

---

# ⚙️ Local Installation

## 1. Clone the repository

```powershell
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd RiskGaurd-AI
```

## 2. Create a virtual environment

```powershell
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🔐 Environment Variables

Create a `.env` file in the project root.

Example:

```env
DATABASE_URL=your_database_connection_string

RAZORPAY_KEY_ID=your_razorpay_key_id
RAZORPAY_KEY_SECRET=your_razorpay_key_secret
RAZORPAY_WEBHOOK_SECRET=your_razorpay_webhook_secret
```

### Never commit secrets

Do not commit:

```text
.env
```

Do not put:

- Razorpay secret keys
- Database passwords
- Webhook secrets
- Private credentials

inside source code.

For production deployment, configure secrets using the hosting provider's environment-variable settings.

---

# ▶️ Run Locally

From the project root:

```powershell
uvicorn backend.main:app --reload
```

The application will normally be available at:

```text
http://127.0.0.1:8000
```

Health check:

```text
http://127.0.0.1:8000/health
```

FastAPI documentation:

```text
http://127.0.0.1:8000/docs
```

---

# 🚀 Production Deployment

The application is structured to run as a FastAPI service.

Production command:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port $PORT
```

Typical architecture:

```text
GitHub
   │
   ▼
Render
   │
   ▼
FastAPI / Uvicorn
   │
   ├── RiskGuard AI Models
   ├── Razorpay Integration
   │
   ▼
Neon PostgreSQL
   │
   ├── Orders
   ├── Transactions
   └── Audit Data
```

---

# 🛡️ Security Considerations

RiskGuard AI follows several important security practices:

### Environment-based secrets

Sensitive credentials are loaded through environment variables.

### Razorpay webhook verification

Webhook signatures are verified server-side before trusted payment processing.

### Separation of simulation and real payment flow

Simulation/evaluation functionality is kept separate from real payment persistence.

### Auditability

Important decision and policy information can be preserved for traceability.

### No secret credentials in source control

`.env` should remain local and should be included in `.gitignore`.

---

# 🧪 Evaluation Philosophy

RiskGuard AI evaluates the model from two perspectives.

## 1. ML performance

Traditional classification metrics:

```text
ROC-AUC
PR-AUC
Precision
Recall
F1
```

## 2. Merchant/business performance

Business-oriented metrics:

```text
Expected Return Loss
Loss Capture Rate
Loss Capture Lift
Review Workload
```

This distinction is important because a model can have reasonable classification performance while still being less useful for a merchant.

RiskGuard AI therefore evaluates whether the model helps answer:

> **"If I can only review a limited number of orders, which orders should I look at first?"**

---

# 📌 Responsible Interpretation of Results

The reported Return Risk metrics are from the documented model-development/evaluation pipeline.

They should not be interpreted as proof that the model will produce the same performance on every merchant's production data.

Performance can change because of:

- Merchant-specific return behavior
- Product categories
- Geography
- Customer demographics
- Shipping policies
- Return policies
- Data quality
- Dataset shift

The current evaluation should therefore be understood as **validated model-development evidence**, not a guarantee of production performance.

---

# ⚠️ Limitations

### Return labels

The model requires historical order/return data for training and evaluation.

### Cold-start customers

A customer with no previous order history has limited behavioral information.

### Dataset dependence

The verified Return Risk model was evaluated on a synthetic e-commerce transaction dataset. Real merchant data may behave differently.

### Model calibration

The output is primarily used for ranking/prioritization and business decisioning. Additional calibration may be appropriate before high-stakes production use.

### Business assumptions

The expected-loss calculation uses:

```text
max(profit_margin, 0) + shipping_cost
```

as the loss proxy when an order is returned.

Actual merchant return cost may include additional factors such as:

- Reverse logistics
- Restocking
- Packaging
- Refund processing
- Customer support
- Inventory depreciation

These can be incorporated in future versions if reliable merchant-specific data is available.

---

# 🔮 Future Improvements

Potential future improvements include:

- Merchant-specific model training
- Better probability calibration
- More detailed return-cost modeling
- Reverse-logistics cost integration
- Real return-outcome feedback loops
- Model drift monitoring
- Automated retraining
- More granular customer behavior features
- Category-specific return models
- Merchant-configurable expected-loss policies
- A/B testing of merchant intervention strategies

---

# 🏆 Why RiskGuard AI?

Traditional return prediction asks:

> **"Will this order be returned?"**

RiskGuard AI asks a more useful merchant question:

> **"Which orders could create the greatest financial impact if returned, and where should the merchant focus attention?"**

That shift turns a raw ML prediction into an operational decision-support system.

---

# 🔄 End-to-End Example

Suppose a merchant receives an order:

```text
Price              = ₹2,000
Quantity           = 1
Shipping Cost      = ₹100
Profit Margin      = ₹300
Return Probability = 30%
```

The system calculates:

```text
Loss If Returned
= ₹300 + ₹100
= ₹400
```

Then:

```text
Expected Return Loss
= 0.30 × ₹400
= ₹120
```

The system compares ₹120 with the configured expected-loss cutoff and assigns an appropriate priority/action.

The merchant can then:

```text
View Order
   ↓
See Return Risk
   ↓
See Expected Loss
   ↓
See SHAP Evidence
   ↓
Take Recommended Action
```

---

# 🧭 Product Workflow

```text
                  ┌──────────────────────┐
                  │     Merchant Order   │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Feature Engineering  │
                  │ Order + Customer     │
                  │ Historical Behavior   │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Return Risk Random   │
                  │ Forest Model         │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Return Probability   │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Loss If Returned     │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Expected Return Loss │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Priority             │
                  │ LOW / MEDIUM / HIGH  │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Merchant Action      │
                  │ SAFE / CONTACT /     │
                  │ REVIEW ORDER         │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ SHAP Explanation     │
                  └──────────────────────┘
```

---

# 📚 Documentation & Reproducibility

The repository contains the model-development notebook and model configuration artifacts so that the evaluation process can be inspected independently from the production application.

Key evidence files:

```text
Model_Developemet&Evalution.ipynb
models/ReturnRisk/return_risk_config.json
models/ReturnRisk/return_risk_rf.joblib
```

The notebook provides the development/evaluation workflow, while the configuration file stores the frozen production evaluation values and expected-loss cutoff.

---

# 👨‍💻 Project

**RiskGuard AI**

**Focus:** AI-powered merchant return-risk and expected-loss prioritization.

**Primary goal:** Help merchants spend limited review capacity on orders with greater potential financial impact.

---

## License

Add the project's chosen license here before publishing if required.

---

## Disclaimer

RiskGuard AI is a decision-support system. Risk scores and recommendations should be used as inputs to merchant review and operational decision-making rather than as an automatic guarantee of customer behavior or financial outcome.
