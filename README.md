# Credit Risk Profiling Engine

A machine learning web application that classifies loan applicants by credit risk and generates a **CIBIL-style credit score (300–900)** with a human-readable rating — built end-to-end with Python, Scikit-learn, and Streamlit.

---

## Live Demo

> Deploy your own or run locally (see below)

---

## What It Does

Given a loan applicant's financial and demographic details, the app:

1. Predicts the **probability of default**
2. Maps that probability to a **credit score** on the 300–900 scale
3. Assigns a **risk rating**: Poor / Average / Good / Excellent

The scoring logic goes beyond standard `.predict_proba()` — it directly applies logistic regression coefficients to derive a scaled, interpretable score, mirroring how real credit bureaus compute scores.

---

## Tech Stack

| Layer | Tools |
|---|---|
| ML Model | Logistic Regression (Scikit-learn) |
| Data Processing | Pandas, NumPy |
| Model Persistence | Joblib |
| Frontend | Streamlit |
| Environment | Python 3.10+ |

---

## Input Features

The model takes 11 inputs:

| Feature | Type |
|---|---|
| Age | Numerical |
| Income | Numerical |
| Loan Amount | Numerical |
| Loan Tenure (months) | Numerical |
| Average DPD per Delinquency | Numerical |
| Delinquency Ratio | Numerical |
| Credit Utilisation Ratio | Numerical |
| Number of Open Accounts | Numerical |
| Residence Type | Categorical (Owned / Rented / Mortgage) |
| Loan Purpose | Categorical (Education / Auto / Home / Personal) |
| Loan Type | Categorical (Secured / Unsecured) |

---

## Credit Score Logic

```python
# Custom scoring — not just predict_proba()
x = np.dot(input_df.values, model.coef_.T) + model.intercept_
default_probability = 1 / (1 + np.exp(-x))
non_default_probability = 1 - default_probability
credit_score = 300 + non_default_probability * 600
```

| Score Range | Rating |
|---|---|
| 300 – 499 | Poor |
| 500 – 649 | Average |
| 650 – 749 | Good |
| 750 – 900 | Excellent |

---

## Project Structure

```
ml-project-credit-risk-model/
│
├── main.py                  # Streamlit app — UI and input handling
├── prediction_helper.py     # Feature engineering, scaling, scoring logic
├── artifacts/
│   └── model_data.joblib    # Serialised model, scaler, and feature list
├── requirements.txt
└── README.md
```

---

## Run Locally

```bash
# Clone the repo
git clone https://github.com/MainHuAj/ml-project-credit-risk-model.git
cd ml-project-credit-risk-model

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run main.py
```

---

## Key ML Concepts Applied

- Logistic Regression for binary classification
- Feature scaling with MinMaxScaler
- One-hot encoding for categorical variables
- Class imbalance handling
- Model serialisation with Joblib
- Custom probability-to-score mapping

---

## Author

**Abhinav Bhatera**
[LinkedIn](https://www.linkedin.com/in/abhinav-bhatera) · [GitHub](https://github.com/MainHuAj)
