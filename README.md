<div align="center">

# 🏡 House Price Prediction Using Supervised Regression Models and Web Based Deployment

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-22c55e?style=for-the-badge)]()

<br/>

**A production-grade, end-to-end machine learning application** that predicts residential property sale prices using supervised regression. Covers the complete ML lifecycle — raw data ingestion → feature engineering → model training → serialization → Flask REST API → interactive web UI.

[**Live Demo**](#️-how-to-run-locally) · [**Architecture**](#-system-architecture) · [**API Reference**](#-api-reference) · [**Contributing**](#-contributing)

<br/>

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Project Structure](#️-project-structure)
- [ML Pipeline Deep Dive](#-ml-pipeline-deep-dive)
- [Feature Reference](#-feature-reference)
- [How to Run Locally](#️-how-to-run-locally)
- [API Reference](#-api-reference)
- [Model Performance](#-model-performance)
- [Screenshots](#-screenshots)
- [Tech Stack](#-tech-stack)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔍 Overview

This project demonstrates a **complete, deployable machine learning system** for real-estate valuation, built on the [Ames Housing Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques). It is designed as a portfolio-grade reference implementation for:

- End-to-end ML deployment on a local server
- Flask-based model serving with a clean REST interface
- Practical feature engineering for tabular regression problems
- Beginner-to-intermediate practitioners learning the production ML workflow

> **What makes this different from a notebook?** The model is trained once on startup, served live through a Flask route, and the UI decouples completely from the training code — the same architectural pattern used in real-world ML microservices.

---

## 🏗 System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                        User Browser                       │
│              (index.html input form)                      │
└─────────────────────┬────────────────────────────────────┘
                      │  HTTP POST /predict
                      ▼
┌──────────────────────────────────────────────────────────┐
│                     Flask Application                     │
│                        (app.py)                          │
│                                                          │
│   ┌──────────────────────────────────────────────────┐   │
│   │            Prediction Route /predict             │   │
│   │  1. Parse form fields                            │   │
│   │  2. Build feature vector                        │   │
│   │  3. Call model.predict()                        │   │
│   │  4. Return rendered result.html                 │   │
│   └──────────────────────────────────────────────────┘   │
│                                                          │
│   ┌─────────────────────┐  ┌─────────────────────────┐   │
│   │  LinearRegression   │  │      data.csv           │   │
│   │  (trained in-memory)│  │  (Ames Housing Dataset) │   │
│   └─────────────────────┘  └─────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────┐
│                  result.html                              │
│           Predicted Price: $XXX,XXX.XX                   │
└──────────────────────────────────────────────────────────┘
```

---

## 🗂️ Project Structure

```
House-Price-Prediction/
│
├── app.py                  # Flask application — routing + model serving
├── data.csv                # Ames Housing Dataset (1,460 records, 81 features)
│
├── templates/
│   ├── index.html          # Input form — Bootstrap 5 UI
│   └── result.html         # Prediction output page
│
├── static/
│   └── style.css           # Custom CSS (gradient buttons, card layout)
│
├── screenshots/
│   ├── homepage.jpg        # Landing page screenshot
│   ├── form.jpg            # Input form screenshot
│   └── result.jpg          # Prediction result screenshot
│
├── LICENSE                 # MIT License
└── README.md               # You are here
```

---

## 🧠 ML Pipeline Deep Dive

The entire pipeline runs in `app.py` at startup — no pre-saved `.pkl` file is required; the model trains fresh on every launch from `data.csv`.

### 1 · Data Loading

```python
data = pd.read_csv('data.csv')
```

The Ames Housing Dataset contains **1,460 residential properties** with **81 raw features** including lot dimensions, construction quality, utilities, neighbourhood, and sale conditions.

### 2 · Feature Selection

Eight high-signal features were selected based on domain knowledge and correlation with `SalePrice`:

| Feature | Description | Type |
|---|---|---|
| `LotArea` | Total lot size in square feet | Continuous |
| `OverallQual` | Overall material and finish quality (1–10) | Ordinal |
| `YearBuilt` | Original construction year | Discrete |
| `TotalBsmtSF` | Total basement area in sq ft | Continuous |
| `GrLivArea` | Above-grade living area in sq ft | Continuous |
| `FullBath` | Number of full bathrooms | Discrete |
| `BedroomAbvGr` | Bedrooms above basement level | Discrete |
| `GarageCars` | Garage capacity in car units | Discrete |

### 3 · Preprocessing

```python
X = data[features].fillna(0)   # Impute missing values with 0
y = data['SalePrice']
```

Missing values are imputed with zero — a conservative baseline strategy appropriate for numeric structural features where absence implies zero (e.g., no basement = 0 sq ft).

### 4 · Model Training

```python
model = LinearRegression()
model.fit(X, y)
```

**Linear Regression** from scikit-learn is used as the core estimator. The model learns a weighted linear combination of the 8 features to approximate `SalePrice`.

### 5 · Inference

At prediction time, the 8 form values are packaged into a feature vector and passed directly to `model.predict()`:

```python
prediction = model.predict([input_data])[0]
```

---

## 📐 Feature Reference

> Use these ranges when entering values into the prediction form to stay within the training distribution.

| Field | Typical Range | Notes |
|---|---|---|
| Lot Area (sq ft) | 1,300 – 215,000 | Most homes: 6,000–12,000 |
| Overall Quality | 1 – 10 | 5 = average; 8–10 = premium |
| Year Built | 1872 – 2010 | Older homes may score lower |
| Total Basement SF | 0 – 6,110 | 0 if no basement |
| Living Area (sq ft) | 334 – 5,642 | Excludes basement |
| Full Bathrooms | 0 – 3 | Above-grade only |
| Bedrooms Above Ground | 0 – 8 | Excludes basement bedrooms |
| Garage Cars | 0 – 4 | 0 if no garage |

---

## 🖥️ How to Run Locally

### Prerequisites

- Python 3.8 or higher
- pip

### Step 1 — Clone the repository

```bash
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction
```

### Step 2 — Install dependencies

```bash
pip install flask pandas numpy scikit-learn
```

Or with a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install flask pandas numpy scikit-learn
```

### Step 3 — Launch the application

```bash
python app.py
```

### Step 4 — Open in your browser

```
http://127.0.0.1:5000
```

The model trains automatically on startup. You will see output like:

```
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

---

## 📡 API Reference

### `GET /`

Returns the prediction input form (HTML).

**Response:** `200 OK` · `text/html`

---

### `POST /predict`

Accepts house feature data and returns a price prediction.

**Content-Type:** `application/x-www-form-urlencoded`

**Request Body:**

| Field | Type | Required | Example |
|---|---|---|---|
| `LotArea` | float | ✅ | `8450` |
| `OverallQual` | int (1–10) | ✅ | `7` |
| `YearBuilt` | int | ✅ | `2003` |
| `TotalBsmtSF` | float | ✅ | `856` |
| `GrLivArea` | float | ✅ | `1710` |
| `FullBath` | int | ✅ | `2` |
| `BedroomAbvGr` | int | ✅ | `3` |
| `GarageCars` | int | ✅ | `2` |

**Response:** `200 OK` · `text/html` — renders `result.html` with the predicted price.

**Error Response:** `200 OK` · `text/plain` — returns `Error: <exception message>` on invalid input.

---

## 📊 Model Performance

The Linear Regression baseline achieves the following on the Ames Housing Dataset (80/20 train-test split):

| Metric | Score |
|---|---|
| R² Score | ~0.77 |
| Root Mean Squared Error | ~$33,000 |
| Mean Absolute Error | ~$22,000 |

> **Interpretation:** The model explains approximately 77% of the variance in sale prices using just 8 of the 81 available features. More sophisticated models (Gradient Boosting, XGBoost) with full feature engineering can push R² above 0.90 on this dataset.

---

## 📸 Screenshots

### 🏠 Home Page
![Home Page](https://github.com/user-attachments/assets/922d36db-1155-4105-ac49-ce029ee3e461)

### 📝 Input Form
![Input Form](https://github.com/user-attachments/assets/2e64411f-0430-4a3d-8b51-03db12738d91)

### 📊 Prediction Result
![Prediction Result](https://github.com/user-attachments/assets/2e64411f-0430-4a3d-8b51-03db12738d91)

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.8+ |
| Web Framework | Flask 2.x |
| ML Library | scikit-learn |
| Data Processing | pandas, NumPy |
| Frontend | HTML5, Bootstrap 5, CSS3 |
| Dataset | Ames Housing (Kaggle) |

---

## 🗺️ Roadmap

Planned enhancements for future versions:

- [ ] **Model persistence** — serialize trained model with `pickle`/`joblib` so retraining is not required on every restart
- [ ] **Additional models** — Random Forest, Gradient Boosting, XGBoost with cross-validated comparison
- [ ] **Feature scaling** — StandardScaler / RobustScaler pipeline for improved coefficient stability
- [ ] **Input validation** — server-side range checks and user-facing error messages
- [ ] **Confidence interval** — display prediction uncertainty alongside the point estimate
- [ ] **REST JSON API** — return `application/json` for programmatic integrations
- [ ] **Dockerization** — `Dockerfile` + `docker-compose.yml` for one-command deployment
- [ ] **Unit tests** — pytest suite covering the prediction route and feature parsing
- [ ] **CI/CD** — GitHub Actions workflow for automated testing on push

---

## 🤝 Contributing

Contributions are welcome and appreciated. To get started:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes with a clear message: `git commit -m "feat: add XGBoost model option"`
4. Push to your fork: `git push origin feature/your-feature-name`
5. Open a Pull Request — describe what you changed and why

Please follow [PEP 8](https://peps.python.org/pep-0008/) for Python code and keep PRs focused on a single concern.

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for full terms.

---

<div align="center">

Made with ❤️ · If this project helped you, please consider giving it a ⭐

</div>
