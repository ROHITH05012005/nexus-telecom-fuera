# Telecom Churn Prediction & Retention System

An end-to-end Machine Learning pipeline and dashboard designed to identify at-risk customers, predict churn probability using XGBoost, and provide actionable retention strategies with SHAP explainability.

## 🌟 Key Features

- **High-Accuracy Churn Prediction**: Utilizes an XGBoost classifier with automated data generation and rigorous feature engineering.
- **Explainable AI (XAI)**: Leverages SHAP to generate transparent, feature-level explanations for every prediction, answering *why* a customer is likely to churn.
- **Actionable Retention Strategies**: Suggests targeted retention actions and calculates expected ROI based on the customer's risk profile.
- **Interactive Dashboard**: A responsive Flask web application for real-time monitoring of churn metrics, high-risk customers, and feature contributions.
- **Data Drift & Performance Monitoring**: Built-in endpoints to track data variation over time and identify drops in model performance.
- **Batch Processing & Trigger Alerts**: Endpoints for bulk predictions and automated alerts when risk thresholds are exceeded.

## 🏗️ Architecture Stack

- **Backend / API**: Python, Flask, Gunicorn
- **Machine Learning**: `scikit-learn`, `xgboost`, `shap`
- **Data Persistence**: `tinydb` for structured JSON storage (`at_risk_customers.json`)
- **Containerization**: Docker & Docker Compose

## 🚀 Quick Start (Local Setup)

### Option 1: Using Docker (Recommended)

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd "ML PROJECT"
   ```

2. **Run with Docker Compose**:
   ```bash
   docker-compose up --build -d
   ```

3. **Access the Application**:
   Navigate to [http://localhost:5000](http://localhost:5000)

### Option 2: Using Python Virtual Environment

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate data and train the initial model**:
   ```bash
   python data_generator.py
   python train_and_explain.py
   ```

3. **Start the Flask server**:
   ```bash
   python app.py
   ```
4. **Access the Application**:
   Navigate to [http://localhost:5000](http://localhost:5000)

## 📡 API Endpoints Summary

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/api/customers` | Fetch customer data, filterable by risk level. |
| `GET`  | `/api/statistics` | Dashboard summary metrics (total at-risk, avg probability). |
| `GET`  | `/api/customer/<id>`| Detailed SHAP values, ROI, and timeline for a customer. |
| `POST` | `/api/predict` | Predict churn for a highly-configurable scenario payload. |
| `POST` | `/api/predict/batch`| Send JSON arrays of customers for bulk prediction. |
| `POST` | `/api/predict/excel`| Bulk upload predictions using a CSV/XLSX file format. |
| `POST` | `/api/drift` | Check data drift vs the training dataset. |
| `GET`  | `/api/monitoring/performance` | Retrieve historical model accuracy and degradation alerts. |

*Note: For endpoints marked with `require_api_key` in the code, supply your token either via the `X-API-Key` header or the `?api_key=` URL parameter.*

## 🐳 Deployment

This app is natively containerized. To deploy to platforms like **Render**, **Heroku**, or **Hugging Face Spaces**:
1. Connect your GitHub repository to your provider.
2. Select **Docker** as the runtime environment.
3. Define your internal API port (e.g. `5000`).
4. (Optional) Set the `API_KEY` and `AUTH_ENABLED` environment variables in your deployment settings.

## 📜 License
This project is for educational or internal enterprise purposes. See individual dependencies for their licensing.
