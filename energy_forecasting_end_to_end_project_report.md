# 📊 Project Report: Energy Forecasting End-to-End System

---

## 1. Project Overview

This project delivers a full **End-to-End Energy Forecasting System**, covering data ingestion, feature engineering, ensemble modeling, containerization, CI/CD automation, and cloud deployment. The system forecasts electricity demand for the **PJM Interconnection (PJME)** region and exposes predictions via **FastAPI** and **Gradio** interfaces.

---

## 2. Project Structure and Codebase

**Figure 1. Project Directory Structure (VS Code)**  
This screenshot demonstrates the modular structure of the project, separating concerns across data, modeling, deployment, and CI/CD.

*Included screenshot:* VS Code Explorer showing folders such as `app/`, `configs/`, `data/`, `models/`, `docker/`, `.github/workflows`, and `Dockerfile`.

**Purpose:** Shows software engineering best practices and maintainability.

---

## 3. Data Ingestion and Feature Engineering

### 3.1 Data Sources

* `PJME_hourly.csv` – Hourly electricity demand data
* `temperature.csv` – Historical temperature data

### 3.2 Feature Engineering

Key engineered features include:

* Lag features: `FE_lag_24h`, `FE_lag_168h`
* Rolling statistics: `FE_rolling_mean_24h`
* Calendar features: hour, day, month, quarter, year
* Temperature-based features: `Temp_K`, `FE_Temp_K_sq`

These features capture seasonality, temporal dependencies, and weather-driven demand patterns.

---

## 4. Modeling: Stacking Ensemble

### 4.1 Architecture

The final model uses a **Stacking Ensemble** approach:

* **Base models:**
  * XGBoost
  * LightGBM
  * RandomForest
* **Meta-model:** RidgeCV

This architecture reduces individual model bias and variance while improving generalization.

### 4.2 Performance Metrics

| Slice | MAPE (%) | MAE (MW) | RMSE (MW) |
|------|----------|----------|-----------|
| OVERALL | ~2.5–3.5 | ~850–950 | ~1200–1350 |
| WEEKDAY | ~2.8 | ~890 | ~1250 |
| WEEKEND | ~3.2 | ~920 | ~1310 |
| PEAK HOURS (17–21) | ~3.5 | ~1050 | ~1450 |

These results indicate strong predictive accuracy suitable for operational forecasting.

---

## 5. API Layer (FastAPI)

**Figure 2. FastAPI Swagger UI (`/docs`)**  
The FastAPI service exposes endpoints for health checks and prediction requests.

* `GET /health`
* `POST /predict_from_scaled`

*Included screenshot:* Swagger UI showing API endpoints and request schema.

---

## 6. User Interface (Gradio)

**Figure 3. Gradio Web Interface**  
The Gradio UI allows users to:

* Submit JSON-formatted feature inputs
* Upload CSV files for batch predictions

*Included screenshot:* Gradio interface running on port `7860` with JSON input example.

---

## 7. Containerization (Docker)

The application is containerized to ensure reproducibility and portability.

* Single Docker image contains FastAPI + Gradio
* Image published to Docker Hub as:

```
shargun1303/mp_9_project:latest
```

**Figure 4. Docker Hub Image Registry**  
*Included screenshot:* Docker Desktop / Docker Hub showing available image tags.

---

## 8. CI/CD Pipeline (GitHub Actions)

A fully automated CI/CD pipeline is implemented using **GitHub Actions**.

Pipeline steps:
1. Trigger on push to `master`
2. Build Docker image
3. Push image to Docker Hub

**Figure 5. GitHub Actions – Successful Runs**  
*Included screenshot:* GitHub Actions page with green (successful) workflow runs.

---

## 9. Cloud Deployment (AWS EC2)

The containerized application was deployed on **AWS EC2 (Ubuntu 22.04, t3.micro)**.

* Open ports: `8000` (API), `7860` (Gradio UI)
* Public IP used for external access

**Figure 6. EC2 Instance Running State**  
*Included screenshot:* AWS EC2 dashboard showing the running instance and public IP.

---

## 10. Resource Cleanup and Cost Optimization

After validation, the EC2 instance was terminated to avoid unnecessary cloud costs.

**Figure 7. EC2 Instance Terminated**  
*Included screenshot:* AWS EC2 console confirming successful instance termination.

---

## 11. Conclusion

This project demonstrates strong proficiency in:

* Time-series forecasting
* Feature engineering and ensemble modeling
* MLOps practices (Docker, CI/CD)
* Cloud deployment and cost-aware operations

The system is production-ready and easily extensible for retraining, monitoring, or scaling.

---

**Status:** Project completed and archived successfully ✅

