# Dynamic Pricing Engine for E-commerce

> Real-time ML-powered pricing engine with a FastAPI inference layer, Streamlit simulation dashboard, and full Docker + AWS EC2 deployment.

**Live Demo:** [https://dynamic-pricing.duckdns.org](https://dynamic-pricing.duckdns.org)  
**API Docs:** [https://dynamic-pricing.duckdns.org/api/docs](https://dynamic-pricing.duckdns.org/api/docs)

---

## What It Does

Takes a product's current signals — inventory level, competitor prices, demand velocity, time-of-day, customer segment — and returns a real-time price recommendation with confidence score and pricing breakdown.

The engine combines an ML baseline (Random Forest / XGBoost) with deterministic guardrails: competitor blending, inventory pressure, demand surge detection, flash sale multipliers, and floor/ceiling bounds. No black box — every price change is traceable.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Training | Scikit-learn, XGBoost, Pandas, NumPy |
| API | FastAPI, Uvicorn, Pydantic |
| Dashboard | Streamlit, Plotly |
| Containerization | Docker, Docker Compose |
| Deployment | AWS EC2 (Ubuntu 24.04), Nginx, Let's Encrypt |
| Optional | Redis (caching), Kafka (streaming) |

---

## Architecture

```
scripts/generate_sample_data.py   →  50,000+ synthetic SKU scenarios
scripts/train_model.py            →  Random Forest + XGBoost, best model saved
app/api.py                        →  FastAPI: real-time price recommendations
app/dashboard.py                  →  Streamlit: scenario builder + monitoring
app/pricing_engine.py             →  Core pricing logic with guardrails
app/feature_engineering.py        →  Feature pipeline shared by train + serve
app/streaming.py                  →  Kafka event producer/consumer (optional)
```

**Pricing logic flow:**

```
Raw signals → Feature engineering → ML baseline price
    → Competitor blend (70/30) → Inventory adjustment
    → Demand surge multiplier → Flash sale check
    → Floor/ceiling guardrails → Final recommended price
```

---

## Project Structure

```
dynamic-pricing-engine/
├── app/
│   ├── api.py                  # FastAPI app and route handlers
│   ├── config.py               # Environment config
│   ├── dashboard.py            # Streamlit dashboard
│   ├── feature_engineering.py  # Shared feature pipeline
│   ├── modeling.py             # Model loading and inference
│   ├── pricing_engine.py       # Core pricing guardrails
│   ├── schemas.py              # Pydantic request/response schemas
│   └── streaming.py            # Kafka helpers (optional)
├── data/
│   ├── processed/
│   └── raw/
├── deploy/
│   └── ec2/                    # systemd service files + startup scripts
├── models/                     # Saved model artifacts
├── scripts/
│   ├── generate_sample_data.py
│   └── train_model.py
├── tests/
│   └── smoke_test.py
├── docker-compose.yml
├── Dockerfile
├── Dockerfile.streamlit
└── requirements.txt
```

---

## Quick Start (Local)

### 1. Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Generate data and train

```bash
# Synthetic data
python scripts/generate_sample_data.py --rows 25000
python scripts/train_model.py

# Or train on Kaggle Retail Price Optimization dataset
# Download retail_price.csv → data/raw/kaggle/retail_price.csv
python scripts/train_model.py --profile kaggle_retail --data-path data/raw/kaggle/retail_price.csv
```

### 3. Run services

```bash
# API
uvicorn app.api:app --reload
# → http://127.0.0.1:8000/docs

# Dashboard (separate terminal)
streamlit run app/dashboard.py
# → http://127.0.0.1:8501
```

---

## Docker (Recommended)

```bash
docker compose up --build
```

| Service | URL |
|---------|-----|
| Streamlit Dashboard | http://localhost:8501 |
| FastAPI Docs | http://localhost:8000/docs |

`data/` and `models/` are volume-mounted so retrained artifacts persist outside containers.

---

## API Reference

### `POST /price/recommend`

Real-time price recommendation for the synthetic product profile.

**Example request:**

```json
{
  "sku_id": "SKU-1024",
  "category": "electronics",
  "brand": "brand_b",
  "customer_segment": "premium",
  "hour_of_day": 20,
  "day_of_week": 5,
  "is_weekend": 1,
  "is_festival": 1,
  "inventory_level": 23,
  "inventory_days_cover": 4.1,
  "competitor_price": 1849.0,
  "click_through_rate": 0.074,
  "conversion_rate": 0.038,
  "units_sold_last_5m": 7,
  "units_sold_last_1h": 33,
  "base_cost": 1210.0,
  "current_price": 1799.0
}
```

### Other endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service health and artifact status |
| POST | `/price/recommend` | Synthetic profile price recommendation |
| POST | `/price/recommend/kaggle` | Kaggle retail profile price recommendation |
| POST | `/events/order` | Register order event for flash sale detection |
| GET | `/monitoring/summary` | Pricing and event summary metrics |
| GET | `/metrics` | Prometheus-compatible metrics |

---

## AWS EC2 Deployment

Deployed on a `t2.micro` (Ubuntu 24.04) with Docker Compose + Nginx reverse proxy + Let's Encrypt SSL.

### Infrastructure

```
Internet → Nginx (port 443/80) → Docker containers
                                  ├── FastAPI  (port 8000)
                                  └── Streamlit (port 8501)
```

### Deploy from scratch

```bash
# On EC2 — install Docker
sudo apt update && sudo apt install -y docker.io docker-compose-v2
sudo usermod -aG docker $USER && newgrp docker

# Upload project (from local machine)
scp -i key.pem project.zip ubuntu@<ec2-ip>:/home/ubuntu/

# On EC2 — extract and run
sudo mkdir -p /opt/dynamic-pricing-engine
sudo chown ubuntu:ubuntu /opt/dynamic-pricing-engine
cd /opt/dynamic-pricing-engine
unzip ~/project.zip
docker compose up --build -d
```

### Security group inbound rules

| Port | Purpose | Source |
|------|---------|--------|
| 22 | SSH | Your IP only |
| 80 | HTTP (redirects to HTTPS) | 0.0.0.0/0 |
| 443 | HTTPS | 0.0.0.0/0 |

---

## Extension Roadmap

- [ ] SHAP explainability for per-prediction price breakdowns
- [ ] RL policy optimization for multi-step pricing strategies
- [ ] Flink / Spark Structured Streaming for real-time event windows
- [ ] Amazon Personalize integration for customer-level contextual pricing
- [ ] A/B testing framework for pricing strategy comparison
- [ ] Replace synthetic data with live marketplace telemetry

---

## Author

**Aishwarya Joshi** — AI/ML Engineer  
[GitHub](https://github.com/Aishwarya-J05) · [LinkedIn](https://linkedin.com/in/aishwaryajoshiaiml) · [Hugging Face](https://huggingface.co/AishwaryaNJ)
