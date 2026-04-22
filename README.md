# Dynamic Pricing Engine for E-commerce Website

An end-to-end intermediate MVP for real-time dynamic pricing with:

- Synthetic data generation for 50,000+ SKU style scenarios
- Random Forest and XGBoost training pipelines
- FastAPI inference API for real-time price recommendations across synthetic and Kaggle retail model profiles
- Streamlit dashboard for simulation and monitoring
- Competitor price blending, inventory adjustments, and flash sale detection
- Optional Redis and Kafka integrations

## Architecture

1. `scripts/generate_sample_data.py` creates synthetic click, order, inventory, and competitor signals.
2. `scripts/train_model.py` trains a baseline Random Forest and an XGBoost regressor, then saves the best model.
3. `app/api.py` serves real-time pricing recommendations through FastAPI.
4. `app/dashboard.py` provides a Streamlit control panel for scenario analysis and monitoring.
5. `app/streaming.py` provides optional Kafka event producer and consumer helpers.

## Project Structure

```text
app/
  api.py
  config.py
  dashboard.py
  feature_engineering.py
  modeling.py
  pricing_engine.py
  schemas.py
  streaming.py
data/
  processed/
  raw/
models/
scripts/
  generate_sample_data.py
  train_model.py
tests/
  smoke_test.py
```

## Quick Start

### 1. Create a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Generate sample data

```powershell
python scripts/generate_sample_data.py --rows 25000
```

### 3. Train the model

```powershell
python scripts/train_model.py
```

### Train on a real Kaggle dataset

Recommended dataset:

- Kaggle `Retail Price Optimization`: https://www.kaggle.com/datasets/suddharshan/retail-price-optimization

Download `retail_price.csv` and place it at:

```text
data/raw/kaggle/retail_price.csv
```

Then train with:

```powershell
python scripts/train_model.py --profile kaggle_retail --data-path data/raw/kaggle/retail_price.csv
```

Notes:

- This trains a real-dataset pricing model using the Kaggle retail schema.
- The FastAPI service now exposes a dedicated Kaggle request schema at `/price/recommend/kaggle` when a `kaggle_retail` model is loaded.

### 4. Run the FastAPI service

```powershell
uvicorn app.api:app --reload
```

Docs will be available at `http://127.0.0.1:8000/docs`.

### 5. Run the Streamlit dashboard

```powershell
streamlit run app/dashboard.py
```

## API Endpoints

- `GET /health`: Service health and artifact status
- `POST /price/recommend`: Returns a recommended price, confidence score, and pricing signals for the synthetic profile
- `POST /price/recommend/kaggle`: Returns a recommended price and price gap analysis for the Kaggle retail profile
- `POST /events/order`: Registers an order event for flash sale detection
- `GET /monitoring/summary`: Returns pricing and event summary metrics

## Core Pricing Logic

The engine combines:

- ML-predicted baseline price
- Competitor blending: default `70% model / 30% competitor`
- Inventory pressure adjustments
- Demand surge adjustments
- Flash sale emergency multiplier
- Floor and ceiling guardrails

## Example Request

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

## AWS Deployment Notes

- EC2: Host FastAPI and Streamlit on one instance for a demo, or split them later if needed.
- S3: Good fit for model artifacts and price history snapshots.
- Lambda: Optional competitor polling and scheduled retraining.
- CloudWatch: Use for API logs, dashboard process logs, and alert thresholds.
- Redis: Cache recent recommendations and event counters.
- Kafka: Use self-managed Kafka for demos, or Amazon MSK only after checking cost.

Current AWS note:

- AWS Free Tier changed on July 15, 2025. Official AWS docs say accounts created before July 15, 2025 can use `t2.micro` free tier eligibility in supported regions, while accounts created on or after July 15, 2025 use a newer credit-based/free-plan model with eligible instance types such as `t3.micro`. Verify your account type before launch.
- Amazon S3 is covered through the newer AWS Free Tier credit/free-plan model, not a simple permanent 5 GB rule for all new accounts.
- Amazon MSK pricing is pay-as-you-go on the official AWS pricing page. Treat MSK as a potential paid service unless your account credits cover it.

## Docker Deployment

Build and run the full stack locally or on a VM:

```powershell
docker compose up --build
```

Then open:

- API docs: `http://127.0.0.1:8000/docs`
- Streamlit dashboard: `http://127.0.0.1:8501`

Notes:

- Compose mounts `data/` and `models/` from the host so retrained artifacts persist outside the containers.
- Replace `.env.example` with a real `.env` file before production deployment.

If you want to build each image separately:

```powershell
docker build -t dynamic-pricing-api -f Dockerfile .
docker build -t dynamic-pricing-dashboard -f Dockerfile.streamlit .
```

## EC2 Deployment

Recommended target path:

```text
/opt/dynamic-pricing-engine
```

Quick setup outline on Ubuntu EC2:

```bash
sudo apt update
sudo apt install -y python3.13-venv
cd /opt
sudo mkdir -p dynamic-pricing-engine
sudo chown ubuntu:ubuntu dynamic-pricing-engine
```

Copy the repo to `/opt/dynamic-pricing-engine`, then:

```bash
cd /opt/dynamic-pricing-engine
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
chmod +x deploy/ec2/start_api.sh
chmod +x deploy/ec2/start_dashboard.sh
python scripts/generate_sample_data.py --rows 25000
python scripts/train_model.py --profile kaggle_retail --data-path data/raw/kaggle/retail_price.csv
```

Important:

- The current Streamlit dashboard expects a `kaggle_retail` model artifact.
- If you deploy both API and dashboard together, train the `kaggle_retail` profile and use `POST /price/recommend/kaggle`.
- If you want the synthetic API profile in production, deploy the API by itself or refactor the dashboard to use the synthetic schema.

Install the included `systemd` services:

```bash
sudo cp deploy/ec2/dynamic-pricing-api.service /etc/systemd/system/
sudo cp deploy/ec2/dynamic-pricing-dashboard.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable dynamic-pricing-api
sudo systemctl enable dynamic-pricing-dashboard
sudo systemctl start dynamic-pricing-api
sudo systemctl start dynamic-pricing-dashboard
sudo systemctl status dynamic-pricing-api
sudo systemctl status dynamic-pricing-dashboard
```

Open these after the instance is up and the EC2 security group allows inbound `8000` and `8501`:

```text
http://<your-ec2-public-ip>:8000/docs
http://<your-ec2-public-ip>:8501
```

Suggested EC2 security group inbound rules for a demo:

- `22` from your IP only
- `8000` from your IP or a narrow trusted range
- `8501` from your IP or a narrow trusted range

For public production access, put Nginx in front and expose only `80` and `443`.

## EC2 With Docker

If you prefer containers on EC2 instead of `systemd`, install Docker on Ubuntu and run:

```bash
cd /opt/dynamic-pricing-engine
docker compose up --build -d
docker compose ps
```

Then open:

```text
http://<your-ec2-public-ip>:8000/docs
http://<your-ec2-public-ip>:8501
```

## Extension Ideas

- Replace synthetic data with marketplace telemetry
- Add Flink or Spark Structured Streaming for event windows
- Add SHAP explainability for pricing decisions
- Introduce RL policy optimization for multi-step pricing
- Connect to Amazon Personalize for customer-level contextual pricing
"# Dynamic-Pricing-Engine" 
"# Dynamic-Pricing-Engine" 
