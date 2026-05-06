# Food Delivery ETA Prediction System

An end-to-end machine learning system for predicting food delivery estimated time of arrival (ETA). Built with production-grade engineering practices including modular architecture, comprehensive evaluation, REST API serving, and CI/CD pipeline.

## Highlights

- **Domain-informed feature engineering**: 20+ features spanning temporal, spatial, and contextual (supply-demand) dimensions
- **Multi-model comparison**: Mean/Median baselines → LightGBM → XGBoost with automated training pipeline
- **Scenario-based evaluation**: 30+ slices (distance, time period, demand pressure, order size) for failure-mode analysis
- **Production-ready serving**: FastAPI with Pydantic validation, Docker containerization, health checks
- **Test coverage**: Unit tests for features, metrics, models, and API endpoints

## Project Structure

```
food-delivery-eta/
├── src/
│   ├── data/            # Data loading, validation, splitting
│   │   ├── loader.py    # CSV loading, schema validation, target computation
│   │   └── splitter.py  # Random & time-based train/val/test splits
│   ├── features/        # Feature engineering pipeline
│   │   ├── temporal.py  # Hour, day, peak flags, cyclical encoding
│   │   ├── spatial.py   # Haversine/Manhattan distance, bearing
│   │   ├── contextual.py # Supply-demand ratio, interaction features
│   │   └── pipeline.py  # Orchestration (fit_transform / transform)
│   ├── models/          # Model definitions and training
│   │   ├── baseline.py  # Mean, Median, BucketMedian baselines
│   │   ├── gbm.py       # LightGBM & XGBoost wrappers
│   │   └── trainer.py   # Training loop, evaluation, artifact saving
│   ├── evaluation/      # Metrics and scenario analysis
│   │   ├── metrics.py   # MAE, RMSE, MAPE, on-time rate, bias
│   │   ├── scenario.py  # 30+ slice evaluator
│   │   └── report.py    # Formatted output
│   ├── api/             # REST API
│   │   ├── app.py       # FastAPI endpoints
│   │   └── schemas.py   # Pydantic request/response models
│   └── utils/           # Config loading, logging
├── tests/               # pytest test suite
├── configs/             # YAML experiment configs
├── scripts/             # Data generation & experiment runner
├── notebooks/           # EDA, feature engineering, model experiments
├── docs/                # Evaluation framework documentation
├── Dockerfile
├── Makefile
└── .github/workflows/   # CI pipeline
```

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Generate synthetic data (50K orders, Shanghai area)
make data

# Train models and evaluate
make train

# Run tests
make test

# Start API server
make serve
```

## Data Schema

| Column | Type | Description |
|--------|------|-------------|
| `order_id` | str | Unique order identifier |
| `created_at` | datetime | Order creation timestamp |
| `actual_delivery_time` | datetime | Actual delivery timestamp |
| `store_latitude/longitude` | float | Restaurant coordinates |
| `delivery_latitude/longitude` | float | Customer coordinates |
| `total_items` | int | Number of items ordered |
| `subtotal` | float | Order total amount (CNY) |
| `num_distinct_items` | int | Number of unique items |
| `total_onshift_riders` | int | Available riders in area |
| `total_busy_riders` | int | Currently occupied riders |
| `total_outstanding_orders` | int | Pending orders in area |

**Target**: `delivery_duration_minutes` = (`actual_delivery_time` - `created_at`) in minutes

## Feature Engineering

### Temporal Features
- Hour of day, minute of day, day of week
- Weekend flag, peak lunch (11-13) / dinner (17-20) flags
- Cyclical encoding (sin/cos) for hour and day_of_week

### Spatial Features
- Haversine distance (km), Manhattan distance approximation
- Distance buckets (short < 2km, medium 2-5km, long > 5km)
- Bearing angle between store and delivery point

### Contextual Features
- Rider utilization rate (`busy / onshift`)
- Orders per rider, supply-demand ratio
- Demand pressure buckets (low/medium/high)
- Order size bucket, average item price
- Interaction features: distance × peak, distance × utilization

## Models

| Model | Description | Use Case |
|-------|-------------|----------|
| MeanBaseline | Predicts global mean | Lower bound |
| MedianBaseline | Predicts global median | Robust baseline |
| BucketMedianBaseline | Median per hour bucket | Captures daily patterns |
| **LightGBM** | Gradient boosting with early stopping | **Primary model** |
| XGBoost | Alternative GBDT implementation | Comparison |

## Evaluation Framework

Metrics evaluated across 30+ scenario slices:

- **Distance**: short / medium / long
- **Time period**: morning / lunch / afternoon / dinner / night
- **Day type**: weekday / weekend
- **Peak**: peak hours / off-peak
- **Demand**: low / medium / high pressure
- **Order size**: small / medium / large
- **Rider utilization**: Q1 / Q2 / Q3 / Q4

Key metrics: MAE, RMSE, MAPE, On-time Rate (±5 min), Over-prediction Rate, Mean Bias, P90 Error

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/predict` | Single order ETA prediction |
| POST | `/predict/batch` | Batch predictions |
| GET | `/health` | Service health check |
| GET | `/model/info` | Model metadata & features |

## Tech Stack

- **ML**: LightGBM, XGBoost, scikit-learn
- **Data**: Pandas, NumPy
- **API**: FastAPI, Pydantic, Uvicorn
- **Testing**: pytest, pytest-cov
- **Deployment**: Docker, GitHub Actions CI
- **Config**: PyYAML

## Future Work

- [ ] Wide & Deep model (TensorFlow) for embedding-based features
- [ ] Online learning with streaming data
- [ ] Quantile regression for confidence intervals
- [ ] Feature store integration
- [ ] A/B testing framework
- [ ] Real-time traffic and weather features

## License

MIT
