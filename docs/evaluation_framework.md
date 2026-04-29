# Evaluation Framework

## Overview

The evaluation framework assesses ETA prediction quality across **30+ scenario slices**, reflecting real-world operational concerns.

## Metrics

| Metric | Formula | Business Meaning |
|--------|---------|------------------|
| MAE | mean(\|y - ŷ\|) | Average prediction error in minutes |
| RMSE | sqrt(mean((y - ŷ)²)) | Penalizes large errors |
| MAPE | mean(\|y - ŷ\| / y) × 100 | Percentage error, scale-independent |
| On-time Rate | fraction within ±5 min | Customer satisfaction proxy |
| Over-prediction Rate | fraction where ŷ > y | Conservative bias indicator |
| Mean Bias | mean(ŷ - y) | Systematic over/under-prediction |
| P90 Error | 90th percentile of \|y - ŷ\| | Tail-case reliability |

## Scenario Slices

### Distance-based
- Short (< 2 km), Medium (2–5 km), Long (> 5 km)

### Time-based
- Morning (6–11), Lunch (11–14), Afternoon (14–17), Dinner (17–21), Night (21–6)
- Weekend vs. Weekday
- Peak (lunch 11–13, dinner 17–20) vs. Off-peak

### Demand-based
- Demand pressure: Low, Medium, High
- Rider utilization quartiles (Q1–Q4)

### Order-based
- Small order (1–2 items), Medium (3–4), Large (5+)

## Why Multi-Scenario Evaluation?

A model with good aggregate MAE can still fail in specific scenarios:
- **Peak hours**: Demand surge causes longer preparation and dispatch delays
- **Long distance**: GPS noise and routing variability increase
- **High rider utilization**: Queuing effects cause non-linear delays

By evaluating across slices, we identify failure modes and guide targeted improvements.
