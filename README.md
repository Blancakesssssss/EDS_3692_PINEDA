# EDS_3692_Pineda
## AGR-04: Micro-Climate Volatility Detection and Statistical Analysis in IoT-Based Greenhouse Systems

**Student:** Pineda, Chester Rios  
**Student Number:** TUPM-25-3692  
**Section:** BSECE 1C  
**Course:** Computer Programming 1   


## 📌 Project Overview

This project presents an automated, Object-Oriented Python data analytics pipeline designed to detect, clean, and statistically analyze **micro-climate volatility patterns** in a real IoT greenhouse sensor environment. The pipeline processes raw 10-minute interval sensor telemetry data and applies a unique dual-condition programmatic filter to isolate peak daytime high-heat volatility events.

**Assigned Topic:** AGR-04 — Micro-Climate Volatility  
**Dataset:** Greenhouse Sensor Data (10-Minute Interval) by Marcel Boonman  
**Dataset Source:** [Kaggle](https://www.kaggle.com/datasets/marcelboonman/greenhouse-sensor-data-10-minute-interval)


## 🔍 Unique Filter Logic

The dataset is programmatically filtered using a dual-condition approach:
- **Time Window:** Daytime hours only → Hour ∈ [06:00, 18:00]
- **Temperature Condition:** Indoor temperature > 28°C (high-heat volatility events only)

This targets peak solar radiation periods where thermal stress, humidity collapse, CO₂ accumulation, and VOC buildup are most pronounced.


## ⚙️ Pipeline Modules (OOP Design)

The entire pipeline is encapsulated in the `GreenhousePipeline` class with 4 distinct modules:

| Method | Module | Description |
|---|---|---|
| `load_data()` | Data Ingestion | Loads raw CSV, parses timestamps |
| `clean_data()` | Data Cleaning | Removes duplicates, nulls, applies unique filter |
| `analyze_data()` | Statistical Analysis | NumPy-based descriptive stats, correlation, outliers |
| `visualize_data()` | Visualization | Generates 5 static PNGs + 5 animated GIFs |


## 📊 Statistical Analysis Performed

- ✅ Descriptive Statistics — Mean, Median, Standard Deviation, Variance
- ✅ Distribution Analysis — Skewness, data spread
- ✅ Outlier Detection — IQR Method (Q1, Q3, IQR)
- ✅ Correlation Analysis — Pearson r between all sensor variables
- ✅ Comparative Analysis — Morning (06:00–12:00) vs Afternoon (12:00–18:00)


## 📈 Key Results

| Metric | Value |
|---|---|
| Filtered Records | 5,049 (from 17,562 raw) |
| Mean Indoor Temp | 41.54°C |
| Std Dev (Temp) | 11.07°C |
| Temp Skewness | +0.8293 (right-skewed) |
| Temp vs Humidity (r) | −0.8597 (strong negative) |
| CO₂ vs VOC (r) | +0.8789 (very strong positive) |
| Morning Mean Temp | 39.17°C |
| Afternoon Mean Temp | 42.80°C |
| Temp Difference | 3.63°C |

## 🌿 Dataset Information

| Info | Detail |
|---|---|
| Dataset Name | Greenhouse Sensor Data — 10 Minute Interval |
| Publisher | Marcel Boonman |
| Source | Kaggle |
| Raw Records | 17,562 |
| Date Range | March 3, 2021 — July 3, 2021 |
| Sampling Rate | Every 10 minutes |
| Separator | Semicolon (`;`) |
| Decimal | Comma (`,`) |

**Sensor Variables:**
- `greenhous_temperature_celsius` — Indoor temperature (°C)
- `greenhouse_humidity_percentage` — Indoor humidity (%)
- `greenhouse_illuminance_lux` — Light level (lux)
- `online_temperature_celsius` — Outdoor temperature (°C)
- `online_humidity_percentage` — Outdoor humidity (%)
- `greenhouse_total_volatile_organic_compounds_ppb` — VOC (ppb)
- `greenhouse_equivalent_co2_ppm` — CO₂ (ppm)
- `created` — Timestamp

*Computer Programming 1 — Final Project | Technological University of the Philippines, Manila | 2026*
