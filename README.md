# Car Price Prediction

A regression ML project that predicts the price (in USD) of a used car based on its make, model, year, mileage, fuel type, and other characteristics.

## Dataset

data/cars.csv — used car listings with columns: make, model, priceUSD (target), year, condition, mileage(kilometers), fuel_type, volume(cm3), color, transmission, drive_unit, segment.

## Project structure

car-price-prediction/
- data/
  - cars.csv
  - cars_cleaned.csv
  - cars_cleaned_with_features.csv
- notebooks/
  - 01_eda.ipynb
- src/
  - data_cleaning.py
  - feature_engineering.py
  - data_preprocessing.py
  - model_training.py
  - model_evaluation.py
  - model_comparison.py
- models/
  - car_price_model.joblib
- README.md
- requirements.txt

## How to run

1. Clone the repo and create a virtual environment:
   - python -m venv venv
   - venv\Scripts\activate
   - pip install -r requirements.txt
2. Place cars.csv inside the data/ folder.
3. Run the pipeline in order:
   - python -m src.data_cleaning
   - python -m src.feature_engineering
   - python -m src.model_training
   - python -m src.model_evaluation
   - python -m src.model_comparison
4. Explore notebooks/01_eda.ipynb for the exploratory analysis.

## Data cleaning decisions

- Rows with missing priceUSD were dropped since a row without a target can't be used for training.
- Rows with price below $100 were treated as invalid listings and removed.
- Rows with year outside 1970-2026 were removed as likely data entry errors.
- Negative mileage values were removed.

## Feature engineering

- car_age: current year minus manufacturing year
- mileage_per_year: mileage divided by car age
- engine_volume_liters: engine volume converted from cm3 to liters
- is_newer_car: flag for cars 5 years old or newer
- is_high_mileage: flag for cars over 200,000 km
- model and brand_model were excluded as categorical features due to high cardinality (too many unique values for one-hot encoding to handle cleanly)

## Model comparison results

Fill in the table below using the output you got from running model_comparison.py. It printed a table with columns model, mae, mse, rmse, r2 — copy those numbers here, one row per model.

| Model | MAE | RMSE | R2 |
|---|---|---|---|
| Linear Regression | [2646.09] | [5183.06] | [0.64] |
| Decision Tree | [1568.65] | [3745.33] | [0.81] |
| Random Forest | [1193.61] | [3027.84] | [0.88] |
| Gradient Boosting | [1531.07] | [3301.30] | [0.85] |

## Final model

Write 1-2 sentences here saying which model you picked and why. Example: "Random Forest had the lowest MAE at $1200, meaning its price predictions are off by about $1200 on average, which was better than every other model tested."

## Example prediction

Fill in using 2-3 rows from the "Prediction examples" table that model_evaluation.py printed.

| Actual Price | Predicted Price | Error |
|---|---|---|
| [9500] | [8624.93] | [875.07] |
| [5450] | [5977.38] | [-527.38] |