import joblib
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.data_preprocessing import split_features_and_target

DATA_PATH = "data/cars_cleaned_with_features.csv"
MODEL_PATH = "models/car_price_model.joblib"


def main() -> None:
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    print("Splitting features and target...")
    X, y = split_features_and_target(df)

    print("Creating the same train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("Loading trained model...")
    model = joblib.load(MODEL_PATH)

    print("Making predictions...")
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    metrics = pd.DataFrame({
        "metric": ["MAE", "MSE", "RMSE", "R2"],
        "value": [mae, mse, rmse, r2],
    })

    print("\nRegression metrics:")
    print(metrics)

    prediction_analysis = pd.DataFrame({
        "actual_price_usd": y_test.values,
        "predicted_price_usd": y_pred,
    })

    prediction_analysis["error_usd"] = (
        prediction_analysis["actual_price_usd"]
        - prediction_analysis["predicted_price_usd"]
    )

    prediction_analysis["absolute_error_usd"] = (
        prediction_analysis["error_usd"].abs()
    )

    print("\nPrediction examples:")
    print(prediction_analysis.sample(10, random_state=42))

    print("\nLargest prediction errors:")
    print(
        prediction_analysis
        .sort_values("absolute_error_usd", ascending=False)
        .head(10)
    )


if __name__ == "__main__":
    main()