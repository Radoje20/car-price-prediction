import pandas as pd

CLEANED_DATA_PATH = "data/cars_cleaned.csv"
FEATURES_DATA_PATH = "data/cars_cleaned_with_features.csv"

CURRENT_YEAR = 2026
HIGH_MILEAGE_THRESHOLD_KM = 200000
NEWER_CAR_MAX_AGE = 5


def _add_car_age(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["car_age"] = CURRENT_YEAR - df["year"]
    return df


def _add_mileage_per_year(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    safe_age = df["car_age"].replace(0, 1)
    df["mileage_per_year"] = df["mileage_km"] / safe_age

    return df


def _add_engine_volume_liters(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "volume_cm3" in df.columns:
        df["engine_volume_liters"] = df["volume_cm3"] / 1000

    return df


def _add_is_newer_car(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_newer_car"] = (df["car_age"] <= NEWER_CAR_MAX_AGE).astype(int)
    return df


def _add_is_high_mileage(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["is_high_mileage"] = (
        df["mileage_km"] >= HIGH_MILEAGE_THRESHOLD_KM
    ).astype(int)

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df_features = (
        df
        .pipe(_add_car_age)
        .pipe(_add_mileage_per_year)
        .pipe(_add_engine_volume_liters)
        .pipe(_add_is_newer_car)
        .pipe(_add_is_high_mileage)
        .reset_index(drop=True)
    )

    return df_features


def main() -> None:
    print("Loading cleaned dataset...")
    df_cleaned = pd.read_csv(CLEANED_DATA_PATH)

    print("Building features...")
    df_features = build_features(df_cleaned)

    print("Saving feature-engineered dataset...")
    df_features.to_csv(FEATURES_DATA_PATH, index=False)

    print(f"Feature-engineered dataset saved to: {FEATURES_DATA_PATH}")


if __name__ == "__main__":
    main()