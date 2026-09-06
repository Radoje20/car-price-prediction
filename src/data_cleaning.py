import re
import pandas as pd

RAW_DATA_PATH = "data/cars.csv"
CLEANED_DATA_PATH = "data/cars_cleaned.csv"

MISSING_LIKE_VALUES = {
    "", " ", "nan", "NaN", "NAN",
    "null", "Null", "NULL",
    "none", "None", "NONE",
}

CURRENT_YEAR = 2026
MIN_VALID_YEAR = 1970
MIN_VALID_PRICE = 100


def _standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    new_columns = []
    for col in df.columns:
        clean_col = col.strip().lower()
        clean_col = clean_col.replace("(", "_")
        clean_col = clean_col.replace(")", "")
        clean_col = clean_col.replace("-", "_")
        clean_col = clean_col.replace("/", "_")
        clean_col = re.sub(r"\s+", "_", clean_col)
        clean_col = re.sub(r"[^a-z0-9_]", "", clean_col)
        clean_col = re.sub(r"_+", "_", clean_col)
        clean_col = clean_col.strip("_")
        new_columns.append(clean_col)

    df.columns = new_columns

    rename_map = {
        "priceusd": "price_usd",
        "mileage_kilometers": "mileage_km",
    }
    df = df.rename(columns=rename_map)

    return df


def _strip_string_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    text_columns = df.select_dtypes(include=["object"]).columns
    for col in text_columns:
        df[col] = df[col].astype(str).str.strip()

    return df


def _replace_missing_like_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.replace(list(MISSING_LIKE_VALUES), pd.NA)
    return df


def _convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_columns = [
        "price_usd",
        "year",
        "mileage_km",
        "volume_cm3",
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _clean_categorical_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    categorical_columns = [
        "make",
        "model",
        "condition",
        "fuel_type",
        "color",
        "transmission",
        "drive_unit",
        "segment",
    ]

    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
            df[col] = df[col].replace("nan", pd.NA)

    return df


def _remove_rows_with_missing_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=["price_usd"])
    return df


def _remove_invalid_price(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df["price_usd"] >= MIN_VALID_PRICE]
    return df


def _remove_invalid_year(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    valid_year_mask = (
        df["year"].isna()
        | ((df["year"] >= MIN_VALID_YEAR) & (df["year"] <= CURRENT_YEAR))
    )

    df = df[valid_year_mask]
    return df


def _remove_invalid_mileage(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "mileage_km" in df.columns:
        valid_mileage_mask = (
            df["mileage_km"].isna()
            | (df["mileage_km"] >= 0)
        )
        df = df[valid_mileage_mask]

    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = (
        df
        .pipe(_standardize_column_names)
        .pipe(_strip_string_values)
        .pipe(_replace_missing_like_values)
        .pipe(_convert_numeric_columns)
        .pipe(_clean_categorical_values)
        .pipe(_remove_rows_with_missing_target)
        .pipe(_remove_invalid_price)
        .pipe(_remove_invalid_year)
        .pipe(_remove_invalid_mileage)
        .reset_index(drop=True)
    )

    return df_clean


def main() -> None:
    print("Loading raw dataset...")
    df_raw = pd.read_csv(RAW_DATA_PATH)

    print("Cleaning dataset...")
    df_cleaned = clean(df_raw)

    print("Saving cleaned dataset...")
    df_cleaned.to_csv(CLEANED_DATA_PATH, index=False)

    print(f"Cleaned dataset saved to: {CLEANED_DATA_PATH}")
    print(f"Rows before cleaning: {len(df_raw)}")
    print(f"Rows after cleaning: {len(df_cleaned)}")


if __name__ == "__main__":
    main()