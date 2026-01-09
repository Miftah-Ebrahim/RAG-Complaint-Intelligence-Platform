import pandas as pd
from src.config import RAW_CSV, FILTERED_CSV


def run_etl():
    """Loads raw data, filters for top 5 products, and saves processed CSV."""
    if not RAW_CSV.exists():
        print(f"Error: {RAW_CSV} not found.")
        return

    print("Loading raw data...")
    df = pd.read_csv(RAW_CSV, low_memory=False)

    target_products = [
        "Credit card",
        "Credit card or prepaid card",
        "Checking or savings account",
        "Money transfer, virtual currency, or money service",
        "Personal loan",
    ]

    print("Filtering and cleaning...")
    df = df[df["Product"].isin(target_products)]
    df = df.dropna(subset=["Consumer complaint narrative"])

    FILTERED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FILTERED_CSV, index=False)
    print(f"Saved {len(df)} rows to {FILTERED_CSV}")


if __name__ == "__main__":
    run_etl()
