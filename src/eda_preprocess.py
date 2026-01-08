import pandas as pd
import os
import argparse
import sys


def clean_data(input_path, output_path):
    """
    Loads raw complaints data, filters for specific products, cleans, and saves.
    args:
        input_path: str, path to the raw csv file
        output_path: str, path to save the processed csv file
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found.")
        sys.exit(1)

    print(f"Loading data from {input_path}...")
    try:
        df = pd.read_csv(input_path, low_memory=False)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        sys.exit(1)

    target_products = [
        "Credit card",
        "Credit card or prepaid card",
        "Checking or savings account",
        "Money transfer, virtual currency, or money service",
        "Personal loan",
    ]

    print(f"Filtering for products: {target_products}")
    df = df[df["Product"].isin(target_products)]

    print("Dropping rows with missing 'Consumer complaint narrative'...")
    df = df.dropna(subset=["Consumer complaint narrative"])

    print(f"Saving processed data to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print("Preprocessing completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Complaints Data")
    parser.add_argument(
        "input_path",
        nargs="?",
        default="data/raw/complaints.csv",
        help="Input CSV path",
    )
    parser.add_argument(
        "output_path",
        nargs="?",
        default="data/processed/filtered_complaints.csv",
        help="Output CSV path",
    )

    args = parser.parse_args()

    clean_data(args.input_path, args.output_path)
