"""ETL pipeline for the CrediTrust Complaint Intelligence Platform.

Loads raw CFPB complaint data, filters it to the target product
categories, removes rows without narratives, and persists the
cleaned dataset for downstream ingestion.
"""

from typing import Optional

import pandas as pd

from src.config import FILTERED_CSV, RAW_CSV, TARGET_PRODUCTS
from src.logger import logger


def run_etl() -> Optional[pd.DataFrame]:
    """Execute the extract-transform-load pipeline.

    Steps:
      1. Load the raw complaints CSV.
      2. Filter rows to the product categories defined in
         ``config.TARGET_PRODUCTS``.
      3. Drop rows missing a consumer complaint narrative.
      4. Save the cleaned DataFrame to ``config.FILTERED_CSV``.

    Returns:
        The filtered ``DataFrame`` on success, or ``None`` if the raw
        CSV file is missing.
    """
    if not RAW_CSV.exists():
        logger.error(f"Error: {RAW_CSV} not found.")
        return None

    logger.info("Loading raw data...")
    df: pd.DataFrame = pd.read_csv(RAW_CSV, low_memory=False)

    logger.info("Filtering and cleaning...")
    df = df[df["Product"].isin(TARGET_PRODUCTS)]
    df = df.dropna(subset=["Consumer complaint narrative"])

    FILTERED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FILTERED_CSV, index=False)
    logger.info(f"Saved {len(df)} rows to {FILTERED_CSV}")
    return df


if __name__ == "__main__":
    run_etl()
