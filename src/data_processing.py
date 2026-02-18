"""Data processing utilities for the CrediTrust RAG pipeline.

Provides stratified sampling and DataFrame-to-LangChain-Document
conversion with rich metadata for downstream vector-store ingestion.
"""

from typing import List

import pandas as pd
from langchain_core.documents import Document

from src.logger import logger


def stratified_sample(df: pd.DataFrame, n_per_class: int = 500) -> pd.DataFrame:
    """Perform stratified sampling to balance product categories.

    Samples up to *n_per_class* rows from each unique ``Product``
    value to prevent dominant categories from skewing retrieval.

    Args:
        df: Input DataFrame that must contain a ``Product`` column.
        n_per_class: Maximum number of rows to sample per product
            category.  If a category has fewer rows, all rows are kept.

    Returns:
        A new DataFrame with at most *n_per_class* rows per product.
    """
    logger.info(f"Performing stratified sampling with n={n_per_class} per class...")

    groups = [
        group.sample(n=min(len(group), n_per_class), random_state=42)
        for _, group in df.groupby("Product", group_keys=False)
    ]
    sampled_df: pd.DataFrame = pd.concat(groups, ignore_index=True)
    logger.info(f"Sampled dataframe shape: {sampled_df.shape}")
    return sampled_df


def create_documents(df: pd.DataFrame) -> List[Document]:
    """Convert DataFrame rows into LangChain ``Document`` objects.

    Each row's consumer complaint narrative becomes the document
    ``page_content``, and selected columns are stored as metadata
    for downstream filtering and citation.

    Args:
        df: DataFrame containing at minimum a
            ``Consumer complaint narrative`` column.  Optional metadata
            columns: ``Product``, ``Sub-product``, ``Date received``,
            ``State``, ``Company``, ``Complaint ID``.

    Returns:
        List of ``Document`` objects ready for text splitting and
        vector-store ingestion.
    """
    documents: List[Document] = []
    logger.info("Converting rows to LangChain Documents...")

    for _, row in df.iterrows():
        meta = {
            "product": row.get("Product", "Unknown"),
            "sub_product": row.get("Sub-product", "Unknown"),
            "date": str(row.get("Date received", "")),
            "state": row.get("State", "Unknown"),
            "company": row.get("Company", "Unknown"),
            "complaint_id": str(row.get("Complaint ID", "")),
        }

        # Clean NaN values in metadata
        meta = {k: (v if pd.notna(v) else "Unknown") for k, v in meta.items()}

        doc = Document(
            page_content=str(row["Consumer complaint narrative"]), metadata=meta
        )
        documents.append(doc)

    logger.info(f"Created {len(documents)} source documents.")
    return documents
