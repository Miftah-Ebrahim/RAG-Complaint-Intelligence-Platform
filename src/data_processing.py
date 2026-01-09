import pandas as pd
from langchain_core.documents import Document
from src.logger import logger


def stratified_sample(df: pd.DataFrame, n_per_class: int = 500) -> pd.DataFrame:
    """Samples n rows from each Product category to ensure balanced representation."""
    logger.info(f"Performing stratified sampling with n={n_per_class} per class...")

    def sample_group(group):
        return group.sample(n=min(len(group), n_per_class), random_state=42)

    sampled_df = df.groupby("Product", group_keys=False).apply(sample_group)
    logger.info(f"Sampled dataframe shape: {sampled_df.shape}")
    return sampled_df


def create_documents(df: pd.DataFrame) -> list[Document]:
    """Converts DataFrame rows to LangChain Documents with rich metadata."""
    documents = []
    logger.info("Converting rows to LangChain Documents...")

    for _, row in df.iterrows():
        # metadata filtering handled here
        meta = {
            "product": row.get("Product", "Unknown"),
            "sub_product": row.get("Sub-product", "Unknown"),
            "date": str(row.get("Date received", "")),
            "state": row.get("State", "Unknown"),
            "company": row.get("Company", "Unknown"),
            "complaint_id": str(row.get("Complaint ID", "")),
        }

        # clean nan values in metadata
        meta = {k: (v if pd.notna(v) else "Unknown") for k, v in meta.items()}

        doc = Document(
            page_content=str(row["Consumer complaint narrative"]), metadata=meta
        )
        documents.append(doc)

    logger.info(f"Created {len(documents)} source documents.")
    return documents
