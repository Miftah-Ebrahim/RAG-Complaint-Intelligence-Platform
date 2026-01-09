import pandas as pd
import os
import shutil
from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import FILTERED_CSV, VECTOR_STORE_DIR
from src.logger import logger
from src.data_processing import stratified_sample, create_documents


def ingest_data(reset_db: bool = True):
    """Main ingestion logic."""
    if not FILTERED_CSV.exists():
        logger.error(f"Filtered CSV not found at {FILTERED_CSV}. Run etl.py first.")
        return

    # 1. Load Data
    logger.info(f"Loading data from {FILTERED_CSV}...")
    df = pd.read_csv(FILTERED_CSV, low_memory=False)

    # 2. Stratified Sampling
    # Using 300 to keep it lighter for local machine, but balanced
    df_sampled = stratified_sample(df, n_per_class=300)

    # 3. Create Documents w/ Metadata
    raw_docs = create_documents(df_sampled)

    # 4. Chunking
    logger.info("Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        truncation=True,  # Ensure hard limit if needed
    )
    chunks = splitter.split_documents(raw_docs)
    logger.info(f"Generated {len(chunks)} text chunks.")

    # 5. Embed & Index
    logger.info(f"Initializing Vector Store at {VECTOR_STORE_DIR}...")

    if reset_db and VECTOR_STORE_DIR.exists():
        logger.warning(f"Deleting existing vector store at {VECTOR_STORE_DIR}")
        shutil.rmtree(VECTOR_STORE_DIR)

    embedding_fn = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Batch processing is safer for large datasets, but for 2000 docs standard is fine
    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_fn,
        persist_directory=str(VECTOR_STORE_DIR),
    )

    logger.info("✅ Ingestion Complete. Vector Store is ready.")


if __name__ == "__main__":
    ingest_data()
