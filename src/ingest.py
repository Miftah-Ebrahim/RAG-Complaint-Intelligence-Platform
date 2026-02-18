"""Vector-store ingestion pipeline for the CrediTrust RAG system.

Reads the filtered complaint CSV, performs stratified sampling,
converts rows to LangChain documents, chunks the text, embeds with
HuggingFace embeddings, and persists a Chroma vector store to disk.
"""

import shutil
from typing import Optional

import pandas as pd
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL_NAME,
    FILTERED_CSV,
    SAMPLE_PER_CLASS,
    VECTOR_STORE_DIR,
)
from src.data_processing import create_documents, stratified_sample
from src.logger import logger


def ingest_data(reset_db: bool = True) -> None:
    """Run the full document ingestion pipeline.

    Steps:
      1. Load the filtered CSV produced by ``etl.run_etl()``.
      2. Perform stratified sampling (``config.SAMPLE_PER_CLASS`` per
         product).
      3. Convert rows to LangChain ``Document`` objects.
      4. Split documents into chunks of ``config.CHUNK_SIZE`` characters.
      5. Embed chunks and persist them in a Chroma vector store.

    Args:
        reset_db: If ``True``, delete the existing vector store before
            ingesting.  Defaults to ``True``.

    Returns:
        None.  Side-effect: writes a Chroma vector store to
        ``config.VECTOR_STORE_DIR``.
    """
    if not FILTERED_CSV.exists():
        logger.error(f"Filtered CSV not found at {FILTERED_CSV}. Run etl.py first.")
        return

    # 1. Load Data
    logger.info(f"Loading data from {FILTERED_CSV}...")
    df: pd.DataFrame = pd.read_csv(FILTERED_CSV, low_memory=False)

    # 2. Stratified Sampling
    df_sampled: pd.DataFrame = stratified_sample(df, n_per_class=SAMPLE_PER_CLASS)

    # 3. Create Documents w/ Metadata
    raw_docs = create_documents(df_sampled)

    # 4. Chunking
    logger.info("Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(raw_docs)
    logger.info(f"Generated {len(chunks)} text chunks.")

    # 5. Embed & Index
    logger.info(f"Initializing Vector Store at {VECTOR_STORE_DIR}...")

    if reset_db and VECTOR_STORE_DIR.exists():
        logger.warning(f"Deleting existing vector store at {VECTOR_STORE_DIR}")
        shutil.rmtree(VECTOR_STORE_DIR)

    embedding_fn = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_fn,
        persist_directory=str(VECTOR_STORE_DIR),
    )

    logger.info("✅ Ingestion Complete. Vector Store is ready.")


if __name__ == "__main__":
    ingest_data()
