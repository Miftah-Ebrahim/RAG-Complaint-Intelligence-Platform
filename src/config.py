"""Configuration constants for the CrediTrust RAG Complaint Intelligence Platform.

Centralizes all paths, model parameters, and processing constants
to eliminate magic numbers throughout the codebase.
"""

from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# Base project directory
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Data Directories
# ---------------------------------------------------------------------------
DATA_RAW: Path = BASE_DIR / "data" / "raw"
DATA_PROCESSED: Path = BASE_DIR / "data" / "processed"
IMAGES_DIR: Path = BASE_DIR / "images"

# ---------------------------------------------------------------------------
# File Paths
# ---------------------------------------------------------------------------
RAW_CSV: Path = DATA_RAW / "complaints.csv"
FILTERED_CSV: Path = DATA_PROCESSED / "filtered_complaints.csv"
VECTOR_STORE_DIR: Path = DATA_PROCESSED / "vector_store"

# ---------------------------------------------------------------------------
# Embedding & Retriever Settings
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
RETRIEVER_K: int = 3

# ---------------------------------------------------------------------------
# LLM Settings (HuggingFace Router / DeepSeek-R1)
# ---------------------------------------------------------------------------
LLM_REPO_ID: str = "deepseek-ai/DeepSeek-R1"
LLM_TEMPERATURE: float = 0.1
LLM_MAX_TOKENS: int = 500
LLM_TIMEOUT_SECONDS: int = 120

# ---------------------------------------------------------------------------
# Text Chunking Settings
# ---------------------------------------------------------------------------
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 50

# ---------------------------------------------------------------------------
# Data Sampling
# ---------------------------------------------------------------------------
SAMPLE_PER_CLASS: int = 300

# ---------------------------------------------------------------------------
# ETL Target Products
# ---------------------------------------------------------------------------
TARGET_PRODUCTS: List[str] = [
    "Credit card",
    "Credit card or prepaid card",
    "Checking or savings account",
    "Money transfer, virtual currency, or money service",
    "Personal loan",
]
