from pathlib import Path

# Base project directory
BASE_DIR = Path(__file__).parent.parent

# Data Directories
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
IMAGES_DIR = BASE_DIR / "images"

# File Paths
RAW_CSV = DATA_RAW / "complaints.csv"
FILTERED_CSV = DATA_PROCESSED / "filtered_complaints.csv"
VECTOR_STORE_DIR = DATA_PROCESSED / "vector_store"
