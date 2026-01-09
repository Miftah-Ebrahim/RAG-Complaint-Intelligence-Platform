from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import VECTOR_STORE_DIR
import sys


def verify_metadata():
    if not VECTOR_STORE_DIR.exists():
        print(f"FAILED: {VECTOR_STORE_DIR} does not exist.")
        sys.exit(1)

    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(
        persist_directory=str(VECTOR_STORE_DIR), embedding_function=embedding
    )

    # Query for something generic
    results = vector_db.similarity_search("credit card fee", k=1)

    if not results:
        print("FAILED: No results return.")
        sys.exit(1)

    doc = results[0]
    print("Retrieved Document Metadata:")
    print(doc.metadata)

    required_keys = ["product", "date", "complaint_id"]
    for key in required_keys:
        if key not in doc.metadata:
            print(f"FAILED: Missing metadata key '{key}'")
            sys.exit(1)

    print("SUCCESS: Metadata verified.")


if __name__ == "__main__":
    verify_metadata()
