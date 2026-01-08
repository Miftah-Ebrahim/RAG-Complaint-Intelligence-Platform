from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
import os


def load_and_chunk(file_path: str, chunk_size: int = 500, overlap: int = 50) -> None:
    """Loads CSV and prints sample chunks."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print("Loading data...")
    # Read only first 100 rows for demonstration speed
    df = pd.read_csv(file_path, nrows=100)
    text_data = df["Consumer complaint narrative"].astype(str).tolist()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap
    )

    print("Chunking data...")
    all_chunks = []
    for text in text_data:
        chunks = splitter.split_text(text)
        all_chunks.extend(chunks)

    print(f"Total chunks generated (from first 100 rows): {len(all_chunks)}")
    print("Sample chunks:")
    for i, chunk in enumerate(all_chunks[:3]):
        print(f"--- Chunk {i + 1} ---\n{chunk}\n")


if __name__ == "__main__":
    load_and_chunk("data/processed/filtered_complaints.csv")
