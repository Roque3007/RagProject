from pathlib import Path
import json

import chromadb
from sentence_transformers import SentenceTransformer


CHUNKS_FILE = Path("data/processed/chunks.jsonl")
DB_DIR = Path("data/chroma")
COLLECTION_NAME = "ai_papers"
MODEL_NAME = "all-MiniLM-L6-v2"


def load_chunks():
    with CHUNKS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def main():
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    chunks = list(load_chunks())

    ids = [chunk["chunk_id"] for chunk in chunks]
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [
        {
            "paper_id": chunk["paper_id"],
            "source_file": chunk["source_file"],
            "page": chunk["page"],
            "chunk_index": chunk["chunk_index"],
        }
        for chunk in chunks
    ]

    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    collection.upsert(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print(f"Indexed {len(texts)} chunks into {DB_DIR}")


if __name__ == "__main__":
    main()