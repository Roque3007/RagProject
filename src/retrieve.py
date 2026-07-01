import argparse

import chromadb
from sentence_transformers import SentenceTransformer


DB_DIR = "data/chroma"
COLLECTION_NAME = "ai_papers"
MODEL_NAME = "all-MiniLM-L6-v2"


def retrieve(query: str, top_k: int = 5):
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)

    query_embedding = model.encode([query]).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append(
            {
                "text": doc,
                "metadata": metadata,
                "distance": distance,
            }
        )

    return hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    print(f"Query: {args.query}")

    hits = retrieve(args.query, top_k=args.top_k)

    print(f"Found {len(hits)} results")

    for i, hit in enumerate(hits, start=1):
        metadata = hit["metadata"]
        print("=" * 80)
        print(
            f"Result {i} | paper={metadata['paper_id']} "
            f"page={metadata['page']} distance={hit['distance']:.4f}"
        )
        print(hit["text"][:1000])

if __name__ == "__main__":
    main()