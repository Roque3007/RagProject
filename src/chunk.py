from pathlib import Path
import json


PAGES_FILE = Path("data/processed/pages.jsonl")
OUT_FILE = Path("data/processed/chunks.jsonl")

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    words = text.split()
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])

        if chunk:
            yield chunk

        if end >= len(words):
            break

        start = end - overlap


def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    chunk_count = 0
    with PAGES_FILE.open("r", encoding="utf-8") as pages, OUT_FILE.open(
        "w", encoding="utf-8"
    ) as chunks:
        for line in pages:
            page = json.loads(line)

            for i, text in enumerate(chunk_text(page["text"])):
                record = {
                    "chunk_id": f"{page['paper_id']}_p{page['page']}_c{i}",
                    "paper_id": page["paper_id"],
                    "source_file": page["source_file"],
                    "page": page["page"],
                    "chunk_index": i,
                    "text": text,
                }
                chunks.write(json.dumps(record, ensure_ascii=False) + "\n")
                chunk_count += 1

    print(f"Wrote {chunk_count} chunks to {OUT_FILE}")


if __name__ == "__main__":
    main()