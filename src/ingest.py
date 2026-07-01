from pathlib import Path
import json
from pypdf import PdfReader


RAW_DIR = Path("data/raw/papers")
OUT_FILE = Path("data/processed/pages.jsonl")


def extract_pages(pdf_path: Path):
    reader = PdfReader(str(pdf_path))

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split())

        if not text:
            continue

        yield {
            "paper_id": pdf_path.stem,
            "source_file": str(pdf_path),
            "page": page_number,
            "text": text,
        }


def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    print(f"Looking for PDFs in: {RAW_DIR.resolve()}")

    print(f"Found PDFs: {list(RAW_DIR.glob('*.pdf'))}")
    count = 0
    with OUT_FILE.open("w", encoding="utf-8") as f:
        for pdf_path in sorted(RAW_DIR.glob("*.pdf")):
            for page in extract_pages(pdf_path):
                f.write(json.dumps(page, ensure_ascii=False) + "\n")
                count += 1

    print(f"Wrote {count} pages to {OUT_FILE}")


if __name__ == "__main__":
    main()