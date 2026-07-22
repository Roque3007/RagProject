from pathlib import Path
import json
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

from retrieve import retrieve  # noqa: E402


QUESTIONS_FILE = PROJECT_ROOT / "eval" / "questions.jsonl"


def load_questions():
    with QUESTIONS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def reciprocal_rank(retrieved_papers: list[str], expected_papers: list[str]) -> float:
    for index, paper_id in enumerate(retrieved_papers, start=1):
        if paper_id in expected_papers:
            return 1 / index

    return 0.0


def main():
    top_k = 5
    total = 0
    hits_at_k = 0
    mrr_total = 0.0

    for item in load_questions():
        question = item["question"]
        expected_papers = item["expected_papers"]

        hits = retrieve(question, top_k=top_k)
        retrieved_papers = [hit["metadata"]["paper_id"] for hit in hits]

        hit = any(paper in expected_papers for paper in retrieved_papers)
        rr = reciprocal_rank(retrieved_papers, expected_papers)

        total += 1
        hits_at_k += int(hit)
        mrr_total += rr

        print("=" * 80)
        print(f"Question: {question}")
        print(f"Expected: {expected_papers}")
        print(f"Retrieved: {retrieved_papers}")
        print(f"Hit@{top_k}: {hit}")
        print(f"Reciprocal rank: {rr:.3f}")

    print("\nSummary")
    print(f"Questions: {total}")
    print(f"Hit@{top_k}: {hits_at_k / total:.2%}")
    print(f"MRR: {mrr_total / total:.3f}")


if __name__ == "__main__":
    main()
