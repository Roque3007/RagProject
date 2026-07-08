import argparse
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError, RateLimitError
from retrieve import retrieve


MODEL_NAME = "gpt-4.1-mini"
load_dotenv()

def format_context(hits: list[dict[str, Any]]) -> str:
    context_blocks = []

    for i, hit in enumerate(hits, start=1):
        metadata = hit["metadata"]
        citation = (
            f"[{i}] paper={metadata['paper_id']}, "
            f"page={metadata['page']}, "
            f"chunk={metadata['chunk_index']}"
        )

        context_blocks.append(
            f"{citation}\n"
            f"{hit['text']}"
        )

    return "\n\n".join(context_blocks)


def build_prompt(question: str, hits: list[dict[str, Any]]) -> str:
    context = format_context(hits)

    return f"""
You are an AI research assistant answering questions using retrieved paper excerpts.

Rules:
- Answer only using the provided context.
- If the context is not enough, say what is missing.
- Cite sources using bracket numbers like [1], [2].
- Be concise but technically accurate.
- Do not invent paper details that are not in the context.

Question:
{question}

Retrieved context:
{context}

Answer:
""".strip()

def build_retrieval_fallback(message: str, hits: list[dict[str, Any]]) -> str:
    fallback = f"{message}\n\n"

    for i, hit in enumerate(hits, start=1):
        metadata = hit["metadata"]
        fallback += (
            f"[{i}] {metadata['paper_id']} page {metadata['page']}\n"
            f"{hit['text'][:700]}\n\n"
        )

    return fallback

def generate_answer(question: str, top_k: int = 5) -> tuple[str, list[dict[str, Any]]]:
    hits = retrieve(question, top_k=top_k)

    if not hits:
        return "I could not find relevant context for that question.", hits

    prompt = build_prompt(question, hits)

    if not os.getenv("OPENAI_API_KEY"):
        return build_retrieval_fallback(
            "No OpenAI API key was found, so here are the most relevant retrieved passages.",
            hits,
        ), hits

    client = OpenAI()

    try:
        response = client.responses.create(
            model=MODEL_NAME,
            input=prompt,
        )
        return response.output_text, hits
    except RateLimitError:
        return build_retrieval_fallback(
            "OpenAI quota was exceeded, so here are the most relevant retrieved passages.",
            hits,
        ), hits
    except OpenAIError as error:
        return build_retrieval_fallback(
            f"OpenAI generation failed: {error}. Here are the most relevant retrieved passages.",
            hits,
        ), hits


def main():
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    answer, hits = generate_answer(args.query, top_k=args.top_k)

    print("\nAnswer:")
    print(answer)

    print("\nSources:")
    for i, hit in enumerate(hits, start=1):
        metadata = hit["metadata"]
        print(
            f"[{i}] {metadata['paper_id']} "
            f"page={metadata['page']} "
            f"chunk={metadata['chunk_index']}"
        )


if __name__ == "__main__":
    main()