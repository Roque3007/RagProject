# AI Papers RAG Assistant

A citation-aware Retrieval-Augmented Generation project for asking questions over AI research papers.

This project ingests AI research paper PDFs, extracts page-level text, chunks the text for retrieval, embeds each chunk, stores the vectors in ChromaDB, and retrieves relevant paper passages for user questions.

## Why This Project

Large language models can answer many AI-related questions, but they may hallucinate or answer without clear sources. This project uses Retrieval-Augmented Generation to ground answers in actual research papers.

The goal is to build a system that can answer questions like:

- What is self-attention?
- How does RAG reduce hallucinations?
- What is ReAct?
- How does Self-RAG decide when to retrieve?
- What problem does Toolformer solve?

## Current Features

- PDF ingestion with `pypdf`
- Page-level text extraction
- JSONL-based processing pipeline
- Chunking with overlap
- Sentence-transformer embeddings
- Persistent ChromaDB vector index
- Semantic retrieval over AI paper chunks
- Source metadata including paper ID, page number, and chunk ID

## Tech Stack

- Python
- pypdf
- sentence-transformers
- ChromaDB
- argparse
- JSONL

## Project Structure

```text
ragProject/
  src/
    ingest.py      # Extracts text from PDFs page by page
    chunk.py       # Splits page text into retrieval chunks
    index.py       # Embeds chunks and stores them in ChromaDB
    retrieve.py    # Retrieves relevant chunks for a user query
    generate.py    # Planned answer generation layer
    app.py         # Planned app/UI layer

  data/
    raw/
      papers/
        .gitkeep   # Add research PDFs here locally
    processed/     # Generated JSONL files, ignored by Git
    chroma/        # Generated vector DB, ignored by Git

  requirements.txt
  .gitignore
  README.md
Papers Used
The local version of this project was tested with the following papers:
Attention Is All You Need
Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
ReAct: Synergizing Reasoning and Acting in Language Models
Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection
Toolformer: Language Models Can Teach Themselves to Use Tools
PDFs are not included in this repository. To reproduce the project, download the papers and place them in:
data/raw/papers/
Setup
Clone the repository:
git clone https://github.com/roque3007/RagProject.git
cd RagProject
Create and activate a virtual environment:
python3 -m venv .venv
source .venv/bin/activate
Install dependencies:
pip install -r requirements.txt
Usage
1. Add PDFs
Place your research paper PDFs in:
data/raw/papers/
Example:
data/raw/papers/attention_is_all_you_need.pdf
data/raw/papers/self_rag.pdf
2. Extract Pages
python src/ingest.py
This creates:
data/processed/pages.jsonl
Each line contains one page of extracted text with metadata.
3. Create Chunks
python src/chunk.py
This creates:
data/processed/chunks.jsonl
Each chunk includes:
chunk_id
paper_id
source_file
page
chunk_index
text
4. Build Vector Index
python src/index.py
This embeds all chunks and stores them in:
data/chroma/
5. Retrieve Relevant Evidence
python src/retrieve.py "What is self-attention?"
Example output:
Found 5 results
================================================================================
Result 1 | paper=attention_is_all_you_need page=7 distance=1.2504
...
How It Works
The pipeline has five main steps:
PDFs
→ page extraction
→ chunking
→ embedding
→ vector search
→ retrieved evidence
First, PDFs are parsed into page-level text records. Then each page is split into overlapping chunks so retrieval can target focused passages instead of entire pages. Each chunk is embedded using a sentence-transformer model and stored in ChromaDB. When a user asks a question, the question is embedded using the same model, and ChromaDB returns the nearest chunks by vector similarity.
Why Chunking Matters
Whole pages often contain multiple topics, which can make retrieval noisy. Smaller chunks improve precision because each embedded passage focuses on a narrower idea. Overlap helps preserve context when important information falls near a chunk boundary.
Current Status
This project currently implements the retrieval foundation of a RAG system:
Document ingestion
Text chunking
Embedding
Vector storage
Semantic retrieval
The next step is to add answer generation using an LLM, where the model answers only from retrieved context and includes citations.
Planned Improvements
Add LLM-based answer generation
Add citations in final answers
Build a Streamlit UI
Add retrieval evaluation metrics
Compare chunk sizes and overlap settings
Add reranking
Add hybrid search with keyword + vector retrieval
Example Interview Explanation
I built a RAG pipeline over AI research papers. The system ingests PDFs, extracts page-level text, chunks the text with overlap, embeds each chunk using a sentence-transformer model, and stores the vectors in ChromaDB. For a user question, it embeds the query, retrieves the most semantically similar chunks, and returns source metadata like paper ID and page number. This creates the retrieval foundation for a citation-grounded question-answering system.
License
This project is for educational and portfolio purposes.
