# Generic RAG Template

A reusable Retrieval-Augmented Generation pipeline you can point at **any**
folder of documents — not tied to any one domain, language, or client. This
is the "adapt-in-a-day" version of the Ayat-Ontology RAG work in
`../ontologyrag_q_kaggle`: same retrieve-then-generate architecture, but
generic document loading/chunking instead of Quran/Tafsir-specific logic.

Meant to be used as a starting point for freelance/consulting RAG work: swap
in a client's documents, run it, and you have a working grounded Q&A system
in minutes.

## What's in this folder

- `rag_pipeline.py` — the whole pipeline as an importable library **and** a
  CLI. No notebook or Kaggle-specific code — runs anywhere Python does.
- `requirements.txt` — dependencies.
- `RAG_Template_Demo.ipynb` — a Kaggle-runnable notebook walking through the
  same pipeline interactively, with a small demo document set.

## Quick start (local / any machine)

```bash
pip install -r requirements.txt

# Point it at a folder of .txt / .md / .csv / .pdf / .docx files
python rag_pipeline.py --docs ./my_client_docs --query "What is the refund policy?"

# Or ask multiple questions interactively
python rag_pipeline.py --docs ./my_client_docs --interactive

# Persist the index so you don't re-embed every run
python rag_pipeline.py --docs ./my_client_docs --save-index ./my_index
python rag_pipeline.py --load-index ./my_index --query "..."
```

By default it uses a **free, open-weight model** (`Qwen2.5-1.5B-Instruct`)
for generation — no API key needed, runs on CPU (slowly) or GPU (fast). Add
`--openai-key sk-...` (or set the `OPENAI_API_KEY` environment variable) to
use GPT instead for higher-quality answers.

## Quick start (as a library)

```python
from rag_pipeline import RAGPipeline

rag = RAGPipeline()                       # free local LLM by default
# rag = RAGPipeline(openai_api_key="sk-...")  # or use GPT

rag.build_index("./my_client_docs")
answer, sources = rag.answer("What is the refund policy?")

print(answer)
print([s.source for s in sources])        # which files it was grounded in
```

## Using it for a consulting engagement

1. **Discovery**: ask the client for a folder export of the documents they
   want searchable (policies, product docs, support tickets, contracts,
   internal wikis — anything readable as text/PDF/docx/CSV).
2. **Adapt**: point `rag_pipeline.py` at that folder. No code changes needed
   for typical text-heavy documents; tune `chunk_size`/`overlap` if answers
   come back missing context (increase chunk_size) or too noisy (decrease
   it).
3. **Demo**: run `--interactive` mode live with the client, asking real
   questions from their domain. `sources` in each answer double as a trust
   signal — show them exactly which document backed each answer.
4. **Harden for production** (this is where the billable work usually is):
   - swap the free local LLM for GPT/Claude for quality, or fine-tune the
     local model
   - add authentication/access control per document if data is sensitive
   - persist the index (`save_index`/`load_index`) instead of rebuilding it
     every run, and add a re-index job when documents change
   - add evaluation: collect a small set of real question/answer pairs from
     the client and measure retrieval + answer quality before going live
   - wrap `RAGPipeline` in an API (FastAPI/Flask) or chat UI

## Extending it

- **Different embedding model**: pass `embedding_model=` to `RAGPipeline` —
  swap in a domain-specific or higher-quality embedding model as needed.
- **More file types**: add a new function to `_READERS` in `rag_pipeline.py`
  (e.g. `.html`, `.eml`).
- **Smarter chunking**: `chunk_text()` is a simple word-count sliding
  window. For structured documents (e.g. verse-by-verse, section-by-section)
  a domain-aware chunker like the one in `../ontologyrag_q_kaggle` will
  retrieve more precisely.
- **Metadata filtering**: `Chunk` currently only tracks `source`; add fields
  (date, category, department) and filter `retrieve()` results by them for
  more targeted retrieval.
