# OntologyRAG-Q on Kaggle

This folder contains a runnable Kaggle notebook that reimplements a simplified
version of the pipeline described in the paper *"OntologyRAG-Q: Resource
Development and Benchmarking for Retrieval-Augmented Question Answering in
Qur'anic Tafsir"* (EMNLP 2025), using the dataset published at
https://github.com/sazani/OntologyRAG-Q/tree/main/Resources

**Why "reimplements":** that repo publishes data only (the ontology-annotated
QA dataset, Tafsir books, and benchmark CSVs) — no pipeline code. The notebook
here builds the retrieval-augmented QA pipeline from scratch based on the
dataset's own structure and the paper's description.

## What's in this folder

- `OntologyRAG_Q_Kaggle.ipynb` — the notebook. Downloads the dataset, builds
  verse-level "Ayat-Ontology" chunks, embeds and indexes them with FAISS,
  answers questions with a free open-weight LLM (or GPT if you provide an API
  key), and evaluates against the gold answers.

## Step by step: running it on Kaggle

1. Go to https://www.kaggle.com/code and click **New Notebook**.
2. Open the notebook's **Settings** (right sidebar):
   - **Accelerator:** GPU T4 x1 (free tier is fine)
   - **Internet:** On (required — the notebook downloads the dataset and
     model weights)
3. Import this notebook: `File -> Import Notebook -> Upload`, and select
   `OntologyRAG_Q_Kaggle.ipynb` from this folder.
4. *(Optional, for better answers)* Add an OpenAI API key: `Add-ons ->
   Secrets -> Add a new secret`, name it `OPENAI_API_KEY`. Without this, the
   notebook automatically falls back to a free open-weight model
   (`Qwen2.5-1.5B-Instruct`) — no cost, no signup required.
5. Click **Run All**.

Expect the first run to take a few minutes (downloading the dataset + model
weights). Subsequent cells (retrieval, generation, evaluation) run in seconds
per question.

## Pipeline overview

1. **Download** — pulls `OntologyQA_v1.json` and the benchmark CSVs straight
   from GitHub's raw content URLs.
2. **Ayat-Ontology chunking** — deduplicates the ~4,200 QA rows down to
   ~1,300 unique verse-level chunks, each tagged with the ontology categories
   (`Q_Type1`, `Q_type2_final_English`) of the questions asked about that
   verse.
3. **Retrieval** — embeds chunks with a multilingual sentence-transformer and
   indexes them in FAISS for nearest-neighbor search.
4. **Generation** — retrieves the top-k relevant chunks for a question and
   generates a grounded answer with an LLM (free local model by default, GPT
   if a key is provided).
5. **Evaluation** — scores generated vs. gold answers with a lexical
   token-F1, plus optional BERTScore (the metric the paper found correlates
   best with human judgment).

## Extending it

- Merge in the full 15 Tafsir books (`Resources/Tafaser/Tafaser_DS1`,
  `Tafaser_DS2`) for a much larger retrieval corpus beyond just the QA
  dataset's own passages.
- Add ontology-category filtering/boosting to retrieval, not just raw
  embedding similarity, to more closely match the paper's approach.
- Swap `gpt-4o-mini` for `gpt-4o` in the generation cell for higher-quality
  answers closer to the paper's best-reported results.
