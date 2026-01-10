# 📚 Backstory Consistency Verification System

A retrieval-based NLP system that verifies whether a given backstory claim is **consistent or contradictory** with a novel, using semantic embeddings and evidence retrieval.

---

## 🔍 Problem Overview

Given:
- A **backstory claim**
- A **primary text (novel)**

The system:
1. Retrieves the most relevant excerpts from the novel
2. Scores semantic similarity
3. Decides **Consistency (1) / Contradiction (0)**
4. Returns **verbatim textual evidence**

---

## 🧠 Approach

- **Chunking**: Overlapping word-based chunks
- **Embedding**: Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Retrieval**: Cosine similarity (top-k)
- **Decision Rule**: Threshold-based verification
- **Explainability**: Top-k excerpts returned as evidence

---

## 📁 Project Structure
backstory-consistency-kds2026/<br>
│<br>
├── data/<br>
│   ├── books/<br>
│   │   ├── The Count of Monte Cristo.txt<br>
│   │   └── In Search of the Castaways.txt<br>
│   │       **→ Primary novels used as factual ground truth**<br>
│   │<br>
│   ├── train.csv<br>
│   │   **→ Labeled backstory claims used for threshold tuning<br>
│   │     Columns: [id, content, book_name, label]**<br>
│   │<br>
│   └── test.csv<br>
│       **→ Unlabeled claims for final evaluation**<br>
│<br>
├── src/<br>
│   ├── chunking.py<br>
│   │   **→ Splits full novels into overlapping word chunks**<br>
│   │<br>
│   ├── embeddings.py<br>
│   │   **→ Generates semantic embeddings using Sentence-Transformers**<br>
│   │<br>
│   ├── retrieval.py<br>
│   │   **→ Retrieves top-k most relevant chunks using cosine similarity**<br>
│   │<br>
│   ├── consistency.py<br>
│   │   **→ Core logic:<br>
│   │     - Compares backstory claim with retrieved chunks<br>
│   │     - Applies threshold to decide consistency / contradiction**<br>
│   │<br>
│   └── io_utils.py<br>
│       **→ Utility functions for loading books by name** <br>
│
├── scripts/<br>
│   ├── run_train_eval.py<br>
│   │   **→ Uses train.csv to:<br>
│   │     - Run consistency checks<br>
│   │     - Sweep similarity thresholds<br>
│   │     - Select best threshold based on accuracy**<br>
│   │<br>
│   └── run_test.py<br>
│       **→ Runs final consistency predictions on test.csv<br>
│         → Outputs decisions and supporting evidence**<br>
│<br>
├── results/<br>
│   └── results.csv<br>
│       **→ Stores generated predictions and evaluation outputs**<br>
│<br>
├── report/<br>
│   └── Report.pdf<br>
│       **→ Final report / submission documents (if required)**<br>
│<br>
├── README.md<br>
│   **→ Project documentation**<br>
│<br>
├── test_consistency.py<br>
│   **→ Quick sanity checks for consistency decision logic**<br>
│<br>
└── .gitignore<br>
    **→ Excludes cache, environments, and generated files**<br>

