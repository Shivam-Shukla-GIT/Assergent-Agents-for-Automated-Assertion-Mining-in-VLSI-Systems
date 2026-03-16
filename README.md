# 🤖 Assergent: Agents for Automated Assertion Mining in VLSI Systems

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Gemini API](https://img.shields.io/badge/API-Gemini%202.0-green.svg)](https://ai.google.dev/)

**Assergent** is an autonomous, retrieval-augmented framework developed at IIIT Delhi's NDCL Lab that generates formally verified SystemVerilog Assertions (SVAs) from natural language specifications. It bridges the gap between high-level design intent and formal verification using semantic retrieval over a curated motif database and iterative closed-loop refinement via Cadence JasperGold.

---

## 🎯 Overview

Writing SystemVerilog Assertions manually demands deep expertise in temporal logic, clock-domain semantics, and reset handling — and is highly error-prone. Assergent addresses this by:

- **Classifying** each property across five orthogonal complexity axes to enable structure-aware retrieval
- **Retrieving** formally verified assertion motifs from a FAISS vector database using semantic similarity
- **Generating** SVA via Gemini 2.0 Flash conditioned on retrieved examples (RAG)
- **Refining** iteratively in a closed loop with Cadence JasperGold using counterexample feedback
- **Thresholding** retrieval confidence to autonomously flag unreliable generations before they reach the designer

---

## ✨ Key Features

### 🔍 **5-Axis Classification Engine**
Every assertion is scored across five orthogonal complexity dimensions — temporal, signal density, sampled value functions, design context, and semantic depth — forming a feature vector used for structure-aware retrieval and stratified dataset splits.

### 🗄️ **RAG Knowledge Base**
A curated database of formally verified assertion motifs is indexed with FAISS using `sentence-transformers` embeddings. At query time, the top-K most semantically similar motifs are retrieved and injected into the LLM prompt.

### 🤖 **LLM Integration (Gemini 2.0 Flash)**
Gemini 2.0 Flash is used for both initial SVA generation and iterative refinement. Prompts are dynamically constructed with retrieved motifs and, on failure, with JasperGold counterexample feedback.

### 🔄 **Closed-Loop Refinement with JasperGold**
Generated assertions are formally verified with Cadence JasperGold. Syntax errors, counterexamples, and unreachable properties are parsed from logs and fed back to the LLM for correction — automatically, across multiple iterations.

### 🛡️ **Confidence Thresholding**
A cosine similarity threshold (τ = 0.4) is used to gate RAG-based generation. Queries below the threshold trigger a warning and fall back to heuristic prompting, preventing hallucinated assertions from silently entering the verification flow.

### 📊 **Stratified Dataset Splits**
The `stratified_splitter.py` ensures train/test splits maintain proportional coverage across all five complexity axes, with Jaccard-based paraphrase leakage detection to ensure evaluation integrity.

---

## 🏗️ Architecture

```
User Spec (English)
        │
        ▼
┌─────────────────────┐
│  5-Axis             │  ← Temporal · Signal · SVF · Design · Semantic
│  Classification     │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐     ┌──────────────────────┐
│  FAISS Knowledge    │◄────│  Golden SVA Database  │
│  Base               │     │  (.txt files)         │
└────────┬────────────┘     └──────────────────────┘
         │  Top-K similar motifs (cosine similarity)
         ▼
┌─────────────────────┐
│  Prompt Builder     │  ← RAG-augmented prompt construction
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Gemini 2.0 Flash   │  ← SVA generation
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  JasperGold FV      │  ← Formal verification
└────────┬────────────┘
         │  Counterexample / error feedback
         └──────────────► Refinement loop (up to N iterations)
```

---

## 📁 Repository Structure

```
assergent/
├── assertion_classifier.py    # 5-axis complexity classification engine
├── prepare_database.py        # Build JSON knowledge base from raw .txt files
├── stratified_splitter.py     # Complexity-stratified train/test split
├── rag_assertion_generator.py # Main RAG pipeline + JasperGold integration
├── complete_workflow.sh       # End-to-end automation script
├── install_packages.py        # Dependency installer
├── database/                  # Raw assertion text files
├── requirements.txt
```

---

## 🛠️ Installation

### Prerequisites

- Python 3.9+
- Conda (recommended)
- Cadence JasperGold (for closed-loop verification)
- Gemini API key ([get one here](https://aistudio.google.com/app/apikey))

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/Shivam-Shukla-GIT/Assergent.git
cd Assergent

# 2. Create and activate environment
conda create -n assergent python=3.9 -y
conda activate assergent

# 3. Install Python dependencies
python install_packages.py

# 4. Install FAISS
conda install -c conda-forge faiss-cpu -y

# 5. Set your Gemini API key
export GEMINI_API_KEY='your-api-key-here'
```

---

## 📂 Database Format & Auto-Extraction

You only need to provide the English specification, the golden SVA, and the RTL module. **The framework automatically extracts the comma-separated signal list and infers the design context.**

Each entry in the `database/` directory should be a plain `.txt` file formatted with specific tags:

```
specs: "Assert that when reset is deasserted, the counter value resets to zero."

property p_reset_counter;
  @(posedge clk) disable iff (!rst_n)
    $rose(rst_n) |=> (count == 0);
endproperty
assert property (p_reset_counter);

RTL: "module counter (
  input logic clk,
  input logic rst_n,
  output logic [7:0] count
);
...
endmodule"
```

*Note: The algorithm processes these tags to dynamically determine signal dependencies and structural context.*

| Section | Content |
|---------|---------|
| 1 | English specification |
| 2 | Golden SVA (formally verified) |
| 3 | RTL Design module |

Place all `.txt` files in a `database/` directory before running the pipeline.

---

## 🚀 Quick Start

### Option A — Full Automated Pipeline

```bash
chmod +x complete_workflow.sh
./complete_workflow.sh --input database/ --output-dir processed/
```

Runs all three setup steps in sequence:
1. Classifies all assertions (5-axis engine)
2. Performs a stratified 2/3–1/3 train/test split
3. Builds `specs_database.json` from the training split

For interactive split configuration:
```bash
./complete_workflow.sh --input database/ --interactive
```

---

### Option B — Step by Step

**Step 1 — Classify:**
```bash
python assertion_classifier.py \
    --input database/ \
    --output processed/classification_report.json
```

**Step 2 — Split:**
```bash
python stratified_splitter.py \
    --classification processed/classification_report.json \
    --input database/ \
    --output-train processed/knowledge_db/ \
    --output-test processed/test_set/ \
    --seed 42
```

**Step 3 — Build database:**
```bash
python prepare_database.py \
    --input processed/knowledge_db/ \
    --output specs_database.json \
    --classification processed/split_metadata.json
```

**Step 4 — Generate assertions:**
```bash
export GEMINI_API_KEY='your-key'
python rag_assertion_generator.py
```

The generator prompts you interactively for:

| Prompt | Default |
|--------|---------|
| English specification (or `@filepath`) | — |
| RTL design file (`.sv`) | — |
| Knowledge base path | `specs_database.json` |
| Max refinement iterations | `10` |
| Number of RAG examples to retrieve (k) | `3` |

---

## 📊 Output Structure

Each run creates a timestamped directory under `results/`:

```
results/
└── <module_name>_<YYYYMMDD_HHMMSS>/
    ├── prompt_iter_1.txt               # RAG-augmented prompt
    ├── llm_response_iter_1.txt         # Raw Gemini response
    ├── assertion_iter_1.sv             # Extracted assertion
    ├── rtl_with_assertion_iter_1.sv    # RTL with assertion embedded
    ├── run_iter_1.jg                   # JasperGold TCL script
    ├── jasper_output_iter_1.log        # Formal verification log
    ├── ...                             # (repeated per iteration)
    └── final_assertion.sv              # Final verified assertion (on success)
```

---

## 🔢 Five-Axis Classification

| Axis | What it measures | Score range |
|------|-----------------|-------------|
| **Temporal (S_T)** | Combinational → complex sequences (`throughout`, `[*]`) | 10 – 90 |
| **Signal (S_S)** | Single signal → 7+ signals or 10+ logical operators | 10 – 90 |
| **SVF (S_SV)** | No sampled-value functions → advanced (`$past`, `$stable`) | 0 – 80 |
| **Design (S_D)** | Simple combinational (<50 LOC) → CPU/pipeline (>500 LOC) | 10 – 95 |
| **Semantic (S_M)** | Direct mapping → cross-module patterns | 20 – 90 |

**Overall score** = 0.25·S_T + 0.20·S_S + 0.20·S_SV + 0.20·S_D + 0.15·S_M

Assertions scoring ≤ 35 are **simple**, 36–65 are **medium**, > 65 are **complex**.

---

## 🛡️ Confidence Thresholding

| Retrieval Confidence | Generation Behavior |
|----------------------|---------------------|
| > 0.4 | RAG-based generation (100% success rate) |
| ≤ 0.4 | Warning issued; falls back to heuristic prompting |

---

## 🔧 Configuration

**Gemini API key** — set as an environment variable (never hardcode):
```bash
export GEMINI_API_KEY='your-key-here'
```

**Max iterations** — passed interactively or configurable in `rag_assertion_generator.py`:
```python
max_iterations = 10   # default
```

**Retrieval threshold** — in `rag_assertion_generator.py`:
```python
min_similarity = 0.4  # confidence gate
```

---

## 🤝 Contributing

Contributions are welcome. Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Follow PEP 8 and add docstrings / type hints
4. Submit a pull request with a clear description

Bug reports and feature requests go through [GitHub Issues](https://github.com/Shivam-Shukla-GIT/Assergent/issues).

---

## 🙏 Acknowledgments

### Development Team
- **Shivam Shukla** — Core development and architecture
- **Utkarsh Chaudhary** — Assertion dataset writing, integration and testing
- **Dr. Sneh Saurabh** — Advisor and guidance

### Institutional Support
- **IIIT Delhi** — Research support
- **NDCL Lab** — Laboratory resources and guidance

### Technology Stack
- **Google Gemini 2.0 Flash** — LLM generation
- **FAISS** — Vector similarity search
- **sentence-transformers** — Semantic embeddings
- **Cadence JasperGold** — Formal verification

---

## 🔗 Related Links

- [SystemVerilog LRM](https://ieeexplore.ieee.org/document/8299595)
- [JasperGold Platform](https://www.cadence.com/en_US/home/tools/system-design-and-verification/formal-and-static-verification/jasper-gold-verification-platform.html)
- [Google Gemini API](https://ai.google.dev/)
- [IIIT Delhi](https://iiitd.ac.in/)
- [NDCL Lab](https://sites.google.com/site/snehsaurabhhome/research/nanoscale-devices-and-circuits-lab/)

---

<div align="center">

**Developed with ❤️ at IIIT Delhi | NDCL Lab**

[![IIITD](https://img.shields.io/badge/IIIT-Delhi-blue.svg)](https://iiitd.ac.in/)
[![NDCL](https://img.shields.io/badge/NDCL-Lab-darkblue.svg)](https://sites.google.com/site/snehsaurabhhome/research/nanoscale-devices-and-circuits-lab/)

</div>
