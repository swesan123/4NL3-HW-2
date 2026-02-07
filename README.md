# Assignment 2 – Corpus Analysis

## Overview

This project implements a complete corpus analysis pipeline for analyzing log datasets from LogHub. It performs bag-of-words conversion, Naive Bayes log-likelihood analysis, LDA topic modeling, and experimentation with different preprocessing configurations.

## Files and Directories

### Core Modules
- `data_collection.py` – Dataset organization and preprocessing
- `bow_processor.py` – Bag-of-words conversion with sparse matrix support
- `naive_bayes_analysis.py` – Naive Bayes LLR calculations
- `topic_modeling.py` – LDA topic modeling using gensim
- `experiments.py` – Experimentation framework for testing variations
- `corpus_analysis.py` – Main pipeline orchestrating all analysis steps
- `normalize_text.py` – Preprocessing functions (reused from HW-1)

### Configuration and Scripts
- `environment.yml` – Conda environment specification
- `scripts/generate_tables.py` – Generate LaTeX tables from analysis results (optional)

### Data and Output
- `data/loghub/` – LogHub datasets organized by category
- `output/` – Generated outputs (BoW matrices, analysis results, visualizations)
  - `output/bow/` – Bag-of-words matrices and vocabulary
  - `output/naive_bayes/` – Naive Bayes LLR results
  - `output/topic_modeling/` – LDA model and topic distributions
  - `output/experiments/` – Experimentation results
- `report/` – LaTeX report source files (optional, for report generation)

## Requirements

- Python 3.12
- Conda (for environment management)
- Required Python packages are listed in `environment.yml`

## Quick Start

If you've cloned this repository, follow these steps to run the analysis:

```bash
# 1. Create and activate conda environment
conda env create -f environment.yml
conda activate 4nl3

# 2. Download required NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"

# 3. Run the complete analysis pipeline
python corpus_analysis.py \
    --data-dir data/loghub \
    --output-dir output \
    --lowercase \
    --stopwords \
    --num-topics 10 \
    --experiments
```

Results will be saved in the `output/` directory.

## Setup

### Dataset Structure

The dataset is organized in `data/loghub/` with subdirectories for each category. Each category directory contains text files (documents). The dataset structure is:
```
data/loghub/
├── error/
│   ├── doc_000001.txt
│   ├── doc_000002.txt
│   └── ...
├── normal/
│   ├── doc_000001.txt
│   ├── doc_000002.txt
│   └── ...
└── dataset_metadata.json
```

## How to Run

### Complete Pipeline (Recommended)

Run the full corpus analysis pipeline with all required steps:

```bash
# Basic run with default settings
python corpus_analysis.py \
    --data-dir data/loghub \
    --output-dir output \
    --lowercase \
    --stopwords \
    --num-topics 10

# With experiments (Section 2.5)
python corpus_analysis.py \
    --data-dir data/loghub \
    --output-dir output \
    --lowercase \
    --stopwords \
    --num-topics 10 \
    --experiments
```

This will execute:
1. **Dataset Loading** – Load documents by category
2. **Bag-of-Words Conversion** (Section 2.2) – Convert documents to BoW format
3. **Naive Bayes Analysis** (Section 2.3) – Compute log-likelihood ratios for top words per category
4. **Topic Modeling** (Section 2.4) – Run LDA and compute topic distributions
5. **Experiments** (Section 2.5) – Test different preprocessing configurations (if `--experiments` flag is used)

### Individual Modules

You can also run each module separately:

#### Bag-of-Words Conversion

```bash
python bow_processor.py \
    --data-dir data/loghub \
    --output-dir output/bow \
    --representation count \
    --lowercase --stopwords
```

#### Naive Bayes Analysis

```bash
python naive_bayes_analysis.py \
    --data-dir data/loghub \
    --output-dir output/naive_bayes \
    --top-k 10 \
    --lowercase --stopwords
```

#### Topic Modeling

```bash
# Standard topic modeling
python topic_modeling.py \
    --data-dir data/loghub \
    --output-dir output/topic_modeling \
    --num-topics 10 \
    --lowercase --stopwords

# Find optimal number of topics
python topic_modeling.py \
    --data-dir data/loghub \
    --find-optimal \
    --lowercase --stopwords
```

#### Run Experiments

```bash
python experiments.py \
    --data-dir data/loghub \
    --output-dir output/experiments
```

## Output Files

After running the pipeline, you'll find:

- **BoW Results** (`output/bow/`):
  - `bow_matrix.npz` – Sparse document-term matrix
  - `bow_vocabulary.json` – Vocabulary mapping
  - `bow_metadata.json` – Metadata about the BoW conversion

- **Naive Bayes Results** (`output/naive_bayes/`):
  - `naive_bayes_results.json` – Top words per category with LLR scores
  - `llr_scores.json` – All LLR scores

- **Topic Modeling Results** (`output/topic_modeling/`):
  - `lda_model.model` – Trained LDA model
  - `topic_modeling_results.json` – Topic distributions and top terms
  - `lda_visualization.html` – Interactive visualization (if pyLDAvis is available)

- **Experiment Results** (`output/experiments/`):
  - Results from different preprocessing configurations

## Dataset Information

This project uses datasets from the [LogHub repository](https://github.com/logpai/loghub). LogHub provides large-scale log parsing datasets including Apache, BGL, Hadoop, HDFS, HealthApp, and others.

**Citation**: 
> He, P., Zhu, J., Zheng, Z., & Lyu, M. R. (2017). Drain: An online log parsing approach with fixed depth tree. In Proceedings of the 38th International Conference on Software Engineering Companion (ICSE-C '16).

## Notes

- All file operations use UTF-8 encoding
- Sparse matrices are used for memory efficiency with large document-term matrices
- The pipeline supports various preprocessing options: `--lowercase`, `--stopwords`, `--stem`, `--lemmatize`

## Generative AI Usage Disclosure

Generative AI tools (specifically Cursor AI Assistant) were used extensively throughout this assignment for conceptual clarification, debugging guidance, code structure assistance, and documentation wording. All final code, decisions, and interpretations were written and verified by the student.

### AI Usage Details
- **Model:** Cursor AI Assistant (Claude-based model)
- **Provider:** Anthropic (via Cursor IDE)
- **Primary uses:**
  - Code structure and architecture design for corpus analysis pipeline
  - Debugging Python import errors and dependency issues
  - Assistance with LaTeX table generation and formatting
  - Clarification of Naive Bayes LLR calculations and topic modeling concepts
  - Documentation and README writing assistance
- **Time used:** Approximately 4-6 hours of active AI assistance over the course of assignment completion
- **Estimated emissions:**
  - Estimated queries/interactions: ~50-75 interactions over 4-6 hours
  - Average emissions per query (Claude-based models): ~4.32 g CO₂ per query
  - Calculation: 50 queries × 4.32 g CO₂ = 216 g CO₂ (0.216 kg CO₂) to 75 queries × 4.32 g CO₂ = 324 g CO₂ (0.324 kg CO₂)
  - **Total estimated emissions:** 0.2-0.3 kg CO₂ equivalent

### Student Contribution
All analysis results, interpretations, and conclusions presented in this work are my own. The AI assistant was used as a tool for implementation and clarification, but all analytical insights, discussion points, and final code decisions were made and verified by me.
