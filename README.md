
# Assignment 2 – Corpus Analysis

## Overview

This project implements a complete corpus analysis pipeline for analyzing log datasets from LogHub. It performs bag-of-words conversion, Naive Bayes log-likelihood analysis, LDA topic modeling, and experimentation with different preprocessing configurations.

## Files and Directories

### Core Modules
- `data_collection.py` – LogHub dataset collection with file size management
- `bow_processor.py` – Bag-of-words conversion with sparse matrix support
- `naive_bayes_analysis.py` – Naive Bayes LLR calculations
- `topic_modeling.py` – LDA topic modeling using gensim
- `experiments.py` – Experimentation framework for testing variations
- `corpus_analysis.py` – Main pipeline orchestrating all analysis steps
- `normalize_text.py` – Preprocessing functions (reused from HW-1)

### Configuration and Scripts
- `environment.yml` – Conda environment specification
- `scripts/check_latex.sh` – Pre-commit LaTeX compilation check
- `scripts/verify_pdf.py` – PDF verification script

### Data and Output
- `data/loghub/` – LogHub datasets organized by category
- `output/` – Generated outputs (BoW matrices, analysis results, visualizations)
- `report/` – Modular LaTeX report structure
  - `main.tex` – Master LaTeX file
  - `sections/` – Individual section files
  - `format/` – Formatting and style files
  - `config/` – Configuration files

### Testing
- `tests/` – Test suite (from HW-1)

## Requirements
- Python 3.12 (gensim compatibility)
- Conda
- Required Python packages are listed in `environment.yml`
- LaTeX distribution (for report compilation)

### Create or update the conda environment
From the repository root:

```bash
conda env create -f environment.yml
conda activate 4nl3
```

## How to Run

### Setup

From the repository root:

```bash
# Create or update conda environment
conda env create -f environment.yml
conda activate 4nl3

# Download NLTK data (if not already downloaded)
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

### Dataset Collection

```bash
# Collect LogHub dataset (adapt based on your dataset structure)
python data_collection.py --local-path /path/to/loghub/data --min-docs 100
```

### Full Pipeline

```bash
# Run complete corpus analysis pipeline
python corpus_analysis.py \
    --data-dir data/loghub \
    --output-dir output \
    --lowercase \
    --stopwords \
    --num-topics 10

# With experiments
python corpus_analysis.py \
    --data-dir data/loghub \
    --output-dir output \
    --lowercase \
    --stopwords \
    --experiments
```

### Individual Modules

```bash
# Bag-of-Words conversion
python bow_processor.py \
    --data-dir data/loghub \
    --output-dir output/bow \
    --representation count \
    --lowercase --stopwords

# Naive Bayes analysis
python naive_bayes_analysis.py \
    --data-dir data/loghub \
    --output-dir output/naive_bayes \
    --top-k 10 \
    --lowercase --stopwords

# Topic modeling
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

# Run experiments
python experiments.py \
    --data-dir data/loghub \
    --output-dir output/experiments
```

### Report Compilation

```bash
# Compile LaTeX report
cd report
pdflatex main.tex
pdflatex main.tex  # Run twice for references
mv main.pdf homework2_report.pdf

# Or use the check script
bash scripts/check_latex.sh

# Verify PDF
python scripts/verify_pdf.py report/homework2_report.pdf
```

## Dataset Information

This project uses datasets from the [LogHub repository](https://github.com/logpai/loghub). LogHub provides large-scale log parsing datasets including Apache, BGL, Hadoop, HDFS, HealthApp, and others.

**Citation**: If using LogHub datasets, please cite:
> He, P., Zhu, J., Zheng, Z., & Lyu, M. R. (2017). Drain: An online log parsing approach with fixed depth tree. In Proceedings of the 38th International Conference on Software Engineering Companion (ICSE-C '16).

## Notes

- All file operations use UTF-8 encoding
- File size limits are enforced during dataset collection (default: 100MB per file)
- Large datasets are sampled to meet requirements without excessive file sizes
- Sparse matrices are used for memory efficiency with large document-term matrices
- LaTeX compilation is checked via pre-commit hook

## Testing (For my personal testing)

A comprehensive test suite is included to ensure correctness and robustness.

### Run Tests

```bash
# activate environment
conda activate 4nl3

# install test dependencies (if not already installed)
pip install pytest pytest-cov

# run all fast tests (recommended)
pytest tests/ -m "not slow" -v

# run all tests including slow combinations with real data
pytest tests/ -v

# run specific test files
pytest tests/test_functions.py -v          # unit tests
pytest tests/test_all_combinations.py -v   # all 48 flag combinations

# run with coverage report
pytest tests/ --cov=normalize_text --cov-report=html
```

### Test Coverage

- **Unit tests** (`test_functions.py`) - Tests for individual functions (tokenization, normalization, file I/O)
- **Integration tests** (`test_integration.py`) - Command-line execution and flag combinations
- **Combination tests** (`test_all_combinations.py`) - All 48 valid flag combinations tested to ensure soundness

The test suite validates:
- All normalization functions work correctly
- All 48 valid flag combinations execute successfully
- Mutually exclusive flags (stem/lemmatize) are properly rejected
- UTF-8 error handling with malformed byte sequences
- Output files are created correctly

## Rationale: UTF-8 with errors="replace"

The file reader uses UTF-8 with `errors="replace"` to ensure that any malformed bytes in large or mixed-encoding corpora do not crash the run. Invalid sequences are replaced with the Unicode replacement character, allowing processing to continue while keeping output deterministic.

## Generative AI Usage Disclosure (Required)

Generative AI tools were used **only for conceptual clarification, debugging guidance, and assistance with code structure and documentation wording**. All final code, decisions, and interpretations were written and verified by the student.

### AI Usage Details
- **Model:** ChatGPT 5.2
- **Provider:** OpenAI
- **Hardware type:** Cloud-based GPU/accelerator
- **Region of compute:** Unknown
- **Time used:** Approximately 2–3 hours total across multiple short interactions
- **How values were estimated:** Time was approximated based on active interaction duration during development and debugging
- **Estimated emissions:**  
  ~4.32 g CO₂ per query × ~25 queries ≈ **108 g CO₂**

