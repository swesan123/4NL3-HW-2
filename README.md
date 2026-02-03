
# Assignment 1 – Text Normalization and Token Frequency Analysis

## Files
- `normalize_text.py` – Main script for normalization, counting, and analysis
- `environment.yml` – Conda environment specification
- `README.md` – Project documentation
- `data/` – Input text corpora (Crime and Punishment, Linux system logs)
- `output/` – Generated outputs (token count files and plots; created automatically)
- `report/` – LaTeX report with analysis, figures, and results
- `tests/` – Comprehensive test suite with unit and integration tests
---

## Requirements
- Python 3.14
- Conda 
- Required Python packages are listed in `environment.yml`

### Create or update the conda environment
From the repository root:

```bash
conda env create -f environment.yml
conda activate 4nl3
```

## How to Run

From the repository root:

```bash
# activate the provided conda environment
conda activate 4nl3

# show help
python normalize_text.py --help

# basic run (tokenize + counts)
python normalize_text.py data/pg2554.txt

# full pipeline with plot
python normalize_text.py data/pg2554.txt -lowercase -myopt -stopwords -stem -analyze

# lemmatization instead of stemming
python normalize_text.py data/Linux.txt -lowercase -lemmatize -analyze
```

Notes:
- Input and output use UTF-8.
- The custom option `-myopt` removes digit-only tokens (e.g., years, timestamps, PIDs) to reduce sparsity.
- `-stem` and `-lemmatize` are mutually exclusive by design.

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

