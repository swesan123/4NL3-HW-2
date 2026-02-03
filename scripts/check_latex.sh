#!/bin/bash
# Pre-commit hook script to check LaTeX compilation
# This script checks if LaTeX files have changed and compiles them

set -e

REPORT_DIR="report"
MAIN_TEX="$REPORT_DIR/main.tex"
OUTPUT_PDF="$REPORT_DIR/homework2_report.pdf"

# Check if LaTeX files have changed
if git diff --cached --name-only | grep -q "\.tex$"; then
    echo "LaTeX files detected in commit. Checking compilation..."
    
    # Change to report directory
    cd "$REPORT_DIR" || exit 1
    
    # Check if pdflatex is available
    if ! command -v pdflatex &> /dev/null; then
        echo "Error: pdflatex not found. Please install a LaTeX distribution."
        exit 1
    fi
    
    # Compile LaTeX (run twice for references)
    echo "Compiling LaTeX..."
    pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 || {
        echo "Error: LaTeX compilation failed. Check main.log for details."
        exit 1
    }
    
    pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 || {
        echo "Error: LaTeX compilation failed on second pass."
        exit 1
    }
    
    # Rename output PDF
    if [ -f "main.pdf" ]; then
        mv main.pdf homework2_report.pdf
        echo "LaTeX compilation successful. PDF generated: $OUTPUT_PDF"
    else
        echo "Error: PDF not generated."
        exit 1
    fi
    
    # Clean up auxiliary files (optional)
    # rm -f main.aux main.log main.out main.toc
    
    cd ..
else
    echo "No LaTeX files changed. Skipping compilation check."
fi

exit 0
