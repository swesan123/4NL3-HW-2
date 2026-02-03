"""
Setup LogHub Dataset for Analysis

This script helps set up LogHub datasets for analysis. It can work with:
1. Existing data files (Linux.txt) by creating categories
2. Actual LogHub datasets when available

For this assignment, we'll create categories from the existing Linux.txt
by splitting into different log types or time periods.

Author: Swesan Pathmanathan
Course: 4NL3
Date: January 2026
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from data_collection import DatasetCollector

def create_categories_from_linux_log(linux_file: Path, output_dir: Path):
    """
    Create categories from Linux log file by splitting into error and normal logs.
    
    Args:
        linux_file: Path to Linux.txt
        output_dir: Directory to save categorized documents
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    error_dir = output_dir / "error"
    normal_dir = output_dir / "normal"
    error_dir.mkdir(exist_ok=True)
    normal_dir.mkdir(exist_ok=True)
    
    error_keywords = [
        'error', 'failed', 'failure', 'killed', 'denied', 
        'refused', 'timeout', 'exception', 'critical', 'alert'
    ]
    
    error_count = 0
    normal_count = 0
    
    with open(linux_file, 'r', encoding='utf-8', errors='replace') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            # Check if line contains error keywords
            is_error = any(keyword.lower() in line.lower() for keyword in error_keywords)
            
            if is_error and error_count < 500:  # Limit to 500 error logs
                doc_path = error_dir / f"doc_{error_count:06d}.txt"
                with open(doc_path, 'w', encoding='utf-8') as doc_file:
                    doc_file.write(line)
                error_count += 1
            elif not is_error and normal_count < 500:  # Limit to 500 normal logs
                doc_path = normal_dir / f"doc_{normal_count:06d}.txt"
                with open(doc_path, 'w', encoding='utf-8') as doc_file:
                    doc_file.write(line)
                normal_count += 1
            
            # Stop if we have enough documents
            if error_count >= 500 and normal_count >= 500:
                break
    
    # Create metadata
    metadata = {
        "categories": {
            "error": error_count,
            "normal": normal_count
        },
        "total_documents": error_count + normal_count,
        "min_docs_per_category": 100,
        "max_docs_per_category": 500,
        "source_info": {
            "dataset_name": "Linux_Log_Categorized",
            "source": str(linux_file),
            "method": "Split by error keywords"
        }
    }
    
    metadata_file = output_dir / "dataset_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"Created dataset with categories:")
    print(f"  error: {error_count} documents")
    print(f"  normal: {normal_count} documents")
    print(f"  Total: {error_count + normal_count} documents")
    print(f"Metadata saved to {metadata_file}")
    
    return metadata

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Setup LogHub dataset for analysis"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/loghub",
        help="Output directory for categorized data"
    )
    parser.add_argument(
        "--linux-file",
        type=str,
        default="data/Linux.txt",
        help="Path to Linux.txt file"
    )
    
    args = parser.parse_args()
    
    linux_file = Path(args.linux_file)
    if not linux_file.exists():
        print(f"Error: {linux_file} not found")
        print("Please ensure Linux.txt exists in data/ directory")
        exit(1)
    
    print(f"Creating categories from {linux_file}...")
    metadata = create_categories_from_linux_log(linux_file, Path(args.data_dir))
    
    print("\nDataset ready for analysis!")
    print(f"Run: python corpus_analysis.py --data-dir {args.data_dir} --output-dir output")
