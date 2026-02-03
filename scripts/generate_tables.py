#!/usr/bin/env python3
"""
Generate LaTeX Tables from Analysis Results

Converts JSON results to LaTeX table format for inclusion in report.

Author: Swesan Pathmanathan
Course: 4NL3
Date: January 2026
"""

import json
from pathlib import Path
from typing import Dict, List


def generate_naive_bayes_table(results_file: Path) -> str:
    """
    Generate LaTeX table for Naive Bayes top words.
    
    Args:
        results_file: Path to naive_bayes_results.json
        
    Returns:
        LaTeX table code
    """
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Top 10 words per category by log-likelihood ratio}",
        "\\begin{tabular}{lll}",
        "\\toprule",
        "Category & Rank & Word (LLR) \\\\",
        "\\midrule"
    ]
    
    for category, words in data["top_words_per_category"].items():
        for rank, word_data in enumerate(words[:10], 1):
            word = word_data["word"]
            llr = word_data["llr"]
            # Escape special LaTeX characters
            word = word.replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")
            lines.append(f"{category.capitalize()} & {rank} & \\texttt{{{word}}} ({llr:.4f}) \\\\")
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    return "\n".join(lines)


def generate_topic_table(results_file: Path, max_topics: int = 10, max_terms: int = 15) -> str:
    """
    Generate LaTeX table for top terms per topic.
    
    Args:
        results_file: Path to topic_modeling_results.json
        max_topics: Maximum number of topics to show
        max_terms: Maximum terms per topic to show
        
    Returns:
        LaTeX table code
    """
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Topic labels (manually assigned based on top terms)
    topic_labels = {
        "0": "System Startup",
        "1": "Kernel Operations",
        "2": "Device Detection",
        "3": "Process Management",
        "4": "Authentication Failures",
        "5": "Network Services",
        "6": "SSH Authentication",
        "7": "Kernel Messages",
        "8": "Kerberos Authentication",
        "9": "Authentication Errors"
    }
    
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Top terms per topic (showing first 15 terms)}",
        "\\begin{tabular}{lll}",
        "\\toprule",
        "Topic & Rank & Term (Probability) \\\\",
        "\\midrule"
    ]
    
    num_topics = min(int(data["num_topics"]), max_topics)
    for topic_id in range(num_topics):
        topic_key = str(topic_id)
        if topic_key in data["topics"]:
            label = topic_labels.get(topic_key, f"Topic {topic_id}")
            terms = data["topics"][topic_key][:max_terms]
            
            for rank, term_data in enumerate(terms, 1):
                term = term_data["term"]
                prob = term_data["probability"]
                # Escape special LaTeX characters
                term = term.replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")
                if rank == 1:
                    lines.append(f"{label} & {rank} & \\texttt{{{term}}} ({prob:.4f}) \\\\")
                else:
                    lines.append(f" & {rank} & \\texttt{{{term}}} ({prob:.4f}) \\\\")
            
            if topic_id < num_topics - 1:
                lines.append("\\midrule")
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    return "\n".join(lines)


def generate_category_topics_table(results_file: Path) -> str:
    """
    Generate LaTeX table for top topics per category.
    
    Args:
        results_file: Path to topic_modeling_results.json
        
    Returns:
        LaTeX table code
    """
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Top 5 topics per category}",
        "\\begin{tabular}{lll}",
        "\\toprule",
        "Category & Rank & Topic (Probability) \\\\",
        "\\midrule"
    ]
    
    for category, topics in data["top_topics_per_category"].items():
        for rank, topic_data in enumerate(topics[:5], 1):
            topic_id = topic_data["topic_id"]
            prob = topic_data["probability"]
            lines.append(f"{category.capitalize()} & {rank} & Topic {topic_id} ({prob:.4f}) \\\\")
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    return "\n".join(lines)


def generate_experiment_table(comparison_file: Path) -> str:
    """
    Generate LaTeX table for experiment comparisons.
    
    Args:
        comparison_file: Path to comparison_tables.json
        
    Returns:
        LaTeX table code
    """
    with open(comparison_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Comparison of preprocessing configurations}",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Configuration & Vocab Size & Coherence \\\\",
        "\\midrule"
    ]
    
    for result in data["topic_modeling"]:
        config = result["configuration"]
        vocab_size = result["vocab_size"]
        coherence = result["coherence"]
        # Escape underscores
        config = config.replace("_", "\\_")
        lines.append(f"{config} & {vocab_size} & {coherence:.4f} \\\\")
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    return "\n".join(lines)


def generate_dataset_stats_table(metadata_file: Path) -> str:
    """
    Generate LaTeX table for dataset statistics.
    
    Args:
        metadata_file: Path to dataset_metadata.json
        
    Returns:
        LaTeX table code
    """
    with open(metadata_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Calculate average tokens per document (approximate)
    # We'll need to compute this from actual documents
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Dataset statistics by category}",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Category & Documents & Total Words \\\\",
        "\\midrule"
    ]
    
    for category, count in data["categories"].items():
        # Get word count from category statistics if available
        # For now, use placeholder - will be filled from actual analysis
        lines.append(f"{category.capitalize()} & {count} & [To be computed] \\\\")
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    return "\n".join(lines)


def compute_avg_tokens_per_doc(data_dir: Path) -> Dict[str, float]:
    """
    Compute average tokens per document for each category.
    
    Args:
        data_dir: Directory containing categorized documents
        
    Returns:
        Dict mapping category to average tokens
    """
    import sys
    from pathlib import Path
    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from bow_processor import load_documents_from_category
    from normalize_text import tokenize
    
    data_dir = Path(data_dir)
    metadata_file = data_dir / "dataset_metadata.json"
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    avg_tokens = {}
    
    for category in metadata["categories"].keys():
        category_dir = data_dir / category
        if category_dir.exists():
            docs, _ = load_documents_from_category(category_dir)
            total_tokens = 0
            for doc in docs:
                tokens = tokenize(doc)
                total_tokens += len(tokens)
            
            avg = total_tokens / len(docs) if docs else 0
            avg_tokens[category] = avg
    
    return avg_tokens


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables from analysis results"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory containing results"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/loghub",
        help="Data directory for computing statistics"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="output/tables.tex",
        help="Output file for all tables"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    data_dir = Path(args.data_dir)
    
    # Generate all tables
    tables = []
    
    # Dataset statistics
    print("Generating dataset statistics table...")
    metadata_file = data_dir / "dataset_metadata.json"
    if metadata_file.exists():
        avg_tokens = compute_avg_tokens_per_doc(data_dir)
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        dataset_table = [
            "\\begin{table}[H]",
            "\\centering",
            "\\caption{Dataset statistics by category}",
            "\\begin{tabular}{lccc}",
            "\\toprule",
            "Category & Documents & Avg Tokens/Doc & Total Words \\\\",
            "\\midrule"
        ]
        
        for category, count in metadata["categories"].items():
            avg = avg_tokens.get(category, 0)
            # Estimate total words (approximate)
            total_words = avg * count
            dataset_table.append(f"{category.capitalize()} & {count} & {avg:.1f} & {total_words:.0f} \\\\")
        
        dataset_table.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ])
        tables.append(("dataset_stats", "\n".join(dataset_table)))
    
    # Naive Bayes table
    print("Generating Naive Bayes table...")
    nb_file = output_dir / "naive_bayes" / "naive_bayes_results.json"
    if nb_file.exists():
        tables.append(("naive_bayes", generate_naive_bayes_table(nb_file)))
    
    # Topic modeling tables
    print("Generating topic modeling tables...")
    tm_file = output_dir / "topic_modeling" / "topic_modeling_results.json"
    if tm_file.exists():
        tables.append(("topics", generate_topic_table(tm_file)))
        tables.append(("category_topics", generate_category_topics_table(tm_file)))
    
    # Experiment table
    print("Generating experiment comparison table...")
    exp_file = output_dir / "experiments" / "comparison_tables.json"
    if exp_file.exists():
        tables.append(("experiments", generate_experiment_table(exp_file)))
    
    # Write all tables to output file
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("% Generated LaTeX tables from analysis results\n")
        f.write("% Use these in the appropriate report sections\n\n")
        
        for name, table in tables:
            f.write(f"% ===== {name.upper()} TABLE =====\n")
            f.write(table)
            f.write("\n\n")
    
    print(f"\nGenerated {len(tables)} tables")
    print(f"Tables saved to {output_file}")
    print("\nTable names:")
    for name, _ in tables:
        print(f"  - {name}")
