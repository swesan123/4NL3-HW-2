"""
Text Normalization and Token Frequency Analysis Tool

This module provides functionality for normalizing natural language text and
analyzing token frequency distributions. It supports various normalization
techniques including lowercasing, stemming, lemmatization, stopword removal,
and custom preprocessing.

Author: Swesan Pathmanathan
Course: 4NL3
Date: January 16, 2026

AI Tool Usage:
Github Copilot Auto was used for coding syntax and commenting of the code. 
Using ~4.32 g CO₂ per query x ~25 queries ≈ 108 g CO₂

External Code Attribution:
- NLTK library: Used for stemming, lemmatization, and stopword lists
- Matplotlib: Used for visualization
- Standard Python libraries: re, argparse, os, string, collections
"""

import nltk
import re
import argparse
import os
import matplotlib.pyplot as plt
import string

from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
from typing import List, Tuple



def read_file(file_path: str) -> str:
    """
    Read and return the contents of a text file.
    
    This function reads a text file using UTF-8 encoding and handles any
    encoding errors by replacing invalid characters.
    
    Args:
        file_path: The path to the input text file
        
    Returns:
        The complete text content of the file as a string
        
    Raises:
        FileNotFoundError: If the specified file does not exist
        IOError: If there are issues reading the file
    """
    with open(f"{file_path}", "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    return text

def write_file(filename: str, header: str, lines: List[str]) -> None:
    """
    Write text content to a file in the output directory.
    
    Creates the output directory if it does not exist and writes the provided
    header and lines to the specified file.
    
    Args:
        filename: Name of the output file (will be created in output/ directory)
        header: Header text to write at the beginning of the file
        lines: List of text lines to write to the file
        
    Returns:
        None
        
    Raises:
        IOError: If there are issues writing to the file
    """
    os.makedirs("output", exist_ok=True)
    with open(f"output/{filename}", "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for line in lines:
            f.write(line + "\n")

def write_token_counts(counts: List[Tuple[str, int]], total: int, suffix: str) -> None:
    """
    Write token frequency counts to an output file.
    
    Creates a tab-separated file containing token frequencies in descending order,
    with a header line showing the total token count.
    
    Args:
        counts: List of (token, frequency) tuples sorted by frequency (descending)
        total: Total number of tokens in the processed text
        suffix: Filename suffix indicating normalization steps applied
        
    Returns:
        None
        
    Side Effects:
        Creates a file named tokens_{suffix}.log in the output directory
        Prints confirmation message to console
    """
    lines = [f"{token}\t{freq}" for token, freq in counts]

    write_file(
        filename=f"tokens_{suffix}.log",
        header=f"The total # of tokens: {total}",
        lines=lines
    )

    print(f"Saved token counts to output/tokens_{suffix}.log")
def analyze_tokens(counts: List[Tuple[str, int]], suffix: str) -> None:
    """
    Generate and save a Zipf's law visualization of token frequencies.
    
    Creates a bar plot with log-log scales showing the relationship between 
    token rank and frequency, which typically follows Zipf's law in natural language.
    
    Args:
        counts: List of (token, frequency) tuples sorted by frequency (descending)
        suffix: Filename suffix indicating normalization steps applied
        
    Returns:
        None
        
    Side Effects:
        Creates a PNG file named zipf_{suffix}.png in the output directory
        Prints confirmation message to console
        
    Note:
        The plot uses logarithmic scales on both axes to visualize Zipf's law,
        which states that frequency is inversely proportional to rank.
    """
    freqs = [freq for _, freq in counts]
    ranks = range(1, len(freqs) + 1)

    out_path = f"output/zipf_{suffix}.png"

    plt.figure(figsize=(10, 6))
    plt.bar(ranks, freqs, width=1.0, edgecolor='none', alpha=0.8)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Rank", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title(f"Token Frequency Distribution (Zipf's Law) - {suffix}", fontsize=14)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved plot to {out_path}")


def tokenize(text: str) -> List[str]:
    """
    Tokenize text by splitting on whitespace.
    
    Implements manual tokenization using regular expressions to split text
    on one or more whitespace characters. Empty tokens are filtered out.
    
    Args:
        text: Input text string to tokenize
        
    Returns:
        List of tokens (strings) extracted from the input text
        
    Note:
        This is a custom tokenization implementation that does not use
        NLTK tokenizers, as required by the assignment specifications.
        Tokens are split only on whitespace; punctuation remains attached.
    """
    tokens: List[str] = re.split(r"\s+", text)
    tokens = [t for t in tokens if t]
    return tokens

def count_tokens(tokens: List[str]) -> List[Tuple[str, int]]:
    """
    Count the frequency of each unique token.
    
    Args:
        tokens: List of tokens to count
        
    Returns:
        List of (token, count) tuples sorted by count in descending order
        
    Note:
        Uses Python's Counter class for efficient counting and sorting.
    """
    counts = Counter(tokens)
    return counts.most_common()


def apply_lowercase(tokens: List[str]) -> List[str]:
    """
    Convert all tokens to lowercase.
    
    Applies lowercase normalization to all tokens and logs the transformations
    that resulted in changes.
    
    Args:
        tokens: List of tokens to normalize
        
    Returns:
        List of lowercase tokens
        
    Side Effects:
        Creates lowercase.log file documenting all transformations
        
    Note:
        This normalization helps reduce vocabulary size by treating words
        like "The" and "the" as the same token.
    """
    before = tokens
    after = [t.lower() for t in tokens]

    changed = [
        f"{b} -> {a}"
        for b, a in zip(before, after)
        if b != a
    ]

    write_file(
        "lowercase.log",
        "Tokens changed by lowercase normalization:",
        changed
    )

    return after

def remove_digits(tokens: List[str]) -> List[str]:
    """
    Remove tokens that consist entirely of numeric digits.
    
    This custom preprocessing step (-myopt) filters out numeric tokens,
    reducing vocabulary sparsity from metadata, timestamps, and formatting
    artifacts. This improves focus on lexical patterns in natural language.
    
    Args:
        tokens: List of tokens to filter
        
    Returns:
        List of tokens with digit-only tokens removed
        
    Side Effects:
        Creates digits_removed.log file documenting removed tokens
        
    Note:
        This is the custom -myopt preprocessing option. Tokens containing
        both letters and digits (e.g., "Unix2") are preserved.
    """
    before = tokens
    after = [
        t for t in tokens
        if not all(c in string.digits for c in t)
    ]

    removed = sorted(set(before) - set(after))

    write_file(
        "digits_removed.log",
        "Digit-only tokens removed:",
        removed
    )

    return after

def remove_stopwords(tokens: List[str]) -> List[str]:
    """
    Remove common English stopwords from the token list.
    
    Filters out high-frequency function words (e.g., "the", "a", "is")
    that typically carry less semantic meaning, using NLTK's English
    stopword list.
    
    Args:
        tokens: List of tokens to filter
        
    Returns:
        List of tokens with stopwords removed
        
    Side Effects:
        Creates stopwords_removed.log file documenting removed stopwords
        
    Note:
        Uses NLTK's English stopword corpus. Stopword removal is typically
        applied after lowercasing for optimal matching.
    """
    stop_words = set(stopwords.words("english"))
    before = tokens
    after = [t for t in tokens if t not in stop_words]

    removed = sorted(set(before) - set(after))

    write_file(
        "stopwords_removed.log",
        "Stopwords removed:",
        removed
    )

    return after

def apply_stemming(tokens: List[str]) -> List[str]:
    """
    Apply Porter stemming algorithm to reduce tokens to their root form.
    
    Uses the Porter stemming algorithm to reduce inflected or derived words
    to their word stem, base, or root form (e.g., "running" -> "run").
    
    Args:
        tokens: List of tokens to stem
        
    Returns:
        List of stemmed tokens
        
    Side Effects:
        Creates stem.log file documenting all transformations
        
    Note:
        Stemming is a heuristic process that may produce non-words.
        Cannot be used simultaneously with lemmatization (-lemmatize flag).
    """
    stemmer = PorterStemmer()
    before = tokens
    after = [stemmer.stem(t) for t in tokens]

    changes = sorted(set(
        f"{b} -> {a}"
        for b, a in zip(before, after)
        if b != a
    ))

    write_file(
        "stem.log",
        "Tokens changed by stemming:",
        changes
    )

    return after

def apply_lemmatization(tokens: List[str]) -> List[str]:
    """
    Apply lemmatization to reduce tokens to their dictionary form.
    
    Uses WordNet lemmatizer to convert words to their base or dictionary
    form (lemma) using vocabulary and morphological analysis.
    
    Args:
        tokens: List of tokens to lemmatize
        
    Returns:
        List of lemmatized tokens
        
    Side Effects:
        Creates lemmatize.log file documenting all transformations
        
    Note:
        Lemmatization produces valid dictionary words unlike stemming.
        Cannot be used simultaneously with stemming (-stem flag).
        Default POS tag is noun; may not reduce all verb forms optimally.
    """
    lemmatizer = WordNetLemmatizer()
    before = tokens
    after = [lemmatizer.lemmatize(t) for t in tokens]

    changes = sorted(set(
        f"{b} -> {a}"
        for b, a in zip(before, after)
        if b != a
    ))

    write_file(
        "lemmatize.log",
        "Tokens changed by lemmatization:",
        changes
    )

    return after



def parse_args() -> argparse.Namespace:
    """
    Parse and validate command-line arguments.
    
    Configures the argument parser with all required normalization options
    and validates that mutually exclusive options are not used together.
    
    Returns:
        Parsed command-line arguments as an argparse.Namespace object
        
    Raises:
        SystemExit: If arguments are invalid or -stem and -lemmatize are both specified
        
    Command-line Usage:
        python normalize_text.py <input_file> [options]
        
    Options:
        -analyze: Generate Zipf's law visualization
        -lowercase: Convert all tokens to lowercase
        -stem: Apply Porter stemming
        -lemmatize: Apply WordNet lemmatization (mutually exclusive with -stem)
        -stopwords: Remove English stopwords
        -myopt: Remove digit-only tokens
    """
    parser = argparse.ArgumentParser(
        description="Normalize text and count token frequencies from a plain text file."
    )
    
    # Positional argument
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input text file (plain text, UTF-8 encoded)"
    )
    

    # Boolean flags
    parser.add_argument(
        "-analyze",
        action="store_true",
        help="Generate Zipf (log-log) frequency plot and save to output directory"
    )

    parser.add_argument(
        "-lowercase",
        action="store_true",
        help="Convert all tokens to lowercase before further processing"
    )

    parser.add_argument(
        "-stem",
        action="store_true",
        help="Apply stemming to tokens using a stemmer (e.g., PorterStemmer)"
    )

    parser.add_argument(
        "-lemmatize",
        action="store_true",
        help="Apply lemmatization to tokens using a lemmatizer (e.g., WordNetLemmatizer)"
    )

    parser.add_argument(
        "-stopwords",
        action="store_true",
        help="Remove common English stopwords from the token list"
    )

    parser.add_argument(
        "-myopt",
        action="store_true",
        help="Remove digit-only tokens"
    )

    args = parser.parse_args()

    if args.stem and args.lemmatize:
        parser.error("Please choose only one of -stem or -lemmatize.")

    return args

def main() -> None:
    """
    Main execution function for text normalization and analysis.
    
    Orchestrates the complete text processing pipeline:
    1. Parse command-line arguments
    2. Read input file
    3. Tokenize text
    4. Apply requested normalization steps in order
    5. Count token frequencies
    6. Write results to output files
    7. Generate visualization if requested
    
    Returns:
        None
        
    Side Effects:
        - Reads input file specified in command-line arguments
        - Creates output directory and writes multiple log files
        - Prints progress messages to console
        - May generate visualization PNG file
        
    Note:
        Normalization steps are applied in a specific order:
        lowercase -> myopt -> stopwords -> stem/lemmatize
    """
    args = parse_args()
    path = args.input_file

    text = read_file(path)
    filename = os.path.splitext(os.path.basename(path))[0]
    suffix = [filename]

    tokens = tokenize(text)

    if args.lowercase:
        suffix.append("lowercase")
        tokens = apply_lowercase(tokens)

    if args.myopt:
        suffix.append("myopt")
        tokens = remove_digits(tokens)

    if args.stopwords:
        suffix.append("stopwords")
        tokens = remove_stopwords(tokens)

    if args.stem:
        suffix.append("stem")
        tokens = apply_stemming(tokens)
    elif args.lemmatize:
        suffix.append("lemmatize")
        tokens = apply_lemmatization(tokens)

    counts = count_tokens(tokens)
    total = len(tokens)

    print(f"The total # of tokens: {total}")

    suffix = "_".join(suffix) if suffix else "raw"
    write_token_counts(counts, total, suffix)

    if args.analyze:
        analyze_tokens(counts, suffix)



if __name__ == "__main__":

    # Doesnt get installed with package
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")

    try:
        WordNetLemmatizer().lemmatize("test")
    except LookupError:
        nltk.download("wordnet")

    main()