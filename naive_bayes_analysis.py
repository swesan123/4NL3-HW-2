"""
Naive Bayes Analysis Module

Computes Naive Bayes probabilities for terms in each category and calculates
log-likelihood ratios (LLR) to identify words most associated with each category.

Implements equations from HW-2 assignment:
- P(w|c) with add-one smoothing
- P(w|C_o) for other categories (equation 2)
- LLR(w,c) = log(P(w|c)) - log(P(w|C_o))

Author: Swesan Pathmanathan
Course: 4NL3
Date: January 2026
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from scipy.sparse import csr_matrix

from bow_processor import BOWProcessor, load_documents_from_category


class NaiveBayesAnalyzer:
    """
    Analyzes documents using Naive Bayes probability model.
    """
    
    def __init__(self, smoothing: float = 1.0):
        """
        Initialize Naive Bayes analyzer.
        
        Args:
            smoothing: Add-k smoothing parameter (default 1.0 for add-one)
        """
        self.smoothing = smoothing
        
        # Word counts per category: {category: {word: count}}
        self.word_counts: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        # Total word counts per category
        self.total_words_per_category: Dict[str, float] = defaultdict(float)
        
        # Vocabulary (all words across all categories)
        self.vocabulary: set = set()
        
        # Categories
        self.categories: List[str] = []
        
        # Computed probabilities
        self.p_w_given_c: Dict[str, Dict[str, float]] = {}
        self.p_w_given_c_other: Dict[str, Dict[str, float]] = {}
        self.llr_scores: Dict[str, Dict[str, float]] = {}
    
    def compute_word_counts(
        self,
        documents_by_category: Dict[str, List[str]],
        processor: BOWProcessor
    ):
        """
        Compute word counts for each category.
        
        Args:
            documents_by_category: Dict mapping category names to document lists
            processor: BOWProcessor instance with vocabulary built
        """
        self.categories = list(documents_by_category.keys())
        self.vocabulary = set(processor.vocabulary.keys())
        
        for category, documents in documents_by_category.items():
            category_word_counts = defaultdict(float)
            total_count = 0.0
            
            for doc in documents:
                tokens = processor.preprocess_text(doc)
                for token in tokens:
                    if token in processor.vocabulary:
                        category_word_counts[token] += 1.0
                        total_count += 1.0
            
            self.word_counts[category] = category_word_counts
            self.total_words_per_category[category] = total_count
    
    def compute_p_w_given_c(self) -> Dict[str, Dict[str, float]]:
        """
        Compute P(w|c) for each word and category with add-one smoothing.
        
        Returns:
            Dict mapping category to dict mapping word to probability
        """
        self.p_w_given_c = {}
        
        # Total vocabulary size
        vocab_size = len(self.vocabulary)
        
        for category in self.categories:
            word_probs = {}
            total_words = self.total_words_per_category[category]
            
            # Denominator: total_words + smoothing * vocab_size
            denominator = total_words + self.smoothing * vocab_size
            
            for word in self.vocabulary:
                # Count of word in category (with smoothing)
                count_w_c = self.word_counts[category].get(word, 0.0) + self.smoothing
                
                # P(w|c) = (count(w,c) + smoothing) / (total_words_in_c + smoothing * |V|)
                prob = count_w_c / denominator
                word_probs[word] = prob
            
            self.p_w_given_c[category] = word_probs
        
        return self.p_w_given_c
    
    def compute_p_w_given_c_other(self) -> Dict[str, Dict[str, float]]:
        """
        Compute P(w|C_o) for other categories (equation 2 from PDF).
        
        P(w|C_o) = sum(count(w, c_o)) / sum(sum(count(w', c_o)))
        where C_o is the set of categories other than c.
        
        Returns:
            Dict mapping category to dict mapping word to probability
        """
        self.p_w_given_c_other = {}
        
        for category in self.categories:
            # Other categories
            other_categories = [c for c in self.categories if c != category]
            
            # Sum of counts for each word across other categories
            word_counts_other = defaultdict(float)
            total_words_other = 0.0
            
            for other_cat in other_categories:
                for word, count in self.word_counts[other_cat].items():
                    word_counts_other[word] += count
                total_words_other += self.total_words_per_category[other_cat]
            
            # Compute probabilities with smoothing
            vocab_size = len(self.vocabulary)
            denominator = total_words_other + self.smoothing * vocab_size
            
            word_probs = {}
            for word in self.vocabulary:
                count_w_other = word_counts_other.get(word, 0.0) + self.smoothing
                prob = count_w_other / denominator
                word_probs[word] = prob
            
            self.p_w_given_c_other[category] = word_probs
        
        return self.p_w_given_c_other
    
    def compute_llr_scores(self) -> Dict[str, Dict[str, float]]:
        """
        Compute log-likelihood ratios: LLR(w,c) = log(P(w|c)) - log(P(w|C_o))
        
        Returns:
            Dict mapping category to dict mapping word to LLR score
        """
        if not self.p_w_given_c:
            self.compute_p_w_given_c()
        
        if not self.p_w_given_c_other:
            self.compute_p_w_given_c_other()
        
        self.llr_scores = {}
        
        for category in self.categories:
            llr_scores = {}
            
            for word in self.vocabulary:
                p_w_c = self.p_w_given_c[category][word]
                p_w_c_other = self.p_w_given_c_other[category][word]
                
                # LLR = log(P(w|c)) - log(P(w|C_o))
                # Use natural log
                llr = np.log(p_w_c) - np.log(p_w_c_other)
                llr_scores[word] = llr
            
            self.llr_scores[category] = llr_scores
        
        return self.llr_scores
    
    def get_top_words_per_category(
        self,
        top_k: int = 10
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get top K words per category by LLR score.
        
        Args:
            top_k: Number of top words to return
            
        Returns:
            Dict mapping category to list of (word, LLR) tuples
        """
        if not self.llr_scores:
            self.compute_llr_scores()
        
        top_words = {}
        
        for category in self.categories:
            # Sort words by LLR score (descending)
            sorted_words = sorted(
                self.llr_scores[category].items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            top_words[category] = sorted_words[:top_k]
        
        return top_words
    
    def analyze(
        self,
        documents_by_category: Dict[str, List[str]],
        processor: BOWProcessor
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Complete analysis pipeline.
        
        Args:
            documents_by_category: Dict mapping category names to document lists
            processor: BOWProcessor instance
            
        Returns:
            Top words per category
        """
        # Compute word counts
        self.compute_word_counts(documents_by_category, processor)
        
        # Compute probabilities and LLR scores
        self.compute_llr_scores()
        
        # Get top words
        return self.get_top_words_per_category()
    
    def save_results(self, output_dir: Path):
        """
        Save analysis results to files.
        
        Args:
            output_dir: Directory to save results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save top words per category
        top_words = self.get_top_words_per_category()
        
        results = {
            "categories": self.categories,
            "vocabulary_size": len(self.vocabulary),
            "smoothing": self.smoothing,
            "top_words_per_category": {
                cat: [
                    {"word": word, "llr": float(llr)}
                    for word, llr in words
                ]
                for cat, words in top_words.items()
            },
            "category_statistics": {
                cat: {
                    "total_words": float(self.total_words_per_category[cat]),
                    "unique_words": len(self.word_counts[cat])
                }
                for cat in self.categories
            }
        }
        
        results_file = output_dir / "naive_bayes_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save detailed LLR scores
        llr_file = output_dir / "llr_scores.json"
        with open(llr_file, 'w', encoding='utf-8') as f:
            json.dump(
                {
                    cat: {word: float(score) for word, score in scores.items()}
                    for cat, scores in self.llr_scores.items()
                },
                f,
                indent=2,
                ensure_ascii=False
            )
        
        print(f"Saved Naive Bayes results to {output_dir}")


def load_documents_by_category(data_dir: Path) -> Dict[str, List[str]]:
    """
    Load documents organized by category.
    
    Args:
        data_dir: Directory containing categorized documents
        
    Returns:
        Dict mapping category names to document lists
    """
    data_dir = Path(data_dir)
    documents_by_category = {}
    
    # Load metadata
    metadata_file = data_dir / "dataset_metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Load documents for each category
    for category in metadata["categories"].keys():
        category_dir = data_dir / category
        if category_dir.exists():
            docs, _ = load_documents_from_category(category_dir)
            documents_by_category[category] = docs
    
    return documents_by_category


if __name__ == "__main__":
    """
    Main execution for Naive Bayes analysis.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Naive Bayes analysis for corpus categories"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/loghub",
        help="Directory containing categorized documents"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/naive_bayes",
        help="Output directory for results"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top words per category (default: 10)"
    )
    parser.add_argument(
        "--lowercase",
        action="store_true",
        help="Convert to lowercase"
    )
    parser.add_argument(
        "--stopwords",
        action="store_true",
        help="Remove stopwords"
    )
    parser.add_argument(
        "--stem",
        action="store_true",
        help="Apply stemming"
    )
    parser.add_argument(
        "--lemmatize",
        action="store_true",
        help="Apply lemmatization"
    )
    
    args = parser.parse_args()
    
    # Load documents
    print(f"Loading documents from {args.data_dir}...")
    documents_by_category = load_documents_by_category(Path(args.data_dir))
    
    print(f"Categories: {list(documents_by_category.keys())}")
    for cat, docs in documents_by_category.items():
        print(f"  {cat}: {len(docs)} documents")
    
    # Create BOW processor
    processor = BOWProcessor(
        lowercase=args.lowercase,
        remove_stopwords_flag=args.stopwords,
        stem=args.stem,
        lemmatize=args.lemmatize,
        representation="count"
    )
    
    # Build vocabulary from all documents
    all_documents = []
    for docs in documents_by_category.values():
        all_documents.extend(docs)
    
    print("Building vocabulary...")
    processor.build_vocabulary(all_documents)
    print(f"Vocabulary size: {len(processor.vocabulary)}")
    
    # Run Naive Bayes analysis
    print("Running Naive Bayes analysis...")
    analyzer = NaiveBayesAnalyzer()
    top_words = analyzer.analyze(documents_by_category, processor)
    
    # Print results
    print("\nTop words per category (by LLR):")
    for category, words in top_words.items():
        print(f"\n{category}:")
        for i, (word, llr) in enumerate(words, 1):
            print(f"  {i}. {word}: {llr:.4f}")
    
    # Save results
    analyzer.save_results(Path(args.output_dir))
