"""
Main Corpus Analysis Pipeline

Orchestrates the complete corpus analysis pipeline:
1. Load dataset
2. Preprocess documents
3. Convert to BoW
4. Run Naive Bayes analysis
5. Run topic modeling
6. Run experiments
7. Generate output files/tables

Author: Swesan Pathmanathan
Course: 4NL3
Date: January 2026
"""

import argparse
from pathlib import Path
from typing import Optional

from data_collection import DatasetCollector
from naive_bayes_analysis import load_documents_by_category
from bow_processor import BOWProcessor, load_documents_from_category
from naive_bayes_analysis import NaiveBayesAnalyzer
from topic_modeling import TopicModeler
from experiments import ExperimentRunner


def run_full_pipeline(
    data_dir: Path,
    output_dir: Path,
    lowercase: bool = True,
    stopwords: bool = True,
    stem: bool = False,
    lemmatize: bool = False,
    num_topics: int = 10,
    run_experiments: bool = False
):
    """
    Run complete corpus analysis pipeline.
    
    Args:
        data_dir: Directory containing categorized documents
        output_dir: Output directory for results
        lowercase: Whether to lowercase tokens
        stopwords: Whether to remove stopwords
        stem: Whether to apply stemming
        lemmatize: Whether to apply lemmatization
        num_topics: Number of topics for LDA
        run_experiments: Whether to run experiments
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Corpus Analysis Pipeline")
    print("=" * 60)
    
    # Step 1: Load dataset
    print("\n[Step 1] Loading dataset...")
    documents_by_category = load_documents_by_category(data_dir)
    
    print(f"Categories: {list(documents_by_category.keys())}")
    for cat, docs in documents_by_category.items():
        print(f"  {cat}: {len(docs)} documents")
    
    # Step 2: Preprocess and convert to BoW
    print("\n[Step 2] Converting to Bag-of-Words...")
    all_documents = []
    document_ids = []
    document_categories = []
    
    for category, docs in documents_by_category.items():
        category_dir = data_dir / category
        if category_dir.exists():
            docs_list, doc_ids = load_documents_from_category(category_dir)
            all_documents.extend(docs_list)
            document_ids.extend(doc_ids)
            document_categories.extend([category] * len(docs_list))
    
    processor = BOWProcessor(
        lowercase=lowercase,
        remove_stopwords_flag=stopwords,
        stem=stem,
        lemmatize=lemmatize,
        representation="count"
    )
    
    print("Building vocabulary...")
    processor.build_vocabulary(all_documents)
    print(f"Vocabulary size: {len(processor.vocabulary)}")
    
    print("Creating document-term matrix...")
    matrix = processor.process_documents(all_documents, document_ids)
    print(f"Matrix shape: {matrix.shape}")
    print(f"Matrix sparsity: {(1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1])) * 100:.2f}%")
    
    # Save BoW
    bow_output = output_dir / "bow"
    processor.save(bow_output)
    
    # Step 3: Naive Bayes analysis
    print("\n[Step 3] Running Naive Bayes analysis...")
    analyzer = NaiveBayesAnalyzer()
    top_words = analyzer.analyze(documents_by_category, processor)
    
    print("\nTop 10 words per category (by LLR):")
    for category, words in top_words.items():
        print(f"\n{category}:")
        for i, (word, llr) in enumerate(words[:10], 1):
            print(f"  {i}. {word}: {llr:.4f}")
    
    # Save Naive Bayes results
    nb_output = output_dir / "naive_bayes"
    analyzer.save_results(nb_output)
    
    # Step 4: Topic modeling
    print("\n[Step 4] Running topic modeling...")
    modeler = TopicModeler(num_topics=num_topics)
    
    print("Preparing corpus...")
    dictionary, corpus = modeler.prepare_corpus(all_documents, processor)
    print(f"Corpus vocabulary size: {len(dictionary)}")
    
    print(f"Training LDA model with {num_topics} topics...")
    modeler.train_lda(dictionary, corpus)
    
    print("Computing topic distributions...")
    modeler.compute_document_topic_distributions(corpus)
    modeler.compute_category_topic_distributions(document_categories)
    
    # Get top terms per topic
    top_terms = modeler.get_top_terms_per_topic(top_n=25)
    print("\nTop 25 terms per topic (showing first 10):")
    for topic_id, terms in top_terms.items():
        print(f"\nTopic {topic_id}:")
        for i, (term, prob) in enumerate(terms[:10], 1):
            print(f"  {i}. {term} ({prob:.4f})")
    
    # Get top topics per category
    top_topics = modeler.get_top_topics_per_category(top_k=5)
    print("\nTop 5 topics per category:")
    for category, topics in top_topics.items():
        print(f"\n{category}:")
        for topic_id, prob in topics:
            print(f"  Topic {topic_id}: {prob:.4f}")
    
    # Generate visualization
    vis_path = output_dir / "topic_modeling" / "lda_visualization.html"
    print(f"\nGenerating visualization...")
    modeler.generate_visualization(vis_path)
    
    # Save topic modeling results
    tm_output = output_dir / "topic_modeling"
    modeler.save_results(tm_output)
    
    # Step 5: Experiments (optional)
    if run_experiments:
        print("\n[Step 5] Running experiments...")
        runner = ExperimentRunner(data_dir, output_dir / "experiments")
        runner.run_all_experiments()
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run complete corpus analysis pipeline"
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
        default="output",
        help="Output directory for results"
    )
    parser.add_argument(
        "--lowercase",
        action="store_true",
        default=True,
        help="Convert to lowercase"
    )
    parser.add_argument(
        "--no-lowercase",
        dest="lowercase",
        action="store_false",
        help="Do not convert to lowercase"
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
    parser.add_argument(
        "--num-topics",
        type=int,
        default=10,
        help="Number of topics for LDA (default: 10)"
    )
    parser.add_argument(
        "--experiments",
        action="store_true",
        help="Run experiments with different configurations"
    )
    
    args = parser.parse_args()
    
    run_full_pipeline(
        Path(args.data_dir),
        Path(args.output_dir),
        lowercase=args.lowercase,
        stopwords=args.stopwords,
        stem=args.stem,
        lemmatize=args.lemmatize,
        num_topics=args.num_topics,
        run_experiments=args.experiments
    )
