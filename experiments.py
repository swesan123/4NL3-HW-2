"""
Experimentation Module

Tests different combinations of text normalization and bag-of-words
representations to compare their effects on analysis results.

Author: Swesan Pathmanathan
Course: 4NL3
Date: January 2026
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import itertools

from bow_processor import BOWProcessor, load_documents_from_category
from naive_bayes_analysis import NaiveBayesAnalyzer, load_documents_by_category
from topic_modeling import TopicModeler


class ExperimentRunner:
    """
    Runs experiments with different preprocessing and representation combinations.
    """
    
    def __init__(self, data_dir: Path, output_dir: Path):
        """
        Initialize experiment runner.
        
        Args:
            data_dir: Directory containing categorized documents
            output_dir: Directory to save experiment results
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load documents once
        self.documents_by_category = load_documents_by_category(self.data_dir)
        
        # Experiment configurations
        self.normalization_configs = [
            {"lowercase": False, "stopwords": False, "stem": False, "lemmatize": False},
            {"lowercase": True, "stopwords": False, "stem": False, "lemmatize": False},
            {"lowercase": True, "stopwords": True, "stem": False, "lemmatize": False},
            {"lowercase": True, "stopwords": True, "stem": True, "lemmatize": False},
            {"lowercase": True, "stopwords": True, "stem": False, "lemmatize": True},
        ]
        
        self.bow_representations = ["count", "binary", "tfidf"]
    
    def run_naive_bayes_experiment(
        self,
        config: Dict,
        representation: str
    ) -> Dict:
        """
        Run Naive Bayes analysis with given configuration.
        
        Args:
            config: Normalization configuration
            representation: BOW representation type
            
        Returns:
            Results dictionary
        """
        # Create processor
        processor = BOWProcessor(
            lowercase=config["lowercase"],
            remove_stopwords_flag=config["stopwords"],
            stem=config["stem"],
            lemmatize=config["lemmatize"],
            representation=representation
        )
        
        # Build vocabulary
        all_docs = []
        for docs in self.documents_by_category.values():
            all_docs.extend(docs)
        processor.build_vocabulary(all_docs)
        
        # Run Naive Bayes
        analyzer = NaiveBayesAnalyzer()
        top_words = analyzer.analyze(self.documents_by_category, processor)
        
        # Extract results
        results = {
            "config": config,
            "representation": representation,
            "vocab_size": len(processor.vocabulary),
            "top_words_per_category": {
                cat: [
                    {"word": word, "llr": float(llr)}
                    for word, llr in words
                ]
                for cat, words in top_words.items()
            }
        }
        
        return results
    
    def run_topic_modeling_experiment(
        self,
        config: Dict,
        num_topics: int = 10
    ) -> Dict:
        """
        Run topic modeling with given configuration.
        
        Args:
            config: Normalization configuration
            num_topics: Number of topics
            
        Returns:
            Results dictionary
        """
        # Flatten documents
        all_documents = []
        document_categories = []
        
        for category, docs in self.documents_by_category.items():
            all_documents.extend(docs)
            document_categories.extend([category] * len(docs))
        
        # Create processor
        processor = BOWProcessor(
            lowercase=config["lowercase"],
            remove_stopwords_flag=config["stopwords"],
            stem=config["stem"],
            lemmatize=config["lemmatize"],
            representation="count"
        )
        
        # Create modeler
        modeler = TopicModeler(num_topics=num_topics)
        
        # Prepare corpus and train
        dictionary, corpus = modeler.prepare_corpus(all_documents, processor)
        modeler.train_lda(dictionary, corpus)
        
        # Compute distributions
        modeler.compute_document_topic_distributions(corpus)
        modeler.compute_category_topic_distributions(document_categories)
        
        # Get results
        top_terms = modeler.get_top_terms_per_topic(top_n=25)
        top_topics = modeler.get_top_topics_per_category(top_k=5)
        
        # Compute coherence
        tokenized_docs = [processor.preprocess_text(doc) for doc in all_documents]
        coherence = modeler.compute_coherence(tokenized_docs, dictionary, modeler.lda_model)
        
        results = {
            "config": config,
            "num_topics": num_topics,
            "coherence": float(coherence),
            "vocab_size": len(dictionary),
            "top_terms_per_topic": {
                topic_id: [
                    {"term": term, "probability": float(prob)}
                    for term, prob in terms
                ]
                for topic_id, terms in top_terms.items()
            },
            "top_topics_per_category": {
                cat: [
                    {"topic_id": topic_id, "probability": float(prob)}
                    for topic_id, prob in topics
                ]
                for cat, topics in top_topics.items()
            }
        }
        
        return results
    
    def run_all_experiments(self):
        """
        Run all experiment combinations.
        """
        print("Running experiments...")
        print(f"Normalization configs: {len(self.normalization_configs)}")
        print(f"BOW representations: {len(self.bow_representations)}")
        
        nb_results = []
        tm_results = []
        
        # Naive Bayes experiments
        print("\n=== Naive Bayes Experiments ===")
        for i, config in enumerate(self.normalization_configs, 1):
            for representation in self.bow_representations:
                config_name = self._config_to_name(config, representation)
                print(f"\n[{i}/{len(self.normalization_configs)}] {config_name}")
                
                try:
                    result = self.run_naive_bayes_experiment(config, representation)
                    nb_results.append(result)
                    print(f"  Vocab size: {result['vocab_size']}")
                except Exception as e:
                    print(f"  Error: {e}")
        
        # Topic modeling experiments
        print("\n=== Topic Modeling Experiments ===")
        for i, config in enumerate(self.normalization_configs, 1):
            config_name = self._config_to_name(config, "count")
            print(f"\n[{i}/{len(self.normalization_configs)}] {config_name}")
            
            try:
                result = self.run_topic_modeling_experiment(config, num_topics=10)
                tm_results.append(result)
                print(f"  Coherence: {result['coherence']:.4f}")
                print(f"  Vocab size: {result['vocab_size']}")
            except Exception as e:
                print(f"  Error: {e}")
        
        # Save results
        self._save_results(nb_results, tm_results)
        
        # Generate comparison tables
        self._generate_comparison_tables(nb_results, tm_results)
    
    def _config_to_name(self, config: Dict, representation: str) -> str:
        """Convert config to readable name."""
        parts = []
        if config["lowercase"]:
            parts.append("lowercase")
        if config["stopwords"]:
            parts.append("stopwords")
        if config["stem"]:
            parts.append("stem")
        if config["lemmatize"]:
            parts.append("lemmatize")
        if not parts:
            parts.append("raw")
        parts.append(representation)
        return "_".join(parts)
    
    def _save_results(self, nb_results: List[Dict], tm_results: List[Dict]):
        """Save experiment results."""
        results = {
            "naive_bayes": nb_results,
            "topic_modeling": tm_results
        }
        
        results_file = self.output_dir / "experiment_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nSaved experiment results to {results_file}")
    
    def _generate_comparison_tables(self, nb_results: List[Dict], tm_results: List[Dict]):
        """Generate comparison tables for report."""
        # Naive Bayes comparison
        nb_comparison = []
        for result in nb_results:
            config_name = self._config_to_name(result["config"], result["representation"])
            nb_comparison.append({
                "configuration": config_name,
                "vocab_size": result["vocab_size"],
                "top_words_sample": {
                    cat: [word for word, _ in words[:5]]
                    for cat, words in result["top_words_per_category"].items()
                }
            })
        
        # Topic modeling comparison
        tm_comparison = []
        for result in tm_results:
            config_name = self._config_to_name(result["config"], "count")
            tm_comparison.append({
                "configuration": config_name,
                "coherence": result["coherence"],
                "vocab_size": result["vocab_size"]
            })
        
        comparison = {
            "naive_bayes": nb_comparison,
            "topic_modeling": tm_comparison
        }
        
        comparison_file = self.output_dir / "comparison_tables.json"
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        
        print(f"Saved comparison tables to {comparison_file}")


if __name__ == "__main__":
    """
    Main execution for experiments.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run experiments with different preprocessing configurations"
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
        default="output/experiments",
        help="Output directory for experiment results"
    )
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(Path(args.data_dir), Path(args.output_dir))
    runner.run_all_experiments()
