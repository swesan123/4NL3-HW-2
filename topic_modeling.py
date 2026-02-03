"""
Topic Modeling Module

Implements Latent Dirichlet Allocation (LDA) using gensim for topic modeling.
Extracts topics, computes topic distributions per document and category,
and generates visualizations.

Author: Swesan Pathmanathan
Course: 4NL3
Date: January 2026
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import gensim
from gensim import corpora, models
from gensim.models import CoherenceModel
from gensim.models.ldamodel import LdaModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

from bow_processor import BOWProcessor, load_documents_from_category


class TopicModeler:
    """
    Performs topic modeling using LDA.
    """
    
    def __init__(
        self,
        num_topics: int = 10,
        random_state: int = 42,
        passes: int = 10,
        alpha: str = "auto",
        eta: str = "auto"
    ):
        """
        Initialize topic modeler.
        
        Args:
            num_topics: Number of topics to extract
            random_state: Random seed for reproducibility
            passes: Number of passes through corpus during training
            alpha: Prior for document-topic distribution
            eta: Prior for topic-word distribution
        """
        self.num_topics = num_topics
        self.random_state = random_state
        self.passes = passes
        self.alpha = alpha
        self.eta = eta
        
        self.dictionary: Optional[gensim.corpora.Dictionary] = None
        self.corpus: Optional[List[List[Tuple[int, int]]]] = None
        self.lda_model: Optional[LdaModel] = None
        
        # Topic distributions
        self.document_topic_distributions: List[np.ndarray] = []
        self.category_topic_distributions: Dict[str, np.ndarray] = {}
        
        # Document IDs and category labels
        self.document_ids: List[str] = []
        self.document_categories: List[str] = []
    
    def prepare_corpus(
        self,
        documents: List[str],
        processor: BOWProcessor
    ) -> Tuple[gensim.corpora.Dictionary, List[List[Tuple[int, int]]]]:
        """
        Prepare corpus for gensim LDA.
        
        Args:
            documents: List of document texts
            processor: BOWProcessor instance
            
        Returns:
            Tuple of (dictionary, corpus)
        """
        # Convert documents to token lists
        tokenized_docs = [
            processor.preprocess_text(doc)
            for doc in documents
        ]
        
        # Create dictionary
        self.dictionary = corpora.Dictionary(tokenized_docs)
        
        # Filter extremes (optional, but recommended)
        # Remove words that appear in less than 2 documents or more than 50% of documents
        self.dictionary.filter_extremes(no_below=2, no_above=0.5)
        
        # Create corpus (bag-of-words format)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in tokenized_docs]
        
        return self.dictionary, self.corpus
    
    def train_lda(
        self,
        dictionary: gensim.corpora.Dictionary,
        corpus: List[List[Tuple[int, int]]]
    ) -> LdaModel:
        """
        Train LDA model.
        
        Args:
            dictionary: Gensim dictionary
            corpus: Gensim corpus
            
        Returns:
            Trained LDA model
        """
        self.lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=self.num_topics,
            random_state=self.random_state,
            passes=self.passes,
            alpha=self.alpha,
            eta=self.eta,
            per_word_topics=True
        )
        
        return self.lda_model
    
    def compute_coherence(
        self,
        texts: List[List[str]],
        dictionary: gensim.corpora.Dictionary,
        model: LdaModel
    ) -> float:
        """
        Compute coherence score for model.
        
        Args:
            texts: Tokenized documents
            dictionary: Gensim dictionary
            model: Trained LDA model
            
        Returns:
            Coherence score
        """
        coherence_model = CoherenceModel(
            model=model,
            texts=texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        
        return coherence_model.get_coherence()
    
    def find_optimal_topics(
        self,
        documents: List[str],
        processor: BOWProcessor,
        topic_range: Tuple[int, int] = (5, 20),
        step: int = 5
    ) -> Tuple[int, float]:
        """
        Find optimal number of topics using coherence score.
        
        Args:
            documents: List of document texts
            processor: BOWProcessor instance
            topic_range: Range of topics to test (min, max)
            step: Step size for topic range
            
        Returns:
            Tuple of (optimal_num_topics, best_coherence)
        """
        # Prepare corpus
        tokenized_docs = [processor.preprocess_text(doc) for doc in documents]
        dictionary, corpus = self.prepare_corpus(documents, processor)
        
        min_topics, max_topics = topic_range
        coherence_scores = []
        topic_numbers = list(range(min_topics, max_topics + 1, step))
        
        print(f"Testing topics: {topic_numbers}")
        
        for num_topics in topic_numbers:
            print(f"  Training model with {num_topics} topics...")
            
            model = LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=num_topics,
                random_state=self.random_state,
                passes=self.passes,
                alpha=self.alpha,
                eta=self.eta
            )
            
            coherence = self.compute_coherence(tokenized_docs, dictionary, model)
            coherence_scores.append(coherence)
            print(f"    Coherence: {coherence:.4f}")
        
        # Find best
        best_idx = np.argmax(coherence_scores)
        best_num_topics = topic_numbers[best_idx]
        best_coherence = coherence_scores[best_idx]
        
        print(f"\nBest number of topics: {best_num_topics} (coherence: {best_coherence:.4f})")
        
        return best_num_topics, best_coherence
    
    def get_top_terms_per_topic(
        self,
        top_n: int = 25
    ) -> Dict[int, List[Tuple[str, float]]]:
        """
        Get top N terms for each topic with probabilities.
        
        Args:
            top_n: Number of top terms per topic
            
        Returns:
            Dict mapping topic ID to list of (term, probability) tuples
        """
        if self.lda_model is None:
            raise ValueError("Model not trained. Call train_lda() first.")
        
        topics = {}
        
        for topic_id in range(self.num_topics):
            # Get topic terms
            topic_terms = self.lda_model.show_topic(topic_id, topn=top_n)
            
            # Convert to list of tuples
            topics[topic_id] = [
                (term, prob) for term, prob in topic_terms
            ]
        
        return topics
    
    def compute_document_topic_distributions(
        self,
        corpus: List[List[Tuple[int, int]]]
    ) -> List[np.ndarray]:
        """
        Compute topic distribution for each document.
        
        Args:
            corpus: Gensim corpus
            
        Returns:
            List of topic distribution arrays (one per document)
        """
        distributions = []
        
        for doc_bow in corpus:
            # Get topic distribution for document
            topic_dist = self.lda_model.get_document_topics(
                doc_bow,
                minimum_probability=0.0
            )
            
            # Convert to numpy array
            dist_array = np.zeros(self.num_topics)
            for topic_id, prob in topic_dist:
                dist_array[topic_id] = prob
            
            distributions.append(dist_array)
        
        self.document_topic_distributions = distributions
        return distributions
    
    def compute_category_topic_distributions(
        self,
        document_categories: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        Compute average topic distribution per category.
        
        Args:
            document_categories: List of category labels for each document
            
        Returns:
            Dict mapping category to average topic distribution
        """
        if not self.document_topic_distributions:
            raise ValueError("Document topic distributions not computed.")
        
        # Group documents by category
        category_docs = defaultdict(list)
        for idx, category in enumerate(document_categories):
            category_docs[category].append(idx)
        
        # Compute average distribution per category
        category_distributions = {}
        
        for category, doc_indices in category_docs.items():
            # Get distributions for documents in this category
            cat_dists = [
                self.document_topic_distributions[idx]
                for idx in doc_indices
            ]
            
            # Average
            avg_dist = np.mean(cat_dists, axis=0)
            category_distributions[category] = avg_dist
        
        self.category_topic_distributions = category_distributions
        return category_distributions
    
    def get_top_topics_per_category(
        self,
        top_k: int = 5
    ) -> Dict[str, List[Tuple[int, float]]]:
        """
        Get top K topics for each category.
        
        Args:
            top_k: Number of top topics per category
            
        Returns:
            Dict mapping category to list of (topic_id, probability) tuples
        """
        if not self.category_topic_distributions:
            raise ValueError("Category topic distributions not computed.")
        
        top_topics = {}
        
        for category, dist in self.category_topic_distributions.items():
            # Sort topics by probability
            topic_probs = [
                (topic_id, prob)
                for topic_id, prob in enumerate(dist)
            ]
            topic_probs.sort(key=lambda x: x[1], reverse=True)
            
            top_topics[category] = topic_probs[:top_k]
        
        return top_topics
    
    def generate_visualization(
        self,
        output_path: Path,
        mds: str = "pcoa"
    ):
        """
        Generate pyLDAvis visualization.
        
        Args:
            output_path: Path to save HTML visualization
            mds: Multidimensional scaling method
        """
        if self.lda_model is None or self.corpus is None:
            raise ValueError("Model and corpus must be prepared first.")
        
        vis = gensimvis.prepare(
            self.lda_model,
            self.corpus,
            self.dictionary,
            mds=mds
        )
        
        pyLDAvis.save_html(vis, str(output_path))
        print(f"Saved visualization to {output_path}")
    
    def save_results(self, output_dir: Path):
        """
        Save topic modeling results.
        
        Args:
            output_dir: Directory to save results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get top terms per topic
        top_terms = self.get_top_terms_per_topic(top_n=25)
        
        # Get top topics per category
        top_topics = self.get_top_topics_per_category(top_k=5)
        
        # Prepare results
        results = {
            "num_topics": self.num_topics,
            "topics": {
                topic_id: [
                    {"term": term, "probability": float(prob)}
                    for term, prob in terms
                ]
                for topic_id, terms in top_terms.items()
            },
            "category_topic_distributions": {
                cat: {
                    topic_id: float(prob)
                    for topic_id, prob in enumerate(dist)
                }
                for cat, dist in self.category_topic_distributions.items()
            },
            "top_topics_per_category": {
                cat: [
                    {"topic_id": topic_id, "probability": float(prob)}
                    for topic_id, prob in topics
                ]
                for cat, topics in top_topics.items()
            }
        }
        
        # Save JSON
        results_file = output_dir / "topic_modeling_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save model
        if self.lda_model:
            model_file = output_dir / "lda_model.model"
            self.lda_model.save(str(model_file))
        
        # Save dictionary
        if self.dictionary:
            dict_file = output_dir / "dictionary.dict"
            self.dictionary.save(str(dict_file))
        
        print(f"Saved topic modeling results to {output_dir}")


def load_documents_by_category(data_dir: Path) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Load documents organized by category with document IDs.
    
    Args:
        data_dir: Directory containing categorized documents
        
    Returns:
        Tuple of (documents_by_category, document_ids_by_category)
    """
    data_dir = Path(data_dir)
    documents_by_category = {}
    doc_ids_by_category = {}
    
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
            docs, doc_ids = load_documents_from_category(category_dir)
            documents_by_category[category] = docs
            doc_ids_by_category[category] = doc_ids
    
    return documents_by_category, doc_ids_by_category


if __name__ == "__main__":
    """
    Main execution for topic modeling.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Topic modeling using LDA"
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
        default="output/topic_modeling",
        help="Output directory for results"
    )
    parser.add_argument(
        "--num-topics",
        type=int,
        default=10,
        help="Number of topics (default: 10)"
    )
    parser.add_argument(
        "--find-optimal",
        action="store_true",
        help="Find optimal number of topics using coherence"
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
    documents_by_category, doc_ids_by_category = load_documents_by_category(Path(args.data_dir))
    
    # Flatten to single list with category labels
    all_documents = []
    document_categories = []
    document_ids = []
    
    for category, docs in documents_by_category.items():
        all_documents.extend(docs)
        document_categories.extend([category] * len(docs))
        document_ids.extend(doc_ids_by_category[category])
    
    print(f"Total documents: {len(all_documents)}")
    print(f"Categories: {set(document_categories)}")
    
    # Create BOW processor
    processor = BOWProcessor(
        lowercase=args.lowercase,
        remove_stopwords_flag=args.stopwords,
        stem=args.stem,
        lemmatize=args.lemmatize,
        representation="count"
    )
    
    # Create topic modeler
    modeler = TopicModeler(num_topics=args.num_topics)
    
    # Find optimal topics if requested
    if args.find_optimal:
        optimal_topics, coherence = modeler.find_optimal_topics(
            all_documents,
            processor,
            topic_range=(5, 20),
            step=5
        )
        modeler.num_topics = optimal_topics
    
    # Prepare corpus
    print("Preparing corpus...")
    dictionary, corpus = modeler.prepare_corpus(all_documents, processor)
    print(f"Vocabulary size: {len(dictionary)}")
    print(f"Corpus size: {len(corpus)}")
    
    # Train model
    print(f"Training LDA model with {modeler.num_topics} topics...")
    modeler.train_lda(dictionary, corpus)
    
    # Compute document topic distributions
    print("Computing document topic distributions...")
    modeler.compute_document_topic_distributions(corpus)
    
    # Compute category topic distributions
    print("Computing category topic distributions...")
    modeler.compute_category_topic_distributions(document_categories)
    
    # Get top terms per topic
    print("Extracting top terms per topic...")
    top_terms = modeler.get_top_terms_per_topic(top_n=25)
    
    print("\nTop 25 terms per topic:")
    for topic_id, terms in top_terms.items():
        print(f"\nTopic {topic_id}:")
        for i, (term, prob) in enumerate(terms[:10], 1):  # Show first 10
            print(f"  {i}. {term} ({prob:.4f})")
    
    # Get top topics per category
    print("\nTop topics per category:")
    top_topics = modeler.get_top_topics_per_category(top_k=5)
    for category, topics in top_topics.items():
        print(f"\n{category}:")
        for topic_id, prob in topics:
            print(f"  Topic {topic_id}: {prob:.4f}")
    
    # Generate visualization
    vis_path = Path(args.output_dir) / "lda_visualization.html"
    print(f"\nGenerating visualization...")
    modeler.generate_visualization(vis_path)
    
    # Save results
    modeler.save_results(Path(args.output_dir))
