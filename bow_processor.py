"""
Bag-of-Words Processor Module

This module converts documents to bag-of-words format with support for
count, binary, and TF-IDF representations. Uses scipy sparse matrices
for memory efficiency with large document-term matrices.

Reuses preprocessing functions from normalize_text.py.

Author: Swesan Pathmanathan
Course: 4NL3
Date: January 2026
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Import preprocessing functions from normalize_text
from normalize_text import (
    tokenize,
    apply_lowercase,
    remove_stopwords,
    apply_stemming,
    apply_lemmatization,
    remove_digits
)


class BOWProcessor:
    """
    Processes documents into bag-of-words representations.
    """
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_stopwords_flag: bool = False,
        stem: bool = False,
        lemmatize: bool = False,
        remove_digits_flag: bool = False,
        representation: str = "count"
    ):
        """
        Initialize BOW processor.
        
        Args:
            lowercase: Whether to lowercase tokens
            remove_stopwords_flag: Whether to remove stopwords
            stem: Whether to apply stemming
            lemmatize: Whether to apply lemmatization
            remove_digits_flag: Whether to remove digit-only tokens
            representation: Type of representation ("count", "binary", "tfidf")
        """
        self.lowercase = lowercase
        self.remove_stopwords_flag = remove_stopwords_flag
        self.stem = stem
        self.lemmatize = lemmatize
        self.remove_digits_flag = remove_digits_flag
        self.representation = representation
        
        if stem and lemmatize:
            raise ValueError("Cannot use both stemming and lemmatization")
        
        self.vocabulary: Dict[str, int] = {}
        self.vocabulary_reverse: Dict[int, str] = {}
        self.document_term_matrix: Optional[csr_matrix] = None
        self.document_ids: List[str] = []
        
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text using configured normalization options.
        
        Args:
            text: Input text
            
        Returns:
            List of preprocessed tokens
        """
        tokens = tokenize(text)
        
        if self.lowercase:
            tokens = apply_lowercase(tokens)
        
        if self.remove_digits_flag:
            tokens = remove_digits(tokens)
        
        if self.remove_stopwords_flag:
            tokens = remove_stopwords(tokens)
        
        if self.stem:
            tokens = apply_stemming(tokens)
        elif self.lemmatize:
            tokens = apply_lemmatization(tokens)
        
        return tokens
    
    def build_vocabulary(self, documents: List[str]) -> Dict[str, int]:
        """
        Build vocabulary from documents.
        
        Args:
            documents: List of document texts
            
        Returns:
            Vocabulary mapping tokens to indices
        """
        vocab_set = set()
        
        for doc in documents:
            tokens = self.preprocess_text(doc)
            vocab_set.update(tokens)
        
        # Sort vocabulary for consistent ordering
        sorted_vocab = sorted(vocab_set)
        self.vocabulary = {token: idx for idx, token in enumerate(sorted_vocab)}
        self.vocabulary_reverse = {idx: token for token, idx in self.vocabulary.items()}
        
        return self.vocabulary
    
    def document_to_bow_vector(
        self,
        document: str,
        vocab: Optional[Dict[str, int]] = None
    ) -> np.ndarray:
        """
        Convert a single document to bag-of-words vector.
        
        Args:
            document: Document text
            vocab: Vocabulary dict (uses self.vocabulary if None)
            
        Returns:
            BOW vector as numpy array
        """
        if vocab is None:
            vocab = self.vocabulary
        
        tokens = self.preprocess_text(document)
        
        if self.representation == "binary":
            # Binary representation: 1 if token appears, 0 otherwise
            vector = np.zeros(len(vocab))
            for token in set(tokens):  # Use set to count unique tokens only
                if token in vocab:
                    vector[vocab[token]] = 1
        else:
            # Count representation
            vector = np.zeros(len(vocab))
            for token in tokens:
                if token in vocab:
                    vector[vocab[token]] += 1
        
        return vector
    
    def documents_to_matrix(
        self,
        documents: List[str],
        document_ids: Optional[List[str]] = None
    ) -> csr_matrix:
        """
        Convert list of documents to document-term matrix.
        
        Args:
            documents: List of document texts
            document_ids: Optional list of document identifiers
            
        Returns:
            Sparse CSR matrix (documents x vocabulary)
        """
        if not self.vocabulary:
            self.build_vocabulary(documents)
        
        if document_ids is None:
            document_ids = [f"doc_{i}" for i in range(len(documents))]
        
        self.document_ids = document_ids
        
        # Build matrix row by row
        rows = []
        cols = []
        data = []
        
        for doc_idx, doc in enumerate(documents):
            tokens = self.preprocess_text(doc)
            
            if self.representation == "binary":
                # Count unique tokens only
                token_counts = Counter(set(tokens))
            else:
                # Count all tokens
                token_counts = Counter(tokens)
            
            for token, count in token_counts.items():
                if token in self.vocabulary:
                    rows.append(doc_idx)
                    cols.append(self.vocabulary[token])
                    data.append(count if self.representation != "binary" else 1)
        
        # Create sparse matrix
        self.document_term_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(len(documents), len(self.vocabulary))
        )
        
        return self.document_term_matrix
    
    def apply_tfidf(self, matrix: csr_matrix) -> csr_matrix:
        """
        Apply TF-IDF transformation to document-term matrix.
        
        Args:
            matrix: Document-term matrix (counts)
            
        Returns:
            TF-IDF transformed matrix
        """
        from sklearn.feature_extraction.text import TfidfTransformer
        
        transformer = TfidfTransformer()
        tfidf_matrix = transformer.fit_transform(matrix)
        
        return tfidf_matrix
    
    def process_documents(
        self,
        documents: List[str],
        document_ids: Optional[List[str]] = None,
        use_tfidf: bool = False
    ) -> csr_matrix:
        """
        Process documents into final representation.
        
        Args:
            documents: List of document texts
            document_ids: Optional list of document identifiers
            use_tfidf: Whether to apply TF-IDF transformation
            
        Returns:
            Document-term matrix
        """
        # Build vocabulary and matrix
        matrix = self.documents_to_matrix(documents, document_ids)
        
        # Apply TF-IDF if requested
        if use_tfidf or self.representation == "tfidf":
            matrix = self.apply_tfidf(matrix)
            self.representation = "tfidf"
        
        return matrix
    
    def save(self, output_dir: Path, prefix: str = "bow"):
        """
        Save BOW processor state and matrices.
        
        Args:
            output_dir: Directory to save files
            prefix: Prefix for output files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save vocabulary
        vocab_file = output_dir / f"{prefix}_vocabulary.json"
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self.vocabulary, f, indent=2, ensure_ascii=False)
        
        # Save matrix
        if self.document_term_matrix is not None:
            matrix_file = output_dir / f"{prefix}_matrix.npz"
            save_npz(matrix_file, self.document_term_matrix)
        
        # Save metadata
        metadata = {
            "lowercase": self.lowercase,
            "remove_stopwords": self.remove_stopwords_flag,
            "stem": self.stem,
            "lemmatize": self.lemmatize,
            "remove_digits": self.remove_digits_flag,
            "representation": self.representation,
            "vocab_size": len(self.vocabulary),
            "num_documents": len(self.document_ids),
            "document_ids": self.document_ids
        }
        
        metadata_file = output_dir / f"{prefix}_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
    
    def load(self, input_dir: Path, prefix: str = "bow"):
        """
        Load BOW processor state and matrices.
        
        Args:
            input_dir: Directory containing saved files
            prefix: Prefix for input files
        """
        input_dir = Path(input_dir)
        
        # Load vocabulary
        vocab_file = input_dir / f"{prefix}_vocabulary.json"
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self.vocabulary = json.load(f)
        self.vocabulary_reverse = {idx: token for token, idx in self.vocabulary.items()}
        
        # Load matrix
        matrix_file = input_dir / f"{prefix}_matrix.npz"
        if matrix_file.exists():
            self.document_term_matrix = load_npz(matrix_file)
        
        # Load metadata
        metadata_file = input_dir / f"{prefix}_metadata.json"
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        self.lowercase = metadata["lowercase"]
        self.remove_stopwords_flag = metadata["remove_stopwords"]
        self.stem = metadata["stem"]
        self.lemmatize = metadata["lemmatize"]
        self.remove_digits_flag = metadata["remove_digits"]
        self.representation = metadata["representation"]
        self.document_ids = metadata.get("document_ids", [])


def load_documents_from_category(
    category_dir: Path
) -> Tuple[List[str], List[str]]:
    """
    Load documents from a category directory.
    
    Args:
        category_dir: Directory containing document files
        
    Returns:
        Tuple of (document_texts, document_ids)
    """
    category_dir = Path(category_dir)
    documents = []
    doc_ids = []
    
    for doc_file in sorted(category_dir.glob("*.txt")):
        with open(doc_file, 'r', encoding='utf-8', errors='replace') as f:
            documents.append(f.read())
        doc_ids.append(doc_file.stem)
    
    return documents, doc_ids


if __name__ == "__main__":
    """
    Example usage of BOW processor.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert documents to bag-of-words format"
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
        default="output/bow",
        help="Output directory for BOW matrices"
    )
    parser.add_argument(
        "--representation",
        type=str,
        choices=["count", "binary", "tfidf"],
        default="count",
        help="BOW representation type"
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
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    # Load metadata to get categories
    metadata_file = data_dir / "dataset_metadata.json"
    if not metadata_file.exists():
        print(f"Metadata file not found: {metadata_file}")
        print("Please run data_collection.py first")
        exit(1)
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Process each category
    all_documents = []
    all_doc_ids = []
    category_boundaries = []
    
    for category in metadata["categories"].keys():
        category_dir = data_dir / category
        if category_dir.exists():
            docs, doc_ids = load_documents_from_category(category_dir)
            all_documents.extend(docs)
            all_doc_ids.extend(doc_ids)
            category_boundaries.append((category, len(all_documents)))
    
    # Create BOW processor
    processor = BOWProcessor(
        lowercase=args.lowercase,
        remove_stopwords_flag=args.stopwords,
        stem=args.stem,
        lemmatize=args.lemmatize,
        representation=args.representation
    )
    
    # Process documents
    print(f"Processing {len(all_documents)} documents...")
    matrix = processor.process_documents(
        all_documents,
        all_doc_ids,
        use_tfidf=(args.representation == "tfidf")
    )
    
    print(f"Vocabulary size: {len(processor.vocabulary)}")
    print(f"Matrix shape: {matrix.shape}")
    print(f"Matrix sparsity: {(1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1])) * 100:.2f}%")
    
    # Save
    processor.save(output_dir)
    print(f"Saved BOW representation to {output_dir}")
