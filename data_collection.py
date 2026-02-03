"""
Dataset Collection Module for HW-2 Corpus Analysis

This module handles downloading and organizing LogHub datasets with file size
management to avoid massive files. It identifies categories, ensures minimum
document counts, and organizes data for corpus analysis.

Author: Swesan Pathmanathan
Course: 4NL3
Date: January 2026

File Size Management:
- Checks file sizes before downloading (configurable limit, default 100MB)
- Implements streaming/chunked reading for large files
- Samples documents if dataset is too large
- Logs file sizes and document counts for transparency
"""

import os
import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import urllib.request
import zipfile
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetCollector:
    """
    Collects and organizes LogHub datasets with size management.
    """
    
    def __init__(
        self,
        data_dir: str = "data/loghub",
        max_file_size_mb: int = 100,
        min_docs_per_category: int = 100,
        max_docs_per_category: Optional[int] = 5000,
        sample_if_large: bool = True
    ):
        """
        Initialize dataset collector.
        
        Args:
            data_dir: Directory to store datasets
            max_file_size_mb: Maximum file size in MB to download (default 100MB)
            min_docs_per_category: Minimum documents required per category
            max_docs_per_category: Maximum documents per category (None = no limit)
            sample_if_large: Whether to sample documents if category exceeds max
        """
        self.data_dir = Path(data_dir)
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.min_docs_per_category = min_docs_per_category
        self.max_docs_per_category = max_docs_per_category
        self.sample_if_large = sample_if_large
        
        # Create data directory structure
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata file path
        self.metadata_file = self.data_dir / "dataset_metadata.json"
        
    def check_file_size(self, file_path: Path) -> Tuple[bool, int]:
        """
        Check if file size is within limits.
        
        Args:
            file_path: Path to file
            
        Returns:
            Tuple of (is_within_limit, file_size_bytes)
        """
        if not file_path.exists():
            return False, 0
        
        size = file_path.stat().st_size
        is_within_limit = size <= self.max_file_size_bytes
        
        return is_within_limit, size
    
    def download_file(
        self,
        url: str,
        destination: Path,
        check_size: bool = True
    ) -> Tuple[bool, Optional[int]]:
        """
        Download file from URL with size checking.
        
        Args:
            url: URL to download from
            destination: Destination path
            check_size: Whether to check file size before downloading
            
        Returns:
            Tuple of (success, file_size_bytes)
        """
        try:
            # Check URL file size first if possible
            if check_size:
                req = urllib.request.Request(url, method='HEAD')
                with urllib.request.urlopen(req) as response:
                    content_length = response.headers.get('Content-Length')
                    if content_length:
                        size = int(content_length)
                        if size > self.max_file_size_bytes:
                            logger.warning(
                                f"File {url} is too large ({size / 1024 / 1024:.2f}MB), skipping"
                            )
                            return False, size
            
            # Download file
            logger.info(f"Downloading {url} to {destination}")
            urllib.request.urlretrieve(url, destination)
            
            # Verify downloaded file size
            is_within_limit, size = self.check_file_size(destination)
            if not is_within_limit:
                logger.warning(
                    f"Downloaded file {destination} exceeds limit, removing"
                )
                destination.unlink()
                return False, size
            
            logger.info(f"Successfully downloaded {destination} ({size / 1024 / 1024:.2f}MB)")
            return True, size
            
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            return False, None
    
    def extract_zip(self, zip_path: Path, extract_to: Path) -> bool:
        """
        Extract zip file to destination.
        
        Args:
            zip_path: Path to zip file
            extract_to: Destination directory
            
        Returns:
            True if successful
        """
        try:
            extract_to.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            logger.info(f"Extracted {zip_path} to {extract_to}")
            return True
        except Exception as e:
            logger.error(f"Error extracting {zip_path}: {e}")
            return False
    
    def split_log_file(
        self,
        file_path: Path,
        category: str,
        split_by: str = "line"
    ) -> List[Path]:
        """
        Split log file into individual documents.
        
        Args:
            file_path: Path to log file
            category: Category name for organization
            split_by: How to split ("line", "entry", "time_window")
            
        Returns:
            List of document file paths
        """
        category_dir = self.data_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        
        documents = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                if split_by == "line":
                    # Each line is a document
                    for idx, line in enumerate(f):
                        if line.strip():  # Skip empty lines
                            doc_path = category_dir / f"doc_{idx:06d}.txt"
                            with open(doc_path, 'w', encoding='utf-8') as doc_file:
                                doc_file.write(line.strip())
                            documents.append(doc_path)
                
                elif split_by == "entry":
                    # Group lines by log entry (lines starting with timestamp or pattern)
                    current_entry = []
                    doc_idx = 0
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        # Simple heuristic: if line starts with timestamp-like pattern
                        # or is continuation, group it
                        if current_entry and not line[0].isdigit():
                            current_entry.append(line)
                        else:
                            if current_entry:
                                doc_path = category_dir / f"doc_{doc_idx:06d}.txt"
                                with open(doc_path, 'w', encoding='utf-8') as doc_file:
                                    doc_file.write('\n'.join(current_entry))
                                documents.append(doc_path)
                                doc_idx += 1
                            current_entry = [line]
                    
                    # Handle last entry
                    if current_entry:
                        doc_path = category_dir / f"doc_{doc_idx:06d}.txt"
                        with open(doc_path, 'w', encoding='utf-8') as doc_file:
                            doc_file.write('\n'.join(current_entry))
                        documents.append(doc_path)
        
        except Exception as e:
            logger.error(f"Error splitting {file_path}: {e}")
        
        return documents
    
    def sample_documents(
        self,
        documents: List[Path],
        max_count: Optional[int] = None
    ) -> List[Path]:
        """
        Sample documents if list exceeds maximum count.
        
        Args:
            documents: List of document paths
            max_count: Maximum number of documents to keep
            
        Returns:
            Sampled list of documents
        """
        if max_count is None or len(documents) <= max_count:
            return documents
        
        logger.info(
            f"Sampling {max_count} documents from {len(documents)} total"
        )
        sampled = random.sample(documents, max_count)
        
        # Remove unsampled documents
        unsampled = set(documents) - set(sampled)
        for doc in unsampled:
            doc.unlink()
        
        return sampled
    
    def organize_dataset(
        self,
        source_dir: Path,
        categories: Dict[str, str]
    ) -> Dict[str, List[Path]]:
        """
        Organize dataset files into categories.
        
        Args:
            source_dir: Directory containing raw dataset files
            categories: Dict mapping category names to file patterns/paths
            
        Returns:
            Dict mapping category names to lists of document paths
        """
        organized = defaultdict(list)
        
        for category, pattern in categories.items():
            category_files = list(source_dir.glob(pattern))
            
            if not category_files:
                logger.warning(f"No files found for category {category} with pattern {pattern}")
                continue
            
            for file_path in category_files:
                # Check file size
                is_within_limit, size = self.check_file_size(file_path)
                if not is_within_limit:
                    logger.warning(
                        f"File {file_path} ({size / 1024 / 1024:.2f}MB) exceeds limit, skipping"
                    )
                    continue
                
                # Split into documents
                documents = self.split_log_file(file_path, category)
                
                # Sample if necessary
                if self.sample_if_large and self.max_docs_per_category:
                    documents = self.sample_documents(documents, self.max_docs_per_category)
                
                organized[category].extend(documents)
        
        return dict(organized)
    
    def validate_categories(
        self,
        category_docs: Dict[str, List[Path]]
    ) -> Tuple[bool, Dict[str, int]]:
        """
        Validate that each category has minimum required documents.
        
        Args:
            category_docs: Dict mapping categories to document lists
            
        Returns:
            Tuple of (is_valid, category_counts)
        """
        category_counts = {cat: len(docs) for cat, docs in category_docs.items()}
        
        is_valid = all(
            count >= self.min_docs_per_category
            for count in category_counts.values()
        )
        
        if not is_valid:
            logger.warning(
                f"Some categories do not meet minimum requirement of "
                f"{self.min_docs_per_category} documents"
            )
            for cat, count in category_counts.items():
                if count < self.min_docs_per_category:
                    logger.warning(f"  {cat}: {count} documents")
        
        return is_valid, category_counts
    
    def save_metadata(
        self,
        category_docs: Dict[str, List[Path]],
        category_counts: Dict[str, int],
        source_info: Optional[Dict] = None
    ):
        """
        Save dataset metadata to JSON file.
        
        Args:
            category_docs: Dict mapping categories to document lists
            category_counts: Dict mapping categories to document counts
            source_info: Optional metadata about data source
        """
        metadata = {
            "categories": category_counts,
            "total_documents": sum(category_counts.values()),
            "min_docs_per_category": self.min_docs_per_category,
            "max_docs_per_category": self.max_docs_per_category,
            "max_file_size_mb": self.max_file_size_bytes / 1024 / 1024,
            "source_info": source_info or {},
            "document_paths": {
                cat: [str(doc.relative_to(self.data_dir)) for doc in docs]
                for cat, docs in category_docs.items()
            }
        }
        
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved metadata to {self.metadata_file}")
    
    def load_metadata(self) -> Optional[Dict]:
        """
        Load dataset metadata from JSON file.
        
        Returns:
            Metadata dict or None if file doesn't exist
        """
        if not self.metadata_file.exists():
            return None
        
        with open(self.metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)


def collect_loghub_dataset(
    dataset_name: str,
    categories: Dict[str, str],
    download_url: Optional[str] = None,
    local_path: Optional[str] = None
) -> DatasetCollector:
    """
    Collect a LogHub dataset and organize by categories.
    
    This is a convenience function for collecting LogHub datasets.
    Users should adapt this based on the specific dataset structure.
    
    Args:
        dataset_name: Name of the dataset
        categories: Dict mapping category names to file patterns
        download_url: Optional URL to download dataset
        local_path: Optional local path if dataset is already downloaded
        
    Returns:
        DatasetCollector instance with organized data
    """
    collector = DatasetCollector()
    
    # If local path provided, use it
    if local_path:
        source_dir = Path(local_path)
    elif download_url:
        # Download dataset
        zip_path = collector.data_dir / f"{dataset_name}.zip"
        success, size = collector.download_file(download_url, zip_path)
        
        if not success:
            raise ValueError(f"Failed to download dataset from {download_url}")
        
        # Extract
        extract_dir = collector.data_dir / dataset_name
        if not collector.extract_zip(zip_path, extract_dir):
            raise ValueError(f"Failed to extract {zip_path}")
        
        source_dir = extract_dir
    else:
        raise ValueError("Either download_url or local_path must be provided")
    
    # Organize by categories
    category_docs = collector.organize_dataset(source_dir, categories)
    
    # Validate
    is_valid, category_counts = collector.validate_categories(category_docs)
    
    if not is_valid:
        logger.warning(
            "Dataset validation failed. Some categories may not meet requirements."
        )
    
    # Save metadata
    collector.save_metadata(
        category_docs,
        category_counts,
        source_info={
            "dataset_name": dataset_name,
            "source": download_url or local_path
        }
    )
    
    return collector


if __name__ == "__main__":
    """
    Example usage for LogHub datasets.
    
    Note: Users need to adapt this based on actual LogHub dataset structure.
    LogHub datasets are typically available from:
    https://github.com/logpai/loghub
    
    Example categories might be:
    - Error vs Normal logs
    - Different system components
    - Different time periods
    - Different log types
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Collect and organize LogHub datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset name"
    )
    parser.add_argument(
        "--local-path",
        type=str,
        help="Local path to dataset files"
    )
    parser.add_argument(
        "--max-size-mb",
        type=int,
        default=100,
        help="Maximum file size in MB (default: 100)"
    )
    parser.add_argument(
        "--min-docs",
        type=int,
        default=100,
        help="Minimum documents per category (default: 100)"
    )
    
    args = parser.parse_args()
    
    # Example: This would need to be adapted based on actual dataset structure
    # For now, this is a template that users should modify
    logger.info(
        "Please adapt this script based on your specific LogHub dataset structure.\n"
        "LogHub datasets: https://github.com/logpai/loghub"
    )
    
    # Example categories (users should modify based on actual dataset)
    example_categories = {
        "error": "*.error.log",
        "normal": "*.log"
    }
    
    if args.local_path:
        collector = DatasetCollector(
            max_file_size_mb=args.max_size_mb,
            min_docs_per_category=args.min_docs
        )
        category_docs = collector.organize_dataset(
            Path(args.local_path),
            example_categories
        )
        is_valid, category_counts = collector.validate_categories(category_docs)
        collector.save_metadata(category_docs, category_counts)
        
        print("\nDataset Summary:")
        print(f"Categories: {list(category_counts.keys())}")
        print(f"Document counts: {category_counts}")
        print(f"Total documents: {sum(category_counts.values())}")
