"""
Pytest configuration and shared fixtures
"""
import pytest
import os
import shutil
from pathlib import Path


@pytest.fixture(autouse=True)
def cleanup_output_dir():
    """Clean up output directory before each test"""
    output_dir = Path("output")
    if output_dir.exists():
        # Save existing output if needed
        backup_dir = Path("output_backup")
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(output_dir, backup_dir)
    
    yield
    
    # Optionally restore or clean up
    # You can modify this based on your needs


@pytest.fixture
def sample_text():
    """Provide sample text for testing"""
    return """The quick brown fox jumps over the lazy dog.
    The dog was running and jumping in 2024.
    Running dogs and cats are playing.
    The cats were running quickly in 2025."""


@pytest.fixture
def sample_tokens():
    """Provide sample token list for testing"""
    return ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]


@pytest.fixture
def temp_test_file(tmp_path, sample_text):
    """Create a temporary test file with sample text"""
    test_file = tmp_path / "test_input.txt"
    test_file.write_text(sample_text, encoding="utf-8")
    return test_file
