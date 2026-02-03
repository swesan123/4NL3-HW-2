"""
Unit tests for individual functions in normalize_text.py
"""
import pytest
import os
import sys
from pathlib import Path

# Add parent directory to path to import normalize_text
sys.path.insert(0, str(Path(__file__).parent.parent))

import normalize_text


class TestTokenize:
    """Test the tokenize function"""
    
    def test_simple_tokenization(self):
        text = "Hello world this is a test"
        result = normalize_text.tokenize(text)
        assert result == ["Hello", "world", "this", "is", "a", "test"]
    
    def test_multiple_whitespace(self):
        text = "Hello    world\t\ttest\n\nend"
        result = normalize_text.tokenize(text)
        assert result == ["Hello", "world", "test", "end"]
    
    def test_empty_string(self):
        result = normalize_text.tokenize("")
        assert result == []
    
    def test_punctuation_attached(self):
        text = "Hello, world! How are you?"
        result = normalize_text.tokenize(text)
        assert result == ["Hello,", "world!", "How", "are", "you?"]


class TestCountTokens:
    """Test the count_tokens function"""
    
    def test_count_simple(self):
        tokens = ["the", "cat", "the", "dog"]
        result = normalize_text.count_tokens(tokens)
        assert result == [("the", 2), ("cat", 1), ("dog", 1)]
    
    def test_count_sorted(self):
        tokens = ["a", "b", "c", "a", "a", "b"]
        result = normalize_text.count_tokens(tokens)
        assert result[0] == ("a", 3)
        assert result[1] == ("b", 2)
        assert result[2] == ("c", 1)
    
    def test_empty_list(self):
        result = normalize_text.count_tokens([])
        assert result == []


class TestApplyLowercase:
    """Test the apply_lowercase function"""
    
    def test_mixed_case(self):
        tokens = ["Hello", "WORLD", "TeSt"]
        result = normalize_text.apply_lowercase(tokens)
        assert result == ["hello", "world", "test"]
    
    def test_already_lowercase(self):
        tokens = ["hello", "world"]
        result = normalize_text.apply_lowercase(tokens)
        assert result == ["hello", "world"]
    
    def test_with_punctuation(self):
        tokens = ["Hello,", "World!"]
        result = normalize_text.apply_lowercase(tokens)
        assert result == ["hello,", "world!"]


class TestRemoveDigits:
    """Test the remove_digits function"""
    
    def test_remove_pure_digits(self):
        tokens = ["hello", "2005", "world", "123", "test"]
        result = normalize_text.remove_digits(tokens)
        assert result == ["hello", "world", "test"]
    
    def test_keep_alphanumeric(self):
        tokens = ["hello", "2005", "test123", "abc"]
        result = normalize_text.remove_digits(tokens)
        assert result == ["hello", "test123", "abc"]
    
    def test_no_digits(self):
        tokens = ["hello", "world"]
        result = normalize_text.remove_digits(tokens)
        assert result == ["hello", "world"]
    
    def test_empty_list(self):
        result = normalize_text.remove_digits([])
        assert result == []


class TestRemoveStopwords:
    """Test the remove_stopwords function"""
    
    def test_remove_common_stopwords(self):
        tokens = ["the", "cat", "is", "on", "the", "mat"]
        result = normalize_text.remove_stopwords(tokens)
        assert "the" not in result
        assert "is" not in result
        assert "on" not in result
        assert "cat" in result
        assert "mat" in result
    
    def test_case_sensitive(self):
        # Stopwords are lowercase by default
        tokens = ["The", "cat", "the", "dog"]
        result = normalize_text.remove_stopwords(tokens)
        # "The" (capitalized) might remain if not in lowercase stopword list
        assert "the" not in result
        assert "cat" in result
        assert "dog" in result


class TestApplyStemming:
    """Test the apply_stemming function"""
    
    def test_stem_verbs(self):
        tokens = ["running", "runs", "ran"]
        result = normalize_text.apply_stemming(tokens)
        # Porter stemmer should reduce these
        assert "run" in result
    
    def test_stem_nouns(self):
        tokens = ["cats", "cat", "dogs", "dog"]
        result = normalize_text.apply_stemming(tokens)
        assert "cat" in result
        assert "dog" in result


class TestApplyLemmatization:
    """Test the apply_lemmatization function"""
    
    def test_lemmatize_nouns(self):
        tokens = ["cats", "cat"]
        result = normalize_text.apply_lemmatization(tokens)
        assert result == ["cat", "cat"]
    
    def test_lemmatize_unchanged(self):
        tokens = ["hello", "world"]
        result = normalize_text.apply_lemmatization(tokens)
        assert result == ["hello", "world"]


class TestReadFile:
    """Test file reading functionality"""
    
    def test_read_existing_file(self, tmp_path):
        # Create a temporary test file
        test_file = tmp_path / "test.txt"
        test_content = "Hello world\nTest file"
        test_file.write_text(test_content, encoding="utf-8")
        
        result = normalize_text.read_file(str(test_file))
        assert result == test_content
    
    def test_read_utf8_with_errors(self, tmp_path):
        # Create a file with some invalid UTF-8
        test_file = tmp_path / "test_invalid.txt"
        # Write valid UTF-8 + some invalid bytes
        with open(test_file, "wb") as f:
            f.write(b"Hello \xf7\xff\xbf world")
        
        # Should not crash, should replace invalid chars
        result = normalize_text.read_file(str(test_file))
        assert "Hello" in result
        assert "world" in result
    
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            normalize_text.read_file("nonexistent_file.txt")


class TestWriteFile:
    """Test file writing functionality"""
    
    def test_write_creates_output_dir(self, tmp_path, monkeypatch):
        # Change to temp directory
        monkeypatch.chdir(tmp_path)
        
        normalize_text.write_file("test.log", "Header", ["line1", "line2"])
        
        assert (tmp_path / "output" / "test.log").exists()
        content = (tmp_path / "output" / "test.log").read_text()
        assert "Header" in content
        assert "line1" in content
        assert "line2" in content


class TestWriteTokenCounts:
    """Test write_token_counts functionality"""
    
    def test_write_token_counts(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        
        counts = [("the", 10), ("cat", 5), ("dog", 2)]
        normalize_text.write_token_counts(counts, 17, "test")
        
        output_file = tmp_path / "output" / "tokens_test.log"
        assert output_file.exists()
        
        content = output_file.read_text()
        assert "The total # of tokens: 17" in content
        assert "the\t10" in content
        assert "cat\t5" in content
        
        # Check console output
        captured = capsys.readouterr()
        assert "Saved token counts" in captured.out
