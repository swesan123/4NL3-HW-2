"""
Comprehensive test for all valid flag combinations
Tests every possible combination of normalization flags
"""
import pytest
import subprocess
import sys
from pathlib import Path


# All valid flag combinations
FLAG_COMBINATIONS = [
    [],
    ["-lowercase"],
    ["-myopt"],
    ["-stopwords"],
    ["-stem"],
    ["-lemmatize"],
    ["-analyze"],
    ["-lowercase", "-myopt"],
    ["-lowercase", "-stopwords"],
    ["-lowercase", "-stem"],
    ["-lowercase", "-lemmatize"],
    ["-lowercase", "-analyze"],
    ["-myopt", "-stopwords"],
    ["-myopt", "-stem"],
    ["-myopt", "-lemmatize"],
    ["-myopt", "-analyze"],
    ["-stopwords", "-stem"],
    ["-stopwords", "-lemmatize"],
    ["-stopwords", "-analyze"],
    ["-stem", "-analyze"],
    ["-lemmatize", "-analyze"],
    ["-lowercase", "-myopt", "-stopwords"],
    ["-lowercase", "-myopt", "-stem"],
    ["-lowercase", "-myopt", "-lemmatize"],
    ["-lowercase", "-myopt", "-analyze"],
    ["-lowercase", "-stopwords", "-stem"],
    ["-lowercase", "-stopwords", "-lemmatize"],
    ["-lowercase", "-stopwords", "-analyze"],
    ["-lowercase", "-stem", "-analyze"],
    ["-lowercase", "-lemmatize", "-analyze"],
    ["-myopt", "-stopwords", "-stem"],
    ["-myopt", "-stopwords", "-lemmatize"],
    ["-myopt", "-stopwords", "-analyze"],
    ["-myopt", "-stem", "-analyze"],
    ["-myopt", "-lemmatize", "-analyze"],
    ["-stopwords", "-stem", "-analyze"],
    ["-stopwords", "-lemmatize", "-analyze"],
    ["-lowercase", "-myopt", "-stopwords", "-stem"],
    ["-lowercase", "-myopt", "-stopwords", "-lemmatize"],
    ["-lowercase", "-myopt", "-stopwords", "-analyze"],
    ["-lowercase", "-myopt", "-stem", "-analyze"],
    ["-lowercase", "-myopt", "-lemmatize", "-analyze"],
    ["-lowercase", "-stopwords", "-stem", "-analyze"],
    ["-lowercase", "-stopwords", "-lemmatize", "-analyze"],
    ["-myopt", "-stopwords", "-stem", "-analyze"],
    ["-myopt", "-stopwords", "-lemmatize", "-analyze"],
    ["-lowercase", "-myopt", "-stopwords", "-stem", "-analyze"],
    ["-lowercase", "-myopt", "-stopwords", "-lemmatize", "-analyze"],
]


@pytest.fixture(scope="session")
def small_test_file(tmp_path_factory):
    """Create a small test file for combination testing"""
    test_dir = tmp_path_factory.mktemp("combo_test_data")
    test_file = test_dir / "test_combo.txt"
    
    # Small corpus for faster testing
    content = """The quick brown fox jumps over the lazy dog in 2024.
    The dog was running and jumping.
    Running dogs and cats are playing in 2025.
    The cats were running quickly."""
    
    test_file.write_text(content, encoding="utf-8")
    return test_file


class TestAllFlagCombinations:
    """Test all valid flag combinations with a small test file"""
    
    @pytest.mark.parametrize("flags", FLAG_COMBINATIONS)
    def test_flag_combination_small_file(self, small_test_file, flags):
        """Test each flag combination with small test file"""
        cmd = [sys.executable, "normalize_text.py", str(small_test_file)] + flags
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(Path(__file__).parent.parent)
        )
        
        assert result.returncode == 0, \
            f"Failed with flags {flags}\nStderr: {result.stderr}\nStdout: {result.stdout}"
        assert "The total # of tokens:" in result.stdout


class TestAllCombinationsWithRealData:
    """Test all combinations with actual data files (slower)"""
    
    @pytest.mark.slow
    @pytest.mark.parametrize("flags", FLAG_COMBINATIONS)
    def test_crime_and_punishment_all_combinations(self, flags):
        """Test all combinations with Crime and Punishment"""
        data_file = Path(__file__).parent.parent / "data" / "pg2554.txt"
        if not data_file.exists():
            pytest.skip("Crime and Punishment data file not available")
        
        cmd = [sys.executable, "normalize_text.py", str(data_file)] + flags
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(Path(__file__).parent.parent)
        )
        
        assert result.returncode == 0, \
            f"Failed with flags {flags}\nStderr: {result.stderr}"
        assert "The total # of tokens:" in result.stdout
    
    @pytest.mark.slow
    @pytest.mark.parametrize("flags", FLAG_COMBINATIONS)
    def test_linux_log_all_combinations(self, flags):
        """Test all combinations with Linux log"""
        data_file = Path(__file__).parent.parent / "data" / "Linux.txt"
        if not data_file.exists():
            pytest.skip("Linux log data file not available")
        
        cmd = [sys.executable, "normalize_text.py", str(data_file)] + flags
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(Path(__file__).parent.parent)
        )
        
        assert result.returncode == 0, \
            f"Failed with flags {flags}\nStderr: {result.stderr}"
        assert "The total # of tokens:" in result.stdout


class TestErrorCases:
    """Test error cases and invalid flag combinations"""
    
    def test_stem_and_lemmatize_conflict(self, small_test_file):
        """Test that -stem and -lemmatize cannot be used together"""
        cmd = [sys.executable, "normalize_text.py", str(small_test_file), 
               "-stem", "-lemmatize"]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent)
        )
        
        assert result.returncode != 0, "Should fail with conflicting flags"
        assert "error:" in result.stderr.lower() or "choose only one" in result.stderr.lower()
    
    def test_missing_input_file(self):
        """Test that missing input file is caught"""
        cmd = [sys.executable, "normalize_text.py"]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent)
        )
        
        assert result.returncode != 0
    
    def test_nonexistent_input_file(self):
        """Test that nonexistent input file is caught"""
        cmd = [sys.executable, "normalize_text.py", "nonexistent_file_xyz123.txt"]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent)
        )
        
        assert result.returncode != 0


def test_help_command():
    """Test --help command"""
    result = subprocess.run(
        [sys.executable, "normalize_text.py", "--help"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent.parent)
    )
    
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()
    assert "normalize_text.py" in result.stdout or "input_file" in result.stdout


# Summary test that counts all combinations
def test_combination_count():
    """Verify we're testing all expected combinations"""
    # Total combinations should be 48
    assert len(FLAG_COMBINATIONS) == 48, \
        f"Expected 48 flag combinations, got {len(FLAG_COMBINATIONS)}"
    
    # Verify no duplicates
    unique_combos = set(tuple(sorted(flags)) for flags in FLAG_COMBINATIONS)
    assert len(unique_combos) == len(FLAG_COMBINATIONS), \
        "Found duplicate flag combinations"
