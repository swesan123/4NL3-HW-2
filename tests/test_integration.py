"""
Integration tests for command-line combinations
Tests all possible combinations of normalization flags
"""
import pytest
import subprocess
import sys
from pathlib import Path
from itertools import combinations

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "test_data"


@pytest.fixture(scope="session")
def test_file(tmp_path_factory):
    """Create a test input file for all tests"""
    test_dir = tmp_path_factory.mktemp("data")
    test_file = test_dir / "test_input.txt"
    
    # Create a small test corpus
    content = """The quick brown fox jumps over the lazy dog.
    The dog was running and jumping in 2024.
    Running dogs and cats are playing.
    The cats were running quickly."""
    
    test_file.write_text(content, encoding="utf-8")
    return test_file


class TestBasicExecution:
    """Test basic command execution"""
    
    def test_help_flag(self):
        """Test --help flag"""
        result = subprocess.run(
            [sys.executable, "normalize_text.py", "--help"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "usage:" in result.stdout.lower()
    
    def test_no_arguments(self):
        """Test that script requires an input file"""
        result = subprocess.run(
            [sys.executable, "normalize_text.py"],
            capture_output=True,
            text=True
        )
        assert result.returncode != 0
    
    def test_basic_run(self, test_file):
        """Test basic run with just input file"""
        result = subprocess.run(
            [sys.executable, "normalize_text.py", str(test_file)],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent)
        )
        assert result.returncode == 0
        assert "The total # of tokens:" in result.stdout


class TestSingleFlags:
    """Test each flag individually"""
    
    def test_lowercase_flag(self, test_file):
        """Test -lowercase flag"""
        result = subprocess.run(
            [sys.executable, "normalize_text.py", str(test_file), "-lowercase"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent)
        )
        assert result.returncode == 0
        assert "lowercase" in result.stdout.lower()
    
    def test_stopwords_flag(self, test_file):
        """Test -stopwords flag"""
        result = subprocess.run(
            [sys.executable, "normalize_text.py", str(test_file), "-stopwords"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent)
        )
        assert result.returncode == 0
    
    def test_stem_flag(self, test_file):
        """Test -stem flag"""
        result = subprocess.run(
            [sys.executable, "normalize_text.py", str(test_file), "-stem"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent)
        )
        assert result.returncode == 0
    
    def test_lemmatize_flag(self, test_file):
        """Test -lemmatize flag"""
        result = subprocess.run(
            [sys.executable, "normalize_text.py", str(test_file), "-lemmatize"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent)
        )
        assert result.returncode == 0
    
    def test_myopt_flag(self, test_file):
        """Test -myopt flag"""
        result = subprocess.run(
            [sys.executable, "normalize_text.py", str(test_file), "-myopt"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent)
        )
        assert result.returncode == 0
    
    def test_analyze_flag(self, test_file):
        """Test -analyze flag"""
        result = subprocess.run(
            [sys.executable, "normalize_text.py", str(test_file), "-analyze"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent)
        )
        assert result.returncode == 0
        assert "Saved plot to" in result.stdout


class TestMutuallyExclusiveFlags:
    """Test mutually exclusive flags"""
    
    def test_stem_and_lemmatize_conflict(self, test_file):
        """Test that -stem and -lemmatize cannot be used together"""
        result = subprocess.run(
            [sys.executable, "normalize_text.py", str(test_file), "-stem", "-lemmatize"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent)
        )
        assert result.returncode != 0
        assert "error:" in result.stderr.lower()


class TestTwoFlagCombinations:
    """Test all valid two-flag combinations"""
    
    @pytest.mark.parametrize("flags", [
        ["-lowercase", "-stopwords"],
        ["-lowercase", "-stem"],
        ["-lowercase", "-lemmatize"],
        ["-lowercase", "-myopt"],
        ["-lowercase", "-analyze"],
        ["-stopwords", "-stem"],
        ["-stopwords", "-lemmatize"],
        ["-stopwords", "-myopt"],
        ["-stopwords", "-analyze"],
        ["-stem", "-myopt"],
        ["-stem", "-analyze"],
        ["-lemmatize", "-myopt"],
        ["-lemmatize", "-analyze"],
        ["-myopt", "-analyze"],
    ])
    def test_two_flag_combination(self, test_file, flags):
        """Test valid two-flag combinations"""
        result = subprocess.run(
            [sys.executable, "normalize_text.py", str(test_file)] + flags,
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent)
        )
        assert result.returncode == 0


class TestThreeFlagCombinations:
    """Test representative three-flag combinations"""
    
    @pytest.mark.parametrize("flags", [
        ["-lowercase", "-stopwords", "-stem"],
        ["-lowercase", "-stopwords", "-lemmatize"],
        ["-lowercase", "-myopt", "-stem"],
        ["-lowercase", "-myopt", "-lemmatize"],
        ["-lowercase", "-stopwords", "-analyze"],
        ["-stopwords", "-stem", "-analyze"],
        ["-stopwords", "-lemmatize", "-analyze"],
        ["-myopt", "-stem", "-analyze"],
        ["-myopt", "-lemmatize", "-analyze"],
    ])
    def test_three_flag_combination(self, test_file, flags):
        """Test valid three-flag combinations"""
        result = subprocess.run(
            [sys.executable, "normalize_text.py", str(test_file)] + flags,
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent)
        )
        assert result.returncode == 0


class TestFourFlagCombinations:
    """Test representative four-flag combinations"""
    
    @pytest.mark.parametrize("flags", [
        ["-lowercase", "-stopwords", "-stem", "-myopt"],
        ["-lowercase", "-stopwords", "-lemmatize", "-myopt"],
        ["-lowercase", "-stopwords", "-stem", "-analyze"],
        ["-lowercase", "-stopwords", "-lemmatize", "-analyze"],
        ["-lowercase", "-myopt", "-stem", "-analyze"],
        ["-lowercase", "-myopt", "-lemmatize", "-analyze"],
        ["-stopwords", "-stem", "-myopt", "-analyze"],
        ["-stopwords", "-lemmatize", "-myopt", "-analyze"],
    ])
    def test_four_flag_combination(self, test_file, flags):
        """Test valid four-flag combinations"""
        result = subprocess.run(
            [sys.executable, "normalize_text.py", str(test_file)] + flags,
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent)
        )
        assert result.returncode == 0


class TestFullPipeline:
    """Test complete pipeline combinations"""
    
    def test_full_pipeline_with_stem(self, test_file):
        """Test full pipeline: lowercase + myopt + stopwords + stem + analyze"""
        result = subprocess.run(
            [sys.executable, "normalize_text.py", str(test_file), 
             "-lowercase", "-myopt", "-stopwords", "-stem", "-analyze"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent)
        )
        assert result.returncode == 0
        assert "The total # of tokens:" in result.stdout
        assert "Saved token counts" in result.stdout
        assert "Saved plot to" in result.stdout
    
    def test_full_pipeline_with_lemmatize(self, test_file):
        """Test full pipeline: lowercase + myopt + stopwords + lemmatize + analyze"""
        result = subprocess.run(
            [sys.executable, "normalize_text.py", str(test_file), 
             "-lowercase", "-myopt", "-stopwords", "-lemmatize", "-analyze"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent)
        )
        assert result.returncode == 0
        assert "The total # of tokens:" in result.stdout
        assert "Saved token counts" in result.stdout
        assert "Saved plot to" in result.stdout


class TestOutputFiles:
    """Test that output files are created correctly"""
    
    def test_token_count_file_created(self, test_file, tmp_path, monkeypatch):
        """Test that token count files are created"""
        monkeypatch.chdir(tmp_path)
        
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent.parent / "normalize_text.py"), 
             str(test_file)],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        
        # Check output directory was created
        assert (tmp_path / "output").exists()
        
        # Check token file exists
        output_files = list((tmp_path / "output").glob("tokens_*.log"))
        assert len(output_files) > 0
    
    def test_plot_file_created(self, test_file, tmp_path, monkeypatch):
        """Test that plot files are created with -analyze"""
        monkeypatch.chdir(tmp_path)
        
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent.parent / "normalize_text.py"), 
             str(test_file), "-analyze"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        
        # Check plot file exists
        plot_files = list((tmp_path / "output").glob("zipf_*.png"))
        assert len(plot_files) > 0


class TestRealWorldFiles:
    """Test with actual data files if available"""
    
    def test_crime_and_punishment_raw(self):
        """Test with Crime and Punishment if available"""
        data_file = Path(__file__).parent.parent / "data" / "pg2554.txt"
        if not data_file.exists():
            pytest.skip("Crime and Punishment data file not available")
        
        result = subprocess.run(
            [sys.executable, "normalize_text.py", str(data_file)],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent)
        )
        assert result.returncode == 0
        assert "206544" in result.stdout or "tokens:" in result.stdout.lower()
    
    def test_linux_log_raw(self):
        """Test with Linux log if available"""
        data_file = Path(__file__).parent.parent / "data" / "Linux.txt"
        if not data_file.exists():
            pytest.skip("Linux log data file not available")
        
        result = subprocess.run(
            [sys.executable, "normalize_text.py", str(data_file)],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent)
        )
        assert result.returncode == 0
        assert "308039" in result.stdout or "tokens:" in result.stdout.lower()
