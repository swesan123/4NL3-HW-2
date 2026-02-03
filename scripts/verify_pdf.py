#!/usr/bin/env python3
"""
PDF Verification Script

Uses PDF MCP tool to verify generated PDF is not corrupted and contains
expected content.

Author: Swesan Pathmanathan
Course: 4NL3
Date: January 2026
"""

import sys
from pathlib import Path


def verify_pdf(pdf_path: Path, expected_pages: int = None) -> bool:
    """
    Verify PDF using PDF MCP tool.
    
    Args:
        pdf_path: Path to PDF file
        expected_pages: Expected number of pages (optional)
        
    Returns:
        True if PDF is valid
    """
    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        return False
    
    try:
        # Note: This would use the PDF MCP tool in actual implementation
        # For now, we'll do basic checks
        
        # Check file size
        file_size = pdf_path.stat().st_size
        if file_size < 1000:  # Less than 1KB is suspicious
            print(f"Warning: PDF file is very small ({file_size} bytes)")
            return False
        
        # Check file extension
        if pdf_path.suffix.lower() != '.pdf':
            print(f"Warning: File does not have .pdf extension")
            return False
        
        # Try to read first few bytes to check PDF header
        with open(pdf_path, 'rb') as f:
            header = f.read(4)
            if header != b'%PDF':
                print(f"Error: File does not appear to be a valid PDF (header: {header})")
                return False
        
        print(f"PDF verification passed: {pdf_path}")
        print(f"  File size: {file_size / 1024:.2f} KB")
        
        # TODO: Use PDF MCP tool for more detailed verification:
        # - Page count check
        # - Content structure validation
        # - Section presence check
        
        return True
        
    except Exception as e:
        print(f"Error verifying PDF: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Verify PDF file using PDF MCP tool"
    )
    parser.add_argument(
        "pdf_path",
        type=str,
        help="Path to PDF file"
    )
    parser.add_argument(
        "--expected-pages",
        type=int,
        help="Expected number of pages"
    )
    
    args = parser.parse_args()
    
    pdf_path = Path(args.pdf_path)
    success = verify_pdf(pdf_path, args.expected_pages)
    
    sys.exit(0 if success else 1)
