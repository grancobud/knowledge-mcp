#!/usr/bin/env python3
"""Minimal test script for DocumentManager _extract_text method."""

import logging
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from markitdown import MarkItDown

# Set up logging to see the detailed output
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Simulate the DocumentManager's _extract_text method
class TestDocumentManager:
    """Minimal DocumentManager for testing _extract_text method."""
    
    # Supported extensions from DocumentManager
    SUPPORTED_EXTENSIONS = {
        # Office documents
        '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls',
        # PDFs
        '.pdf',
        # Text files
        '.txt', '.md', '.rst', '.csv', '.json', '.xml', '.yaml', '.yml',
        # Web files
        '.html', '.htm',
        # Email files
        '.eml', '.msg',
        # Audio files (requires transcription)
        '.mp3', '.wav', '.m4a',
        # Images (requires vision model)
        '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'
    }
    
    def __init__(self):
        self.markitdown = MarkItDown()
        self.logger = logging.getLogger(__name__)
    
    def _extract_text(self, doc_path: Path) -> str:
        """Extracts text content from a document using MarkItDown."""
        # Check if file extension is supported
        file_ext = doc_path.suffix.lower()
        if file_ext not in self.SUPPORTED_EXTENSIONS:
            self.logger.warning(f"File type {file_ext} not in explicitly supported list, attempting extraction with MarkItDown anyway.")
        else:
            self.logger.info(f"Processing supported file type: {file_ext}")
        
        # Log file information
        try:
            file_size = doc_path.stat().st_size
            self.logger.info(f"Starting MarkItDown extraction for: {doc_path.name} (Size: {file_size:,} bytes, Type: {file_ext})")
        except OSError:
            self.logger.info(f"Starting MarkItDown extraction for: {doc_path.name} (Type: {file_ext})")
        
        try:
            self.logger.debug(f"Calling MarkItDown.convert() for: {doc_path}")
            result = self.markitdown.convert(str(doc_path))
            
            # Extract and validate content
            text_content = result.text_content
            content_length = len(text_content) if text_content else 0
            
            if content_length == 0:
                self.logger.warning(f"MarkItDown extraction returned empty content for: {doc_path.name}")
            else:
                self.logger.info(f"MarkItDown extraction successful for: {doc_path.name} (Extracted {content_length:,} characters)")
                self.logger.debug(f"First 100 characters of extracted content: {text_content[:100]!r}...")
            
            return text_content
        except FileNotFoundError as e:
            msg = f"Document file not found: {doc_path}"
            self.logger.error(msg)
            raise Exception(msg) from e
        except PermissionError as e:
            msg = f"Permission denied accessing file: {doc_path}"
            self.logger.error(msg)
            raise Exception(msg) from e
        except (ValueError, TypeError) as e:
            # Handle invalid file format or corrupted files
            msg = f"Invalid or corrupted file format for {doc_path}: {e}"
            self.logger.error(msg)
            raise Exception(msg) from e
        except ImportError as e:
            # Handle missing optional dependencies for specific file types
            msg = f"Missing required dependency for processing {doc_path}: {e}. Try installing with 'pip install markitdown[all]'"
            self.logger.error(msg)
            raise Exception(msg) from e
        except Exception as e:
            # Catch any other MarkItDown-specific or unexpected errors
            error_type = type(e).__name__
            msg = f"Failed to extract text from {doc_path} using MarkItDown ({error_type}): {e}"
            self.logger.exception(msg)
            raise Exception(msg) from e

def test_document_manager_extract_text():
    """Test DocumentManager _extract_text method with various file types."""
    print("=== Testing DocumentManager _extract_text Method ===\n")
    
    # Initialize test DocumentManager
    doc_manager = TestDocumentManager()
    
    # Test files directory
    test_files_dir = Path("tests/test-files")
    
    # Test files to process
    test_files = [
        "iphone.md",      # Markdown file
        "math.md",        # Another markdown file
        "math.png",       # PNG image (requires LLM for description)
        "math_origin.pdf" # PDF file
    ]
    
    print(f"Testing DocumentManager._extract_text() with files from {test_files_dir}:\n")
    
    # Test each file
    success_count = 0
    total_count = 0
    
    for test_file in test_files:
        file_path = test_files_dir / test_file
        
        if not file_path.exists():
            print(f"❌ Skipping {test_file} - file not found")
            continue
            
        total_count += 1
        print(f"🧪 Testing DocumentManager._extract_text() with: {test_file}")
        
        try:
            # Test the _extract_text method
            text_content = doc_manager._extract_text(file_path)
            
            if text_content:
                content_length = len(text_content)
                print(f"✅ SUCCESS: Extracted {content_length:,} characters")
                success_count += 1
                
                # Show first 150 characters as preview
                preview = text_content[:150].replace('\n', '\\n')
                print(f"   Preview: {preview}...")
            else:
                print("⚠️  WARNING: Extraction returned empty content")
                
        except Exception as e:
            print(f"❌ FAILED: {type(e).__name__}: {e}")
            
        print("-" * 60)
    
    print(f"\n=== Test Results ===")
    print(f"Successful extractions: {success_count}/{total_count}")
    print(f"Success rate: {(success_count/total_count)*100:.1f}%" if total_count > 0 else "No files tested")
    print("DocumentManager MarkItDown integration test completed.")

if __name__ == "__main__":
    test_document_manager_extract_text()
