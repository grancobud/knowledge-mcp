# tests/test_document_manager.py
"""Unit tests for the DocumentManager class."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

from knowledge_mcp.document_manager import (
    DocumentManager,
    TextExtractionError,
    DocumentManagerError
)
from knowledge_mcp.rag_manager import RAGManager, RAGInitializationError

# Mark all tests in this module as async tests
pytestmark = pytest.mark.asyncio

@pytest.fixture
def mock_rag_manager():
    """Fixture for a mocked RAGManager."""
    manager = MagicMock(spec=RAGManager)
    # Mock the async method get_rag_instance
    manager.get_rag_instance = AsyncMock()
    return manager

@pytest.fixture
def mock_lightrag_instance():
    """Fixture for a mocked LightRAG instance."""
    instance = MagicMock()
    # Mock the async method insert (or add if that's the actual method used)
    # Based on the code change you made, it seems LightRAG uses 'insert'
    instance.insert = AsyncMock()
    instance.add = AsyncMock() # Keep add mock just in case, though insert seems correct now
    return instance

@pytest.fixture
def document_manager(mock_rag_manager): # Depends on the mocked manager
    """Fixture for DocumentManager initialized with a mocked RAGManager."""
    return DocumentManager(rag_manager=mock_rag_manager)

async def test_process_document_success(document_manager, mock_rag_manager, mock_lightrag_instance, tmp_path):
    """Test successful processing of a document."""
    # Arrange
    kb_name = "test_kb"
    doc_content = "This is the test document content."
    doc_path = tmp_path / "test_doc.txt"
    doc_path.write_text(doc_content, encoding='utf-8')
    filename = doc_path.name

    # Configure mocks
    mock_rag_manager.get_rag_instance.return_value = mock_lightrag_instance

    # Patch textract.process within the test scope
    with patch("knowledge_mcp.document_manager.textract.process", return_value=doc_content.encode('utf-8')) as mock_textract:
        # Act
        await document_manager.add(doc_path, kb_name)

        # Assert
        mock_rag_manager.get_rag_instance.assert_awaited_once_with(kb_name)
        mock_textract.assert_called_once_with(str(doc_path)) # textract expects string path
        # Assert insert was called, matching the updated code
        mock_lightrag_instance.insert.assert_awaited_once_with(
            input=doc_content, # Expect string
            ids=filename,      # Expect string
            file_paths=filename
        )
        mock_lightrag_instance.add.assert_not_awaited() # Ensure old 'add' wasn't called

async def test_process_document_file_not_found(document_manager):
    """Test processing when the document file does not exist."""
    # Arrange
    kb_name = "test_kb"
    non_existent_path = Path("/path/to/non/existent/file.txt")

    # Act & Assert
    with pytest.raises(FileNotFoundError):
        await document_manager.add(non_existent_path, kb_name)

async def test_process_document_text_extraction_error(document_manager, mock_rag_manager, tmp_path):
    """Test handling of errors during text extraction."""
    # Arrange
    kb_name = "test_kb"
    doc_path = tmp_path / "test_doc.pdf" # Use a different extension for clarity
    doc_path.touch() # Create an empty file

    # Configure mock RAG manager (needed even if extraction fails first)
    mock_rag_manager.get_rag_instance.return_value = MagicMock() # Return a dummy mock

    # Patch textract.process to raise an exception
    extraction_exception = Exception("textract failed")
    with patch("knowledge_mcp.document_manager.textract.process", side_effect=extraction_exception) as mock_textract:

        # Act & Assert
        with pytest.raises(TextExtractionError) as excinfo:
            await document_manager.add(doc_path, kb_name)

        assert "Failed to extract text" in str(excinfo.value)
        assert excinfo.value.__cause__ is extraction_exception # Check the cause
        mock_textract.assert_called_once_with(str(doc_path))
        mock_rag_manager.get_rag_instance.assert_awaited_once_with(kb_name) # RAG instance is fetched before extraction


async def test_process_document_rag_init_error(document_manager, mock_rag_manager, tmp_path):
    """Test handling of errors during RAG instance initialization."""
    # Arrange
    kb_name = "test_kb"
    doc_path = tmp_path / "test_doc.txt"
    doc_path.write_text("content")

    # Configure mock RAG manager to raise error
    init_exception = RAGInitializationError("Failed to connect to vector DB")
    mock_rag_manager.get_rag_instance.side_effect = init_exception

    # Act & Assert
    # Expect the wrapped error, not the original one
    with pytest.raises(DocumentManagerError) as excinfo:
        await document_manager.add(doc_path, kb_name)

    # Optionally check the error message or the wrapped cause
    assert kb_name in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, RAGInitializationError)

    mock_rag_manager.get_rag_instance.assert_awaited_once_with(kb_name)


async def test_process_document_ingestion_error(document_manager, mock_rag_manager, mock_lightrag_instance, tmp_path):
    """Test handling of errors during document ingestion (rag.insert)."""
    # Arrange
    kb_name = "test_kb"
    doc_content = "Test content"
    doc_path = tmp_path / "test_doc.txt"
    doc_path.write_text(doc_content)

    # Configure mocks
    mock_rag_manager.get_rag_instance.return_value = mock_lightrag_instance
    ingestion_exception = Exception("Vector DB write failed")
    mock_lightrag_instance.insert.side_effect = ingestion_exception # Set side effect on insert

    with patch("knowledge_mcp.document_manager.textract.process", return_value=doc_content.encode('utf-8')) as mock_textract:
        # Act & Assert
        with pytest.raises(DocumentManagerError) as excinfo:
            await document_manager.add(doc_path, kb_name)

        assert "Failed to ingest document" in str(excinfo.value)
        assert excinfo.value.__cause__ is ingestion_exception
        mock_rag_manager.get_rag_instance.assert_awaited_once_with(kb_name)
        mock_textract.assert_called_once_with(str(doc_path))
        mock_lightrag_instance.insert.assert_awaited_once() # Check insert was called
        mock_lightrag_instance.add.assert_not_awaited() # Ensure old 'add' wasn't called

async def test_process_document_empty_content(document_manager, mock_rag_manager, mock_lightrag_instance, tmp_path):
    """Test that ingestion is skipped if extracted content is empty or whitespace only."""
    # Arrange
    kb_name = "test_kb"
    doc_content_empty = "   \n  \t  "
    doc_path = tmp_path / "empty_doc.txt"
    doc_path.write_text(doc_content_empty)

    # Configure mocks
    mock_rag_manager.get_rag_instance.return_value = mock_lightrag_instance

    with patch("knowledge_mcp.document_manager.textract.process", return_value=doc_content_empty.encode('utf-8')) as mock_textract:
        # Act
        # Should complete without raising ingestion error
        await document_manager.add(doc_path, kb_name)

        # Assert
        mock_rag_manager.get_rag_instance.assert_awaited_once_with(kb_name)
        mock_textract.assert_called_once_with(str(doc_path))
        mock_lightrag_instance.insert.assert_not_awaited() # Ensure insert was NOT called
        mock_lightrag_instance.add.assert_not_awaited() # Ensure old 'add' was NOT called
