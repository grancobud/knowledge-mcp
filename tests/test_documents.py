import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock, call

from knowledge_mcp.documents import (
    DocumentManager,
    DocumentManagerError,
    TextExtractionError,
    UnsupportedFileTypeError, # Although not explicitly raised in current code, good to be aware
    DocumentProcessingError,
    SUPPORTED_EXTENSIONS, # For checking if textract is called
    TEXT_EXTENSIONS
)
from knowledge_mcp.rag import RagManager, RAGInitializationError # For error simulation
from knowledge_mcp.knowledgebases import KnowledgeBaseNotFoundError # For error simulation
# Assuming LightRAG has an insert method, and we'll mock it.
# from lightrag.core import LightRAG # Not strictly needed if fully mocked

# --- Fixtures ---

@pytest.fixture
def mock_rag_manager():
    """Provides a MagicMock for RagManager."""
    return MagicMock(spec=RagManager)

@pytest.fixture
def mock_lightrag_instance():
    """Provides a MagicMock for a LightRAG instance."""
    # insert is called via to_thread, so it's a sync mock
    mock_rag = MagicMock() 
    mock_rag.insert = MagicMock()
    return mock_rag

@pytest.fixture
def document_manager(mock_rag_manager):
    """Provides an initialized DocumentManager instance."""
    return DocumentManager(mock_rag_manager)

# --- Helper Function to Create Temp Files ---
def create_temp_doc(tmp_path: Path, filename: str, content: str | bytes = "test content", encoding: str | None = "utf-8"):
    doc_path = tmp_path / filename
    if isinstance(content, str) and encoding:
        doc_path.write_text(content, encoding=encoding)
    elif isinstance(content, bytes):
        doc_path.write_bytes(content)
    else: # fallback for str without encoding, or other types
        doc_path.write_text(str(content))
    return doc_path

# --- Tests for DocumentManager ---

# 1. Initialization
def test_dm_init_success(mock_rag_manager):
    """Test successful initialization with RagManager."""
    dm = DocumentManager(mock_rag_manager)
    assert dm.rag_manager is mock_rag_manager

# 2. Text Extraction (_extract_text via add) - primarily for non-TEXT_EXTENSIONS
@pytest.mark.asyncio
@patch('knowledge_mcp.documents.textract.process')
async def test_dm_extract_text_pdf_success(mock_textract_process, document_manager, mock_rag_manager, mock_lightrag_instance, tmp_path, caplog):
    """Test text extraction from PDF using mocked textract.process."""
    doc_path = create_temp_doc(tmp_path, "test.pdf", b"") # Content not important, textract is mocked
    mock_textract_process.return_value = b"extracted pdf text"
    mock_rag_manager.get_rag_instance = AsyncMock(return_value=mock_lightrag_instance)

    await document_manager.add(doc_path, "test_kb")

    mock_textract_process.assert_called_once_with(str(doc_path))
    mock_lightrag_instance.insert.assert_called_once()
    # Check if "extracted pdf text" is in the input for insert
    args, _ = mock_lightrag_instance.insert.call_args
    assert args[0] == "extracted pdf text"
    assert "File type .pdf not in explicitly supported list, attempting extraction with textract." not in caplog.text # .pdf is in SUPPORTED_EXTENSIONS

@pytest.mark.asyncio
@patch('knowledge_mcp.documents.textract.process')
async def test_dm_extract_text_docx_success(mock_textract_process, document_manager, mock_rag_manager, mock_lightrag_instance, tmp_path, caplog):
    """Test text extraction from DOCX using mocked textract.process."""
    doc_path = create_temp_doc(tmp_path, "test.docx", b"")
    mock_textract_process.return_value = b"extracted docx text"
    mock_rag_manager.get_rag_instance = AsyncMock(return_value=mock_lightrag_instance)

    await document_manager.add(doc_path, "test_kb")

    mock_textract_process.assert_called_once_with(str(doc_path))
    mock_lightrag_instance.insert.assert_called_once_with(input="extracted docx text", ids=[doc_path.name], file_paths=[doc_path.name])
    assert "File type .docx not in explicitly supported list, attempting extraction with textract." not in caplog.text # .docx is in SUPPORTED_EXTENSIONS


@pytest.mark.asyncio
@patch('knowledge_mcp.documents.textract.process', side_effect=Exception("textract failed!"))
async def test_dm_extract_text_textract_fails(mock_textract_process, document_manager, mock_rag_manager, tmp_path):
    """Test TextExtractionError if textract.process fails."""
    doc_path = create_temp_doc(tmp_path, "test.rtf", b"") # .rtf is in SUPPORTED_EXTENSIONS
    mock_rag_manager.get_rag_instance = AsyncMock() # To avoid error before extraction attempt

    with pytest.raises(TextExtractionError, match="Failed to extract text from .*test.rtf: textract failed!"):
        await document_manager.add(doc_path, "test_kb")
    mock_textract_process.assert_called_once_with(str(doc_path))

@pytest.mark.asyncio
@patch('knowledge_mcp.documents.textract.process')
async def test_dm_extract_text_unsupported_extension_warning(mock_textract_process, document_manager, mock_rag_manager, mock_lightrag_instance, tmp_path, caplog):
    """Test warning for file types not in SUPPORTED_EXTENSIONS but attempted by textract."""
    doc_path = create_temp_doc(tmp_path, "test.unknownext", b"")
    mock_textract_process.return_value = b"extracted unknown data"
    mock_rag_manager.get_rag_instance = AsyncMock(return_value=mock_lightrag_instance)

    await document_manager.add(doc_path, "test_kb")
    
    mock_textract_process.assert_called_once_with(str(doc_path))
    assert f"File type {doc_path.suffix} not in explicitly supported list, attempting extraction with textract." in caplog.text
    mock_lightrag_instance.insert.assert_called_once_with(input="extracted unknown data", ids=[doc_path.name], file_paths=[doc_path.name])

# 3. Direct Text Reading (for TEXT_EXTENSIONS)
@pytest.mark.asyncio
@patch('knowledge_mcp.documents.textract.process') # To ensure it's NOT called
async def test_dm_direct_read_txt_utf8(mock_textract_process, document_manager, mock_rag_manager, mock_lightrag_instance, tmp_path):
    """Test direct UTF-8 read for .txt files."""
    content = "Hello, this is a UTF-8 text file. 😊"
    doc_path = create_temp_doc(tmp_path, "test.txt", content, encoding="utf-8")
    mock_rag_manager.get_rag_instance = AsyncMock(return_value=mock_lightrag_instance)

    await document_manager.add(doc_path, "test_kb")

    mock_textract_process.assert_not_called()
    mock_lightrag_instance.insert.assert_called_once_with(input=content, ids=[doc_path.name], file_paths=[doc_path.name])

@pytest.mark.asyncio
@patch('knowledge_mcp.documents.textract.process')
async def test_dm_direct_read_py_md_etc(mock_textract_process, document_manager, mock_rag_manager, mock_lightrag_instance, tmp_path):
    """Test direct read for other TEXT_EXTENSIONS like .py, .md."""
    files_content = {
        "script.py": "print('Hello Python!')",
        "notes.md": "# Markdown Notes\n- Point 1",
        "config.yaml": "key: value"
    }
    mock_rag_manager.get_rag_instance = AsyncMock(return_value=mock_lightrag_instance)

    for filename, content in files_content.items():
        doc_path = create_temp_doc(tmp_path, filename, content, encoding="utf-8")
        mock_lightrag_instance.reset_mock() # Reset for each iteration

        await document_manager.add(doc_path, "test_kb")
        
        mock_lightrag_instance.insert.assert_called_once_with(input=content, ids=[doc_path.name], file_paths=[doc_path.name])
    
    mock_textract_process.assert_not_called() # textract should never be called for these

@pytest.mark.asyncio
@patch('knowledge_mcp.documents.textract.process')
async def test_dm_direct_read_fallback_to_latin1(mock_textract_process, document_manager, mock_rag_manager, mock_lightrag_instance, tmp_path, caplog):
    """Test fallback to 'latin-1' if UTF-8 decoding fails."""
    # Content that is valid in latin-1 but not in utf-8 (e.g., 0xA9 is © in latin-1)
    latin1_content_bytes = b"This is latin-1 text with a copyright symbol: \xa9"
    latin1_content_str = latin1_content_bytes.decode('latin-1')
    
    doc_path = create_temp_doc(tmp_path, "test_latin1.txt", latin1_content_bytes) # Write as bytes
    mock_rag_manager.get_rag_instance = AsyncMock(return_value=mock_lightrag_instance)

    await document_manager.add(doc_path, "test_kb")

    assert f"UTF-8 decoding failed for {doc_path}. Trying latin-1." in caplog.text
    mock_textract_process.assert_not_called()
    mock_lightrag_instance.insert.assert_called_once_with(input=latin1_content_str, ids=[doc_path.name], file_paths=[doc_path.name])

@pytest.mark.asyncio
@patch('knowledge_mcp.documents.textract.process')
async def test_dm_direct_read_fails_even_with_latin1(mock_textract_process, document_manager, mock_rag_manager, tmp_path, caplog):
    """Test DocumentProcessingError if both UTF-8 and latin-1 decoding fail."""
    # Use bytes that are invalid for both utf-8 and latin-1 (e.g., some control chars or undefined sequences)
    # For instance, 0x81 is undefined in Latin-1 and invalid as a standalone byte in UTF-8.
    invalid_bytes = b"Invalid sequence: \x81"
    doc_path = create_temp_doc(tmp_path, "test_invalid_encoding.txt", invalid_bytes)
    mock_rag_manager.get_rag_instance = AsyncMock() # To avoid error before read attempt

    with pytest.raises(DocumentProcessingError, match=f"Failed to read text file {doc_path} with latin-1"):
        await document_manager.add(doc_path, "test_kb")

    assert f"UTF-8 decoding failed for {doc_path}. Trying latin-1." in caplog.text
    mock_textract_process.assert_not_called()

# 4. Document Addition (add - async) - Comprehensive Scenarios
@pytest.mark.asyncio
async def test_dm_add_file_not_found(document_manager, tmp_path):
    """Test FileNotFoundError if doc_path does not exist."""
    non_existent_path = tmp_path / "ghost.txt"
    with pytest.raises(FileNotFoundError, match=f"Document not found or is not a file: {non_existent_path}"):
        await document_manager.add(non_existent_path, "test_kb")

@pytest.mark.asyncio
async def test_dm_add_rag_instance_fails_kb_not_found(document_manager, mock_rag_manager, tmp_path):
    """Test DocumentManagerError if get_rag_instance raises KnowledgeBaseNotFoundError."""
    doc_path = create_temp_doc(tmp_path, "test.txt")
    kb_name = "non_existent_kb"
    mock_rag_manager.get_rag_instance.side_effect = KnowledgeBaseNotFoundError(f"KB {kb_name} not found.")

    with pytest.raises(DocumentManagerError, match=f"Failed to get RAG instance for KB '{kb_name}': KB {kb_name} not found."):
        await document_manager.add(doc_path, kb_name)

@pytest.mark.asyncio
async def test_dm_add_rag_instance_fails_rag_init_error(document_manager, mock_rag_manager, tmp_path):
    """Test DocumentManagerError if get_rag_instance raises RAGInitializationError."""
    doc_path = create_temp_doc(tmp_path, "test.txt")
    kb_name = "fail_kb"
    mock_rag_manager.get_rag_instance.side_effect = RAGInitializationError("RAG init failed.")

    with pytest.raises(DocumentManagerError, match=f"Failed to get RAG instance for KB '{kb_name}': RAG init failed."):
        await document_manager.add(doc_path, kb_name)

@pytest.mark.asyncio
@patch('knowledge_mcp.documents.textract.process', side_effect=TextExtractionError("Extraction failed during add"))
async def test_dm_add_extraction_fails_propagates(mock_textract, document_manager, mock_rag_manager, tmp_path):
    """Test that TextExtractionError from _extract_text propagates during add."""
    doc_path = create_temp_doc(tmp_path, "test.pdf", b"") # PDF uses textract
    mock_rag_manager.get_rag_instance = AsyncMock() 

    with pytest.raises(TextExtractionError, match="Extraction failed during add"):
        await document_manager.add(doc_path, "test_kb")

@pytest.mark.asyncio
async def test_dm_add_ingestion_fails(document_manager, mock_rag_manager, mock_lightrag_instance, tmp_path):
    """Test DocumentManagerError if rag.insert fails."""
    doc_path = create_temp_doc(tmp_path, "test.txt", "some content")
    kb_name = "test_kb"
    mock_rag_manager.get_rag_instance = AsyncMock(return_value=mock_lightrag_instance)
    mock_lightrag_instance.insert.side_effect = Exception("DB write error")

    with pytest.raises(DocumentManagerError, match=f"Failed to ingest document {doc_path} into KB '{kb_name}': DB write error"):
        await document_manager.add(doc_path, kb_name)

@pytest.mark.asyncio
@patch('knowledge_mcp.documents.textract.process')
async def test_dm_add_empty_content_from_textract(mock_textract, document_manager, mock_rag_manager, mock_lightrag_instance, tmp_path, caplog):
    """Test rag.insert is not called if textract returns empty or whitespace content."""
    doc_path = create_temp_doc(tmp_path, "test.pdf", b"")
    mock_rag_manager.get_rag_instance = AsyncMock(return_value=mock_lightrag_instance)
    
    # Scenario 1: Empty content
    mock_textract.return_value = b""
    await document_manager.add(doc_path, "test_kb")
    assert f"Skipping ingestion for {doc_path} due to empty extracted content." in caplog.text
    mock_lightrag_instance.insert.assert_not_called()
    
    caplog.clear()
    mock_lightrag_instance.reset_mock()

    # Scenario 2: Whitespace content
    mock_textract.return_value = b"   \n\t   "
    await document_manager.add(doc_path, "test_kb")
    assert f"Skipping ingestion for {doc_path.name}: Extracted content is empty or whitespace only." in caplog.text
    mock_lightrag_instance.insert.assert_not_called()

@pytest.mark.asyncio
async def test_dm_add_empty_content_from_direct_read(document_manager, mock_rag_manager, mock_lightrag_instance, tmp_path, caplog):
    """Test rag.insert is not called if direct read results in empty or whitespace content."""
    mock_rag_manager.get_rag_instance = AsyncMock(return_value=mock_lightrag_instance)

    # Scenario 1: Empty content
    doc_path_empty = create_temp_doc(tmp_path, "empty.txt", "")
    await document_manager.add(doc_path_empty, "test_kb")
    assert f"Skipping ingestion for {doc_path_empty} due to empty extracted content." in caplog.text
    mock_lightrag_instance.insert.assert_not_called()

    caplog.clear()
    mock_lightrag_instance.reset_mock()

    # Scenario 2: Whitespace content
    doc_path_whitespace = create_temp_doc(tmp_path, "whitespace.txt", "   \n\t   ")
    await document_manager.add(doc_path_whitespace, "test_kb")
    assert f"Skipping ingestion for {doc_path_whitespace.name}: Extracted content is empty or whitespace only." in caplog.text
    mock_lightrag_instance.insert.assert_not_called()

# 5. Error Types and Wrapping (already covered in many above tests, but can add specific ones if needed)
# This is mostly to ensure the hierarchy and specific messages are as expected.

@pytest.mark.asyncio
@patch('knowledge_mcp.documents.textract.process', side_effect=Exception("Generic textract problem"))
async def test_dm_text_extraction_error_wrapping(mock_textract, document_manager, mock_rag_manager, tmp_path):
    """Verify TextExtractionError wraps underlying textract errors."""
    doc_path = create_temp_doc(tmp_path, "test.pdf", b"")
    mock_rag_manager.get_rag_instance = AsyncMock()
    with pytest.raises(TextExtractionError) as excinfo:
        await document_manager.add(doc_path, "test_kb")
    assert isinstance(excinfo.value.__cause__, Exception)
    assert "Generic textract problem" in str(excinfo.value.__cause__)

@pytest.mark.asyncio
async def test_dm_document_manager_error_wrapping_rag_get_instance(document_manager, mock_rag_manager, tmp_path):
    """Verify DocumentManagerError wraps errors from rag_manager.get_rag_instance."""
    doc_path = create_temp_doc(tmp_path, "test.txt")
    original_error = RAGInitializationError("Underlying RAG init problem")
    mock_rag_manager.get_rag_instance.side_effect = original_error
    
    with pytest.raises(DocumentManagerError) as excinfo:
        await document_manager.add(doc_path, "test_kb")
    assert excinfo.value.__cause__ is original_error

@pytest.mark.asyncio
async def test_dm_document_manager_error_wrapping_rag_insert(document_manager, mock_rag_manager, mock_lightrag_instance, tmp_path):
    """Verify DocumentManagerError wraps errors from rag.insert."""
    doc_path = create_temp_doc(tmp_path, "test.txt", "content")
    original_error = Exception("Underlying DB insert problem")
    mock_rag_manager.get_rag_instance = AsyncMock(return_value=mock_lightrag_instance)
    mock_lightrag_instance.insert.side_effect = original_error

    with pytest.raises(DocumentManagerError) as excinfo:
        await document_manager.add(doc_path, "test_kb")
    assert excinfo.value.__cause__ is original_error

@pytest.mark.asyncio
@patch('knowledge_mcp.documents.open', side_effect=IOError("File read permission denied"))
async def test_dm_document_processing_error_on_direct_read_io_error(mock_open, document_manager, mock_rag_manager, tmp_path):
    """Test DocumentProcessingError for direct read IO errors."""
    doc_path = create_temp_doc(tmp_path, "test.txt", "content") # Content doesn't matter, open is mocked
    mock_rag_manager.get_rag_instance = AsyncMock() # Avoid error before read attempt

    with pytest.raises(DocumentProcessingError, match=f"Failed to read text file {doc_path}: File read permission denied"):
        await document_manager.add(doc_path, "test_kb")
    mock_open.assert_any_call(doc_path, "r", encoding="utf-8") # Check that it attempted to open

"""
This test suite covers the DocumentManager class from knowledge_mcp.documents.

Key areas tested:
- Initialization: Correct setup with RagManager.
- Text Extraction (via textract for non-TEXT_EXTENSIONS):
    - Successful extraction for .pdf, .docx (mocked textract).
    - TextExtractionError on textract failure.
    - Warning for unsupported extensions still attempted by textract.
- Direct Text Reading (for TEXT_EXTENSIONS):
    - Successful UTF-8 read for .txt, .py, .md, .yaml.
    - Fallback to 'latin-1' on UTF-8 decode error.
    - DocumentProcessingError on read failure (even with latin-1).
    - Verified textract.process is NOT called for these.
- Document Addition (add method - async):
    - Successful ingestion path (mocked RagManager.get_rag_instance and LightRAG.insert).
    - FileNotFoundError for non-existent documents.
    - DocumentManagerError wrapping for failures in RagManager.get_rag_instance.
    - Propagation of TextExtractionError/DocumentProcessingError from extraction/read steps.
    - DocumentManagerError wrapping for failures in LightRAG.insert.
    - Skipping ingestion (no call to LightRAG.insert) for empty or whitespace-only content.
- Error Types and Wrapping:
    - Ensured specific custom exceptions are raised.
    - Verified wrapping of underlying errors in DocumentManagerError.

Utilized pytest.mark.asyncio, tmp_path, unittest.mock (MagicMock, AsyncMock, patch),
and helper for creating temporary document files.
"""
