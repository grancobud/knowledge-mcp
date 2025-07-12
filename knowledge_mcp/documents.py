"""Handles document loading, processing, and ingestion into knowledge bases."""
import logging
from pathlib import Path
from markitdown import MarkItDown
from knowledge_mcp.rag import RagManager 

logger = logging.getLogger(__name__)

# Supported file extensions based on MarkItDown capabilities
SUPPORTED_EXTENSIONS = {
    # Office documents
    ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls",
    # PDF files
    ".pdf",
    # Text files
    ".txt", ".md", ".markdown", ".rst",
    # Web files
    ".html", ".htm", ".xml",
    # Email files
    ".eml", ".msg",
    # Audio files (for transcription)
    ".wav", ".mp3",
    # Image files (with LLM support)
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif",
    # Other common formats
    ".rtf", ".csv", ".tsv", ".json"
}



class DocumentManagerError(Exception):
    """Base exception for document management errors."""

class TextExtractionError(DocumentManagerError):
    """Raised when text extraction fails."""

class UnsupportedFileTypeError(DocumentManagerError):
    """Raised when the document file type is not supported."""

class DocumentProcessingError(Exception):
    """Custom exception for errors during document processing."""

class DocumentManager:
    """Processes and ingests documents into a specified knowledge base."""

    def __init__(self, rag_manager: RagManager): 
        """Initializes the DocumentManager."""
        self.rag_manager = rag_manager
        self.markitdown = MarkItDown()
        logger.info("DocumentManager initialized with MarkItDown.")

    def _extract_text(self, doc_path: Path) -> str:
        """Extracts text content from a document using MarkItDown.

        Args:
            doc_path: Path to the document file.

        Returns:
            The extracted text content as a string.

        Raises:
            TextExtractionError: If MarkItDown fails to process the file.
            UnsupportedFileTypeError: If the file extension is not supported.
        """
        # Check if file extension is supported
        file_ext = doc_path.suffix.lower()
        if file_ext not in SUPPORTED_EXTENSIONS:
            logger.warning(f"File type {file_ext} not in explicitly supported list, attempting extraction with MarkItDown anyway.")
        else:
            logger.info(f"Processing supported file type: {file_ext}")
        
        # Log file information
        try:
            file_size = doc_path.stat().st_size
            logger.info(f"Starting MarkItDown extraction for: {doc_path.name} (Size: {file_size:,} bytes, Type: {file_ext})")
        except OSError:
            logger.info(f"Starting MarkItDown extraction for: {doc_path.name} (Type: {file_ext})")
        
        try:
            logger.debug(f"Calling MarkItDown.convert() for: {doc_path}")
            result = self.markitdown.convert(str(doc_path))
            
            # Extract and validate content
            text_content = result.text_content
            content_length = len(text_content) if text_content else 0
            
            if content_length == 0:
                logger.warning(f"MarkItDown extraction returned empty content for: {doc_path.name}")
            else:
                logger.info(f"MarkItDown extraction successful for: {doc_path.name} (Extracted {content_length:,} characters)")
                logger.debug(f"First 100 characters of extracted content: {text_content[:100]!r}...")
            
            return text_content
        except FileNotFoundError as e:
            msg = f"Document file not found: {doc_path}"
            logger.error(msg)
            raise TextExtractionError(msg) from e
        except PermissionError as e:
            msg = f"Permission denied accessing file: {doc_path}"
            logger.error(msg)
            raise TextExtractionError(msg) from e
        except (ValueError, TypeError) as e:
            # Handle invalid file format or corrupted files
            msg = f"Invalid or corrupted file format for {doc_path}: {e}"
            logger.error(msg)
            raise UnsupportedFileTypeError(msg) from e
        except ImportError as e:
            # Handle missing optional dependencies for specific file types
            msg = f"Missing required dependency for processing {doc_path}: {e}. Try installing with 'pip install markitdown[all]'"
            logger.error(msg)
            raise TextExtractionError(msg) from e
        except Exception as e:
            # Catch any other MarkItDown-specific or unexpected errors
            error_type = type(e).__name__
            msg = f"Failed to extract text from {doc_path} using MarkItDown ({error_type}): {e}"
            logger.exception(msg)
            raise TextExtractionError(msg) from e


    async def add_multimodal(self, doc_path: Path, kb_name: str) -> None: 
        """Ingests a document into the specified knowledge base using multimodal processing.
        
        This method uses RAGAnything for full multimodal document processing,
        which can handle images, complex layouts, and other multimodal content.

        Args:
            doc_path: The path to the document file.
            kb_name: The name of the target knowledge base.

        Raises:
            FileNotFoundError: If the document path does not exist.
            DocumentManagerError: If RAG instance creation or ingestion fails.
        """
        logger.info(f"Ingesting document: {doc_path} into KB: {kb_name} using multimodal processing")

        if not doc_path.is_file(): 
            msg = f"Document not found or is not a file: {doc_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        try:
            # Use RagManager's ingest_document method for multimodal processing
            logger.info(f"Starting multimodal ingestion for {doc_path.name}...")
            doc_id = await self.rag_manager.ingest_document(
                kb_name=kb_name,
                file_path=doc_path,
                doc_id=doc_path.name,
                parse_method="multimodal"
            )
            logger.info(f"Successfully completed multimodal ingestion of {doc_path.name} as '{doc_id}'")
            
        except Exception as e:
            msg = f"Failed to ingest document {doc_path} into KB '{kb_name}' using multimodal processing: {e}"
            logger.exception(msg)
            raise DocumentManagerError(msg) from e

    async def add_text_only(self, doc_path: Path, kb_name: str) -> None:
        """Ingests a document into the specified knowledge base using text-only processing.
        
        This method uses MarkItDown for text extraction and LightRAG directly for ingestion,
        bypassing multimodal processing. This is more efficient for text-only documents.

        Args:
            doc_path: The path to the document file.
            kb_name: The name of the target knowledge base.

        Raises:
            FileNotFoundError: If the document path does not exist.
            TextExtractionError: If MarkItDown text extraction fails.
            UnsupportedFileTypeError: If the file type is not supported.
            DocumentManagerError: If ingestion fails.
        """
        logger.info(f"Ingesting document: {doc_path} into KB: {kb_name} using text-only processing")

        if not doc_path.is_file(): 
            msg = f"Document not found or is not a file: {doc_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        # Extract text content using MarkItDown
        logger.info(f"Extracting text content from {doc_path.name} using MarkItDown...")
        try:
            text_content = self._extract_text(doc_path)
        except (TextExtractionError, UnsupportedFileTypeError) as e:
            # Log the specific error from _extract_text and re-raise
            logger.error(f"MarkItDown extraction failed for {doc_path.name}: {e}")
            raise  # Re-raise the caught exception
        except Exception as e:
            # Catch any other unexpected errors during extraction
            msg = f"Unexpected error during MarkItDown text extraction for {doc_path.name}: {e}"
            logger.exception(msg)
            raise DocumentProcessingError(msg) from e

        # Validate extracted content
        if not text_content or not text_content.strip():
            logger.warning(f"Skipping ingestion for {doc_path.name}: Extracted content is empty or whitespace only.")
            return  # Skip ingestion for empty content

        # Ingest using text-only method
        try:
            logger.info(f"Starting text-only ingestion for {doc_path.name}...")
            doc_id = await self.rag_manager.ingest_text_only(
                kb_name=kb_name,
                text_content=text_content,
                doc_id=doc_path.name  # Use filename as document ID
            )
            logger.info(f"Successfully completed text-only ingestion of {doc_path.name} as '{doc_id}'")
            
        except Exception as e:
            msg = f"Failed to ingest document {doc_path} into KB '{kb_name}' using text-only processing: {e}"
            logger.exception(msg)
            raise DocumentManagerError(msg) from e

    async def add(self, doc_path: Path, kb_name: str, method: str = "multimodal") -> None:
        """Generic document ingestion method that routes to appropriate processing method.
        
        This method provides a unified interface for document ingestion while allowing
        selection of the processing method. Maintains backward compatibility by defaulting
        to multimodal processing.

        Args:
            doc_path: The path to the document file.
            kb_name: The name of the target knowledge base.
            method: Processing method to use. Options:
                - "multimodal": Use RAGAnything for full multimodal processing (default)
                - "text": Use MarkItDown + LightRAG for text-only processing

        Raises:
            ValueError: If an unsupported method is specified.
            FileNotFoundError: If the document path does not exist.
            TextExtractionError: If text extraction fails (text-only method).
            UnsupportedFileTypeError: If the file type is not supported (text-only method).
            DocumentManagerError: If ingestion fails.
        """
        logger.info(f"Ingesting document: {doc_path} into KB: {kb_name} using '{method}' method")
        
        if method == "multimodal":
            await self.add_multimodal(doc_path, kb_name)
        elif method == "text":
            await self.add_text_only(doc_path, kb_name)
        else:
            msg = f"Unsupported processing method: '{method}'. Use 'multimodal' or 'text'."
            logger.error(msg)
            raise ValueError(msg)

    # Placeholder for other potential helper methods