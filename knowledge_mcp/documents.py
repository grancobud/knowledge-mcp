"""Handles document loading, processing, and ingestion into knowledge bases."""
import logging
import asyncio
from pathlib import Path
import textract 
from knowledge_mcp.rag import RagManager 

logger = logging.getLogger(__name__)

# Define supported extensions explicitly if needed, otherwise rely on textract
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".rtf", ".xlsx", ".pptx", ".html", ".htm", ".eml", ".msg"}

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
        logger.info("DocumentManager initialized.")

    def _extract_text(self, doc_path: Path) -> str:
        """Extracts text content from a document using textract.

        Args:
            doc_path: Path to the document file.

        Returns:
            The extracted text content as a string.

        Raises:
            TextExtractionError: If textract fails to process the file.
            UnsupportedFileTypeError: If the file extension is not supported (optional check).
        """
        if doc_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
             # You might choose to rely solely on textract's capabilities
             # raise UnsupportedFileTypeError(f"File type {doc_path.suffix} not explicitly supported.")
             logger.warning(f"File type {doc_path.suffix} not in explicitly supported list, attempting extraction with textract.")

        try:
            logger.debug(f"Extracting text from: {doc_path}")
            # textract handles various file types internally
            # Specify encoding if known issues arise, otherwise default utf-8 is usually fine
            byte_content = textract.process(str(doc_path)) # textract might expect string path
            text_content = byte_content.decode('utf-8', errors='replace') # Decode bytes to string
            logger.debug(f"Successfully extracted text from: {doc_path} (Length: {len(text_content)})" )
            return text_content
        except Exception as e:
            # Catching a broad exception as textract can raise various errors
            msg = f"Failed to extract text from {doc_path}: {e}"
            logger.exception(msg) # Log with stack trace
            raise TextExtractionError(msg) from e

    async def add(self, doc_path: Path, kb_name: str) -> None: 
        """Ingests a document into the specified knowledge base.

        Args:
            doc_path: The path to the document file.
            kb_name: The name of the target knowledge base.

        Raises:
            FileNotFoundError: If the document path does not exist.
            TextExtractionError: If text extraction fails.
            Exception: For errors during RAG instantiation or ingestion.
        """
        logger.info(f"Inserting document: {doc_path} into KB: {kb_name}")

        if not doc_path.is_file(): 
            msg = f"Document not found or is not a file: {doc_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        try:
            rag = await self.rag_manager.get_rag_instance(kb_name)
        except Exception as e:
            msg = f"Failed to get RAG instance for KB '{kb_name}': {e}"
            logger.exception(msg)
            raise DocumentManagerError(msg) from e

        text_content = self._extract_text(doc_path)
        

        if text_content: # Ensure there is content to ingest
            logger.info(f"Ingesting content from {doc_path} into KB '{kb_name}'...")
            try:
                # Assuming LightRAG's add method takes content and optional metadata/ID
                # We might need to chunk the text_content first if LightRAG doesn't handle it.
                # For now, pass the full text. We can refine chunking later if needed.
                # We should also consider a unique ID for the document. Using the path for now.
                if not text_content.strip():
                    logger.warning(
                        f"Skipping ingestion for {doc_path.name}: Extracted content is empty or whitespace only."
                    )
                    return # Skip ingestion for empty content
                await asyncio.to_thread(
                    rag.insert,
                    input=text_content,
                    ids=[doc_path.name], 
                    file_paths=[doc_path.name]
                )
                logger.info(f"Successfully ingested content from {doc_path} into {kb_name}.")
            except Exception as e:
                msg = f"Failed to ingest document {doc_path} into KB '{kb_name}': {e}"
                logger.exception(msg)
                # Decide if this should raise a specific IngestionError or re-raise
                raise DocumentManagerError(msg) from e # Wrap in our custom error
        else:
            logger.warning(f"Skipping ingestion for {doc_path} due to empty extracted content.")

        logger.info(f"Finished processing for document: {doc_path}")

    # Placeholder for other potential helper methods