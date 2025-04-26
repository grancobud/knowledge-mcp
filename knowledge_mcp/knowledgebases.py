"""Manages knowledge bases, including creation, loading, and querying."""

import logging
from pathlib import Path
from typing import List
import shutil  # Import shutil for rmtree

# from .config import ConfigService # No longer needed directly
from knowledge_mcp.config import Config # Import Config

logger = logging.getLogger(__name__)


class KnowledgeBaseError(Exception):
    """Base exception for knowledge base operations."""


class KnowledgeBaseExistsError(KnowledgeBaseError):
    """Raised when trying to create a knowledge base that already exists."""


class KnowledgeBaseNotFoundError(KnowledgeBaseError):
    """Raised when trying to operate on a knowledge base that does not exist."""


class KnowledgeBaseManager:
    """Manages knowledge base directories."""

    def __init__(self, config: Config) -> None: # Accept Config object
        """
        Initializes the KnowledgeBaseManager.
        The base directory for knowledge bases is retrieved from the Config object.

        Args:
            config: The application config object.

        Raises:
            TypeError: If the resolved base directory path is not valid or accessible.
        """
        if not config.knowledge_base or not config.knowledge_base.base_dir:
            msg = "Knowledge base base_dir not configured in config."
            logger.error(msg)
            raise ValueError(msg)

        resolved_base_dir: Path = config.knowledge_base.base_dir.resolve()
        logger.info(f"Using base directory from config: {resolved_base_dir}")

        # Ensure the base directory exists
        try:
            resolved_base_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create or access base directory {resolved_base_dir}: {e}")
            raise TypeError(f"Invalid base directory path or permissions: {resolved_base_dir}") from e

        self.base_dir: Path = resolved_base_dir
        logger.debug(f"KnowledgeBaseManager initialized with base_dir: {self.base_dir}")
        self.config = config

    def get_kb_path(self, name: str) -> Path:
        """Returns the full path for a given knowledge base name."""
        return self.base_dir / name

    def kb_exists(self, name: str) -> bool:
        """Checks if a knowledge base directory exists."""
        return self.get_kb_path(name).is_dir()

    def create_kb(self, name: str) -> Path:
        """Creates a new knowledge base directory."""
        kb_path = self.get_kb_path(name)
        if self.kb_exists(name):
            logger.warning(f"Attempted to create existing knowledge base: {name}")
            raise KnowledgeBaseExistsError(f"Knowledge base '{name}' already exists at {kb_path}")

        try:
            kb_path.mkdir(exist_ok=False)
            logger.info(f"Created knowledge base directory: {kb_path}")
            # Create subdirectories if needed (e.g., 'docs', 'index')
            # (kb_path / 'docs').mkdir(exist_ok=True)
            # (kb_path / 'index').mkdir(exist_ok=True)
            return kb_path
        except OSError as e:
            logger.error(f"Failed to create knowledge base directory '{name}' at {kb_path}: {e}")
            # Re-raise as a generic KnowledgeBaseError or handle specific OS errors
            raise KnowledgeBaseError(f"OS error creating knowledge base '{name}': {e}") from e

    def list_kbs(self) -> List[str]:
        """Lists existing knowledge base directories."""
        # Base directory existence checked in __init__
        try:
            return [d.name for d in self.base_dir.iterdir() if d.is_dir()]
        except OSError as e:
            logger.error(f"Error listing knowledge bases in {self.base_dir}: {e}")
            # Depending on requirements, could return [] or raise
            raise KnowledgeBaseError(f"Error listing knowledge bases: {e}") from e

    def delete_kb(self, name: str) -> None:
        """Deletes a knowledge base directory and its contents."""
        kb_path = self.get_kb_path(name)
        if not self.kb_exists(name):
            logger.warning(f"Attempted to delete non-existent knowledge base: {name}")
            raise KnowledgeBaseNotFoundError(f"Knowledge base '{name}' not found at {kb_path}")

        try:
            shutil.rmtree(kb_path)
            logger.info(f"Deleted knowledge base directory and contents: {kb_path}")
        except OSError as e:
            logger.error(f"Failed to delete knowledge base '{name}' at {kb_path}: {e}")
            raise KnowledgeBaseError(f"OS error deleting knowledge base '{name}': {e}") from e

    # --- Document Management (Requires RagManager Interaction) ---
    # Placeholder methods - Implementation requires RagManager
    def add_document(self, kb_name: str, doc_path: Path, doc_name: str | None = None):
        # 1. Check if kb exists (using self.kb_exists)
        # 2. Validate doc_path
        # 3. Determine final document name/ID
        # 4. Potentially copy/store the original doc inside kb_path/docs ?
        # 5. Call RagManager to process and index the document for this KB
        logger.info(f"Placeholder: Add document {doc_path} to KB {kb_name} (doc_name: {doc_name})")
        if not self.kb_exists(kb_name):
             raise KnowledgeBaseNotFoundError(f"Knowledge base '{kb_name}' not found.")
        # ... further implementation needed with RagManager ...

    def remove_document(self, kb_name: str, doc_name: str):
        # 1. Check if kb exists
        # 2. Call RagManager to remove the document and its index data for this KB
        # 3. Potentially remove original doc from kb_path/docs ?
        logger.info(f"Placeholder: Remove document {doc_name} from KB {kb_name}")
        if not self.kb_exists(kb_name):
             raise KnowledgeBaseNotFoundError(f"Knowledge base '{kb_name}' not found.")
        # ... further implementation needed with RagManager ...

    # Add query_kb etc. later, likely involving RagManager
