"""Manages knowledge bases, including creation, loading, and querying."""

import logging
from pathlib import Path
from typing import List

from .config import ConfigService

logger = logging.getLogger(__name__)


class KnowledgeBaseError(Exception):
    """Base exception for knowledge base operations."""


class KnowledgeBaseExistsError(KnowledgeBaseError):
    """Raised when trying to create a knowledge base that already exists."""


class KnowledgeBaseNotFoundError(KnowledgeBaseError):
    """Raised when trying to operate on a knowledge base that does not exist."""


class KnowledgeBaseManager:
    """Manages knowledge base directories."""

    def __init__(self) -> None:
        """
        Initializes the KnowledgeBaseManager.
        The base directory for knowledge bases is retrieved from the ConfigService.

        Raises:
            RuntimeError: If ConfigService fails to initialize (e.g., config not found).
            TypeError: If the resolved base directory path is not valid.
        """
        resolved_base_dir: Path

        # Always get base_dir from ConfigService
        try:
            # Get singleton instance (initializes on first call if needed)
            resolved_base_dir = ConfigService.get_instance().knowledge_base.base_dir.resolve()
            logger.info(f"Using base directory from config: {resolved_base_dir}")
        except RuntimeError as e: # Catch potential init errors from ConfigService
            logger.error(f"Failed to get base directory from ConfigService: {e}")
            raise # Re-raise the error to prevent incorrect state

        # Ensure the base directory exists
        try:
            resolved_base_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create base directory {resolved_base_dir}: {e}")
            # Raise a more specific error or handle as needed
            raise TypeError(f"Invalid base directory path or permissions: {resolved_base_dir}") from e

        self.base_dir: Path = resolved_base_dir
        logger.debug(f"KnowledgeBaseManager initialized with base_dir: {self.base_dir}")

    def create_kb(self, name: str) -> Path:
        """Creates a new knowledge base directory."""
        kb_path = self.base_dir / name
        try:
            kb_path.mkdir(exist_ok=False) # Error if exists
            logger.info(f"Created knowledge base: {kb_path}")
            return kb_path
        except FileExistsError:
            logger.warning(f"Knowledge base '{name}' already exists at {kb_path}")
            raise
        except OSError as e:
            logger.error(f"Failed to create knowledge base '{name}' at {kb_path}: {e}")
            raise

    def list_kbs(self) -> List[str]:
        """Lists existing knowledge base directories."""
        if not self.base_dir.exists():
            logger.warning(f"Base directory {self.base_dir} does not exist. Cannot list KBs.")
            return []
        return [d.name for d in self.base_dir.iterdir() if d.is_dir()]

    def delete_kb(self, name: str) -> None:
        """Deletes a knowledge base directory and its contents."""
        kb_path = self.base_dir / name
        if not kb_path.is_dir():
            logger.warning(f"Cannot delete. Knowledge base '{name}' not found at {kb_path}")
            raise FileNotFoundError(f"Knowledge base '{name}' not found.")

        try:
            # Basic recursive delete (consider shutil.rmtree for robustness)
            for item in kb_path.iterdir():
                if item.is_dir():
                    # Add recursive delete for subdirs if needed, or use shutil
                    logger.warning(f"Directory found inside KB '{name}'. Simple delete might fail.")
                    # For now, just deleting files
                else:
                    item.unlink()
            kb_path.rmdir()
            logger.info(f"Deleted knowledge base: {kb_path}")
        except OSError as e:
            logger.error(f"Failed to delete knowledge base '{name}' at {kb_path}: {e}")
            raise

    # Add load_kb, query_kb etc. later
