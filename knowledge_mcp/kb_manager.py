"""Manages knowledge bases, including creation, loading, and querying."""

import logging
import shutil
from pathlib import Path

from knowledge_mcp.config import settings

logger = logging.getLogger(__name__)


class KnowledgeBaseError(Exception):
    """Base exception for knowledge base operations."""


class KnowledgeBaseExistsError(KnowledgeBaseError):
    """Raised when trying to create a knowledge base that already exists."""


class KnowledgeBaseNotFoundError(KnowledgeBaseError):
    """Raised when trying to operate on a knowledge base that does not exist."""


class KnowledgeBaseManager:
    """Handles operations related to knowledge bases."""

    def __init__(self, base_dir: Path | None = None) -> None:
        """Initializes the KnowledgeBaseManager.

        Args:
            base_dir: The base directory for storing knowledge bases.
                      Defaults to the path specified in settings.
        """
        self.base_dir = base_dir or settings.knowledge_base_dir
        if not self.base_dir.exists():
            logger.info(f"Creating knowledge base directory: {self.base_dir}")
            self.base_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"KnowledgeBaseManager initialized with base directory: {self.base_dir}")

    def create_kb(self, name: str) -> Path:
        """Creates a new knowledge base directory.

        Args:
            name: The name of the knowledge base to create.

        Returns:
            The path to the newly created knowledge base directory.

        Raises:
            KnowledgeBaseExistsError: If a knowledge base with the given name already exists.
            OSError: If there is an issue creating the directory.
        """
        kb_path = self.base_dir / name
        if kb_path.exists():
            msg = f"Knowledge base '{name}' already exists at {kb_path}"
            logger.error(msg)
            raise KnowledgeBaseExistsError(msg)

        try:
            logger.info(f"Creating knowledge base '{name}' at {kb_path}")
            kb_path.mkdir(parents=False, exist_ok=False)  # Don't create parents, fail if exists
            # Future: Initialize internal structure/files if needed (e.g., index config)
            logger.info(f"Successfully created knowledge base '{name}' at {kb_path}")
            return kb_path
        except OSError as e:
            logger.exception(f"Failed to create knowledge base directory '{kb_path}': {e}")
            raise  # Re-raise the exception after logging

    def list_kbs(self) -> list[str]:
        """Lists available knowledge bases."""
        if not self.base_dir.exists():
            return []
        # Only list directories, ignore files like .DS_Store
        return [d.name for d in self.base_dir.iterdir() if d.is_dir()]

    def delete_kb(self, name: str) -> None:
        """Deletes an existing knowledge base directory and all its contents.

        Args:
            name: The name of the knowledge base to delete.

        Raises:
            KnowledgeBaseNotFoundError: If the knowledge base directory does not exist.
            OSError: If there is an issue deleting the directory.
        """
        kb_path = self.base_dir / name
        if not kb_path.is_dir():  # Check if it's a directory specifically
            msg = f"Knowledge base '{name}' not found or is not a directory at {kb_path}"
            logger.error(msg)
            raise KnowledgeBaseNotFoundError(msg)

        try:
            logger.warning(f"Attempting to delete knowledge base '{name}' at {kb_path}")
            shutil.rmtree(kb_path)
            logger.info(f"Successfully deleted knowledge base '{name}' at {kb_path}")
        except OSError as e:
            logger.exception(f"Failed to delete knowledge base directory '{kb_path}': {e}")
            raise  # Re-raise the exception after logging

    # Add load_kb, query_kb etc. later
