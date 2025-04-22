"""Unit tests for the KnowledgeBaseManager class."""

import pytest
import shutil
from pathlib import Path

from knowledge_mcp.kb_manager import (
    KnowledgeBaseManager,
    KnowledgeBaseExistsError,
    KnowledgeBaseNotFoundError,
)


@pytest.fixture
def kb_manager(tmp_path: Path) -> KnowledgeBaseManager:
    """Provides a KnowledgeBaseManager instance using a temporary directory."""
    # Create a subdirectory within tmp_path for the base directory
    base_dir = tmp_path / "kbs"
    return KnowledgeBaseManager(base_dir=base_dir)


@pytest.fixture
def existing_kb(kb_manager: KnowledgeBaseManager, tmp_path: Path) -> str:
    """Creates a dummy KB directory for testing deletion/existence checks."""
    kb_name = "existing_kb"
    kb_path = kb_manager.base_dir / kb_name
    kb_path.mkdir()
    # Add a dummy file inside
    (kb_path / "dummy.txt").touch()
    return kb_name

# === Test Initialization ===

def test_kb_manager_init_creates_base_dir(tmp_path: Path):
    """Test that the base directory is created if it doesn't exist."""
    base_dir = tmp_path / "new_kbs"
    assert not base_dir.exists()
    KnowledgeBaseManager(base_dir=base_dir)
    assert base_dir.exists()
    assert base_dir.is_dir()

def test_kb_manager_init_uses_existing_base_dir(kb_manager: KnowledgeBaseManager):
    """Test that an existing base directory is used."""
    assert kb_manager.base_dir.exists()

# === Test Creation ===

def test_create_kb_success(kb_manager: KnowledgeBaseManager):
    """Test successful creation of a knowledge base."""
    kb_name = "test_kb_create"
    kb_path = kb_manager.base_dir / kb_name

    assert not kb_path.exists()
    created_path = kb_manager.create_kb(kb_name)

    assert created_path == kb_path
    assert kb_path.exists()
    assert kb_path.is_dir()

def test_create_kb_already_exists(kb_manager: KnowledgeBaseManager, existing_kb: str):
    """Test creating a KB that already exists raises KnowledgeBaseExistsError."""
    kb_path = kb_manager.base_dir / existing_kb
    assert kb_path.exists()

    with pytest.raises(KnowledgeBaseExistsError) as excinfo:
        kb_manager.create_kb(existing_kb)

    assert existing_kb in str(excinfo.value)
    assert str(kb_path) in str(excinfo.value)

# === Test Listing ===

def test_list_kbs_empty(kb_manager: KnowledgeBaseManager):
    """Test listing KBs when none exist."""
    assert kb_manager.list_kbs() == []

def test_list_kbs_one(kb_manager: KnowledgeBaseManager, existing_kb: str):
    """Test listing KBs when one exists."""
    assert kb_manager.list_kbs() == [existing_kb]

def test_list_kbs_multiple(kb_manager: KnowledgeBaseManager):
    """Test listing KBs when multiple exist."""
    kb_names = ["kb1", "kb2", "kb3"]
    for name in kb_names:
        kb_manager.create_kb(name)

    # Create a dummy file in the base dir - should be ignored
    (kb_manager.base_dir / "dummy_file.txt").touch()

    listed_kbs = sorted(kb_manager.list_kbs())
    assert listed_kbs == sorted(kb_names)

# === Test Deletion ===

def test_delete_kb_success(kb_manager: KnowledgeBaseManager, existing_kb: str):
    """Test successful deletion of an existing knowledge base."""
    kb_path = kb_manager.base_dir / existing_kb
    assert kb_path.exists()

    kb_manager.delete_kb(existing_kb)

    assert not kb_path.exists()
    assert kb_manager.list_kbs() == []

def test_delete_kb_not_found(kb_manager: KnowledgeBaseManager):
    """Test deleting a KB that does not exist raises KnowledgeBaseNotFoundError."""
    kb_name = "non_existent_kb"
    kb_path = kb_manager.base_dir / kb_name
    assert not kb_path.exists()

    with pytest.raises(KnowledgeBaseNotFoundError) as excinfo:
        kb_manager.delete_kb(kb_name)

    assert kb_name in str(excinfo.value)
    assert str(kb_path) in str(excinfo.value)

def test_delete_kb_is_file(kb_manager: KnowledgeBaseManager):
    """Test deleting something that exists but is a file raises KnowledgeBaseNotFoundError."""
    file_name = "not_a_kb.txt"
    file_path = kb_manager.base_dir / file_name
    file_path.touch()
    assert file_path.exists()
    assert file_path.is_file()

    with pytest.raises(KnowledgeBaseNotFoundError) as excinfo:
        kb_manager.delete_kb(file_name)

    assert file_name in str(excinfo.value)
    assert str(file_path) in str(excinfo.value)
    assert file_path.exists() # Ensure the file wasn't deleted
