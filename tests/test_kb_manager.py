"""Unit tests for the KnowledgeBaseManager class."""

import pytest
import shutil
from pathlib import Path
import yaml
from knowledge_mcp.kb_manager import KnowledgeBaseManager
from knowledge_mcp.config import ConfigService


# --- Fixtures --- 

# Fixture to reset the singleton state before each test
# Ideally, this would be in tests/conftest.py
@pytest.fixture(autouse=True)
def reset_config_service_singleton():
    """Ensures each test gets a fresh ConfigService state."""
    ConfigService._instance = None
    ConfigService._initialized = False
    ConfigService._config_data = None
    ConfigService._config_path = None
    yield # Run the test
    ConfigService._instance = None
    ConfigService._initialized = False
    ConfigService._config_data = None
    ConfigService._config_path = None

@pytest.fixture
def kb_manager(tmp_path: Path) -> KnowledgeBaseManager:
    """Provides a KnowledgeBaseManager instance.
    Requires a valid config to be loaded, typically by a test function
    or assumes a default config.yaml exists.
    """
    # Manager now relies on ConfigService, which should be set up by the specific test
    # or a default config file.
    # We create a dummy config here *if* no other test has initialized ConfigService,
    # otherwise tests depending *only* on this fixture might fail if run in isolation.
    if not ConfigService._initialized:
        print("\nWARNING (kb_manager fixture): ConfigService not initialized. Creating dummy config.")
        dummy_config_path = tmp_path / "fixture_dummy_config.yaml"
        dummy_data = {
            'knowledge_base': {'base_dir': str(tmp_path / "fixture_kbs")},
            'embedding_model': {'provider': 'dummy', 'model_name': 'dummy', 'api_key': 'dummy'},
            'language_model': {'provider': 'dummy', 'model_name': 'dummy', 'api_key': 'dummy'},
            'logging': {'level': 'INFO'}
        }
        dummy_config_path.write_text(yaml.dump(dummy_data))
        ConfigService.get_instance(str(dummy_config_path))
    
    return KnowledgeBaseManager() # Initialize without args

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

def test_kb_manager_init_uses_config_default(tmp_path: Path):
    """Test KBM uses base_dir from ConfigService when none is provided."""
    # 1. Create a temporary config file for this test
    config_base_dir = tmp_path / "config_kbs"
    config_data = {
        'knowledge_base': {'base_dir': str(config_base_dir)},
        # Add minimal required sections for config validation to pass
        'embedding_model': {'provider': 'dummy', 'model_name': 'dummy', 'api_key': 'dummy'},
        'language_model': {'provider': 'dummy', 'model_name': 'dummy', 'api_key': 'dummy'},
        'logging': {'level': 'INFO'}
    }
    config_file = tmp_path / "test_default_config.yaml"
    config_file.write_text(yaml.dump(config_data))

    # 2. Ensure ConfigService loads this config (fixture already resets state)
    # This call initializes the singleton for this test
    ConfigService.get_instance(str(config_file))

    # 3. Initialize KnowledgeBaseManager WITHOUT providing base_dir
    assert not config_base_dir.exists() # Ensure target dir doesn't exist yet
    kb_manager_default = KnowledgeBaseManager() # Use default init path

    # 4. Assert that the manager's base_dir matches the config and was created
    assert kb_manager_default.base_dir == config_base_dir.resolve()
    assert config_base_dir.exists()
    assert config_base_dir.is_dir()

def test_kb_manager_init_creates_base_dir(tmp_path: Path):
    """Test that the base directory specified in config is created if it doesn't exist."""
    # 1. Define the base directory path for this test
    base_dir = tmp_path / "new_kbs"
    assert not base_dir.exists()

    # 2. Create a temporary config file pointing to this base_dir
    config_data = {
        'knowledge_base': {'base_dir': str(base_dir)},
        # Add minimal required sections for config validation to pass
        'embedding_model': {'provider': 'dummy', 'model_name': 'dummy', 'api_key': 'dummy'},
        'language_model': {'provider': 'dummy', 'model_name': 'dummy', 'api_key': 'dummy'},
        'logging': {'level': 'INFO'}
    }
    config_file = tmp_path / "create_dir_config.yaml"
    config_file.write_text(yaml.dump(config_data))

    # 3. Initialize ConfigService with this config
    # Fixture should ensure clean state, but good practice to be explicit if needed
    # ConfigService._reset_instance() # Uncomment if tests interfere
    ConfigService.get_instance(str(config_file))

    # 4. Initialize KnowledgeBaseManager without arguments
    kb_manager_creates = KnowledgeBaseManager()

    # 5. Assert the directory was created and matches
    assert kb_manager_creates.base_dir == base_dir.resolve()
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
    """Test creating a KB that already exists raises FileExistsError."""
    kb_path = kb_manager.base_dir / existing_kb
    assert kb_path.exists()

    with pytest.raises(FileExistsError):
        kb_manager.create_kb(existing_kb)

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
    """Test deleting a KB that does not exist raises FileNotFoundError."""
    kb_name = "non_existent_kb"
    kb_path = kb_manager.base_dir / kb_name
    assert not kb_path.exists()

    with pytest.raises(FileNotFoundError):
        kb_manager.delete_kb(kb_name)

def test_delete_kb_is_file(kb_manager: KnowledgeBaseManager):
    """Test deleting something that exists but is a file raises FileNotFoundError."""
    file_name = "not_a_kb.txt"
    file_path = kb_manager.base_dir / file_name
    file_path.touch()
    assert file_path.exists()
    assert file_path.is_file()

    with pytest.raises(FileNotFoundError): # Current implementation raises this
        kb_manager.delete_kb(file_name)

    assert file_path.exists() # Ensure the file wasn't deleted
