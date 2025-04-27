"""Unit tests for the KnowledgeBaseManager class and related functions."""

import pytest
from pathlib import Path
import yaml

# Import from the correct module
from knowledge_mcp.knowledgebases import (
    KnowledgeBaseManager,
    KnowledgeBaseExistsError,
    KnowledgeBaseError,  # Base error for not found etc.
    load_kb_query_config,
    DEFAULT_QUERY_PARAMS
)
from knowledge_mcp.config import Config  # Import Config


# --- Fixtures ---

@pytest.fixture
def temp_kb_base_dir(tmp_path: Path) -> Path:
    """Provides a temporary, empty directory for KB tests."""
    base_dir = tmp_path / "kb_test_base"
    base_dir.mkdir()
    return base_dir

@pytest.fixture
def kb_manager(temp_kb_base_dir: Path) -> KnowledgeBaseManager:
    """Provides a KnowledgeBaseManager instance configured with a temporary base directory."""
    # Create a minimal Config object pointing to the temp base directory
    # Include minimal dummy data for required fields to pass validation
    test_config_data = {
        'knowledge_base': {'base_dir': str(temp_kb_base_dir)},
        'lightrag': { # Dummy data
            'embedding': {
                'provider': 'dummy',
                'model_name': 'dummy',
                'api_key': 'dummy',
                'embedding_dim': 128, # Added dummy
                'max_token_size': 512 # Added dummy
            },
            'llm': {
                'provider': 'dummy',
                'model_name': 'dummy',
                'api_key': 'dummy',
                'max_token_size': 4096 # Added dummy
            },
            'embedding_cache': {
                'enabled': False, # Added dummy
                'similarity_threshold': 0.9 # Added dummy
            } # Added dummy
        },
        'logging': {'level': 'INFO'}, # Dummy data
        'env_file': '.env.test' # Dummy data
    }
    test_config = Config(**test_config_data)
    # Pass the config object directly to the manager
    manager = KnowledgeBaseManager(config=test_config)
    assert manager.base_dir == temp_kb_base_dir # Ensure it's set
    return manager

@pytest.fixture
def existing_kb(kb_manager: KnowledgeBaseManager) -> str:
    """Creates a dummy KB directory for testing deletion/existence checks."""
    kb_name = "existing_kb"
    # Use the manager's method to create it, ensuring consistency
    kb_manager.create_kb(kb_name)
    kb_path = kb_manager.get_kb_path(kb_name)
    # Add a dummy file inside to simulate content
    (kb_path / "dummy.txt").touch()
    return kb_name

# === Test KnowledgeBaseManager Initialization ===

def test_kb_manager_init_creates_base_dir(tmp_path: Path):
    """Test that the base directory specified in config is created if it doesn't exist."""
    base_dir = tmp_path / "new_kbs"
    assert not base_dir.exists()

    # Include minimal dummy data for required fields
    test_config_data = {
        'knowledge_base': {'base_dir': str(base_dir)},
        'lightrag': { # Dummy data
            'embedding': {
                'provider': 'dummy',
                'model_name': 'dummy',
                'api_key': 'dummy',
                'embedding_dim': 128, # Added dummy
                'max_token_size': 512 # Added dummy
            },
            'llm': {
                'provider': 'dummy',
                'model_name': 'dummy',
                'api_key': 'dummy',
                'max_token_size': 4096 # Added dummy
            },
            'embedding_cache': {
                'enabled': False, # Added dummy
                'similarity_threshold': 0.9 # Added dummy
            } # Added dummy
        },
        'logging': {'level': 'INFO'}, # Dummy data
        'env_file': '.env.test' # Dummy data
    }
    test_config = Config(**test_config_data)
    kb_manager_instance = KnowledgeBaseManager(config=test_config)

    assert kb_manager_instance.base_dir == base_dir.resolve()
    assert base_dir.exists()
    assert base_dir.is_dir()

def test_kb_manager_init_uses_existing_base_dir(temp_kb_base_dir: Path):
    """Test that an existing base directory is used without error."""
    assert temp_kb_base_dir.exists()
    # Include minimal dummy data for required fields
    test_config_data = {
        'knowledge_base': {'base_dir': str(temp_kb_base_dir)},
        'lightrag': { # Dummy data
            'embedding': {
                'provider': 'dummy',
                'model_name': 'dummy',
                'api_key': 'dummy',
                'embedding_dim': 128, # Added dummy
                'max_token_size': 512 # Added dummy
            },
            'llm': {
                'provider': 'dummy',
                'model_name': 'dummy',
                'api_key': 'dummy',
                'max_token_size': 4096 # Added dummy
            },
            'embedding_cache': {
                'enabled': False, # Added dummy
                'similarity_threshold': 0.9 # Added dummy
            } # Added dummy
        },
        'logging': {'level': 'INFO'}, # Dummy data
        'env_file': '.env.test' # Dummy data
    }
    test_config = Config(**test_config_data)
    # Should initialize without raising errors
    kb_manager_instance = KnowledgeBaseManager(config=test_config)
    assert kb_manager_instance.base_dir == temp_kb_base_dir

def test_kb_manager_init_missing_config_raises_error():
    """Test that initializing without a config raises an error."""
    # KnowledgeBaseManager now requires a Config object
    with pytest.raises(TypeError): # Or specific error if __init__ enforces it
        KnowledgeBaseManager() # type: ignore

def test_kb_manager_init_invalid_config_type_raises_error():
    """Test passing wrong type as config raises error"""
    with pytest.raises(TypeError):
        KnowledgeBaseManager(config="not a config object") # type: ignore

# === Test KnowledgeBaseManager Creation ===

def test_create_kb_success(kb_manager: KnowledgeBaseManager):
    """Test successful creation of a knowledge base and its config.yaml."""
    kb_name = "test_kb_create"
    kb_path = kb_manager.get_kb_path(kb_name)
    config_path = kb_path / "config.yaml"

    assert not kb_path.exists()
    assert not config_path.exists()

    created_path = kb_manager.create_kb(kb_name)

    assert created_path == kb_path
    assert kb_path.exists()
    assert kb_path.is_dir()

    # Assert config.yaml was created with default content
    assert config_path.exists()
    assert config_path.is_file()
    with open(config_path, 'r', encoding='utf-8') as f:
        loaded_config = yaml.safe_load(f)
    assert loaded_config == DEFAULT_QUERY_PARAMS

def test_create_kb_already_exists(kb_manager: KnowledgeBaseManager, existing_kb: str):
    """Test creating a KB that already exists raises KnowledgeBaseExistsError."""
    kb_path = kb_manager.get_kb_path(existing_kb)
    assert kb_path.exists()

    with pytest.raises(KnowledgeBaseExistsError):
        kb_manager.create_kb(existing_kb)

# === Test KnowledgeBaseManager Listing ===

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

# === Test KnowledgeBaseManager Deletion ===

def test_delete_kb_success(kb_manager: KnowledgeBaseManager, existing_kb: str):
    """Test successful deletion of an existing knowledge base."""
    kb_path = kb_manager.get_kb_path(existing_kb)
    assert kb_path.exists()

    kb_manager.delete_kb(existing_kb)

    assert not kb_path.exists()
    assert kb_manager.list_kbs() == []

def test_delete_kb_not_found(kb_manager: KnowledgeBaseManager):
    """Test deleting a KB that does not exist raises KnowledgeBaseError."""
    kb_name = "non_existent_kb"
    kb_path = kb_manager.get_kb_path(kb_name)
    assert not kb_path.exists()

    with pytest.raises(KnowledgeBaseError): # Changed from FileNotFoundError
        kb_manager.delete_kb(kb_name)

def test_delete_kb_is_file(kb_manager: KnowledgeBaseManager):
    """Test deleting something that exists but is a file raises KnowledgeBaseError."""
    file_name = "not_a_kb.txt"
    file_path = kb_manager.base_dir / file_name
    file_path.touch()
    assert file_path.exists()
    assert file_path.is_file()

    with pytest.raises(KnowledgeBaseError): # Changed from FileNotFoundError
        kb_manager.delete_kb(file_name)

    assert file_path.exists() # Ensure the file wasn't deleted

# === Test load_kb_query_config ===

# Helper function to create a KB dir with optional config
def create_test_kb_dir(tmp_path: Path, name: str, config_content: dict | str | None = None) -> Path:
    kb_dir = tmp_path / name
    kb_dir.mkdir()
    if config_content is not None:
        config_path = kb_dir / "config.yaml"
        if isinstance(config_content, dict):
            config_path.write_text(yaml.dump(config_content), encoding='utf-8')
        else:
            config_path.write_text(config_content, encoding='utf-8') # For invalid content tests
    return kb_dir

def test_load_config_missing_file(tmp_path: Path):
    """Test loading config when config.yaml is missing returns defaults."""
    kb_dir = create_test_kb_dir(tmp_path, "kb_no_config")
    config = load_kb_query_config(kb_dir)
    assert config == DEFAULT_QUERY_PARAMS

def test_load_config_empty_file(tmp_path: Path):
    """Test loading config when config.yaml is empty returns defaults."""
    kb_dir = create_test_kb_dir(tmp_path, "kb_empty_config", config_content={})
    config = load_kb_query_config(kb_dir)
    # An empty dict loaded results in defaults
    assert config == DEFAULT_QUERY_PARAMS

def test_load_config_none_in_file(tmp_path: Path):
    """Test loading config when config.yaml contains only 'null' returns defaults."""
    kb_dir = create_test_kb_dir(tmp_path, "kb_none_config", config_content="null")
    config = load_kb_query_config(kb_dir)
    # Loading 'null' results in None, should use defaults
    assert config == DEFAULT_QUERY_PARAMS

def test_load_config_partial_override(tmp_path: Path):
    """Test loading config with some valid overrides."""
    overrides = {
        "mode": "local",
        "top_k": 10
    }
    kb_dir = create_test_kb_dir(tmp_path, "kb_partial", config_content=overrides)
    config = load_kb_query_config(kb_dir)

    expected_config = DEFAULT_QUERY_PARAMS.copy()
    expected_config.update(overrides)

    assert config == expected_config
    assert config["mode"] == "local"
    assert config["top_k"] == 10
    assert config["history_turns"] == DEFAULT_QUERY_PARAMS["history_turns"] # Check one default

def test_load_config_full_override(tmp_path: Path):
    """Test loading config with all valid keys overridden."""
    overrides = {
        "mode": "hybrid",
        "only_need_context": True,
        "only_need_prompt": True,
        "response_type": "Single Paragraph",
        "top_k": 5,
        "max_token_for_text_unit": 100,
        "max_token_for_global_context": 200,
        "max_token_for_local_context": 300,
        "history_turns": 0,
    }
    kb_dir = create_test_kb_dir(tmp_path, "kb_full", config_content=overrides)
    config = load_kb_query_config(kb_dir)
    assert config == overrides # All keys are overridden

def test_load_config_with_extra_keys(tmp_path: Path):
    """Test loading config ignores keys not in DEFAULT_QUERY_PARAMS."""
    content = {
        "mode": "local",
        "top_k": 15,
        "extra_key": "should be ignored",
        "embedding_model": "test"
    }
    kb_dir = create_test_kb_dir(tmp_path, "kb_extra", config_content=content)
    config = load_kb_query_config(kb_dir)

    expected_config = DEFAULT_QUERY_PARAMS.copy()
    expected_config.update({"mode": "local", "top_k": 15})

    assert config == expected_config
    assert "extra_key" not in config
    assert "embedding_model" not in config

def test_load_config_invalid_yaml(tmp_path: Path):
    """Test loading config with invalid YAML returns defaults."""
    invalid_yaml_content = "key: value: another_value\n  unindented_key: oops"
    kb_dir = create_test_kb_dir(tmp_path, "kb_invalid_yaml", config_content=invalid_yaml_content)
    config = load_kb_query_config(kb_dir)
    # Should log an error and return defaults
    assert config == DEFAULT_QUERY_PARAMS

def test_load_config_not_a_dict(tmp_path: Path):
    """Test loading config where YAML is valid but not a dict returns defaults."""
    not_a_dict_content = "- item1\n- item2"
    kb_dir = create_test_kb_dir(tmp_path, "kb_not_dict", config_content=not_a_dict_content)
    config = load_kb_query_config(kb_dir)
    # Should log an error and return defaults
    assert config == DEFAULT_QUERY_PARAMS
