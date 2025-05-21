import asyncio
import pytest
import yaml
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from knowledge_mcp.knowledgebases import (
    KnowledgeBaseManager,
    KnowledgeBaseError,
    KnowledgeBaseExistsError,
    KnowledgeBaseNotFoundError,
    DEFAULT_QUERY_PARAMS,
    load_kb_query_config,
)
from knowledge_mcp.config import Config, KnowledgeBaseConfig as PydanticKnowledgeBaseConfig

# --- Fixtures ---

@pytest.fixture
def mock_config(tmp_path):
    """Provides a mock Config object with knowledge_base.base_dir set to a temp path."""
    kb_base_dir = tmp_path / "kb_root"
    # kb_base_dir.mkdir(parents=True, exist_ok=True) # KBM should create it

    # Create Pydantic models for the config structure
    pydantic_kb_config = PydanticKnowledgeBaseConfig(base_dir=kb_base_dir)
    
    # Create a MagicMock for the top-level Config object
    config_mock = MagicMock(spec=Config)
    config_mock.knowledge_base = pydantic_kb_config
    
    # Ensure the base_dir attribute itself is a Path object for resolve()
    # This simulates how the actual Config object would behave after Pydantic validation and path resolution
    config_mock.knowledge_base.base_dir = kb_base_dir
    return config_mock

@pytest.fixture
def kb_manager(mock_config):
    """Provides an initialized KnowledgeBaseManager instance."""
    return KnowledgeBaseManager(mock_config)

# --- Tests for KnowledgeBaseManager ---

# 1. Initialization (__init__)
def test_kbm_init_success(mock_config, tmp_path):
    """Test successful initialization and base_dir creation."""
    kb_root_path = tmp_path / "kb_root"
    assert not kb_root_path.exists() # Should not exist before KBM init
    manager = KnowledgeBaseManager(mock_config)
    assert manager.base_dir == kb_root_path.resolve()
    assert kb_root_path.is_dir() # KBM should create it

def test_kbm_init_existing_base_dir(mock_config, tmp_path):
    """Test initialization when base_dir already exists."""
    kb_root_path = tmp_path / "kb_root"
    kb_root_path.mkdir(parents=True, exist_ok=True) # Create it beforehand
    manager = KnowledgeBaseManager(mock_config)
    assert manager.base_dir == kb_root_path.resolve()
    assert kb_root_path.is_dir()

def test_kbm_init_type_error_invalid_config():
    """Test TypeError if config is not a Config instance."""
    with pytest.raises(TypeError, match="Expected a Config instance, but got str"):
        KnowledgeBaseManager("not_a_config_object")

def test_kbm_init_value_error_missing_kb_config_attr(mock_config):
    """Test ValueError if config.knowledge_base is missing."""
    del mock_config.knowledge_base 
    with pytest.raises(ValueError, match="Knowledge base base_dir not configured"):
        KnowledgeBaseManager(mock_config)

def test_kbm_init_value_error_missing_base_dir_attr(mock_config):
    """Test ValueError if config.knowledge_base.base_dir is missing."""
    mock_config.knowledge_base.base_dir = None
    with pytest.raises(ValueError, match="Knowledge base base_dir not configured"):
        KnowledgeBaseManager(mock_config)


def test_kbm_init_base_dir_is_file(mock_config, tmp_path):
    """Test TypeError (or KnowledgeBaseError) if base_dir path points to a file."""
    file_path = tmp_path / "kb_root_file"
    file_path.write_text("I am a file.")
    
    # Update mock_config to point to this file
    # Create a new Pydantic model instance for knowledge_base
    pydantic_kb_config_file = PydanticKnowledgeBaseConfig(base_dir=file_path)
    mock_config_file = MagicMock(spec=Config)
    mock_config_file.knowledge_base = pydantic_kb_config_file
    mock_config_file.knowledge_base.base_dir = file_path


    # The error is caught by Path.mkdir raising an OSError, which is wrapped
    with pytest.raises(TypeError, match="Invalid base directory path or permissions"):
        KnowledgeBaseManager(mock_config_file)

@patch('pathlib.Path.mkdir', side_effect=OSError("Test OS permission error"))
def test_kbm_init_os_error_on_mkdir(mock_mkdir, mock_config):
    """Test OSError during base_dir creation is wrapped in TypeError."""
    with pytest.raises(TypeError, match="Invalid base directory path or permissions"):
        KnowledgeBaseManager(mock_config)


# 2. KB Path Management (get_kb_path, kb_exists)
def test_kbm_get_kb_path(kb_manager, mock_config):
    """Test get_kb_path returns correct Path object."""
    expected_path = (mock_config.knowledge_base.base_dir / "my_kb").resolve()
    assert kb_manager.get_kb_path("my_kb") == expected_path

def test_kbm_kb_exists(kb_manager):
    """Test kb_exists for existing and non-existing KBs."""
    assert not kb_manager.kb_exists("non_existent_kb")
    kb_path = kb_manager.get_kb_path("existing_kb")
    kb_path.mkdir()
    assert kb_manager.kb_exists("existing_kb")

# 3. KB Creation (create_kb)
def test_kbm_create_kb_success(kb_manager):
    """Test successful KB creation and config.yaml generation."""
    kb_name = "new_kb"
    kb_path = kb_manager.create_kb(kb_name)
    assert kb_path.is_dir()
    assert kb_path.name == kb_name

    config_file = kb_path / "config.yaml"
    assert config_file.is_file()
    with open(config_file, 'r') as f:
        content = yaml.safe_load(f)
    assert content == DEFAULT_QUERY_PARAMS

def test_kbm_create_kb_with_description(kb_manager):
    """Test KB creation with a custom description."""
    kb_name = "desc_kb"
    description = "This is a test description."
    kb_path = kb_manager.create_kb(kb_name, description=description)
    
    config_file = kb_path / "config.yaml"
    assert config_file.is_file()
    with open(config_file, 'r') as f:
        content = yaml.safe_load(f)
    
    expected_config = DEFAULT_QUERY_PARAMS.copy()
    expected_config["description"] = description
    assert content == expected_config

def test_kbm_create_kb_already_exists(kb_manager):
    """Test KnowledgeBaseExistsError if KB already exists."""
    kb_name = "existing_kb"
    kb_manager.create_kb(kb_name) # Create it once
    with pytest.raises(KnowledgeBaseExistsError, match=f"Knowledge base '{kb_name}' already exists"):
        kb_manager.create_kb(kb_name) # Try creating again

@patch('pathlib.Path.mkdir', side_effect=OSError("Simulated mkdir failure"))
def test_kbm_create_kb_mkdir_fails(mock_mkdir, kb_manager):
    """Test KnowledgeBaseError if directory creation fails."""
    with pytest.raises(KnowledgeBaseError, match="Could not create directory for KB 'fail_kb'"):
        kb_manager.create_kb("fail_kb")

@patch('yaml.dump', side_effect=yaml.YAMLError("Simulated YAML dump error"))
def test_kbm_create_kb_config_yaml_dump_fails(mock_yaml_dump, kb_manager):
    """Test error handling if yaml.dump fails for config.yaml."""
    kb_name = "yaml_fail_kb"
    with pytest.raises(KnowledgeBaseError, match="KB directory created, but failed to write config.yaml"):
        kb_manager.create_kb(kb_name)
    # Check if KB directory was created but config.yaml might be missing or incomplete
    kb_path = kb_manager.get_kb_path(kb_name)
    assert kb_path.is_dir() # Directory should exist
    assert not (kb_path / "config.yaml").exists() # Config file should not exist or be empty

@patch('builtins.open', side_effect=IOError("Simulated file open error"))
def test_kbm_create_kb_config_yaml_open_fails(mock_open, kb_manager):
    """Test error handling if opening config.yaml for writing fails."""
    kb_name = "open_fail_kb"
    with pytest.raises(KnowledgeBaseError, match="KB directory created, but failed to write config.yaml"):
        kb_manager.create_kb(kb_name)
    kb_path = kb_manager.get_kb_path(kb_name)
    assert kb_path.is_dir()

# 4. KB Listing (list_kbs - async)
@pytest.mark.asyncio
async def test_kbm_list_kbs_empty(kb_manager):
    """Test listing when no KBs exist."""
    assert await kb_manager.list_kbs() == {}

@pytest.mark.asyncio
async def test_kbm_list_kbs_multiple(kb_manager):
    """Test listing with multiple KBs and correct description parsing."""
    kb1_path = kb_manager.create_kb("kb1", description="Description for KB1")
    kb2_path = kb_manager.create_kb("kb2", description="Second KB here")
    
    # Create a KB with default description (no explicit description on create)
    kb3_path = kb_manager.create_kb("kb3")


    # Create a KB with a config.yaml but no 'description' key
    kb4_path = kb_manager.get_kb_path("kb4")
    kb4_path.mkdir()
    with open(kb4_path / "config.yaml", 'w') as f:
        yaml.dump({"mode": "test"}, f)

    # Create a KB with a malformed config.yaml
    kb5_path = kb_manager.get_kb_path("kb5")
    kb5_path.mkdir()
    with open(kb5_path / "config.yaml", 'w') as f:
        f.write("key: value: another_value # Invalid YAML")
        
    # Create a KB with a missing config.yaml
    kb6_path = kb_manager.get_kb_path("kb6")
    kb6_path.mkdir()
    
    # Add a non-directory file in base_dir, should be ignored
    (kb_manager.base_dir / "some_file.txt").write_text("ignore me")


    kbs = await kb_manager.list_kbs()
    
    assert len(kbs) == 6
    assert kbs.get("kb1") == "Description for KB1"
    assert kbs.get("kb2") == "Second KB here"
    assert kbs.get("kb3") == DEFAULT_QUERY_PARAMS["description"] # Default description
    assert kbs.get("kb4") == "No description found." # No description key
    assert "Error reading description: " in kbs.get("kb5") # Malformed YAML
    assert kbs.get("kb6") == "No description found." # Missing config.yaml

@patch('asyncio.to_thread', side_effect=OSError("Simulated listdir error"))
@pytest.mark.asyncio
async def test_kbm_list_kbs_os_error(mock_to_thread, kb_manager):
    """Test error handling if os.listdir (or equivalent) fails during list_kbs."""
    # This test is a bit tricky because the error is in base_dir.iterdir()
    # We'll mock iterdir itself on the Path object instance used by kb_manager
    with patch.object(kb_manager.base_dir, 'iterdir', side_effect=OSError("Simulated iterdir error")):
        with pytest.raises(KnowledgeBaseError, match="Error listing knowledge bases"):
            await kb_manager.list_kbs()


# 5. KB Deletion (delete_kb)
def test_kbm_delete_kb_success(kb_manager):
    """Test successful deletion of an existing KB."""
    kb_name = "to_delete_kb"
    kb_path = kb_manager.create_kb(kb_name)
    assert kb_path.exists()
    
    kb_manager.delete_kb(kb_name)
    assert not kb_path.exists()

def test_kbm_delete_kb_not_found(kb_manager):
    """Test KnowledgeBaseNotFoundError when deleting non-existent KB."""
    with pytest.raises(KnowledgeBaseNotFoundError, match="Knowledge base 'non_existent_kb' not found"):
        kb_manager.delete_kb("non_existent_kb")

@patch('shutil.rmtree', side_effect=OSError("Simulated rmtree failure"))
def test_kbm_delete_kb_rmtree_fails(mock_rmtree, kb_manager):
    """Test KnowledgeBaseError if shutil.rmtree fails."""
    kb_name = "rmtree_fail_kb"
    kb_manager.create_kb(kb_name) # Create it first
    
    with pytest.raises(KnowledgeBaseError, match=f"OS error deleting knowledge base '{kb_name}'"):
        kb_manager.delete_kb(kb_name)
    
    # Check that the directory still exists after the mocked failure
    assert kb_manager.kb_exists(kb_name)

# 6. Placeholder Document Methods
def test_kbm_add_document_kb_not_found(kb_manager):
    """Test add_document raises KnowledgeBaseNotFoundError if KB doesn't exist."""
    with pytest.raises(KnowledgeBaseNotFoundError, match="Knowledge base 'ghost_kb' not found."):
        kb_manager.add_document("ghost_kb", Path("dummy_doc.txt"))

def test_kbm_remove_document_kb_not_found(kb_manager):
    """Test remove_document raises KnowledgeBaseNotFoundError if KB doesn't exist."""
    with pytest.raises(KnowledgeBaseNotFoundError, match="Knowledge base 'phantom_kb' not found."):
        kb_manager.remove_document("phantom_kb", "dummy_doc_name")

# --- Tests for load_kb_query_config ---

@pytest.fixture
def temp_kb_path(tmp_path):
    """Creates a temporary KB directory path for load_kb_query_config tests."""
    kb_dir = tmp_path / "test_kb_for_load"
    kb_dir.mkdir()
    return kb_dir

def test_load_query_config_valid_file(temp_kb_path):
    """1. Test loading with a valid config.yaml, parameters override defaults."""
    custom_params = {
        "mode": "custom_mode",
        "top_k": 5,
        "description": "This should be ignored by load_kb_query_config"
    }
    config_file = temp_kb_path / "config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(custom_params, f)

    loaded = load_kb_query_config(temp_kb_path)
    
    expected = DEFAULT_QUERY_PARAMS.copy()
    expected.update(custom_params)
    del expected["description"] # Description should not be in query params

    assert loaded["mode"] == "custom_mode"
    assert loaded["top_k"] == 5
    assert "description" not in loaded
    assert loaded["response_type"] == DEFAULT_QUERY_PARAMS["response_type"] # Check a default is still there

def test_load_query_config_missing_file(temp_kb_path, caplog):
    """2. Test when config.yaml is missing (returns defaults, logs debug)."""
    expected_defaults_no_desc = DEFAULT_QUERY_PARAMS.copy()
    del expected_defaults_no_desc["description"]
    
    loaded = load_kb_query_config(temp_kb_path)
    assert loaded == expected_defaults_no_desc
    assert f"Config file not found for KB '{temp_kb_path.name}'" in caplog.text

def test_load_query_config_empty_file(temp_kb_path, caplog):
    """2. Test when config.yaml is empty (returns defaults, logs warning)."""
    config_file = temp_kb_path / "config.yaml"
    config_file.write_text("")
    
    expected_defaults_no_desc = DEFAULT_QUERY_PARAMS.copy()
    del expected_defaults_no_desc["description"]

    loaded = load_kb_query_config(temp_kb_path)
    assert loaded == expected_defaults_no_desc
    assert f"Config file for KB '{temp_kb_path.name}' is empty." in caplog.text

def test_load_query_config_null_content(temp_kb_path, caplog):
    """2. Test when config.yaml contains 'null' (returns defaults, logs warning)."""
    config_file = temp_kb_path / "config.yaml"
    config_file.write_text("null")

    expected_defaults_no_desc = DEFAULT_QUERY_PARAMS.copy()
    del expected_defaults_no_desc["description"]

    loaded = load_kb_query_config(temp_kb_path)
    assert loaded == expected_defaults_no_desc
    assert f"Config file for KB '{temp_kb_path.name}' is empty." in caplog.text # 'null' loads as None

def test_load_query_config_partial_params(temp_kb_path):
    """3. Test with partial params in file - unspecified ones use defaults."""
    partial_params = {"top_k": 10} # Only top_k specified
    config_file = temp_kb_path / "config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(partial_params, f)

    loaded = load_kb_query_config(temp_kb_path)
    
    assert loaded["top_k"] == 10
    assert loaded["mode"] == DEFAULT_QUERY_PARAMS["mode"] # Default
    assert "description" not in loaded

def test_load_query_config_extra_keys_ignored(temp_kb_path):
    """3. Test with extra keys in file - they should be ignored."""
    extra_params = {
        "top_k": 7,
        "unknown_param": "some_value",
        "another_extra": 123
    }
    config_file = temp_kb_path / "config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(extra_params, f)

    loaded = load_kb_query_config(temp_kb_path)
    
    assert loaded["top_k"] == 7
    assert "unknown_param" not in loaded
    assert "another_extra" not in loaded
    assert "description" not in loaded

def test_load_query_config_invalid_yaml(temp_kb_path, caplog):
    """4. Test with invalid YAML (returns defaults, logs error)."""
    config_file = temp_kb_path / "config.yaml"
    config_file.write_text("mode: mix\ntop_k: broken: yaml")

    expected_defaults_no_desc = DEFAULT_QUERY_PARAMS.copy()
    del expected_defaults_no_desc["description"]
    
    loaded = load_kb_query_config(temp_kb_path)
    assert loaded == expected_defaults_no_desc
    assert f"Error parsing YAML file {config_file}" in caplog.text

def test_load_query_config_not_a_dictionary(temp_kb_path, caplog):
    """4. Test when config.yaml content is not a dictionary (returns defaults, logs error)."""
    config_file = temp_kb_path / "config.yaml"
    yaml.dump(["item1", "item2"], open(config_file, 'w')) # Content is a list

    expected_defaults_no_desc = DEFAULT_QUERY_PARAMS.copy()
    del expected_defaults_no_desc["description"]

    loaded = load_kb_query_config(temp_kb_path)
    assert loaded == expected_defaults_no_desc
    assert f"Invalid config format in {config_file}. Expected a dictionary" in caplog.text

def test_load_query_config_description_in_defaults_is_ignored(temp_kb_path):
    """Ensure 'description' from DEFAULT_QUERY_PARAMS is not in the final result."""
    # This test makes sure that even if config.yaml is missing, the 'description'
    # from DEFAULT_QUERY_PARAMS itself is correctly excluded.
    loaded = load_kb_query_config(temp_kb_path) # No config.yaml
    assert "description" not in loaded
    assert loaded["mode"] == DEFAULT_QUERY_PARAMS["mode"] # Other defaults are present
    
    # Also test with an empty config file
    (temp_kb_path / "config.yaml").write_text("")
    loaded_empty_config = load_kb_query_config(temp_kb_path)
    assert "description" not in loaded_empty_config

# Final check for test coverage and structure
# The file tests/test_kb_manager.py would be implicitly replaced by this tests/test_knowledgebases.py

"""
This test suite covers the KnowledgeBaseManager and load_kb_query_config function
from knowledge_mcp.knowledgebases module.

KnowledgeBaseManager tests include:
- Initialization: success, existing base_dir, invalid config types, missing config attributes, base_dir as file, OS errors.
- Path Management: get_kb_path, kb_exists.
- KB Creation: success, with description, already exists error, mkdir failure, config.yaml creation failure.
- KB Listing (async): empty list, multiple KBs, description parsing, missing/malformed config.yaml, non-directory items, OS errors.
- KB Deletion: success, not found error, rmtree failure.
- Placeholder Document Methods: basic KnowledgeBaseNotFoundError checks.

load_kb_query_config tests include:
- Valid config file: parameter overriding, 'description' exclusion.
- Config file scenarios: missing, empty, null content.
- Parameter merging: partial parameters, extra keys ignored.
- Invalid config file: invalid YAML, non-dictionary content.
- Ensure 'description' from DEFAULT_QUERY_PARAMS is always excluded from the result.
"""
