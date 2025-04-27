"""Integration tests for the 'config' command in the shell."""

import pytest
from unittest.mock import patch, MagicMock
import yaml
from io import StringIO
from pathlib import Path
import subprocess

from knowledge_mcp.shell import Shell
from knowledge_mcp.knowledgebases import KnowledgeBaseManager, DEFAULT_QUERY_PARAMS
from knowledge_mcp.config import Config
from knowledge_mcp.rag import RagManager

# --- Fixtures ---

@pytest.fixture
def temp_kb_base_dir(tmp_path: Path) -> Path:
    """Provides a temporary, empty directory for KB tests."""
    base_dir = tmp_path / "kb_shell_test_base"
    base_dir.mkdir()
    return base_dir

@pytest.fixture
def test_config(temp_kb_base_dir: Path) -> Config:
    """Creates a minimal valid Config object for testing."""
    config_data = {
        'knowledge_base': {'base_dir': str(temp_kb_base_dir)},
        'lightrag': { # Dummy data
            'embedding': {
                'provider': 'dummy',
                'model_name': 'dummy',
                'api_key': 'dummy',
                'embedding_dim': 128,
                'max_token_size': 512
            },
            'llm': {
                'provider': 'dummy',
                'model_name': 'dummy',
                'api_key': 'dummy',
                'max_token_size': 4096
            },
            'embedding_cache': {'enabled': False, 'similarity_threshold': 0.9}
        },
        'logging': {'level': 'INFO'},
        'env_file': '.env.test'
    }
    return Config(**config_data)

@pytest.fixture
def kb_manager(test_config: Config) -> KnowledgeBaseManager:
    """Provides a KnowledgeBaseManager instance."""
    return KnowledgeBaseManager(config=test_config)

@pytest.fixture
def mock_rag_manager() -> MagicMock:
    """Provides a mock RagManager."""
    return MagicMock(spec=RagManager)

@pytest.fixture
def shell_instance(kb_manager: KnowledgeBaseManager, mock_rag_manager: MagicMock) -> Shell:
    """Provides a Shell instance with mocked dependencies."""
    # Patch stdin to avoid blocking
    # Explicitly create StringIO for stdout and pass it to Shell constructor
    mock_stdout = StringIO()
    with patch('sys.stdin', StringIO()):
        shell = Shell(kb_manager, mock_rag_manager, stdout=mock_stdout)
        # Shell instance's self.stdout should now be the mock_stdout object
        yield shell # Yield the shell instance for the test
        # Cleanup: Stop background loop if shell has one and it's running
        if hasattr(shell, '_stop_background_loop'):
            shell._stop_background_loop()

@pytest.fixture
def kb_with_config(kb_manager: KnowledgeBaseManager) -> str:
    """Creates a KB with a default config.yaml file."""
    kb_name = "kb_with_cfg"
    kb_manager.create_kb(kb_name) # This should now create the config
    return kb_name

@pytest.fixture
def kb_without_config(kb_manager: KnowledgeBaseManager) -> str:
    """Creates a KB directory *without* a config.yaml file."""
    kb_name = "kb_no_cfg"
    kb_path = kb_manager.get_kb_path(kb_name)
    kb_path.mkdir() # Just create the dir, not the config
    return kb_name

# --- Test Cases ---

def test_config_show_existing(shell_instance: Shell, kb_with_config: str, kb_manager: KnowledgeBaseManager):
    """Test 'config show <kb>' when config exists."""
    kb_path = kb_manager.get_kb_path(kb_with_config)
    config_path = kb_path / "config.yaml"

    shell_instance.onecmd(f"config {kb_with_config} show")
    output = shell_instance.stdout.getvalue()

    assert f"Config file path: {config_path.resolve()}" in output
    assert "--- Config Content ---" in output
    # Check if default params are dumped correctly
    assert yaml.dump(DEFAULT_QUERY_PARAMS, default_flow_style=False, indent=2) in output
    assert "--- End Config Content ---" in output

def test_config_show_default_subcommand(shell_instance: Shell, kb_with_config: str, kb_manager: KnowledgeBaseManager):
    """Test 'config <kb>' defaults to 'show' when config exists."""
    kb_path = kb_manager.get_kb_path(kb_with_config)
    config_path = kb_path / "config.yaml"

    shell_instance.onecmd(f"config {kb_with_config}") # No subcommand
    output = shell_instance.stdout.getvalue()

    assert f"Config file path: {config_path.resolve()}" in output
    assert "--- Config Content ---" in output
    assert yaml.dump(DEFAULT_QUERY_PARAMS, default_flow_style=False, indent=2) in output
    assert "--- End Config Content ---" in output

def test_config_show_missing_config(shell_instance: Shell, kb_without_config: str, kb_manager: KnowledgeBaseManager):
    """Test 'config show <kb>' when config.yaml is missing."""
    kb_path = kb_manager.get_kb_path(kb_without_config)
    config_path = kb_path / "config.yaml"

    shell_instance.onecmd(f"config {kb_without_config} show")
    output = shell_instance.stdout.getvalue()

    assert f"Config file path: {config_path.resolve()}" in output
    assert "Config file does not exist" in output
    assert f"KB '{kb_without_config}' will use default query parameters." in output
    assert "--- Config Content ---" not in output

def test_config_show_nonexistent_kb(shell_instance: Shell):
    """Test 'config show <kb>' for a KB that doesn't exist."""
    shell_instance.onecmd("config non_existent_kb show")
    output = shell_instance.stdout.getvalue()
    assert "Error: Knowledge base 'non_existent_kb' not found." in output

@patch('knowledge_mcp.shell.subprocess.run')
@patch('knowledge_mcp.shell.os.getenv')
def test_config_edit_existing(mock_getenv: MagicMock, mock_subprocess_run: MagicMock, shell_instance: Shell, kb_with_config: str, kb_manager: KnowledgeBaseManager):
    """Test 'config edit <kb>' when config exists."""
    kb_path = kb_manager.get_kb_path(kb_with_config)
    config_path = kb_path / "config.yaml"
    mock_getenv.side_effect = lambda key, default=None: 'vim' if key in ['EDITOR', 'VISUAL'] else default

    shell_instance.onecmd(f"config {kb_with_config} edit")
    output = shell_instance.stdout.getvalue()

    mock_subprocess_run.assert_called_once_with(['vim', str(config_path)], check=True)
    assert f"Attempting to open '{config_path.resolve()}' with editor 'vim'" in output
    assert "Editor closed." in output

@patch('knowledge_mcp.shell.subprocess.run')
def test_config_edit_missing_config(mock_subprocess_run: MagicMock, shell_instance: Shell, kb_without_config: str):
    """Test 'config edit <kb>' when config.yaml is missing."""
    shell_instance.onecmd(f"config {kb_without_config} edit")
    output = shell_instance.stdout.getvalue()

    mock_subprocess_run.assert_not_called()
    assert "Error: Config file" in output
    assert "does not exist" in output

@patch('knowledge_mcp.shell.subprocess.run')
def test_config_edit_nonexistent_kb(mock_subprocess_run: MagicMock, shell_instance: Shell):
    """Test 'config edit <kb>' for a KB that doesn't exist."""
    shell_instance.onecmd("config non_existent_kb edit")
    output = shell_instance.stdout.getvalue()

    mock_subprocess_run.assert_not_called()
    assert "Error: Knowledge base 'non_existent_kb' not found." in output

def test_config_invalid_subcommand(shell_instance: Shell, kb_with_config: str):
    """Test 'config <kb> <invalid_subcommand>'."""
    shell_instance.onecmd(f"config {kb_with_config} foobar")
    output = shell_instance.stdout.getvalue()
    assert "Error: Unknown config subcommand 'foobar'. Use 'show' or 'edit'." in output

def test_config_missing_kb_name(shell_instance: Shell):
    """Test 'config' command with no arguments."""
    shell_instance.onecmd("config")
    output = shell_instance.stdout.getvalue()
    assert "Usage: config <kb_name> [show|edit]" in output

@patch('knowledge_mcp.shell.subprocess.run', side_effect=FileNotFoundError("Editor not found"))
@patch('knowledge_mcp.shell.os.getenv')
def test_config_edit_editor_not_found(mock_getenv: MagicMock, mock_subprocess_run: MagicMock, shell_instance: Shell, kb_with_config: str):
    """Test 'config edit' when the editor binary is not found."""
    mock_getenv.side_effect = lambda key, default=None: 'fake_editor' if key in ['EDITOR', 'VISUAL'] else default

    shell_instance.onecmd(f"config {kb_with_config} edit")
    output = shell_instance.stdout.getvalue()

    mock_subprocess_run.assert_called_once()
    assert "Error: Editor 'fake_editor' not found." in output

@patch('knowledge_mcp.shell.subprocess.run', side_effect=subprocess.CalledProcessError(1, 'cmd'))
@patch('knowledge_mcp.shell.os.getenv')
def test_config_edit_editor_error(mock_getenv: MagicMock, mock_subprocess_run: MagicMock, shell_instance: Shell, kb_with_config: str):
    """Test 'config edit' when the editor returns an error."""
    mock_getenv.side_effect = lambda key, default=None: 'bad_editor' if key in ['EDITOR', 'VISUAL'] else default

    shell_instance.onecmd(f"config {kb_with_config} edit")
    output = shell_instance.stdout.getvalue()

    mock_subprocess_run.assert_called_once()
    assert "Error running editor 'bad_editor'" in output
