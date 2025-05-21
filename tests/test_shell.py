import pytest
import asyncio
import cmd
import shlex
import logging
import threading
from pathlib import Path
import yaml
import os
import subprocess
from io import StringIO
from unittest.mock import patch, MagicMock, AsyncMock, call

from knowledge_mcp.shell import Shell
from knowledge_mcp.knowledgebases import (
    KnowledgeBaseManager,
    KnowledgeBaseExistsError,
    KnowledgeBaseNotFoundError,
    KnowledgeBaseError,
    DEFAULT_QUERY_PARAMS # For config tests
)
from knowledge_mcp.rag import (
    RagManager,
    RAGInitializationError,
    ConfigurationError,
    RAGManagerError
)
from knowledge_mcp.documents import DocumentManager, DocumentProcessingError, TextExtractionError # Assuming these are relevant for doc_manager.add
from knowledge_mcp.config import Config as AppConfig # For config tests from test_shell_config

# --- Fixtures ---

@pytest.fixture
def mock_kb_manager():
    """Provides a MagicMock for KnowledgeBaseManager."""
    manager = MagicMock(spec=KnowledgeBaseManager)
    manager.create_kb = MagicMock() # Sync method in shell's direct call
    manager.list_kbs = AsyncMock(return_value={}) # Async method
    manager.delete_kb = MagicMock()
    manager.get_kb_path = MagicMock(return_value=Path("/fake/kb_base/test_kb"))
    return manager

@pytest.fixture
def mock_rag_manager():
    """Provides a MagicMock for RagManager."""
    manager = MagicMock(spec=RagManager)
    manager.create_rag_instance = AsyncMock() # Async method
    manager.remove_rag_instance = MagicMock()
    manager.query = AsyncMock(return_value="Mocked query result") # Async method
    # Assuming remove_document is on RagManager as per problem description
    # If it's on DocumentManager, this mock would be different.
    manager.remove_document = MagicMock(return_value=True) 
    return manager

@pytest.fixture
def mock_document_manager():
    """Provides a MagicMock for DocumentManager.
       Note: DocumentManager is created inside Shell, so we'll patch its constructor."""
    manager_instance_mock = MagicMock(spec=DocumentManager)
    manager_instance_mock.add = AsyncMock(return_value="doc_123") # Async method
    return manager_instance_mock


@pytest.fixture
def shell_instance(mock_kb_manager, mock_rag_manager, mock_document_manager):
    """
    Provides a Shell instance with mocked dependencies and patched background loop.
    Patches DocumentManager constructor to inject mock_document_manager.
    """
    # Patch threading.Thread to prevent actual thread start, but allow checks
    with patch('threading.Thread', MagicMock(spec=threading.Thread)) as MockThread, \
         patch('knowledge_mcp.shell.DocumentManager', return_value=mock_document_manager) as MockDocManagerCls, \
         patch('sys.stdout', new_callable=StringIO) as mock_stdout:
        
        shell = Shell(kb_manager=mock_kb_manager, rag_manager=mock_rag_manager, stdout=mock_stdout)
        # Store mocks for assertions if needed, though they are also available from fixture args
        shell.mock_stdout = mock_stdout 
        shell.MockThread = MockThread
        shell.MockDocManagerCls = MockDocManagerCls
        
        yield shell # Test runs here

        # Teardown: attempt to stop loop, though mock thread won't really run
        # This ensures _stop_background_loop is at least called in tests like do_exit
        if hasattr(shell, '_stop_background_loop'):
             shell._stop_background_loop(test_mode=True) # Add a flag to avoid join issues with mock


# Modify _stop_background_loop in Shell for testability with mock thread
# This is a common pattern: allow a "test_mode" to bypass problematic parts with mocks.
def _test_stop_background_loop(self, test_mode=False):
    if test_mode and hasattr(self, '_async_thread') and isinstance(self._async_thread, MagicMock):
        logging.info("Mock background thread: Skipping join for test mode.")
        return
    # Original logic from shell.py for actual thread
    if hasattr(self, '_async_thread') and self._async_thread.is_alive():
        logging.info("Stopping background thread...")
        self._async_thread.join() # This would block if thread was real and not stopping
        logging.info("Background thread joined.")
    else:
        logging.info("Background thread not running or not initialized.")
Shell._stop_background_loop = _test_stop_background_loop


# --- Tests for Shell Initialization ---
def test_shell_init(shell_instance, mock_kb_manager, mock_rag_manager, mock_document_manager):
    """Test Shell initializes correctly, creates DocumentManager, and starts background loop."""
    assert shell_instance.kb_manager is mock_kb_manager
    assert shell_instance.rag_manager is mock_rag_manager
    
    # Check DocumentManager instantiation
    shell_instance.MockDocManagerCls.assert_called_once_with(mock_rag_manager)
    assert shell_instance.document_manager is mock_document_manager

    # Check background loop start
    shell_instance.MockThread.assert_called_once()
    # args[0] of the call_args_list is the first positional arg to Thread() constructor
    # which is `target`. Check if it's `shell_instance._run_background_loop`.
    assert shell_instance.MockThread.call_args[1]['target'] == shell_instance._run_background_loop
    shell_instance.MockThread.return_value.start.assert_called_once()


# --- Tests for Basic Commands ---

def test_do_exit(shell_instance):
    """Test do_exit returns True, prints message, and calls _stop_background_loop."""
    with patch.object(shell_instance, '_stop_background_loop', MagicMock()) as mock_stop_loop:
        assert shell_instance.onecmd("exit") is True
    output = shell_instance.mock_stdout.getvalue()
    assert "Exiting shell." in output
    mock_stop_loop.assert_called_once()

def test_do_EOF(shell_instance):
    """Test do_EOF returns True, prints message, and calls _stop_background_loop."""
    with patch.object(shell_instance, '_stop_background_loop', MagicMock()) as mock_stop_loop:
        assert shell_instance.onecmd("EOF") is True # onecmd will handle EOF correctly
    output = shell_instance.mock_stdout.getvalue()
    assert "\nExiting shell." in output # EOF adds a newline
    mock_stop_loop.assert_called_once()

@patch('os.system')
def test_do_clear_windows(mock_os_system, shell_instance):
    """Test do_clear calls os.system with 'cls' on Windows."""
    with patch('os.name', 'nt'):
        shell_instance.onecmd("clear")
    mock_os_system.assert_called_once_with('cls')

@patch('os.system')
def test_do_clear_posix(mock_os_system, shell_instance):
    """Test do_clear calls os.system with 'clear' on non-Windows."""
    with patch('os.name', 'posix'):
        shell_instance.onecmd("clear")
    mock_os_system.assert_called_once_with('clear')


# --- Tests for KB Management Commands ---

@patch('asyncio.run') # Patch asyncio.run where shell commands use it
def test_do_create_success(mock_asyncio_run, shell_instance, mock_kb_manager, mock_rag_manager):
    """Test successful KB creation with description."""
    kb_name = "mykb"
    description = "A test KB"
    
    shell_instance.onecmd(f'create {kb_name} "{description}"')
    
    mock_kb_manager.create_kb.assert_called_once_with(kb_name, description=description)
    # Check that asyncio.run was called with rag_manager.create_rag_instance
    # This is a bit indirect. We check the first arg of the first call to asyncio.run
    assert mock_asyncio_run.call_args_list[0][0][0] == mock_rag_manager.create_rag_instance(kb_name)
    
    output = shell_instance.mock_stdout.getvalue()
    assert f"Knowledge base '{kb_name}' created and RAG instance initialized successfully." in output

@patch('asyncio.run')
def test_do_create_no_description(mock_asyncio_run, shell_instance, mock_kb_manager, mock_rag_manager):
    """Test KB creation without description."""
    kb_name = "mykb_no_desc"
    shell_instance.onecmd(f"create {kb_name}")
    mock_kb_manager.create_kb.assert_called_once_with(kb_name, description=None)
    assert mock_asyncio_run.call_args_list[0][0][0] == mock_rag_manager.create_rag_instance(kb_name)
    assert f"Knowledge base '{kb_name}' created and RAG instance initialized successfully." in shell_instance.mock_stdout.getvalue()

def test_do_create_kb_exists(shell_instance, mock_kb_manager):
    """Test create when KB already exists."""
    kb_name = "existing_kb"
    mock_kb_manager.create_kb.side_effect = KnowledgeBaseExistsError
    shell_instance.onecmd(f"create {kb_name}")
    assert f"Error: Knowledge base '{kb_name}' already exists." in shell_instance.mock_stdout.getvalue()

def test_do_create_kb_error(shell_instance, mock_kb_manager):
    """Test create with general KnowledgeBaseError."""
    mock_kb_manager.create_kb.side_effect = KnowledgeBaseError("Disk full")
    shell_instance.onecmd("create anykb")
    assert "Error creating knowledge base: Disk full" in shell_instance.mock_stdout.getvalue()

@patch('asyncio.run')
def test_do_create_rag_init_error(mock_asyncio_run, shell_instance, mock_kb_manager, mock_rag_manager):
    """Test create with RAGInitializationError during RAG instance creation."""
    kb_name = "rag_fail_kb"
    # create_kb succeeds, but create_rag_instance fails
    mock_rag_manager.create_rag_instance.side_effect = RAGInitializationError("RAG setup failed")
    # Make asyncio.run propagate this error
    mock_asyncio_run.side_effect = lambda coro: asyncio.get_event_loop().run_until_complete(coro)


    shell_instance.onecmd(f"create {kb_name}")
    
    mock_kb_manager.create_kb.assert_called_once_with(kb_name, description=None)
    mock_rag_manager.create_rag_instance.assert_awaited_once_with(kb_name)
    output = shell_instance.mock_stdout.getvalue()
    assert f"Warning: Knowledge base '{kb_name}' created, but RAG initialization failed: RAG setup failed" in output

def test_do_create_invalid_args(shell_instance):
    """Test create with invalid arguments."""
    shell_instance.onecmd("create") # No name
    assert "Usage: create <name>" in shell_instance.mock_stdout.getvalue()
    shell_instance.mock_stdout.truncate(0); shell_instance.mock_stdout.seek(0) # Clear output
    shell_instance.onecmd('create kb1 "desc1" "extra arg"') # Too many args
    assert "Usage: create <name>" in shell_instance.mock_stdout.getvalue()


@patch('asyncio.run')
def test_do_list_success(mock_asyncio_run, shell_instance, mock_kb_manager):
    """Test successful listing of KBs."""
    kbs_data = {"kb1": "Description 1", "longkbname": "A longer description"}
    mock_kb_manager.list_kbs.return_value = kbs_data
    # Make asyncio.run return the future's result
    mock_asyncio_run.side_effect = lambda coro: asyncio.get_event_loop().run_until_complete(coro)


    shell_instance.onecmd("list")
    
    mock_kb_manager.list_kbs.assert_awaited_once()
    output = shell_instance.mock_stdout.getvalue()
    assert "Available knowledge bases:" in output
    assert "- kb1        : Description 1" in output # Check formatting
    assert "- longkbname : A longer description" in output

@patch('asyncio.run')
def test_do_list_empty(mock_asyncio_run, shell_instance, mock_kb_manager):
    """Test list when no KBs exist."""
    mock_kb_manager.list_kbs.return_value = {}
    mock_asyncio_run.side_effect = lambda coro: asyncio.get_event_loop().run_until_complete(coro)

    shell_instance.onecmd("list")
    assert "No knowledge bases found." in shell_instance.mock_stdout.getvalue()

@patch('asyncio.run')
def test_do_list_kb_error(mock_asyncio_run, shell_instance, mock_kb_manager):
    """Test list with KnowledgeBaseError."""
    mock_kb_manager.list_kbs.side_effect = KnowledgeBaseError("Cannot access directory")
    mock_asyncio_run.side_effect = lambda coro: asyncio.get_event_loop().run_until_complete(coro)

    shell_instance.onecmd("list")
    assert "Error listing knowledge bases: Cannot access directory" in shell_instance.mock_stdout.getvalue()


@patch('builtins.input', return_value='yes')
def test_do_delete_success(mock_input, shell_instance, mock_kb_manager, mock_rag_manager):
    """Test successful deletion of a KB."""
    kb_name = "deleteme"
    shell_instance.onecmd(f"delete {kb_name}")
    
    mock_input.assert_called_once_with(f"Are you sure you want to delete knowledge base '{kb_name}' and all its contents? (yes/no): ")
    mock_kb_manager.delete_kb.assert_called_once_with(kb_name)
    mock_rag_manager.remove_rag_instance.assert_called_once_with(kb_name)
    assert f"Knowledge base '{kb_name}' deleted successfully." in shell_instance.mock_stdout.getvalue()

@patch('builtins.input', return_value='no')
def test_do_delete_cancelled(mock_input, shell_instance, mock_kb_manager, mock_rag_manager):
    """Test deletion cancelled by user."""
    shell_instance.onecmd("delete anykb")
    assert "Deletion cancelled." in shell_instance.mock_stdout.getvalue()
    mock_kb_manager.delete_kb.assert_not_called()
    mock_rag_manager.remove_rag_instance.assert_not_called()

def test_do_delete_kb_not_found(shell_instance, mock_kb_manager):
    """Test delete when KB not found."""
    kb_name = "ghostkb"
    mock_kb_manager.delete_kb.side_effect = KnowledgeBaseNotFoundError
    with patch('builtins.input', return_value='yes'): # Assume user confirms
        shell_instance.onecmd(f"delete {kb_name}")
    assert f"Error: Knowledge base '{kb_name}' not found." in shell_instance.mock_stdout.getvalue()

def test_do_delete_kb_error(shell_instance, mock_kb_manager):
    """Test delete with general KnowledgeBaseError."""
    mock_kb_manager.delete_kb.side_effect = KnowledgeBaseError("Permission denied")
    with patch('builtins.input', return_value='yes'):
        shell_instance.onecmd("delete anykb")
    assert "Error deleting knowledge base: Permission denied" in shell_instance.mock_stdout.getvalue()

def test_do_delete_invalid_args(shell_instance):
    """Test delete with invalid arguments."""
    shell_instance.onecmd("delete") # No name
    assert "Usage: delete <name>" in shell_instance.mock_stdout.getvalue()


# --- Tests for KB Config Management (do_config) ---
# Merging logic from tests/test_shell_config.py

@pytest.fixture
def temp_kb_for_config(tmp_path: Path, mock_kb_manager):
    """Creates a temporary KB directory structure for config tests."""
    kb_name = "config_test_kb"
    kb_dir = tmp_path / kb_name
    kb_dir.mkdir()
    # Make get_kb_path return this specific path for this KB name
    def side_effect_get_kb_path(name_arg):
        if name_arg == kb_name:
            return kb_dir
        return MagicMock(spec=Path) # Default mock for other names
    mock_kb_manager.get_kb_path.side_effect = side_effect_get_kb_path
    return kb_name, kb_dir

def test_config_show_existing(shell_instance, temp_kb_for_config, mock_kb_manager):
    """Test 'config show <kb>' when config exists."""
    kb_name, kb_path = temp_kb_for_config
    config_file = kb_path / "config.yaml"
    test_config_content = {"mode": "test", "top_k": 10}
    with open(config_file, 'w') as f:
        yaml.dump(test_config_content, f)

    shell_instance.onecmd(f"config {kb_name} show")
    output = shell_instance.mock_stdout.getvalue()

    assert f"Config file path: {config_file.resolve()}" in output
    assert "--- Config Content ---" in output
    assert yaml.dump(test_config_content, default_flow_style=False, indent=2) in output

def test_config_show_default_subcommand(shell_instance, temp_kb_for_config):
    """Test 'config <kb>' defaults to 'show'."""
    kb_name, kb_path = temp_kb_for_config
    config_file = kb_path / "config.yaml"
    test_config_content = {"mode": "default_show"}
    with open(config_file, 'w') as f:
        yaml.dump(test_config_content, f)
    
    shell_instance.onecmd(f"config {kb_name}") # No subcommand
    output = shell_instance.mock_stdout.getvalue()
    assert f"Config file path: {config_file.resolve()}" in output
    assert yaml.dump(test_config_content, default_flow_style=False, indent=2) in output

def test_config_show_missing_config(shell_instance, temp_kb_for_config):
    """Test 'config show <kb>' when config.yaml is missing."""
    kb_name, kb_path = temp_kb_for_config
    # Ensure config.yaml does not exist
    if (kb_path / "config.yaml").exists(): (kb_path / "config.yaml").unlink()

    shell_instance.onecmd(f"config {kb_name} show")
    output = shell_instance.mock_stdout.getvalue()
    assert f"Config file path: {(kb_path / 'config.yaml').resolve()}" in output
    assert "Config file does not exist." in output

def test_config_show_nonexistent_kb(shell_instance, mock_kb_manager):
    """Test 'config show <kb>' for a KB that doesn't exist."""
    mock_kb_manager.get_kb_path.side_effect = KnowledgeBaseNotFoundError
    shell_instance.onecmd("config non_existent_kb show")
    assert "Error: Knowledge base 'non_existent_kb' not found." in shell_instance.mock_stdout.getvalue()

@patch('knowledge_mcp.shell.subprocess.run')
@patch('knowledge_mcp.shell.os.getenv')
def test_config_edit_existing(mock_getenv, mock_subprocess_run, shell_instance, temp_kb_for_config):
    kb_name, kb_path = temp_kb_for_config
    config_file = kb_path / "config.yaml"
    config_file.write_text("mode: edit_me")
    mock_getenv.side_effect = lambda key, default=None: 'vim' if key in ['EDITOR', 'VISUAL'] else default

    shell_instance.onecmd(f"config {kb_name} edit")
    output = shell_instance.mock_stdout.getvalue()
    mock_subprocess_run.assert_called_once_with(['vim', str(config_file)], check=True)
    assert f"Attempting to open '{config_file.resolve()}' with editor 'vim'" in output

@patch('knowledge_mcp.shell.subprocess.run')
def test_config_edit_missing_config(mock_subprocess_run, shell_instance, temp_kb_for_config):
    kb_name, kb_path = temp_kb_for_config
    if (kb_path / "config.yaml").exists(): (kb_path / "config.yaml").unlink() # Ensure no config
    
    shell_instance.onecmd(f"config {kb_name} edit")
    output = shell_instance.mock_stdout.getvalue()
    mock_subprocess_run.assert_not_called()
    assert f"Error: Config file '{(kb_path / 'config.yaml')}' does not exist" in output

@patch('knowledge_mcp.shell.subprocess.run', side_effect=FileNotFoundError("Editor not found"))
@patch('knowledge_mcp.shell.os.getenv')
def test_config_edit_editor_not_found(mock_getenv, mock_subprocess_run, shell_instance, temp_kb_for_config):
    kb_name, kb_path = temp_kb_for_config
    (kb_path / "config.yaml").write_text("content") # Config must exist for editor attempt
    mock_getenv.return_value = 'fake_editor'
    shell_instance.onecmd(f"config {kb_name} edit")
    assert "Error: Editor 'fake_editor' not found." in shell_instance.mock_stdout.getvalue()

@patch('knowledge_mcp.shell.subprocess.run', side_effect=subprocess.CalledProcessError(1, 'cmd'))
@patch('knowledge_mcp.shell.os.getenv')
def test_config_edit_editor_error(mock_getenv, mock_subprocess_run, shell_instance, temp_kb_for_config):
    kb_name, kb_path = temp_kb_for_config
    (kb_path / "config.yaml").write_text("content")
    mock_getenv.return_value = 'bad_editor'
    shell_instance.onecmd(f"config {kb_name} edit")
    assert "Error running editor 'bad_editor'" in shell_instance.mock_stdout.getvalue()

def test_config_invalid_subcommand(shell_instance, temp_kb_for_config):
    kb_name, _ = temp_kb_for_config
    shell_instance.onecmd(f"config {kb_name} foobar")
    assert "Error: Unknown config subcommand 'foobar'" in shell_instance.mock_stdout.getvalue()

def test_config_missing_kb_name(shell_instance):
    shell_instance.onecmd("config")
    assert "Usage: config <kb_name> [show|edit]" in shell_instance.mock_stdout.getvalue()


# --- Tests for Document Management Commands ---

@patch('asyncio.run')
def test_do_add_success(mock_asyncio_run, shell_instance, mock_document_manager, tmp_path):
    """Test successful document addition."""
    kb_name = "doc_kb"
    file_content = "Test document content."
    doc_file = tmp_path / "test_doc.txt"
    doc_file.write_text(file_content)
    
    mock_document_manager.add.return_value = "doc_id_123"
    # Make asyncio.run return the future's result
    mock_asyncio_run.side_effect = lambda coro: asyncio.get_event_loop().run_until_complete(coro)


    shell_instance.onecmd(f"add {kb_name} {str(doc_file)}")
    
    mock_document_manager.add.assert_awaited_once_with(doc_file, kb_name)
    output = shell_instance.mock_stdout.getvalue()
    assert f"Adding document '{doc_file.name}' to KB '{kb_name}'..." in output
    assert "Document added successfully with ID: doc_id_123" in output

@patch('asyncio.run')
def test_do_add_kb_not_found(mock_asyncio_run, shell_instance, mock_document_manager, tmp_path):
    """Test add when KB not found."""
    doc_file = tmp_path / "test.txt"; doc_file.touch()
    mock_document_manager.add.side_effect = KnowledgeBaseNotFoundError("Target KB missing")
    mock_asyncio_run.side_effect = lambda coro: asyncio.get_event_loop().run_until_complete(coro)

    shell_instance.onecmd(f"add missing_kb {str(doc_file)}")
    assert "Error: Knowledge base 'missing_kb' not found." in shell_instance.mock_stdout.getvalue()

def test_do_add_file_not_found(shell_instance):
    """Test add when document file not found."""
    shell_instance.onecmd("add any_kb /path/to/ghost_file.txt")
    assert "Error: File not found at '/path/to/ghost_file.txt'" in shell_instance.mock_stdout.getvalue()

def test_do_add_invalid_args(shell_instance):
    shell_instance.onecmd("add mykb") # Missing file_path
    assert "Usage: add <kb_name> <file_path>" in shell_instance.mock_stdout.getvalue()
    shell_instance.mock_stdout.truncate(0); shell_instance.mock_stdout.seek(0)
    shell_instance.onecmd("add") # Missing kb_name and file_path
    assert "Usage: add <kb_name> <file_path>" in shell_instance.mock_stdout.getvalue()


def test_do_remove_success(shell_instance, mock_rag_manager):
    """Test successful document removal."""
    kb_name = "doc_kb_rem"
    doc_id = "doc_to_remove"
    mock_rag_manager.remove_document.return_value = True # Simulate successful removal

    shell_instance.onecmd(f"remove {kb_name} {doc_id}")
    
    mock_rag_manager.remove_document.assert_called_once_with(kb_name, doc_id)
    assert f"Document '{doc_id}' removed successfully." in shell_instance.mock_stdout.getvalue()

def test_do_remove_not_found_in_kb(shell_instance, mock_rag_manager):
    """Test remove when document not found in KB (manager returns False)."""
    kb_name = "doc_kb_rem"
    doc_id = "non_existent_doc"
    mock_rag_manager.remove_document.return_value = False # Simulate doc not found

    shell_instance.onecmd(f"remove {kb_name} {doc_id}")
    assert f"Document '{doc_id}' not found in KB '{kb_name}' or could not be removed." in shell_instance.mock_stdout.getvalue()

def test_do_remove_kb_not_found(shell_instance, mock_rag_manager):
    """Test remove when KB itself not found."""
    kb_name = "ghost_kb_rem"
    doc_id = "any_doc"
    mock_rag_manager.remove_document.side_effect = KnowledgeBaseNotFoundError(f"KB {kb_name} missing")
    
    shell_instance.onecmd(f"remove {kb_name} {doc_id}")
    assert f"Error: Knowledge base '{kb_name}' not found." in shell_instance.mock_stdout.getvalue()

def test_do_remove_invalid_args(shell_instance):
    shell_instance.onecmd("remove mykb") # Missing doc_id
    assert "Usage: remove <kb_name> <doc_id>" in shell_instance.mock_stdout.getvalue()


# --- Tests for Query Command ---

@patch('asyncio.run')
def test_do_query_success(mock_asyncio_run, shell_instance, mock_rag_manager):
    """Test successful query execution."""
    kb_name = "querykb"
    query_text = "What is the answer?"
    mock_rag_manager.query.return_value = "The answer is 42."
    mock_asyncio_run.side_effect = lambda coro: asyncio.get_event_loop().run_until_complete(coro)


    shell_instance.onecmd(f"query {kb_name} {query_text}")

    mock_rag_manager.query.assert_awaited_once_with(kb_name, query_text)
    output = shell_instance.mock_stdout.getvalue()
    assert f"Querying KB '{kb_name}' with: \"{query_text}\"" in output
    assert " [running query] ... [done]" in output
    assert "--- Query Result ---" in output
    assert "The answer is 42." in output
    assert "--- End Result ---" in output

@patch('asyncio.run')
def test_do_query_kb_not_found(mock_asyncio_run, shell_instance, mock_rag_manager):
    """Test query with KnowledgeBaseNotFoundError."""
    kb_name = "ghost_query_kb"
    query_text = "Query for ghost."
    mock_rag_manager.query.side_effect = KnowledgeBaseNotFoundError(f"KB {kb_name} does not exist.")
    mock_asyncio_run.side_effect = lambda coro: asyncio.get_event_loop().run_until_complete(coro)

    shell_instance.onecmd(f"query {kb_name} {query_text}")
    output = shell_instance.mock_stdout.getvalue()
    assert " [running query] ... [failed]" in output
    assert f"Error querying KB '{kb_name}': KB {kb_name} does not exist." in output

@patch('asyncio.run')
def test_do_query_rag_error(mock_asyncio_run, shell_instance, mock_rag_manager):
    """Test query with RAGManagerError."""
    kb_name = "rag_error_kb"
    query_text = "This query will fail."
    mock_rag_manager.query.side_effect = RAGManagerError("RAG system exploded.")
    mock_asyncio_run.side_effect = lambda coro: asyncio.get_event_loop().run_until_complete(coro)

    shell_instance.onecmd(f"query {kb_name} {query_text}")
    output = shell_instance.mock_stdout.getvalue()
    assert f"Error querying KB '{kb_name}': RAG system exploded." in output


def test_do_query_invalid_args(shell_instance):
    shell_instance.onecmd("query mykb") # Missing query_text
    assert "Usage: query <kb_name> <query_text>" in shell_instance.mock_stdout.getvalue()
    shell_instance.mock_stdout.truncate(0); shell_instance.mock_stdout.seek(0)
    shell_instance.onecmd("query") # Missing kb_name and query_text
    assert "Usage: query <kb_name> <query_text>" in shell_instance.mock_stdout.getvalue()

"""
This test suite covers the Shell class from knowledge_mcp.shell.

Key areas tested:
- Initialization: Correct setup with mocked managers, DocumentManager creation, background loop start.
- Basic Commands: do_exit, do_EOF (return True, print message, stop loop), do_clear (mock os.system).
- KB Management:
    - do_create: Success (with/without description), KBExistsError, KBError, RAGInitializationError (warning), arg parsing.
    - do_list: Success (empty/with data), KBError.
    - do_delete: Success (yes/no input), KBNotFoundError, KBError, arg parsing.
- KB Config Management (do_config):
    - show: Existing config, missing config, KB not found.
    - edit: Existing config, missing config, editor not found, editor error.
    - Invalid subcommand, missing KB name. (Merged from tests/test_shell_config.py)
- Document Management:
    - do_add: Success, KBNotFoundError, FileNotFoundError, arg parsing.
    - do_remove: Success, document not found, KBNotFoundError, arg parsing.
- Query Command (do_query):
    - Success, error handling (KBNotFoundError, RAG-related errors), arg parsing.

Utilized pytest, unittest.mock (MagicMock, AsyncMock, patch), io.StringIO for stdout,
and fixtures for dependencies. Asyncio.run is patched for commands that use it.
The background thread logic is minimally tested by checking Thread instantiation and start,
and ensuring _stop_background_loop is callable (especially for exit commands).
"""
