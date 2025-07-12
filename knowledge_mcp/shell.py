import asyncio
import cmd
import shlex
import logging
import threading
from pathlib import Path
import yaml
import os
import subprocess

from knowledge_mcp.knowledgebases import KnowledgeBaseManager, KnowledgeBaseExistsError, KnowledgeBaseNotFoundError, KnowledgeBaseError
from knowledge_mcp.rag import RagManager, RAGInitializationError, ConfigurationError, RAGManagerError, DeletionResult
from knowledge_mcp.documents import DocumentManager

logger = logging.getLogger(__name__)

class Shell(cmd.Cmd):
    """Interactive shell for Knowledge MCP."""
    intro = 'Welcome to the Knowledge MCP shell. Type help or ? to list commands.\n'
    prompt = '(kbmcp) '

    def __init__(self, kb_manager: KnowledgeBaseManager, rag_manager: RagManager, stdout=None):
        super().__init__(stdout=stdout)
        self.kb_manager = kb_manager
        self.rag_manager = rag_manager
        self.document_manager = DocumentManager(rag_manager)
        self._loop = None
        self._running_servers = {}  # Track running lightrag servers: {kb_name: subprocess.Popen}
        self._start_background_loop()

    def _run_background_loop(self):
        """Target function for the background thread to run the event loop."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_forever()
        finally:
            self._loop.close()
            logger.info("Background thread stopped.")

    def _start_background_loop(self):
        """Starts the background thread."""
        self._async_thread = threading.Thread(
            target=self._run_background_loop,
            daemon=True, # Allow program to exit even if thread is running
            name="AsyncLoopThread"
        )
        self._async_thread.start()
        # Wait a bit for the loop to be created
        import time
        time.sleep(0.1)
        logger.info("Background thread started.")

    def _stop_background_loop(self):
        """Signals the background thread to stop."""
        if hasattr(self, '_loop') and self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if hasattr(self, '_async_thread') and self._async_thread.is_alive():
            logger.info("Stopping background thread...")
            # Wait for the thread to finish
            self._async_thread.join()
            logger.info("Background thread joined.")
        else:
            logger.info("Background thread not running or not initialized.")

    def _run_async_task(self, coro):
        """Run an async task in the background event loop and wait for the result."""
        if not self._loop or self._loop.is_closed():
            raise RuntimeError("Background event loop is not running")
        
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()  # This will block until the coroutine completes

    # --- Basic Commands ---

    def do_exit(self, arg: str) -> bool:
        """Exit the shell."""
        print("Exiting shell.")
        self._cleanup_servers()
        self._stop_background_loop()
        return True # Returning True stops the cmdloop

    def do_EOF(self, arg: str) -> bool:
        """Exit the shell when EOF (Ctrl+D) is received."""
        print() # Print a newline for cleaner exit
        self._cleanup_servers()
        self._stop_background_loop()
        return self.do_exit(arg)

    # --- KB Management Commands ---

    def do_create(self, arg: str):
        """Create a new knowledge base. Usage: create <name> ["description"]"""
        try:
            args = shlex.split(arg)
            if not 1 <= len(args) <= 2:
                print('Usage: create <name> ["description"]')
                return

            name = args[0]
            description = args[1] if len(args) == 2 else None

            self.kb_manager.create_kb(name, description=description) # Pass description
            # Optionally initialize the RAG instance immediately (if desired)
            # Consider if this should be async or handled differently
            # For now, assuming synchronous initialization might block, but let's keep it simple
            # If create_rag_instance becomes async, this needs `asyncio.run`
            try:
                print(f"Initializing RAG instance for '{name}'...")
                # Assuming create_rag_instance is now async
                self._run_async_task(self.rag_manager.create_rag_instance(name))
                print(f"Knowledge base '{name}' created and RAG instance initialized successfully.")
            except (RAGInitializationError, ConfigurationError, RAGManagerError) as rag_e:
                logger.warning(f"KB '{name}' created, but failed to initialize RAG instance: {rag_e}")
                print(f"Warning: Knowledge base '{name}' created, but RAG initialization failed: {rag_e}")
                print("You may need to configure LLM/Embedding settings before using this KB.")
            # Removed original asyncio.run(self.rag_manager.create_rag_instance(name))
            # print(f"Knowledge base '{name}' created successfully.")
        except KnowledgeBaseExistsError:
            print(f"Error: Knowledge base '{name}' already exists.")
        except KnowledgeBaseError as e:
            print(f"Error creating knowledge base: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error in create: {e}")
            print(f"An unexpected error occurred: {e}")

    def do_list(self, arg: str):
        """List all available knowledge bases and their descriptions."""
        try:
            # list_kbs is now async, run it in the event loop
            kbs_with_desc = self._run_async_task(self.kb_manager.list_kbs())

            if not kbs_with_desc:
                print("No knowledge bases found.")
                return

            print("Available knowledge bases:")
            # Determine max name length for alignment
            max_len = 0
            if kbs_with_desc: # Check if dict is not empty
                 max_len = max(len(name) for name in kbs_with_desc.keys()) if kbs_with_desc else 0

            for name, desc in kbs_with_desc.items():
                print(f"- {name:<{max_len}} : {desc}") # Print name and description

        except KnowledgeBaseError as e:
            print(f"Error listing knowledge bases: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error in list: {e}")
            print(f"An unexpected error occurred: {e}")

    def do_delete(self, arg: str):
        """Delete a knowledge base. Usage: delete <name>"""
        try:
            args = shlex.split(arg)
            if len(args) != 1:
                print("Usage: delete <name>")
                return
            name = args[0]
            confirm = input(f"Are you sure you want to delete knowledge base '{name}' and all its contents? (yes/no): ").lower()
            if confirm == 'yes':
                self.kb_manager.delete_kb(name)
                self.rag_manager.remove_rag_instance(name)
                print(f"Knowledge base '{name}' deleted successfully.")
            else:
                print("Deletion cancelled.")
        except KnowledgeBaseNotFoundError:
            print(f"Error: Knowledge base '{name}' not found.")
        except KnowledgeBaseError as e:
            print(f"Error deleting knowledge base: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error in delete: {e}")
            print(f"An unexpected error occurred: {e}")

    # --- KB Config Management ---

    def do_config(self, arg: str):
        """Manage knowledge base configuration.
        Usage: config <kb_name> [show|edit]

        show: Display the path and content of the KB's config.yaml (default).
        edit: Open the KB's config.yaml in the default editor.
        """
        try:
            args = shlex.split(arg)
            if not 1 <= len(args) <= 2:
                print("Usage: config <kb_name> [show|edit]")
                return

            kb_name = args[0]
            subcommand = args[1].lower() if len(args) == 2 else "show"

            if subcommand not in ["show", "edit"]:
                print(f"Error: Unknown config subcommand '{args[1]}'. Use 'show' or 'edit'.", file=self.stdout)
                return

            # Get KB path and config path
            try:
                kb_path = self.kb_manager.get_kb_path(kb_name)
                if not kb_path.is_dir(): # Should be caught by get_kb_path if strict=True, but double check
                    print(f"Error: Knowledge base '{kb_name}' not found or is not a directory.", file=self.stdout)
                    return
            except KnowledgeBaseNotFoundError:
                 print(f"Error: Knowledge base '{kb_name}' not found.", file=self.stdout)
                 return

            config_path = kb_path / "config.yaml"

            # --- Handle 'show' subcommand ---
            if subcommand == "show":
                print(f"Config file path: {config_path.resolve()}", file=self.stdout)
                if config_path.is_file():
                    try:
                        with open(config_path, 'r', encoding='utf-8') as f:
                            content = yaml.safe_load(f)
                        print("--- Config Content ---", file=self.stdout)
                        print(yaml.dump(content, default_flow_style=False, indent=2), file=self.stdout)
                        print("--- End Config Content ---", file=self.stdout)
                    except (IOError, yaml.YAMLError) as e:
                        print(f"Error reading or parsing config file: {e}", file=self.stdout)
                else:
                    print(f"Config file does not exist. KB '{kb_name}' will use default query parameters.", file=self.stdout)

            # --- Handle 'edit' subcommand ---
            elif subcommand == "edit":
                if not config_path.is_file():
                    print(f"Error: Config file '{config_path}' does not exist for KB '{kb_name}'.", file=self.stdout)
                    # Future improvement: Offer to create it?
                    # For now, just error out.
                    return

                editor = os.getenv('EDITOR') or os.getenv('VISUAL') or 'nano' # Default to nano/vim for mac/linux
                print(f"Attempting to open '{config_path.resolve()}' with editor '{editor}'...", file=self.stdout)
                try:
                    # Use check=True to raise CalledProcessError on failure
                    subprocess.run([editor, str(config_path)], check=True)
                    print("Editor closed.", file=self.stdout)
                except FileNotFoundError:
                    print(f"Error: Editor '{editor}' not found. Set EDITOR or VISUAL environment variable.", file=self.stdout)
                except subprocess.CalledProcessError as e:
                    print(f"Error running editor '{editor}': {e}", file=self.stdout)
                except Exception as e:
                    logger.exception(f"Unexpected error opening editor: {e}")
                    print(f"An unexpected error occurred while trying to open the editor: {e}", file=self.stdout)

        except Exception as e:
            logger.exception(f"Unexpected error in config command: {e}")
            print(f"An unexpected error occurred: {e}")

    # --- Document Management Commands ---

    def do_add(self, arg: str):
        """Add a document to a knowledge base. Usage: add <kb_name> <file_path> [method]"""
        try:
            args = shlex.split(arg)
            if not 2 <= len(args) <= 3:
                print("Usage: add <kb_name> <file_path> [method:auto|multimodal|text|txt|ocr]")
                return

            kb_name = args[0]
            file_path_str = args[1]
            parse_method = args[2] if len(args) == 3 else "auto"

            file_path = Path(file_path_str)
            if not file_path.is_file():
                print(f"Error: File not found at '{file_path_str}'")
                return

            # Map legacy parse_method values to new method parameter
            method_mapping = {
                "auto": "multimodal",
                "multimodal": "multimodal", 
                "txt": "text",
                "text": "text",
                "ocr": "text"  # OCR is closest to text-only processing
            }
            
            method = method_mapping.get(parse_method, "multimodal")
            print(f"Adding document '{file_path.name}' to KB '{kb_name}' using '{method}' processing...")
            
            self._run_async_task(self.document_manager.add(file_path, kb_name, method=method))
            print(f"Document added successfully using '{method}' processing.")

        except KnowledgeBaseNotFoundError:
            print(f"Error: Knowledge base '{kb_name}' not found.")
        except FileNotFoundError:
            print(f"Error: Document file path '{file_path_str}' not found.")
        except Exception as e:
            logger.exception(f"Unexpected error in add: {e}")
            print(f"An unexpected error occurred: {e}")

    def do_add_multimodal(self, arg: str):
        """Add a document using multimodal processing. Usage: add_multimodal <kb_name> <file_path>"""
        try:
            args = shlex.split(arg)
            if len(args) != 2:
                print("Usage: add_multimodal <kb_name> <file_path>")
                return

            kb_name = args[0]
            file_path_str = args[1]

            file_path = Path(file_path_str)
            if not file_path.is_file():
                print(f"Error: File not found at '{file_path_str}'")
                return

            print(f"Adding document '{file_path.name}' to KB '{kb_name}' using multimodal processing...")
            self._run_async_task(self.document_manager.add_multimodal(file_path, kb_name))
            print("Document added successfully using multimodal processing.")

        except KnowledgeBaseNotFoundError:
            print(f"Error: Knowledge base '{kb_name}' not found.")
        except FileNotFoundError:
            print(f"Error: Document file path '{file_path_str}' not found.")
        except Exception as e:
            logger.exception(f"Unexpected error in add_multimodal: {e}")
            print(f"An unexpected error occurred: {e}")

    def do_add_text(self, arg: str):
        """Add a document using text-only processing. Usage: add_text <kb_name> <file_path>"""
        try:
            args = shlex.split(arg)
            if len(args) != 2:
                print("Usage: add_text <kb_name> <file_path>")
                return

            kb_name = args[0]
            file_path_str = args[1]

            file_path = Path(file_path_str)
            if not file_path.is_file():
                print(f"Error: File not found at '{file_path_str}'")
                return

            print(f"Adding document '{file_path.name}' to KB '{kb_name}' using text-only processing...")
            self._run_async_task(self.document_manager.add_text_only(file_path, kb_name))
            print("Document added successfully using text-only processing.")

        except KnowledgeBaseNotFoundError:
            print(f"Error: Knowledge base '{kb_name}' not found.")
        except FileNotFoundError:
            print(f"Error: Document file path '{file_path_str}' not found.")
        except Exception as e:
            logger.exception(f"Unexpected error in add_text: {e}")
            print(f"An unexpected error occurred: {e}")

    def do_remove(self, arg: str):
        """Remove a document from a knowledge base by its ID. Usage: remove <kb_name> <doc_id>"""
        try:
            args = shlex.split(arg)
            if len(args) != 2:
                print("Usage: remove <kb_name> <doc_id>")
                return

            kb_name = args[0]
            doc_id = args[1]

            print(f"Removing document '{doc_id}' from KB '{kb_name}'...")
            result = self._run_async_task(self.rag_manager.remove_document(kb_name, doc_id))
            print(f"Document '{doc_id}' removal result: {result.status}. {result.message}")
        except KnowledgeBaseNotFoundError:
            print(f"Error: Knowledge base '{kb_name}' not found.")
        except Exception as e:
             logger.exception(f"Unexpected error in remove: {e}")
             print(f"An unexpected error occurred: {e}")

    # --- Query Commands --- 

    def do_query(self, arg: str) -> None:
        """Queries a specified knowledge base. Usage: query <kb_name> <query_text>"""
        args = arg.split(maxsplit=1)
        if len(args) < 2:
            print("Usage: query <kb_name> <query_text>", file=self.stdout)
            return

        kb_name = args[0]
        query_text = args[1]

        print(f"\nQuerying KB '{kb_name}' with: \"{query_text}\"", file=self.stdout)
        print(" [running query] ...", end="", flush=True, file=self.stdout)

        try:
            # Call the synchronous query method directly, passing kwargs
            result = self._run_async_task(self.rag_manager.query(kb_name, query_text))
            print(" [done]", file=self.stdout) # Indicate completion
            print("\n--- Query Result ---", file=self.stdout)
            print(result, file=self.stdout) # result should already be a string or printable
            print("--- End Result ---", file=self.stdout)
        except (KnowledgeBaseNotFoundError, RAGInitializationError, ConfigurationError, RAGManagerError) as e:
            # Catch specific known errors from RagManager
            print("\n [failed]", file=self.stdout)
            print(f"\nError querying KB '{kb_name}': {e}", file=self.stdout)
            logger.error(f"Query failed for {kb_name}: {e}") # Log specific known errors
        except Exception as e: # Catch any other unexpected errors
            print("\n [failed]", file=self.stdout)
            print(f"An unexpected error occurred during the query: {e}", file=self.stdout)
            logger.exception("Unexpected query error") # Log full traceback for unknowns

    def do_clear(self, arg: str): 
        """Clear the screen."""
        os.system('cls' if os.name == 'nt' else 'clear')

    def do_server(self, arg: str):
        """Manage LightRAG servers. Usage: server <kb_name> start|stop|status"""
        try:
            args = shlex.split(arg)
            if len(args) != 2:
                print('Usage: server <kb_name> start|stop|status')
                return

            kb_name, action = args
            
            if action == 'start':
                self._start_server(kb_name)
            elif action == 'stop':
                self._stop_server(kb_name)
            elif action == 'status':
                self._server_status(kb_name)
            else:
                print(f"Unknown action '{action}'. Use: start, stop, or status")
                
        except Exception as e:
            logger.exception(f"Error in server command: {e}")
            print(f"Error: {e}")

    def _start_server(self, kb_name: str):
        """Start a LightRAG server for the specified knowledge base."""
        try:
            # Check if knowledge base exists
            kb_path = self.kb_manager.base_dir / kb_name
            if not kb_path.exists():
                print(f"Error: Knowledge base '{kb_name}' does not exist.")
                return

            # Check if server is already running
            if kb_name in self._running_servers:
                process = self._running_servers[kb_name]
                if process.poll() is None:  # Process is still running
                    print(f"Server for '{kb_name}' is already running (PID: {process.pid})")
                    return
                else:
                    # Process has terminated, remove from tracking
                    del self._running_servers[kb_name]

            # Get config for environment variables
            from knowledge_mcp.config import Config
            config = Config.get_instance()
            
            # Set up environment variables from config
            env = os.environ.copy()
            
            # LLM configuration
            llm_config = config.lightrag.llm
            env['LLM_BINDING'] = llm_config.provider
            env['LLM_MODEL'] = llm_config.model_name
            env['LLM_BINDING_API_KEY'] = llm_config.api_key
            if llm_config.api_base:
                env['LLM_BINDING_HOST'] = llm_config.api_base
            
            # Embedding configuration
            embedding_config = config.lightrag.embedding
            env['EMBEDDING_BINDING'] = embedding_config.provider
            env['EMBEDDING_MODEL'] = embedding_config.model_name
            env['EMBEDDING_BINDING_API_KEY'] = embedding_config.api_key
            if embedding_config.api_base:
                env['EMBEDDING_BINDING_HOST'] = embedding_config.api_base
            env['EMBEDDING_DIM'] = str(embedding_config.embedding_dim)

            # Start the lightrag-server process
            cmd = ['lightrag-server', '--working-dir', str(kb_path)]
            print(f"Starting LightRAG server for '{kb_name}'...")
            print(f"Command: {' '.join(cmd)}")
            print(f"Working directory: {kb_path}")
            
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Track the running process
            self._running_servers[kb_name] = process
            print(f"Server started for '{kb_name}' (PID: {process.pid})")
            print("Note: Server output is captured. Use 'server <kb_name> status' to check if it's running.")
            
        except Exception as e:
            logger.exception(f"Failed to start server for {kb_name}: {e}")
            print(f"Failed to start server: {e}")

    def _stop_server(self, kb_name: str):
        """Stop the LightRAG server for the specified knowledge base."""
        try:
            if kb_name not in self._running_servers:
                print(f"No server running for '{kb_name}'")
                return

            process = self._running_servers[kb_name]
            if process.poll() is not None:
                # Process has already terminated
                print(f"Server for '{kb_name}' is not running")
                del self._running_servers[kb_name]
                return

            print(f"Stopping server for '{kb_name}' (PID: {process.pid})...")
            process.terminate()
            
            # Wait for graceful shutdown
            try:
                process.wait(timeout=5)
                print(f"Server for '{kb_name}' stopped gracefully")
            except subprocess.TimeoutExpired:
                print(f"Server for '{kb_name}' did not stop gracefully, forcing termination...")
                process.kill()
                process.wait()
                print(f"Server for '{kb_name}' forcefully terminated")
            
            del self._running_servers[kb_name]
            
        except Exception as e:
            logger.exception(f"Failed to stop server for {kb_name}: {e}")
            print(f"Failed to stop server: {e}")

    def _server_status(self, kb_name: str):
        """Check the status of the LightRAG server for the specified knowledge base."""
        try:
            if kb_name not in self._running_servers:
                print(f"No server tracked for '{kb_name}'")
                return

            process = self._running_servers[kb_name]
            if process.poll() is None:
                print(f"Server for '{kb_name}' is running (PID: {process.pid})")
            else:
                print(f"Server for '{kb_name}' has terminated (exit code: {process.returncode})")
                del self._running_servers[kb_name]
                
        except Exception as e:
            logger.exception(f"Failed to check server status for {kb_name}: {e}")
            print(f"Failed to check server status: {e}")

    def _cleanup_servers(self):
        """Stop all running servers before exiting."""
        if not self._running_servers:
            return
            
        print("Stopping running servers...")
        for kb_name in list(self._running_servers.keys()):
            try:
                process = self._running_servers[kb_name]
                if process.poll() is None:  # Still running
                    print(f"Stopping server for '{kb_name}'...")
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                del self._running_servers[kb_name]
            except Exception as e:
                logger.exception(f"Error stopping server for {kb_name}: {e}")
        print("All servers stopped.")

    def help_server(self):
        """Help for the server command."""
        print("""Manage LightRAG servers for knowledge bases.

Usage:
  server <kb_name> start   - Start a LightRAG server for the knowledge base
  server <kb_name> stop    - Stop the running server for the knowledge base
  server <kb_name> status  - Check if the server is running

Examples:
  server csl start    - Start server for 'csl' knowledge base
  server csl status   - Check if 'csl' server is running
  server csl stop     - Stop the 'csl' server

Note: The server uses configuration from config.yaml instead of .env files.
Environment variables are set automatically from your LLM and embedding config.""")