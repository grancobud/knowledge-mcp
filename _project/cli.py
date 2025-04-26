# /Users/olaf/work/projects/knowledge-mcp/knowledge_mcp/cli.py
import argparse
import logging
import sys
from pathlib import Path
import uvicorn  # Keep for potential future server use # noqa: F401
import time     # Keep for placeholder serve mode

# Updated relative imports
from .config import Config, load_and_validate_config
from .knowledgebases import KnowledgeBaseManager # Updated module name
from .rag import RagManager # Updated module name
from .shell import Shell # Updated module and class name

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def initialize_components(config: Config) -> tuple[KnowledgeBaseManager, RagManager]:
    """Initialize and return manager instances."""
    logger.info("Initializing components...")
    kb_manager = KnowledgeBaseManager(config)
    # Initialize RagManager - this might load existing KBs
    rag_manager = RagManager(config, kb_manager)
    logger.info("Components initialized.")
    return kb_manager, rag_manager

def run_serve_mode(config: Config):
    """Runs the application in server mode."""
    logger.info("Starting in serve mode...")
    kb_manager, rag_manager = initialize_components(config)
    # Placeholder for starting the FastMCP server
    logger.info("Starting FastMCP server...")
    # Example: await run_server(kb_manager, rag_manager, config)
    print("MCP Server running (placeholder)... Press Ctrl+C to exit.")
    # Keep running or use uvicorn/hypercorn to run the server
    # For now, just keeps the process alive conceptually
    try:
        # Simulate server running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Server stopped.")

def run_manage_mode(config: Config):
    """Runs the application in management mode."""
    logger.info("Starting in manage mode...")
    kb_manager, rag_manager = initialize_components(config)

    # Placeholder for starting the FastMCP server in the background
    logger.info("Starting FastMCP server in background (placeholder)...")
    # Example: asyncio.create_task(run_server(kb_manager, rag_manager, config))
    # Or use threading

    logger.info("Starting management shell...")
    # Instantiate and run the interactive shell
    shell = Shell(kb_manager, rag_manager) # Use the renamed Shell class
    try:
        shell.cmdloop()
    except KeyboardInterrupt:
        # Catch Ctrl+C during cmdloop if needed, though EOF (Ctrl+D) is handled by the shell
        print("\nExiting management shell (KeyboardInterrupt).")
    finally:
        # Stop background server if necessary
        logger.info("Stopping background server (placeholder)...")
        logger.info("Manage mode finished.")

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Knowledge Base MCP Server and Management Tool")
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="config.yml", # Consider making default relative to project root if cli is run from there
        help="Path to the configuration file (default: config.yml)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True,
                                       help='Available modes: serve, manage')

    # Serve command
    parser_serve = subparsers.add_parser("serve", help="Run the MCP server")
    # Add serve-specific arguments if needed later
    parser_serve.set_defaults(func=run_serve_mode)

    # Manage command
    parser_manage = subparsers.add_parser("manage", help="Run the interactive management shell")
    # Add manage-specific arguments if needed later
    parser_manage.set_defaults(func=run_manage_mode)

    args = parser.parse_args()

    # Load config - config path might need adjustment depending on CWD
    # If cli.py is run via `python -m knowledge_mcp.cli`, paths relative to project root might be okay.
    try:
        # If config is expected relative to project root, and cli.py is in the package,
        # we might need to adjust how the default path is handled or make it absolute.
        # For now, assume it's run from project root or path is absolute.
        config_path = Path(args.config)
        config = load_and_validate_config(config_path)
        logger.info(f"Successfully loaded config from {config_path.resolve()}")
    except FileNotFoundError:
        # Try searching relative to the cli script's parent dir? Or require absolute path?
        logger.critical(f"Configuration file not found: {args.config}")
        sys.exit(1)
    except (ValueError, RuntimeError) as e:
        logger.critical(f"Failed to load or validate configuration: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"An unexpected error occurred during configuration loading: {e}")
        sys.exit(1)

    # Execute the function associated with the chosen command
    args.func(config)

if __name__ == "__main__":
    # This allows running the cli directly for development,
    # but entry point script is preferred for installation.
    main()

    # --- Error Handling ---
    def default(self, line: str):
        """Called on an unknown command."""
        print(f"Unknown command: {line}. Type 'help' for available commands.")

    def emptyline(self):
        """Called when an empty line is entered. Does nothing."""
        pass
