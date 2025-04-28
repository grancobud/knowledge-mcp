import argparse
import logging
import logging.config
import sys

# Updated relative imports
from knowledge_mcp.config import Config
from knowledge_mcp.knowledgebases import KnowledgeBaseManager # Updated module name
from knowledge_mcp.rag import RagManager # Updated module name
from knowledge_mcp.shell import Shell # Updated module and class name
from knowledge_mcp.mcp_server import MCP # Import class and mcp instance

logger = logging.getLogger(__name__)

def initialize_components(config: Config) -> tuple[KnowledgeBaseManager, RagManager]:
    """Initialize and return manager instances."""
    logger.info("Initializing components...")
    kb_manager = KnowledgeBaseManager(config)
    rag_manager = RagManager(config, kb_manager)
    logger.info("Components initialized.")
    return kb_manager, rag_manager

def run_mcp_mode():
    """Runs the application in server mode."""
    logger.info("Starting in serve mode...")
    config = Config.get_instance() # Get the loaded config
    kb_manager, rag_manager = initialize_components(config)
    
    # Instantiate the MCP service class
    logger.info("Instantiating KnowledgeMCP service...")
    mcp = MCP(rag_manager, kb_manager)


def run_shell_mode():
    """Runs the interactive management shell."""
    logger.info("Starting in management shell...")
    kb_manager, rag_manager = initialize_components(Config.get_instance())

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
    parser = argparse.ArgumentParser(description="Knowledge Base MCP Server and Management Shell")
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="config.yml", # Consider making default relative to project root if cli is run from there
        help="Path to the configuration file (default: config.yml)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True, help='Available modes: mcp, shell')

    # MCP command
    parser_mcp = subparsers.add_parser("mcp", help="Run the MCP server")
    parser_mcp.set_defaults(func=run_mcp_mode)

    # Shell command
    parser_shell = subparsers.add_parser("shell", help="Run the interactive management shell")
    parser_shell.set_defaults(func=run_shell_mode)

    args = parser.parse_args()

    # Load config - config path might need adjustment depending on CWD
    # If config is expected relative to project root, and cli.py is in the package,
    # we might need to adjust how the default path is handled or make it absolute.
    # For now, assume it's run from project root or path is absolute.
    try:
        # If config is expected relative to project root, and cli.py is in the package,
        # we might need to adjust how the default path is handled or make it absolute.
        # For now, assume it's run from project root or path is absolute.
        Config.load(args.config)
        # Configure logging AFTER config is loaded
        configure_logging() 
        logger.info(f"Successfully loaded config from {args.config}")
    except FileNotFoundError:
        # Try searching relative to the cli script's parent dir? Or require absolute path?
        logger.error(f"Configuration file not found at {args.config}. Please provide a valid path.")
        sys.exit(1)
    except (ValueError, RuntimeError) as e:
        logger.critical(f"Failed to load or validate configuration: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"An unexpected error occurred during configuration loading: {e}")
        sys.exit(1)

    # Execute the function associated with the chosen command
    args.func()

def configure_logging():
    """Configure logging based on the loaded Config singleton."""
    config = Config.get_instance()
    log_config = config.logging
    kb_config = config.knowledge_base

    # Determine the main log file path (within the knowledge base base dir)
    log_file_path = kb_config.base_dir / "kbmcp.log"

    logger.info(f"Main application log file: {log_file_path}")

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": { # For console
                    "format": log_config.default_format,
                },
                "detailed": {
                    "format": log_config.detailed_format,
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr", # Correct stream specifier
                    "level": log_config.level,
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_config.max_bytes,
                    "backupCount": log_config.backup_count,
                    "encoding": "utf-8",
                    "level": log_config.level,
                },
            },
            "loggers": {
                "lightrag": { # Configure LightRAG's root logger
                    "handlers": ["file"],
                    "level": log_config.level, # Use level from config
                    "propagate": False, # Don't pass to our root logger
                },
                "kbmcp": { # Specific logger for our application modules
                    "handlers": ["file"],
                    "level": log_config.level,
                    "propagate": False, # Don't pass to root logger
                },
                "knowledge_mcp": { # Catch logs from submodules like knowledge_mcp.rag etc.
                    "handlers": ["file"],
                    "level": log_config.level,
                    "propagate": False,
                },
            },
            "root": { # Catch-all for other libraries (unless they disable propagation)
                "handlers": ["file"],
                "level": "WARNING", # Set root level higher to avoid too much noise
            },
        }
    )

if __name__ == "__main__":
    # This allows running the cli directly for development,
    # but entry point script is preferred for installation.
    main()
