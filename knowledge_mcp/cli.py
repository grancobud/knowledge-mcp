import typer
from typing_extensions import Annotated
import logging
from pathlib import Path

# Import project components
from knowledge_mcp.config import load_config
from knowledge_mcp.rag_manager import RAGManager
from knowledge_mcp.document_manager import DocumentManager, DocumentProcessingError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = typer.Typer(    name="knowledge-mcp",
    help="A CLI tool to manage and interact with Knowledge MCP."
)

def get_document_manager() -> DocumentManager:
    """Initializes and returns a DocumentManager instance."""
    try:
        # Assuming config.yaml is in the project root relative to execution
        config_path = Path("config.yaml")
        if not config_path.exists():
            logger.error(f"Configuration file not found at {config_path.resolve()}")
            raise typer.Exit(code=1)
        
        config = load_config(config_path)
        rag_manager = RAGManager(config)
        document_manager = DocumentManager(config, rag_manager)
        return document_manager
    except Exception as e:
        logger.exception("Failed to initialize application components.")
        print(f"Error initializing components: {e}")
        raise typer.Exit(code=1)

@app.command("add", help="Add a document to a knowledge base.")
def add_command(
    doc_path: Annotated[Path, typer.Argument(help="Path to the document file.", exists=True, file_okay=True, dir_okay=False, readable=True)],
    kb_name: Annotated[str, typer.Argument(help="Name of the knowledge base to add the document to.")]
):
    """Processes a single document file and ingests it into the specified knowledge base."""
    logger.info(f"Received request to process document: {doc_path} for knowledge base: {kb_name}")

    try:
        document_manager = get_document_manager()
        document_manager.add(doc_path, kb_name)
        print(f"Successfully processed and ingested {doc_path.name} into knowledge base '{kb_name}'.")
    except DocumentProcessingError as e:
        logger.error(f"Error processing document {doc_path}: {e}", exc_info=True)
        print(f"Error: Failed to process document '{doc_path.name}'. {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        # Catch other unexpected errors during processing
        logger.exception(f"An unexpected error occurred while processing {doc_path} for {kb_name}.")
        print(f"An unexpected error occurred: {e}")
        raise typer.Exit(code=1)
    finally:
        print(">>> Exiting process_document_command") # Debug print

@app.command("create", help="Create a new knowledge base.")
def create_command(
    kb_name: Annotated[str, typer.Argument(help="Name for the new knowledge base.")]
):
    """Creates a new knowledge base directory structure."""
    logger.info(f"Received request to create knowledge base: {kb_name}")
    # TODO: Initialize KBManager
    # TODO: Call kb_manager.create_kb(kb_name)
    print(f"Placeholder: Creating knowledge base '{kb_name}'.")
    print("Actual implementation pending.")


# Add other commands as needed (e.g., query, list-kbs, delete-kb)


def main():
    """Main entry point for the CLI application."""
    app()

if __name__ == "__main__":
    main()
