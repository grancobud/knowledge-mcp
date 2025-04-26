import cmd
import shlex
import logging
import asyncio
from pathlib import Path

from knowledge_mcp.knowledgebases import KnowledgeBaseManager, KnowledgeBaseExistsError, KnowledgeBaseNotFoundError, KnowledgeBaseError
from knowledge_mcp.rag import RagManager
from knowledge_mcp.documents import DocumentManager

logger = logging.getLogger(__name__)

class Shell(cmd.Cmd):
    """Interactive shell for Knowledge MCP."""
    intro = 'Welcome to the Knowledge MCP shell. Type help or ? to list commands.\n'
    prompt = '(kbmcp) '

    def __init__(self, kb_manager: KnowledgeBaseManager, rag_manager: RagManager):
        super().__init__()
        self.kb_manager = kb_manager
        self.rag_manager = rag_manager
        self.document_manager = DocumentManager(rag_manager)

    # --- Basic Commands ---

    def do_exit(self, arg: str) -> bool:
        """Exit the shell."""
        print("Exiting shell.")
        return True # Returning True stops the cmdloop

    def do_EOF(self, arg: str) -> bool:
        """Exit the shell when EOF (Ctrl+D) is received."""
        print() # Print a newline for cleaner exit
        return self.do_exit(arg)

    # --- KB Management Commands ---

    def do_create(self, arg: str):
        """Create a new knowledge base. Usage: create <name>"""
        try:
            args = shlex.split(arg)
            if len(args) != 1:
                print("Usage: create <name>")
                return
            name = args[0]
            self.kb_manager.create_kb(name)
            asyncio.run(self.rag_manager.get_rag_instance(name))
            print(f"Knowledge base '{name}' created successfully.")
        except KnowledgeBaseExistsError:
            print(f"Error: Knowledge base '{name}' already exists.")
        except KnowledgeBaseError as e:
            print(f"Error creating knowledge base: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error in create: {e}")
            print(f"An unexpected error occurred: {e}")

    def do_list(self, arg: str):
        """List all available knowledge bases."""
        try:
            kbs = self.kb_manager.list_kbs()
            if not kbs:
                print("No knowledge bases found.")
                return
            print("Available knowledge bases:")
            for kb_name in kbs:
                print(f"- {kb_name}")
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

    # --- Document Management Commands ---

    def do_add(self, arg: str):
        """Add a document to a knowledge base. Usage: add <kb_name> <file_path>"""
        try:
            args = shlex.split(arg)
            if not 2 <= len(args) <= 3:
                print("Usage: add <kb_name> <file_path>")
                return

            kb_name = args[0]
            file_path_str = args[1]

            file_path = Path(file_path_str)
            if not file_path.is_file():
                print(f"Error: File not found at '{file_path_str}'")
                return

            print(f"Adding document '{file_path.name}' to KB '{kb_name}'...")
            added_doc_id = asyncio.run(self.document_manager.add(file_path, kb_name))
            print(f"Document added successfully with ID: {added_doc_id}")

        except KnowledgeBaseNotFoundError:
            print(f"Error: Knowledge base '{kb_name}' not found.")
        except FileNotFoundError:
            print(f"Error: Document file path '{file_path_str}' not found.")
        except Exception as e:
            logger.exception(f"Unexpected error in add: {e}")
            print(f"An unexpected error occurred: {e}")

    async def do_remove_doc(self, arg: str):
        """Remove a document from a knowledge base by its ID. Usage: remove_doc <kb_name> <doc_id>"""
        try:
            args = shlex.split(arg)
            if len(args) != 2:
                print("Usage: remove_doc <kb_name> <doc_id>")
                return

            kb_name = args[0]
            doc_id = args[1]

            print(f"Removing document '{doc_id}' from KB '{kb_name}'...")
            removed = await self.rag_manager.remove_document(kb_name, doc_id)
            if removed:
                print(f"Document '{doc_id}' removed successfully.")
            else:
                 print(f"Document '{doc_id}' not found in KB '{kb_name}' or could not be removed.")
        except KnowledgeBaseNotFoundError:
            print(f"Error: Knowledge base '{kb_name}' not found.")
        except Exception as e:
             logger.exception(f"Unexpected error in remove_doc: {e}")
             print(f"An unexpected error occurred: {e}")

    # --- Query Commands --- # TODO

    def do_query(self, arg: str):
        """Query a knowledge base. Usage: query <kb_name> <query_text>"""
        print("Query functionality not yet implemented.")
