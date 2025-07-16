.PHONY: help shell insp test main dev-install clean

# Default target
help:
	@echo "Available development commands:"
	@echo "  make shell     - Run knowledge-mcp shell with config"
	@echo "  make insp      - Run MCP inspector with knowledge-mcp server"
	@echo "  make test      - Run pytest tests"
	@echo "  make main      - Run knowledge-mcp CLI directly"
	@echo "  make dev-install - Install package in development mode"
	@echo "  make clean     - Clean build artifacts"

# Run shell with config
shell:
	python -m knowledge_mcp.cli --config ./kbs/config.yaml shell

# Run MCP inspector
insp:
	npx @modelcontextprotocol/inspector uv run knowledge-mcp --config ./kbs/config.yaml mcp

# Run tests
test:
	pytest

# Run main CLI
main:
	python -m knowledge_mcp.cli

# Install in development mode
dev-install:
	uv sync --group dev
	uv pip install -e .

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
