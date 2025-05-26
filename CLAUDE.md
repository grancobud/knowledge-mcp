# Ignore Patterns
archive/
CLAUDE.md
.aider*
data/
docs/

# Coding
- Use Python 3.12 features and syntax
- Follow PEP 8 style guide for Python code
- Use type hints everywhere possible
- Use list, dict, and set comprehensions when appropriate for concise and readable code.
- Prefer pathlib over os.path for file system operation
- Use explicit exception handling. Catch specific exceptions rather than using bare except clauses
- Keep functions and methods small and focused on a single task
- Use docstrings for all public modules, functions, classes, and methods
- Use dataclasses for data containers when appropriate
- Prefer composition over inheritance where possible
- Use logging for debugging and monitoring
- Use meaningful variable and method names

# Development
- Use pytest for unit testing
- Do not create tests unless requested by the user
- Use uv for dependency management

## Using uv 
- use 'uv add <dependency name>' to add dependencies
- use 'uv remove <dependency name>' to remove dependencies
- Create uv scripts for running scripts in pyproject.toml [project.scripts]
- Use hatchling as the build-system

## Running
- Ensure activate the venv before spawning a new console session: `source .venv/bin/activate`
- Use uv scripts to run something

# Using Tools (MCP)
- Use the sequential-thinking tool if the problem is complex and you need to think hard, you need to think step by step. This is mandatory.
- Use the available docs tools to look up the documentation notation. "LightRAG Docs" and "FastMCP Docs"
- Use the fetch tool to fetch the content of web pages, i.e. from an llms.txt sitemap

# Project
- Project sepecification is in _project directory: 
  - prd.md: project requirements
  - plan.md: development plan