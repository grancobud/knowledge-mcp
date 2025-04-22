# Development Plan for knowledge-mcp

**Date:** 2025-04-22

## Overview
This plan breaks down the development of `knowledge-mcp` into five phases, aligning with the PRD roadmap. Each phase includes milestones, tasks, and estimated timelines.

---

## Phase 1: Core Infrastructure (1 week)

**Milestones:**
- Project scaffolding
- Dependency management setup
- Config parsing module
- Basic CLI structure

**Tasks:**
1. Initialize `uv` project and virtual environment
2. Define project structure (folders, modules)
3. Set up `pyproject.toml` (metadata & scripts)
4. Implement `config.py` for YAML/ENV loading
5. Create `cli.py` skeleton with `argparse` or `click`
6. Add commands: `create`, `delete` stub implementations

---

## Phase 2: Document Management (1.5 weeks)

**Milestones:**
- Document add/remove CLI
- LightRAG integration for ingestion

**Tasks:**
1. Implement `kb_manager.py` with filesystem CRUD
2. Integrate `LightRAG` chunking & embedding in `document_processor.py`
3. Support PDF, text, markdown, doc parsing
4. CLI commands: `add`, `remove` with config support
5. Unit tests for file ops & parsing

---

## Phase 3: Search Functionality (1 week)

**Milestones:**
- Search module using LightRAG
- CLI `search` command fully functional

**Tasks:**
1. Implement `search.py` wrapping `LightRAG` in-context mode
2. Format search results (text chunks)
3. CLI `search` integration in `cli.py`
4. Logging setup in `config.yaml`
5. Tests for search outputs & edge cases

---

## Phase 4: MCP Server (1 week)

**Milestones:**
- FastMCP integration
- MCP `search` method exposed

**Tasks:**
1. Add FastMCP dependency & server scaffold (`mcp_server.py`)
2. Implement MCP server method `search(kb, query)` calling `search.py`
3. Update config sample for MCP in docs
4. End-to-end test with sample client

---

## Phase 5: Testing & Documentation (1 week)

**Milestones:**
- Comprehensive test coverage
- User & developer docs

**Tasks:**
1. Expand pytest suite (CLI, processor, search, MCP)
2. Write README with usage examples
3. Create `docs/` with prd summary, config guide
4. CI setup (GitHub Actions) for test & lint

---

## Total Estimated Duration: ~5.5 weeks

_This plan is a draft. Let's refine task breakdowns, timelines, or priorities based on your feedback._
