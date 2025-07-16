#!/usr/bin/env python3
"""Development script runner for knowledge-mcp."""

import sys
import subprocess
import argparse

def run_shell():
    """Run knowledge-mcp shell with config."""
    cmd = [
        sys.executable, "-m", "knowledge_mcp.cli",
        "--config", "./kbs/config.yaml",
        "shell"
    ]
    subprocess.run(cmd)

def run_inspector():
    """Run MCP inspector with knowledge-mcp server."""
    cmd = [
        "npx", "@modelcontextprotocol/inspector",
        "uv", "run", "knowledge-mcp", "--config", "./kbs/config.yaml", "mcp"
    ]
    subprocess.run(cmd)

def run_tests():
    """Run pytest tests."""
    subprocess.run([sys.executable, "-m", "pytest"])

def run_main():
    """Run knowledge-mcp CLI directly."""
    subprocess.run([sys.executable, "-m", "knowledge_mcp.cli"])

def main():
    parser = argparse.ArgumentParser(description="Development script runner")
    parser.add_argument("command", choices=["shell", "insp", "test", "main"],
                       help="Command to run")
    
    args = parser.parse_args()
    
    if args.command == "shell":
        run_shell()
    elif args.command == "insp":
        run_inspector()
    elif args.command == "test":
        run_tests()
    elif args.command == "main":
        run_main()

if __name__ == "__main__":
    main()
