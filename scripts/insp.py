#!/usr/bin/env python3
"""Inspector script wrapper that runs MCP inspector with knowledge-mcp server."""

import sys
import subprocess

def main():
    """Run MCP inspector with knowledge-mcp server and specific config."""
    cmd = [
        "npx", "@modelcontextprotocol/inspector",
        "uv", "run knowledge-mcp --config ./kbs/config.yaml mcp"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(1)

if __name__ == "__main__":
    main()
