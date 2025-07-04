#!/usr/bin/env python3
"""Shell script wrapper that runs knowledge-mcp with specific config."""

import sys
import subprocess

def main():
    """Run knowledge-mcp with the specified config and shell command."""
    cmd = [
        sys.executable, "-m", "knowledge_mcp.cli",
        "--config", "./kbs/config.yaml",
        "shell"
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
