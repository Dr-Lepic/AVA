"""Local stdio MCP servers exposed to the agent as function tools.

Why local and not hosted: OpenRouter only serves the Chat Completions API, and
`HostedMCPTool` is Responses-API-only. A hosted server would fail at runtime
with an endpoint error, so MCP has to go through a stdio subprocess.

The filesystem server is community code (@modelcontextprotocol/server-filesystem)
launched over npx, scoped to a single directory. That directory is the only
thing bounding what the model can read or write, so it is passed alone and
never as a model-supplied argument.
"""

from __future__ import annotations

from pathlib import Path

from agents.mcp import MCPServerStdio

NPM_PACKAGE = "@modelcontextprotocol/server-filesystem"

# First launch fetches the package over the network, which can exceed the SDK's
# 5 second default. Later launches start from the npx cache and are fast.
SESSION_TIMEOUT_SECONDS = 60

MODELS_NOTE = """\
MCP integration notes:

- AVA runs on the OpenRouter Chat Completions path, so HostedMCPTool is
  unavailable: it is Responses-API-only. Use local stdio servers.
- Hosted OpenAI tools (WebSearchTool, FileSearchTool, CodeInterpreterTool)
  are likewise Responses-only and will not work here.
- The filesystem server is scoped to the vault directory. The model cannot
  read or write anywhere else.
"""


def build_servers(vault_path: Path) -> list[MCPServerStdio]:
    """Return the stdio MCP servers the agent may use.

    The filesystem server is scoped to vault_path, which is created if absent
    so the server does not fail to start on a fresh install.
    """
    vault_path.mkdir(parents=True, exist_ok=True)

    return [
        MCPServerStdio(
            name="filesystem",
            params={
                "command": "npx",
                "args": ["-y", NPM_PACKAGE, str(vault_path)],
            },
            # Every run calls list_tools() on each server; caching avoids
            # paying that latency on every message.
            cache_tools_list=True,
            client_session_timeout_seconds=SESSION_TIMEOUT_SECONDS,
        ),
    ]
