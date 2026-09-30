"""Live test of the MCP filesystem server.

Marked slow and skipped by default, because it launches the community server
via npx: the first run downloads the package over the network.

Run it with:

    uv run pytest tests/test_mcp_live.py -m slow -q

It is worth running at least once before trusting the sandbox, because the
scoping claim in mcp_servers.py is only real if the server actually enforces
it. The config tests cannot show that.
"""

import asyncio
from pathlib import Path

import pytest

from ava.mcp_servers import build_servers

pytestmark = pytest.mark.slow


async def _with_server(vault: Path):
    server = build_servers(vault)[0]
    return server


@pytest.mark.asyncio
async def test_server_starts_and_lists_tools(tmp_path):
    vault = tmp_path / "vault"
    server = await _with_server(vault)
    async with server:
        names = {t.name for t in await server.list_tools()}

    # The subset AVA actually depends on. The full set is larger and evolves
    # with the community package, so pinning all of it would be brittle.
    assert {"read_file", "write_file", "list_directory"} <= names


@pytest.mark.asyncio
async def test_server_can_read_inside_the_vault(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir(parents=True, exist_ok=True)
    (vault / "note.md").write_text("hello from inside", encoding="utf-8")

    server = await _with_server(vault)
    async with server:
        result = await server.call_tool("read_file", {"path": str(vault / "note.md")})

    assert "hello from inside" in str(result)


@pytest.mark.asyncio
async def test_server_refuses_to_read_outside_the_vault(tmp_path):
    """The sandbox is the point. A file beside the vault must be unreachable."""
    vault = tmp_path / "vault"
    vault.mkdir(parents=True, exist_ok=True)
    secret = tmp_path / "secret.txt"
    secret.write_text("SECRET-OUTSIDE-VAULT", encoding="utf-8")

    server = await _with_server(vault)
    async with server:
        try:
            result = await server.call_tool("read_file", {"path": str(secret)})
        except Exception:
            return  # refused by raising — also acceptable
        assert "SECRET-OUTSIDE-VAULT" not in str(result)
