"""Local stdio MCP server configuration.

These are config-level tests: they assert the wiring, not a live subprocess.
The server itself is community code (@modelcontextprotocol/server-filesystem)
run over npx, and starting it for real belongs to the live smoke test.

The security-relevant property is scoping: the filesystem server is handed
exactly one directory, and the model must not be able to read or write
outside it.
"""

from ava.mcp_servers import MODELS_NOTE, build_servers

NPM_PACKAGE = "@modelcontextprotocol/server-filesystem"


def test_build_servers_returns_one_server(tmp_path):
    servers = build_servers(tmp_path / "vault")
    assert len(servers) == 1


def test_server_is_named_filesystem(tmp_path):
    servers = build_servers(tmp_path / "vault")
    assert servers[0].name == "filesystem"


def test_server_uses_npx(tmp_path):
    """Launching via npx means no separate install step for the user."""
    params = build_servers(tmp_path / "vault")[0].params
    assert params.command == "npx"


def test_server_pins_the_published_package(tmp_path):
    """An unpinned npx fetch is a supply-chain risk on every launch."""
    args = build_servers(tmp_path / "vault")[0].params.args
    assert args[0] == "-y"
    assert args[1] == NPM_PACKAGE


def test_server_is_scoped_to_the_vault(tmp_path):
    """The vault path is the last argv element, which bounds what it can touch."""
    vault = tmp_path / "vault"
    args = build_servers(vault)[0].params.args
    assert args[-1] == str(vault)


def test_server_is_scoped_to_exactly_one_directory(tmp_path):
    """Extra paths would widen the sandbox; assert only the vault is passed."""
    vault = tmp_path / "vault"
    args = build_servers(vault)[0].params.args
    assert len(args) == 3
    assert str(vault.parent) not in args


def test_build_servers_creates_the_vault_dir(tmp_path):
    vault = tmp_path / "nested" / "vault"
    build_servers(vault)
    assert vault.is_dir()


def test_tools_list_is_cached(tmp_path):
    """Every run calls list_tools() on each server; caching avoids the latency."""
    assert build_servers(tmp_path / "vault")[0].cache_tools_list is True


def test_build_servers_is_repeatable(tmp_path):
    """Called once per process, but must not blow up if called again."""
    vault = tmp_path / "vault"
    build_servers(vault)
    servers = build_servers(vault)
    assert len(servers) == 1


def test_hosted_mcp_is_documented_as_unavailable():
    """Guard against someone reaching for HostedMCPTool on the OpenRouter path."""
    note = MODELS_NOTE.lower()
    assert "chat completions" in note
    assert "hostedmcptool" in note
    assert "responses-only" in note
