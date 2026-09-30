"""AVA terminal REPL.

Two behaviours here were established by probing a real streaming run, and both
would otherwise have been bugs:

- `Runner.run_streamed(...)` is not awaitable. It returns a
  RunResultStreaming directly.
- `raw_response_event.data.delta` carries tool-call argument JSON as well as
  assistant text, so deltas are only printed once the run is emitting message
  output. Otherwise the user sees {"input":"milk"} mid-sentence.
"""

from __future__ import annotations

import asyncio
import json
import sys

from agents import Agent, Runner, SQLiteSession

from ava.agent import build_agent
from ava.budget import Budget
from ava.commands import NEW_SESSION_PREFIX, Context, handle
from ava.config import ConfigError, load_config
from ava.llm import build_model, configure_llm
from ava.mcp_servers import build_servers

BANNER = """AVA — personal assistant. Free models, ~12 delegated turns/day.
Type /quit to exit."""


def write_text(text: str = "", *, stream: bool = False) -> None:
    """Write conversation text to stdout (pipable) and optionally flush."""
    sys.stdout.write(text)
    if stream:
        sys.stdout.flush()
    else:
        sys.stdout.write("\n")


def write_diagnostic(text: str) -> None:
    """Write a diagnostic line to stderr, keeping stdout clean for piping."""
    print(text, file=sys.stderr)


def _format_tool_call(item) -> str | None:
    """One-line summary of a tool call, or None if the item is not one."""
    raw = getattr(item, "raw_item", None)
    name = getattr(raw, "name", None)
    if not name:
        return None
    raw_args = getattr(raw, "arguments", "") or ""
    try:
        args = json.loads(raw_args)
    except (json.JSONDecodeError, TypeError):
        return f"  · {name}({raw_args})"
    if not isinstance(args, dict):
        return f"  · {name}({raw_args})"
    shown = ", ".join(f"{k}={v!r}" for k, v in list(args.items())[:3])
    return f"  · {name}({shown})"


# The chunk type that carries user-visible assistant text. A reasoning model
# also streams `response.reasoning_text.delta`, which is internal deliberation
# and must not be printed — established by probing a real run, where a free
# reasoning model leaked its scratchpad to the terminal.
OUTPUT_TEXT_DELTA = "response.output_text.delta"


def _is_output_text(event) -> bool:
    """True if this raw_response_event carries visible assistant text.

    Some models (the free reasoning ones, commonly) stream their deliberation
    as `response.reasoning_text.delta` through the same event type as the
    answer. Only `response.output_text.delta` is meant for the user.

    Chunks without a `type` are treated as visible text: older or simpler
    providers may not label chunks, and hiding real output would be worse than
    showing an unlabelled delta.
    """
    data = getattr(event, "data", None)
    chunk_type = getattr(data, "type", None)
    if chunk_type is None:
        return True
    return chunk_type == OUTPUT_TEXT_DELTA


def _looks_like_tool_json(chunk: str) -> bool:
    """True if a streamed delta is tool-call argument JSON, not prose.

    Established by probing a real streaming run: the argument delta arrives
    several events BEFORE the corresponding run_item_stream_event, so event
    ordering cannot be relied on to suppress it. The delta's own shape can —
    assistant text is prose, whereas a tool-argument delta is a JSON fragment.
    """
    stripped = chunk.strip()
    if not stripped:
        return False
    # Argument deltas start with a JSON object fragment and never form a
    # complete sentence. Be conservative: only suppress on an opening brace.
    return stripped[0] == "{"


async def _respond(
    agent: Agent, session: SQLiteSession | None, text: str, _ctx
) -> tuple[int, str]:
    """Stream one turn. Returns (model_calls, usage summary).

    run_streamed is not awaitable. Two filters are applied, both established
    by probing a real run: tool-argument deltas are identified by shape (they
    arrive before the run_item event that names the call), and reasoning deltas
    are identified by chunk type.
    """
    result = Runner.run_streamed(agent, text, session=session)

    printed = False

    async for event in result.stream_events():
        if event.type == "run_item_stream_event":
            line = _format_tool_call(event.item)
            if line:
                if printed:
                    write_text()
                write_diagnostic(line)
                printed = False
        elif event.type == "raw_response_event" and _is_output_text(event):
            chunk = getattr(getattr(event, "data", None), "delta", None)
            if chunk and not _looks_like_tool_json(chunk):
                write_text(chunk, stream=True)
                printed = True

    write_text()

    handled = getattr(result, "last_agent", None)
    if handled is not None and handled.name != "AVA":
        write_diagnostic(f"  [handled by {handled.name}]")

    calls = len(getattr(result, "raw_responses", []) or []) or 1
    return calls, f"{calls} model call(s)"


def _report_overdue(conn) -> None:
    """Print reminders that fired while AVA was not running."""
    from ava.tools.reminders import due_reminders

    missed = due_reminders(conn)
    if not missed:
        return
    write_diagnostic(f"You have {len(missed)} overdue reminder(s):")
    for row in missed:
        write_diagnostic(f"  - {row['title']} (was due {row['due_at']})")


async def _run(config) -> int:
    from agents.mcp import MCPServerManager

    from ava.db import connect
    from ava.watcher import run_forever

    conn = connect(config.db_path)
    model = build_model(config.model)
    servers = build_servers(config.vault_path)
    agent = build_agent(
        config.db_path,
        config.vault_path,
        mcp_servers=servers,
        model=model,
    )
    session = SQLiteSession("ava-cli", db_path=str(config.db_path))
    ctx = Context(agent=agent, conn=conn, model=config.model)
    budget = Budget(config.state_path)

    _report_overdue(conn)

    write_text(BANNER)
    write_diagnostic(f"Model: {config.model}")
    write_diagnostic(budget.summary())

    # The watcher runs as a concurrent task. It takes a database path and
    # resolves its own connection per call, because the tools run on worker
    # threads and a sqlite3 connection is bound to its creating thread.
    stop = asyncio.Event()
    watcher_task = asyncio.create_task(run_forever(config.db_path, stop))

    # MCP servers are stdio subprocesses the caller must start and stop. The
    # SDK raises "Server not initialized" if a run starts them unconnected.
    manager = MCPServerManager(servers)
    await manager.connect_all()

    try:
        while True:
            try:
                line = await asyncio.to_thread(input, "\nyou> ")
            except (EOFError, KeyboardInterrupt):
                write_text()
                break

            line = line.strip()
            if not line:
                continue

            # Slash commands bypass the model: instant, free, and
            # deterministic. This matters on a 50/day request budget.
            if line.startswith("/"):
                try:
                    out = handle(line, ctx)
                except SystemExit:
                    break
                if out.startswith(NEW_SESSION_PREFIX):
                    # A /new hands back a new session id; rebuild the session
                    # so the next turn actually uses it.
                    session = SQLiteSession(
                        out[len(NEW_SESSION_PREFIX) :],
                        db_path=str(config.db_path),
                    )
                    out = "Starting a fresh conversation."
                if line.lower() == "/usage":
                    out = f"{out}\n{budget.summary()}"
                write_diagnostic(out)
                continue

            # Slash commands are free, so only a real turn is charged.
            if budget.is_exhausted():
                write_diagnostic(
                    "Daily free limit reached — message not sent. "
                    "Slash commands still work; try /model openrouter/free."
                )
                continue

            try:
                calls, summary = await _respond(agent, session, line, None)
                # Charge the REAL number of model calls this turn made. A
                # delegated turn is ~4, not 1, and undercounting would let the
                # day's budget disappear without a warning.
                budget.record(calls)
                ctx.last_usage = summary
                write_diagnostic(f"  [{summary}]")
                if budget.should_warn():
                    write_diagnostic(f"  [quota] {budget.summary()}")
            except KeyboardInterrupt:
                write_text("\n[interrupted]")
            except Exception as exc:  # keep the REPL alive on model errors
                write_diagnostic(f"[error] {type(exc).__name__}: {exc}")
    finally:
        stop.set()
        watcher_task.cancel()
        await manager.cleanup_all()

    return 0


def main() -> int:
    try:
        config = load_config()
    except ConfigError as exc:
        write_diagnostic(f"Configuration error: {exc}")
        return 1

    configure_llm(config)
    return asyncio.run(_run(config))


if __name__ == "__main__":
    raise SystemExit(main())
