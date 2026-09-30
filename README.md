# AVA

A terminal personal assistant built on the
[OpenAI Agents SDK](https://openai.github.io/openai-agents-python), running
against an **OpenRouter** key with **free models only**.

AVA keeps notes, writes email drafts (never sends them), manages reminders with
macOS notifications, and reads and writes files in your vault through a local
MCP server.

## Setup

```bash
uv sync
cp .env.example .env      # then add your OPENROUTER_API_KEY
uv run ava
```

## Commands

Anything not starting with `/` is sent to the agent. Slash commands bypass the
model entirely — they are instant, free, and deterministic, which matters on a
50-requests-per-day budget.

| Command | What it does |
|---|---|
| `/help` | List the commands |
| `/notes` | Recent notes |
| `/drafts` | Recent email drafts |
| `/reminders` | Pending reminders |
| `/tools` | Tool names the agent can call |
| `/model` | Show the current model |
| `/model <slug>` | Switch model — free models only |
| `/new` | Start a fresh conversation |
| `/usage` | Model calls used today |
| `/quit` | Exit |

```
you> /reminders
No reminders set.

you> What are my notes about wifi?
```

### Examples

```bash
ava                                   # interactive
echo "remind me to stretch at 3pm" | ava   # one-shot, piped
ava > transcript.txt                   # conversation only; diagnostics on stderr
```

## Which models can I use?

Free models only, enforced twice: at startup, and again on every `/model`
switch. A paid slug is refused with an explanation.

To see what is free today and supports tool calling:

```bash
uv run python scripts/list_free_models.py
```

That list rotates, and some free models have no tool support at all — AVA would
answer them in prose and do nothing. If your configured model leaves the free
list, switch to `openrouter/free` and OpenRouter picks whichever free model is
available.

**Free models fail more often than paid ones.** Expect these:

- `503 provider_overloaded` — the endpoint is busy. Wait a few seconds and
  retry; it is not a configuration problem.
- `429 rate limit` — the day's budget is spent. Resets at midnight.

### How much you can actually do

| Turn type | Requests |
|---|---|
| Slash command | **0** |
| Plain chat | 1 |
| Delegated turn (a tool fires) | ~4 |

So roughly **12 delegated turns a day**. AVA counts this locally, warns at
40/50, and tells you when it runs out — before OpenRouter returns a 429.
Failed requests count too, so the local counter also catches a bug that burns
requests.

## How it is put together

AVA routes to three specialist agents, each owning one domain's tools:

```
you ──► AVA (coordinator, no domain tools)
          ├─ handoff ──────────► Notes       (notes)
          ├─ handoff ──────────► Drafting    (email drafts)
          ├─ handoff ──────────► Scheduling  (reminders)
          ├─ as tool ──────────► Notes / Drafting
          └─ MCP filesystem ───► ~/.ava/vault
```

**Handoff** hands the turn to a specialist, which answers you directly. **Agent
as tool** keeps the coordinator in charge: it calls a specialist for a bounded
job, then narrates the result. Pure-domain requests hand off; cross-domain
requests use tools. Scheduling is handoff-only, so reminders always go through
the specialist that owns the timezone reasoning.

## Safety limits

- **Email drafts are never sent.** AVA writes `.eml` files to `~/.ava/vault`
  and has no send capability.
- **Reminders only fire while the CLI is running.** Missed ones are reported on
  the next start. There is no background daemon.
- **The MCP filesystem server is scoped to `~/.ava/vault`.** Nothing outside it
  is readable or writable through the agent.
- **Free endpoints may log your prompts.** Fine for personal notes, not for
  anything confidential.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | — | Required |
| `AVA_MODEL` | a `:free` slug | Must be free; the guard refuses anything else |
| `AVA_TZ` | `Asia/Dhaka` | Timezone for reminders. Naive times are read here |
| `AVA_HOME` | `~/.ava` | Database, vault, and usage counter |

## Development

```bash
uv run pytest -q                 # 300 tests, no network or API key needed
uv run pytest -m slow            # live MCP server tests (launches npx)
uv run python -m ava             # same as `ava`
```

Data lives in `~/.ava/`: `ava.db` (SQLite), `vault/` (drafts and files), and
`usage.json` (delete it to reset the daily counter).
