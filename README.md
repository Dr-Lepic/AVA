# AVA

A terminal personal assistant built on the [OpenAI Agents SDK](https://openai.github.io/openai-agents-python),
running against an **OpenRouter** key with free models only.

AVA keeps notes, writes email drafts (never sends), and manages reminders, and can
read and write files in your vault through a local MCP server.

## Setup

```bash
uv sync
cp .env.example .env      # then add your OPENROUTER_API_KEY
uv run ava
```

## Which models can I use?

Free models only, and AVA refuses to start on a paid slug. To see what is free
today and supports tool calling:

```bash
uv run python scripts/list_free_models.py
```

## Safety limits

- **Email drafts are never sent.** AVA writes `.eml` files to `~/.ava/vault` and
  has no send capability.
- **Reminders only fire while the CLI is running.** Missed ones are reported on
  the next start.
- **Free-tier rate limits apply** (~50 model requests/day). AVA tracks the day's
  usage locally and warns before OpenRouter returns a 429.
- **Free endpoints may log your prompts.** Fine for personal notes, not for
  anything confidential.

## Development

```bash
uv run pytest -q
```
