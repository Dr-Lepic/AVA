"""The agent actually receives the real current time.

The bug being guarded: the model had no clock and answered from training data
where the date was 2025. Injecting the current time into the scheduling
specialist's instructions is the second of three defences, and the one that
stops the model computing dates in the first place.
"""

from pathlib import Path

from ava.agents import build_specialists


def _instructions(tmp_path, tz_name="Asia/Dhaka") -> str:
    """The scheduling specialist's instructions as plain text.

    Agent.instructions may be a callable or None, so coerce before asserting.
    """
    agent = build_specialists(
        tmp_path / "a.db", tmp_path / "vault", model="m", tz_name=tz_name
    )["scheduling"]
    return agent.instructions or ""


def test_scheduling_instructions_contain_the_current_date(tmp_path):
    """The year in the block must be today's, not a stale training year."""
    from datetime import datetime

    from ava.tools.reminders import _local_tz

    instructions = _instructions(tmp_path)
    this_year = str(datetime.now(_local_tz("Asia/Dhaka")).year)
    assert this_year in instructions, f"{this_year} missing from instructions"


def test_instructions_name_the_configured_timezone(tmp_path):
    assert "Asia/Dhaka" in _instructions(tmp_path, "Asia/Dhaka")


def test_tz_override_is_reflected(tmp_path):
    assert "UTC" in _instructions(tmp_path, "UTC")


def test_instructions_tell_the_model_it_has_no_clock(tmp_path):
    text = _instructions(tmp_path).lower()
    assert "no clock" in text
    assert "in 20 minutes" in text


def test_instructions_forbid_computing_the_date(tmp_path):
    """The old instruction told it to convert times. That is the bug."""
    text = _instructions(tmp_path).lower()
    assert "convert relative times" not in text
    assert "straight through" in text


def test_instructions_explain_the_past_due_rejection(tmp_path):
    text = _instructions(tmp_path).lower()
    assert "past" in text
