"""Snapshot tests for prompt rendering (configs/prompts/README.md). First run with --snapshot-update writes
tests/snapshots/*.txt; later runs must match byte-for-byte."""
import pathlib

import pytest

from tbf.data.fixtures import load_shuffle_map, load_windows
from tbf.prompts import SYSTEM, format_value, render_messages

SNAP = pathlib.Path(__file__).parent / "snapshots"


@pytest.fixture
def win():
    ws = {w["window_id"]: w for w in load_windows()}
    return ws, load_shuffle_map()


def test_format_value():
    assert format_value(12345.0, 0) == "12345" and format_value(1.08337, 4) == "1.0834"


@pytest.mark.parametrize("setup", ["direct", "reviser"])
@pytest.mark.parametrize("mode", ["full", "none", "shuffled"])
def test_snapshot(win, setup, mode, request):
    ws, smap = win
    w = ws["web__Influenza__2025-09-05"]
    prior = [round(v * 1.01, 0) for v in w["context"][-12:]] if setup == "reviser" else None
    src = ws[smap[w["window_id"]]] if mode == "shuffled" else None
    msgs = render_messages(w, setup, mode, prior_forecast=prior, shuffle_source=src)
    assert msgs[0] == {"role": "system", "content": SYSTEM}
    text = msgs[1]["content"]
    assert text.count("\n") > 96 and '{"forecast": [v1, v2, ..., v12]}' in text
    if mode == "none":
        assert "## Recent events" not in text and "## Calendar" not in text
    if setup == "reviser":
        assert "## Statistical model forecast" in text
    path = SNAP / f"{setup}__{mode}.txt"
    if request.config.getoption("--snapshot-update", default=False) or not path.exists():
        SNAP.mkdir(exist_ok=True); path.write_text(text)
    assert text == path.read_text()
