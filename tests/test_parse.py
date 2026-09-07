import json
import pathlib

import pytest

from tbf.parse import parse_forecast

CASES = json.loads((pathlib.Path(__file__).parent / "fixtures_answers.json").read_text())


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_fixture_answers(case):
    p = parse_forecast(case["answer"])
    assert p.valid == case["valid"], (case["name"], p.reason)
    if case["valid"]:
        assert len(p.values) == 12
        assert p.n_values_mismatch == case.get("mismatch", False)
        if "first" in case:
            assert p.values[0] == pytest.approx(case["first"])


def test_last_object_wins():
    ans = '{"forecast": [1,1,1,1,1,1,1,1,1,1,1,1]} oops actually {"forecast": [2,2,2,2,2,2,2,2,2,2,2,2]}'
    assert parse_forecast(ans).values[0] == 2.0
