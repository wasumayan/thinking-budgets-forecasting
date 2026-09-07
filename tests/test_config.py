import subprocess
import sys

from tbf.config import RunConfig, load_grid


def test_config_id_roundtrip():
    for c in [RunConfig("freshts26", "qwen3-8b", "direct", True, 2048, "full"),
              RunConfig("freshts26", "qwen3-4b", "reviser", False, 0, "shuffled", 5, 2, "tirex"),
              RunConfig("cik", "qwen3-8b", "direct", True, 8192, "full", 5, 0)]:
        assert RunConfig.parse(c.config_id) == c
    assert RunConfig.parse("freshts26__qwen3-8b__reviser__b512__full__s0__prior-timesfm25").prior == "timesfm25"


def test_grid_expands_to_spec_counts():
    out = subprocess.run([sys.executable, "scripts/expand_grid.py", "--tier", "1"], capture_output=True, text=True, check=True)
    ids = out.stdout.split()
    assert len(ids) == 30, ids  # 24 + 6 (SPEC §5 tier 1)
    out2 = subprocess.run([sys.executable, "scripts/expand_grid.py", "--tier", "2"], capture_output=True, text=True, check=True)
    assert len(out2.stdout.split()) == 24  # 16 + 4 + 4
    out3 = subprocess.run([sys.executable, "scripts/expand_grid.py", "--tier", "3"], capture_output=True, text=True, check=True)
    assert len(out3.stdout.split()) == 16  # 6 + 4 + 6 (stretch excluded)


def test_grid_models_have_hf_ids():
    g = load_grid()
    for k, v in g["models"].items():
        assert "/" in v["hf"], k
