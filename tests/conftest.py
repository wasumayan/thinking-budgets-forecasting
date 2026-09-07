import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))


def pytest_addoption(parser):
    parser.addoption("--snapshot-update", action="store_true", default=False)
