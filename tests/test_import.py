import sys
import os


def test_import_package():
    # Ensure src is on path so imports work in CI and locally
    sys.path.insert(0, os.path.join(os.getcwd(), "src"))
    import importlib

    mod = importlib.import_module("generative_model_unlearning")
    assert mod is not None
