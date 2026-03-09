from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parent.parent
APP_PATH = ROOT / "app.py"


class AppBenchmarkPathTests(unittest.TestCase):
    def test_run_stock_benchmark_passes_config_paths_into_build_context(self) -> None:
        source = APP_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(APP_PATH))

        fn_node = None
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "_run_stock_benchmark":
                fn_node = node
                break
        self.assertIsNotNone(fn_node, "_run_stock_benchmark must exist in app.py")

        build_context_calls = []
        for node in ast.walk(fn_node):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if (
                isinstance(func, ast.Subscript)
                and isinstance(func.value, ast.Name)
                and func.value.id == "helpers"
                and isinstance(func.slice, ast.Constant)
                and func.slice.value == "build_context"
            ):
                build_context_calls.append(node)

        self.assertTrue(build_context_calls, "_run_stock_benchmark must call helpers['build_context']")

        matched = False
        for call in build_context_calls:
            kw_by_name = {kw.arg: kw.value for kw in call.keywords if kw.arg}
            value = kw_by_name.get("config_paths")
            if not isinstance(value, ast.List):
                continue
            if len(value.elts) != 1:
                continue
            elt = value.elts[0]
            if isinstance(elt, ast.Name) and elt.id == "config_path":
                matched = True
                break

        self.assertTrue(
            matched,
            "_run_stock_benchmark must pass config_paths=[config_path] into helpers['build_context']",
        )


if __name__ == "__main__":
    unittest.main()
