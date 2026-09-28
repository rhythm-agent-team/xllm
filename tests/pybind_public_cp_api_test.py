# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import ast
import unittest
from pathlib import Path

from xllm.pybind.args import ArgumentParser

_ROOT = Path(__file__).resolve().parents[1]


def _constructor_defaults(module_name: str, class_name: str) -> dict[str, object]:
    module_path = _ROOT / "xllm" / "pybind" / f"{module_name}.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    init_node = next(node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    args = init_node.args.args
    defaults = init_node.args.defaults
    names = [arg.arg for arg in args[len(args) - len(defaults) :]]
    return dict(zip(names, (ast.literal_eval(value) for value in defaults)))


class PublicContextParallelApiTest(unittest.TestCase):
    def test_offline_cli_defaults_and_accepts_context_parallel_size(self) -> None:
        parser = ArgumentParser().parser
        self.assertEqual(parser.parse_args([]).cp_size, 1)
        self.assertEqual(parser.parse_args(["--cp_size", "4"]).cp_size, 4)

    def test_offline_cli_rejects_removed_spelling(self) -> None:
        parser = ArgumentParser().parser
        removed_option = "--enable_" + "prefill_sp"
        with self.assertRaises(SystemExit):
            parser.parse_args([removed_option])

    def test_python_constructors_default_context_parallel_size_to_one(self) -> None:
        for module_name, class_name in (
            ("llm", "LLM"),
            ("embedding", "Embedding"),
            ("vlm", "VLM"),
        ):
            defaults = _constructor_defaults(module_name, class_name)
            self.assertEqual(defaults["cp_size"], 1)
            self.assertNotIn("enable_" + "prefill_sp", defaults)


if __name__ == "__main__":
    unittest.main()
