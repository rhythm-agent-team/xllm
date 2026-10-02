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

import json
import signal
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import pytest

from xllm.pybind import embedding, llm, utils, vlm
from xllm.pybind.args import ArgumentParser


def test_offline_cli_defaults_and_accepts_context_parallel_size() -> None:
    parser = ArgumentParser().parser
    assert parser.parse_args([]).cp_size == 1
    assert parser.parse_args(["--cp_size", "4"]).cp_size == 4


def test_offline_cli_rejects_removed_spelling() -> None:
    with pytest.raises(SystemExit) as error:
        ArgumentParser().parser.parse_args(["--enable_prefill_sp"])
    assert error.value.code == 2


@pytest.mark.parametrize(
    "api_module,constructor,master_name,model_type,backend",
    [
        (llm, llm.LLM, "LLMMaster", "qwen3", "llm"),
        (embedding, embedding.Embedding, "LLMMaster", "qwen3", "llm"),
        (vlm, vlm.VLM, "VLMMaster", "qwen2_vl", "vlm"),
    ],
    ids=["llm", "embedding", "vlm"],
)
@pytest.mark.parametrize("cp_size", [None, 4], ids=["default", "explicit"])
def test_python_constructors_forward_context_parallel_size(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    api_module: ModuleType,
    constructor: Callable[..., Any],
    master_name: str,
    model_type: str,
    backend: str,
    cp_size: int | None,
) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}), encoding="utf-8")
    master = MagicMock()
    monkeypatch.setattr(api_module, master_name, master)
    monkeypatch.setattr(signal, "signal", MagicMock())
    monkeypatch.setattr(utils, "get_free_port", lambda: 26001)
    monkeypatch.setattr(utils.xllm_export, "get_model_backend", lambda _model_type: backend)
    monkeypatch.setattr(utils.xllm_export, "configure_cpp_chat_template", MagicMock())

    kwargs = {} if cp_size is None else {"cp_size": cp_size}
    instance = constructor(model=str(tmp_path), **kwargs)

    master.assert_called_once()
    options = master.call_args.args[0]
    assert isinstance(options, api_module.Options)
    assert options.model_path == str(tmp_path)
    assert options.cp_size == (1 if cp_size is None else 4)
    assert instance.master is master.return_value

    master.reset_mock()
    with pytest.raises(TypeError, match="Unexpected keyword arguments: enable_prefill_sp"):
        constructor(model=str(tmp_path), enable_prefill_sp=True)
    master.assert_not_called()
