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

"""Scalar observability tests; no device allocation or graph execution."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from xllm.python.model_executor import aclgraph_validation as validation
from xllm.python.model_executor.executor import ModelExecutor


def _metadata(**kwargs: object) -> SimpleNamespace:
    fields = {
        "is_prefill": False,
        "is_chunked_prefill": False,
        "is_spec_verify": False,
        "prepared_attention_state": None,
        "expanded_decode_metadata": None,
        "dp_global_sequence_nums": (),
        "q_seq_lens": SimpleNamespace(numel=lambda: 8),
        "kv_seq_lens": SimpleNamespace(numel=lambda: 8),
    }
    return SimpleNamespace(**(fields | kwargs))


def test_dp_counts_preserve_empty_rank_and_do_not_divide_token_rows() -> None:
    fields = validation.execution_fields(_metadata(dp_global_sequence_nums=(0, 3)), 1, 0, row_expanded=True)
    assert fields["logical_sequences"] == 0
    assert fields["logical_sequence_source"] == "dp_global_sequence_nums"
    assert fields["actual_rows"] == 1
    assert fields["query_length_items"] == 8


@pytest.mark.parametrize(
    "metadata",
    [
        _metadata(prepared_attention_state=object()),
        _metadata(is_spec_verify=True),
        _metadata(),  # Ordinary draft repair rows also have no sequence-level count.
    ],
)
def test_expanded_rows_are_not_reported_as_logical_sequences(metadata: SimpleNamespace) -> None:
    fields = validation.execution_fields(metadata, 8, 0, row_expanded=True)
    assert fields["logical_sequences"] is None
    assert fields["logical_sequence_source"] == "unknown_row_layout"


def test_native_chunked_lengths_count_requests_even_with_separate_expanded_rows() -> None:
    metadata = _metadata(is_chunked_prefill=True, expanded_decode_metadata=SimpleNamespace(enabled=True))
    fields = validation.execution_fields(metadata, 32, 0, row_expanded=True)
    assert fields["logical_sequences"] == 8
    assert fields["logical_sequence_source"] == "q_seq_lens"
    assert fields["phase"] == "decode"


def test_disabled_observation_does_not_read_clock_memory_or_log() -> None:
    with (
        patch.object(validation, "VALIDATION_ENABLED", False),
        patch.object(validation, "torch") as framework,
        patch.object(validation, "time") as clock,
        patch.object(validation.logger, "info") as log,
    ):
        validation.log_capture_memory("before_capture", torch.device("cpu"), {})
        validation.log_event("execution", {})
    assert framework.mock_calls == []
    assert clock.mock_calls == []
    log.assert_not_called()


def test_memory_event_is_json_with_wall_clock_and_unreset_peaks() -> None:
    npu = SimpleNamespace(
        **{
            name: MagicMock(return_value=i)
            for i, name in enumerate(
                (
                    "memory_allocated",
                    "memory_reserved",
                    "max_memory_allocated",
                    "max_memory_reserved",
                )
            )
        }
    )
    device = torch.device("cpu")
    with (
        patch.object(validation, "VALIDATION_ENABLED", True),
        patch.object(validation, "torch", SimpleNamespace(npu=npu)),
        patch.object(validation.time, "time_ns", return_value=123),
        patch.object(validation.logger, "info") as log,
    ):
        validation.log_capture_memory("after_capture", device, {"pool": (1, 2), "mode": "graph"})
    assert log.call_args.args[0] == "XLLM_ACLGRAPH_VALIDATION %s"
    record = json.loads(log.call_args.args[1])
    assert record["event"] == "capture_memory"
    assert record["wall_time_ns"] == 123
    assert record["stage"] == "after_capture"
    assert record["pool"] == [1, 2]
    assert record["max_memory_reserved"] == 3
    for fn in vars(npu).values():
        fn.assert_called_once_with(device)


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("fails", [False, True])
def test_executor_logs_actual_successful_branch_only(graph: bool, fails: bool) -> None:
    executor = ModelExecutor.__new__(ModelExecutor)
    executor._kv_bound = True
    executor.layerwise_split_size = 1
    executor._prepared_mtp = False
    executor._validation_fields = {"dp_rank": 0}
    executor._validation_row_expanded = False
    executor.prepared_graph_runner = None
    executor.inductor_runner = None
    eager, graph_runner = MagicMock(), MagicMock()
    executor.eager_runner = eager
    executor.decode_graph_runner = graph_runner if graph else None
    graph_runner.can_execute.return_value = True
    graph_runner.warmup.return_value = (16,)
    runner = graph_runner if graph else eager
    output = object()
    error = RuntimeError("original execution failure")
    runner.execute.side_effect = error if fails else None
    runner.execute.return_value = output
    tokens = SimpleNamespace(numel=lambda: 8)
    with (
        patch.object(validation, "VALIDATION_ENABLED", True),
        patch.object(validation, "log_event") as log,
    ):
        if fails:
            with pytest.raises(RuntimeError) as raised:
                executor.execute(tokens, tokens, _metadata(is_prefill=not graph))
            assert raised.value is error
            log.assert_not_called()
        else:
            assert executor.execute(tokens, tokens, _metadata(is_prefill=not graph)) is output
            assert log.call_args.args[0] == "execution"
            fields = log.call_args.args[1]
            assert fields["mode"] == ("graph" if graph else "eager")
            assert fields["actual_rows"] == 8
            assert fields["effective_rows"] == (16 if graph else 8)
