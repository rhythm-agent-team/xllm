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

"""Tests for the NPU ACL decode-graph runner."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from xllm.python.model_executor.runners.decode_acl_graph import (
    DecodeAclGraphRunner,
)


def _runner() -> DecodeAclGraphRunner:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    return DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=8,
        max_model_len=8,
    )


def _metadata(linear_state_indices: torch.Tensor) -> SimpleNamespace:
    rows = linear_state_indices.numel()
    lengths = torch.arange(1, rows + 1, dtype=torch.int32)
    pages = lengths * 10
    return SimpleNamespace(
        slot_mapping=torch.arange(rows, dtype=torch.int32),
        paged_kv_indptr=torch.arange(rows + 1, dtype=torch.int32),
        paged_kv_indices=pages,
        paged_kv_last_page_len=lengths,
        block_table=torch.stack((pages, torch.zeros_like(pages)), dim=1),
        kv_seq_lens=lengths,
        kv_seq_lens_host_values=lengths.tolist(),
        kv_cu_seq_lens=torch.cat((torch.zeros(1, dtype=torch.int32), lengths.cumsum(0, dtype=torch.int32))),
        q_cu_seq_lens=None,
        linear_state_indices=linear_state_indices,
        expanded_decode_metadata=None,
        is_prefill=False,
        is_chunked_prefill=False,
    )


def test_linear_state_indices_use_stable_graph_buffer() -> None:
    runner = _runner()
    input_ids = torch.arange(4, dtype=torch.int32)
    positions = torch.arange(4, dtype=torch.int32)
    metadata = _metadata(torch.tensor([3, 7, 11, 15], dtype=torch.int32))
    entry = runner._allocate_entry(
        padded_batch_size=8,
        input_ids=input_ids,
        positions=positions,
        metadata=metadata,
    )
    static_indices = entry.static_metadata.linear_state_indices
    data_ptr = static_indices.data_ptr()

    with patch(
        "xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata",
        create=True,
    ):
        runner._fill_entry(
            entry,
            input_ids,
            positions,
            metadata,
            batch_size=4,
            input_embedding=None,
        )
        assert static_indices.tolist() == [3, 7, 11, 15, 0, 0, 0, 0]

        metadata.linear_state_indices = torch.tensor(
            [4, 8, 12, 16],
            dtype=torch.int32,
        )
        runner._fill_entry(
            entry,
            input_ids,
            positions,
            metadata,
            batch_size=4,
            input_embedding=None,
        )

    assert static_indices.data_ptr() == data_ptr
    assert static_indices.tolist() == [4, 8, 12, 16, 0, 0, 0, 0]


@pytest.mark.parametrize(
    "capacity,peer_rows,padded_rows",
    [(8, 5, 8), (8, 8, 8), (7, 5, 7), (8, 9, None)],
    ids=["peer-bucket", "at-capacity", "partial-bucket", "over-capacity"],
)
def test_dp_empty_rank_uses_group_wide_acl_graph_bucket(capacity: int, peer_rows: int, padded_rows: int | None) -> None:
    runner = _runner()
    runner.dp_size = 2
    runner.dp_rank = 1
    runner.max_batch = capacity
    metadata = _metadata(torch.zeros(1, dtype=torch.int32))
    metadata.dp_execution_token_counts = (peer_rows, 1)
    metadata.dp_is_decode = (1, 1)
    input_ids = torch.zeros(1, dtype=torch.int32)
    assert runner.can_execute(input_ids, metadata) is (padded_rows is not None)
    if padded_rows is None:
        with pytest.raises(ValueError, match="decode batch exceeds ACL graph capacity"):
            runner._padded_batch_size(1, metadata)
    else:
        assert runner._padded_batch_size(1, metadata) == padded_rows


@pytest.mark.parametrize("counts,phases", [((3, 2), (0, 1)), ((3,), (1, 1))])
def test_dp_acl_graph_rejects_invalid_peers(counts: tuple[int, ...], phases: tuple[int, ...]) -> None:
    runner = _runner()
    runner.dp_size = 2
    metadata = _metadata(torch.arange(3, dtype=torch.int32))
    metadata.dp_execution_token_counts = counts
    metadata.dp_is_decode = phases
    if len(counts) != 2:
        with pytest.raises(RuntimeError, match="valid dp_execution_token_counts"):
            runner.can_execute(torch.zeros(3, dtype=torch.int32), metadata)
    else:
        assert not runner.can_execute(torch.zeros(3, dtype=torch.int32), metadata)


def test_dp_graph_variant_ids_distinguish_mtp_input_signatures() -> None:
    runner = _runner()
    runner.dp_size = 2
    ids = torch.ones(1, dtype=torch.int32)
    topk = torch.ones((1, 1, 4), dtype=torch.int32)
    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("xllm.python.distributed.all_gather") as gather,
    ):
        keys = []
        for indices, variants in ((None, (1, 7)), (topk, (2, 11)), (None, (1, 13))):
            gather.return_value = torch.tensor(variants, dtype=torch.int32)
            local_key = runner._graph_key(2, False, None, indices)
            key = runner._synchronize_dp_graph_key(local_key, ids)
            assert key == (*local_key[:-1], variants)
            keys.append(key)
        assert keys[0] != keys[2]
    assert [int(call.args[0].item()) for call in gather.call_args_list] == [1, 2, 1]


def test_mtp_graph_output_slices_and_detaches_replay_buffers() -> None:
    hidden = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    topk = torch.arange(24, dtype=torch.int64).reshape(4, 2, 3)

    sliced = DecodeAclGraphRunner._slice_output((hidden, None, topk), 2)

    assert isinstance(sliced, tuple)
    sliced_hidden, sliced_aux, sliced_topk = sliced
    assert sliced_aux is None
    assert torch.equal(sliced_hidden, hidden[:2])
    assert torch.equal(sliced_topk, topk[:2])
    assert sliced_hidden.data_ptr() != hidden.data_ptr()
    assert sliced_topk.data_ptr() != topk.data_ptr()
    hidden.zero_()
    topk.zero_()
    assert torch.count_nonzero(sliced_hidden) > 0
    assert torch.count_nonzero(sliced_topk) > 0


def test_mtp_graph_key_separates_first_step_and_topk_shapes() -> None:
    topk = torch.ones((4, 1, 8), dtype=torch.int32)
    key = DecodeAclGraphRunner._graph_key(8, False, None, topk)
    assert key != DecodeAclGraphRunner._graph_key(8, False, None)
    assert key == DecodeAclGraphRunner._graph_key(8, False, None, topk + 1)
    assert key != DecodeAclGraphRunner._graph_key(8, False, None, topk[:, :, :4])


def test_mtp_topk_input_changes_without_reallocating_capture_buffer() -> None:
    runner = _runner()
    input_ids = torch.arange(4, dtype=torch.int32)
    positions = input_ids.clone()
    metadata = _metadata(input_ids)
    topk = torch.arange(24, dtype=torch.int32).reshape(4, 2, 3)
    entry = runner._allocate_entry(8, input_ids, positions, metadata, topk)
    address = entry.static_mtp_topk_indices.data_ptr()

    with patch(
        "xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata",
        create=True,
    ):
        for source in (topk, topk.flip(0) + 7):
            runner._fill_entry(entry, input_ids, positions, metadata, 4, None, source)
            assert entry.static_mtp_topk_indices.data_ptr() == address
            torch.testing.assert_close(entry.static_mtp_topk_indices[:4], source)
            assert torch.count_nonzero(entry.static_mtp_topk_indices[4:]) == 0
