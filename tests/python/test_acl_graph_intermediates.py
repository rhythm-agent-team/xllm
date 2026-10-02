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

"""Contracts and native consumers of forward-local ACLGraph intermediates."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from xllm.python import kernels
from xllm.python.attention import npu_paged_attention
from xllm.python.attention.npu_paged_attention import NpuPagedAttentionBackend
from xllm.python.kernels_npu import moe
from xllm.python.model_executor.forward_context import AclGraphExecutionState, ForwardContext, forward_context
from xllm.python.models import deepseek_v32, glm5_2


@pytest.mark.parametrize("graph_mode", [False, True])
def test_shared_expert_output_is_forward_local(graph_mode: bool) -> None:
    layer = deepseek_v32.DeepseekV3MoE.__new__(deepseek_v32.DeepseekV3MoE)
    torch.nn.Module.__init__(layer)
    layer.hidden = 16
    layer.layer_id = 3
    layer._fuse_shared_expert = True
    state = AclGraphExecutionState({}) if graph_mode else None

    def project(hidden: torch.Tensor, *, output: torch.Tensor | None) -> torch.Tensor:
        expected = hidden.to(torch.bfloat16) * 2
        if output is None:
            assert not graph_mode
            return expected
        assert graph_mode
        assert output.shape == hidden.shape
        assert output.dtype == torch.bfloat16
        assert output.device == hidden.device
        output.copy_(expected)
        return output

    layer.shared_experts = SimpleNamespace(forward_dequant_swiglu_quant=MagicMock(side_effect=project))
    with forward_context(ForwardContext(None, torch.device("cpu"), None, [], execution_state=state)):
        for rows in (2, 5, 2):
            hidden = torch.arange(rows * 16).reshape(rows, 16).to(torch.bfloat16)
            actual = layer._run_shared_experts(hidden)
            torch.testing.assert_close(actual, hidden * 2)
    assert layer.shared_experts.forward_dequant_swiglu_quant.call_count == 3
    if state is not None:
        assert not state.persistent_buffers


@pytest.mark.parametrize("graph_mode", [False, True])
def test_sparse_attention_keeps_out_contract_without_retention(
    monkeypatch: pytest.MonkeyPatch, graph_mode: bool
) -> None:
    backend = NpuPagedAttentionBackend(4, 1, 16, 0.25, 0, True, torch.device("cpu"), torch.bfloat16)
    state = AclGraphExecutionState({}) if graph_mode else None
    calls: list[torch.Tensor] = []

    def attention(*args: object) -> torch.Tensor:
        query, output = args[0], args[-1]
        assert isinstance(query, torch.Tensor) and isinstance(output, torch.Tensor)
        assert output.shape == query.shape
        assert output.dtype == query.dtype
        assert output.device == query.device
        output.copy_(query + 1)
        calls.append(output)
        return output

    monkeypatch.setattr(npu_paged_attention.kernels, "sparse_flash_attention_out", attention)
    with forward_context(ForwardContext(backend, torch.device("cpu"), None, [], execution_state=state)):
        for rows in (2, 5, 2):
            query = torch.zeros(rows, 4, 16, dtype=torch.bfloat16)
            cache = torch.zeros(1, 128, 1, 16, dtype=torch.bfloat16)
            for layer_id in (0, 1):
                actual = backend._mla_sparse(
                    query,
                    None,
                    cache,
                    None,
                    torch.zeros(rows, 1, 4, dtype=torch.int32),
                    torch.zeros(1, 1, dtype=torch.int32),
                    torch.tensor([rows], dtype=torch.int32),
                    torch.tensor([128], dtype=torch.int32),
                    layer_id,
                )
                assert actual is calls[-1]
                torch.testing.assert_close(actual, query + 1)
    assert len(calls) == 6
    if state is not None:
        assert not state.persistent_buffers


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("cp", [False, True])
def test_indexer_returns_native_or_cp_scatter_without_retention(
    monkeypatch: pytest.MonkeyPatch, quantized: bool, cp: bool
) -> None:
    cfg = glm5_2.Glm52Config(
        hidden_size=16,
        q_lora_rank=16,
        index_n_heads=2,
        index_head_dim=128,
        qk_rope_head_dim=64,
        index_topk=4,
        indexer_rope_interleave=False,
    )
    indexer = glm5_2.Glm52Indexer(cfg, torch.bfloat16, torch.device("cpu"))
    state = AclGraphExecutionState({})
    returned: list[torch.Tensor] = []

    def select(*args: object) -> torch.Tensor:
        query = args[0]
        assert isinstance(query, torch.Tensor)
        result = torch.arange(query.shape[0] * 4, dtype=torch.int32).view(-1, 1, 4)
        returned.append(result)
        return result

    def select_out(*args: object) -> torch.Tensor:
        indices, values = args[-2:]
        assert isinstance(indices, torch.Tensor) and isinstance(values, torch.Tensor)
        query = args[0]
        assert isinstance(query, torch.Tensor)
        assert indices.shape == (query.shape[0], 1, 4)
        assert indices.dtype == torch.int32 and values.dtype == torch.bfloat16
        assert values.shape == indices.shape and indices.device == values.device == query.device
        indices.copy_(select(*args[:-2]))
        values.zero_()
        returned[-1] = indices
        return indices

    monkeypatch.setattr(kernels, "quant_lightning_indexer", select)
    monkeypatch.setattr(kernels, "lightning_indexer_out", select_out)
    monkeypatch.setattr(kernels, "dynamic_quant", lambda value: (value.to(torch.int8), torch.ones(value.shape[:-1])))
    with (
        patch.object(
            indexer,
            "_project_index_inputs",
            side_effect=lambda hidden, _: (torch.zeros(hidden.shape[0], 128), torch.ones(hidden.shape[0], 2)),
        ),
        patch.object(
            indexer, "_project_query", side_effect=lambda qr, _: torch.ones(qr.shape[0], 2, 128, dtype=torch.bfloat16)
        ),
        patch.object(indexer, "_update_index_cache"),
        forward_context(ForwardContext(None, torch.device("cpu"), None, [], execution_state=state)),
    ):
        for rows in (2, 5, 2):
            cache = torch.zeros(1, 128, 1, 128, dtype=torch.int8 if quantized else torch.bfloat16)
            scales = torch.ones(1, 128, 1, 1, dtype=torch.float16) if quantized else None
            plan = SimpleNamespace(total_local=rows + 1, query_index=torch.arange(rows)) if cp else None
            ctx = SimpleNamespace(
                actual_seq_q=torch.tensor([rows], dtype=torch.int32),
                actual_seq_kv=torch.tensor([128], dtype=torch.int32),
                cp_context=plan,
                index_cache=cache,
                index_cache_scale=scales,
                materialize_index_cache=lambda cache=cache, scales=scales: (
                    cache,
                    scales,
                    torch.zeros(1, 1, dtype=torch.int32),
                ),
                get_quant_indexer_metadata=lambda *_: torch.empty(0, dtype=torch.int32),
            )
            hidden = torch.zeros(rows, 16, dtype=torch.bfloat16)
            rope = (torch.ones(rows, 64), torch.zeros(rows, 64))
            actual = indexer.select_qli(hidden, hidden, ctx, rope, rope)
            if cp:
                assert actual.shape == (rows + 1, 1, 4)
                torch.testing.assert_close(actual[:rows], returned[-1])
                assert actual[-1].eq(-1).all()
            else:
                assert actual is returned[-1]
            assert not state.persistent_buffers


def test_shared_dsa_layer_passes_prev_topk_without_copy() -> None:
    layer = glm5_2.Glm52MLAAttention.__new__(glm5_2.Glm52MLAAttention)
    torch.nn.Module.__init__(layer)
    layer.cfg = glm5_2.Glm52Config(indexer_rope_interleave=False)
    layer.layer_id = 1
    layer.indexer = None
    backend = MagicMock()
    state = AclGraphExecutionState({})
    hidden = torch.zeros(2, 16)
    rope = torch.ones(2, 64)
    topk = torch.arange(8, dtype=torch.int32).reshape(2, 1, 4)
    with forward_context(
        ForwardContext(
            backend,
            hidden.device,
            SimpleNamespace(is_prefill=False, is_chunked_prefill=False),
            [],
            execution_state=state,
        )
    ):
        actual = layer._select_topk(hidden, hidden, backend, rope, rope, rope, rope, (rope, rope), topk, False)
    assert actual is topk
    backend.mla_index_context.assert_not_called()
    assert not state.persistent_buffers


@pytest.fixture
def npu_device() -> torch.device:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("native intermediate tests require an available Ascend NPU")
    return torch.device("npu", torch.npu.current_device())


@pytest.mark.parametrize("group_list_type", [0, 2])
@torch.inference_mode()
def test_native_gmm2_temporary_output_reaches_unpermute(npu_device: torch.device, group_list_type: int) -> None:
    torch.manual_seed(2403)
    entries = []
    for rows in (4, 8):
        hidden = torch.randn(rows, 128, dtype=torch.bfloat16, device=npu_device)
        ids = torch.tensor([[0, 2]], dtype=torch.int32, device=npu_device).expand(rows, -1).contiguous()
        probs = torch.full((rows, 2), 0.5, dtype=torch.bfloat16, device=npu_device)
        weight = moe.format_cast_nz(torch.randint(-3, 4, (4, 128, 128), dtype=torch.int8, device=npu_device))
        weight_scale = torch.full((4, 128), 1 / 64, dtype=torch.bfloat16, device=npu_device)
        # Sparse groups place active experts before the zero-count tail.
        expert_ids = torch.tensor([0, 2, 1, 3], device=npu_device, dtype=torch.int64)
        state = AclGraphExecutionState({})

        def run(
            hidden: torch.Tensor = hidden,
            ids: torch.Tensor = ids,
            probs: torch.Tensor = probs,
            weight: torch.Tensor = weight,
            weight_scale: torch.Tensor = weight_scale,
            expert_ids: torch.Tensor = expert_ids,
            state: AclGraphExecutionState | None = state,
        ) -> torch.Tensor:
            with forward_context(ForwardContext(None, npu_device, None, [], execution_state=state)):
                routed, row_ids, groups, scales = moe.torch_npu.npu_moe_init_routing_v2(
                    hidden,
                    ids,
                    scale=None,
                    active_num=hidden.shape[0] * 2,
                    expert_num=4,
                    expert_tokens_num_type=0,
                    expert_tokens_num_flag=True,
                    active_expert_range=[0, 4],
                    quant_mode=1,
                )
                if group_list_type == 2:
                    counts = torch.diff(torch.cat((groups.new_zeros(1), groups)))
                    groups = torch.stack((expert_ids, counts.index_select(0, expert_ids)), -1)
                output = moe._grouped_matmul_gmm2(
                    act_i8=routed,
                    act_pertoken_scale=scales,
                    weight=weight,
                    weight_scale=weight_scale,
                    group_list=groups,
                    group_list_type=group_list_type,
                )
                return moe.torch_npu.npu_moe_token_unpermute(
                    permuted_tokens=output,
                    sorted_indices=row_ids.abs(),
                    probs=probs,
                )

        for _ in range(3):
            run()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            actual = run()
        assert not state.persistent_buffers
        entries.append((hidden, ids, probs, weight, weight_scale, state, run, graph, actual))

    snapshots: dict[int, torch.Tensor] = {}
    for entry_id in (0, 1, 0):
        hidden, ids, probs, weight, weight_scale, state, run, graph, actual = entries[entry_id]
        hidden.copy_(torch.randn_like(hidden))
        eager = run(state=None)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(actual, eager, rtol=2e-2, atol=2e-2)
        # Independent FP32 matmuls of the quantized inputs, not another out call.
        quantized, scales = kernels.dynamic_quant(hidden)
        x = quantized.cpu().float()
        scale = scales.cpu().float().reshape(-1, 1)
        w = weight.cpu().float()
        ws = weight_scale.cpu().float()
        reference = sum((x @ w[expert] * scale * ws[expert]).to(torch.bfloat16).float() * 0.5 for expert in (0, 2)).to(
            torch.bfloat16
        )
        torch.testing.assert_close(actual.cpu(), reference, rtol=3e-2, atol=3e-2)
        assert actual.shape == hidden.shape and actual.dtype == torch.bfloat16
        assert not state.persistent_buffers
        for previous_id, snapshot in snapshots.items():
            if previous_id != entry_id:
                torch.testing.assert_close(entries[previous_id][-1].cpu(), snapshot, rtol=0, atol=0)
        snapshots[entry_id] = actual.cpu().clone()


@pytest.mark.parametrize("quantized", [False, True])
@torch.inference_mode()
def test_native_indexer_temporary_output_reaches_sparse_attention(npu_device: torch.device, quantized: bool) -> None:
    torch.manual_seed(2405)
    # Production QLI has G_SIZE=64, D=128 and topk=2048; SFA consumes
    # four local heads, latent rank 512 and a separate 64-wide RoPE cache.
    indexer = glm5_2.Glm52Indexer(
        glm5_2.Glm52Config(
            hidden_size=128, q_lora_rank=128, index_n_heads=64, index_head_dim=128, qk_rope_head_dim=64, index_topk=2048
        ),
        torch.bfloat16,
        npu_device,
    )
    shared = glm5_2.Glm52MLAAttention.__new__(glm5_2.Glm52MLAAttention)
    torch.nn.Module.__init__(shared)
    shared.cfg = glm5_2.Glm52Config(indexer_rope_interleave=False)
    shared.layer_id = 1
    shared.indexer = None
    backend = NpuPagedAttentionBackend(4, 1, 512, 576**-0.5, 0, True, npu_device, torch.bfloat16)
    entries = []
    for rows in (1, 2):
        iq = (
            torch.randint(-8, 9, (rows, 64, 128), device=npu_device, dtype=torch.int8)
            if quantized
            else torch.randn(rows, 64, 128, device=npu_device, dtype=torch.bfloat16)
        )
        ik = (
            torch.randint(-8, 9, (1, 128, 1, 128), device=npu_device, dtype=torch.int8)
            if quantized
            else torch.randn(1, 128, 1, 128, device=npu_device, dtype=torch.bfloat16)
        )
        weights = torch.ones(rows, 64, device=npu_device, dtype=torch.float16 if quantized else torch.bfloat16)
        qs = torch.full((rows, 64), 1 / 8, device=npu_device, dtype=torch.float16)
        ks = torch.full((1, 128, 1), 1 / 8, device=npu_device, dtype=torch.float16)
        q_ends = torch.tensor([rows], device=npu_device, dtype=torch.int32)
        kv_lengths = torch.tensor([128], device=npu_device, dtype=torch.int32)
        blocks = torch.zeros(1, 1, device=npu_device, dtype=torch.int32)
        metadata = (
            kernels.quant_lightning_indexer_metadata(
                64,
                1,
                128,
                q_ends,
                kv_lengths,
                rows,
                128,
                2048,
                1,
            )
            if quantized
            else None
        )
        q = torch.randn(rows, 4, 512, device=npu_device, dtype=torch.bfloat16) / 8
        q_rope = torch.randn(rows, 4, 64, device=npu_device, dtype=torch.bfloat16) / 8
        caches = [torch.randn(1, 128, 1, 512, device=npu_device, dtype=torch.bfloat16) for _ in range(2)]
        rope_caches = [torch.randn(1, 128, 1, 64, device=npu_device, dtype=torch.bfloat16) / 8 for _ in range(2)]
        uv = [torch.randn(4, 512, 128, device=npu_device, dtype=torch.bfloat16) / 32 for _ in range(2)]
        state = AclGraphExecutionState({})
        ctx = SimpleNamespace(actual_seq_q=q_ends, actual_seq_kv=kv_lengths)

        def run(
            iq: torch.Tensor = iq,
            ik: torch.Tensor = ik,
            weights: torch.Tensor = weights,
            qs: torch.Tensor = qs,
            ks: torch.Tensor = ks,
            metadata: torch.Tensor | None = metadata,
            q_ends: torch.Tensor = q_ends,
            kv_lengths: torch.Tensor = kv_lengths,
            blocks: torch.Tensor = blocks,
            q: torch.Tensor = q,
            q_rope: torch.Tensor = q_rope,
            caches: list[torch.Tensor] = caches,
            rope_caches: list[torch.Tensor] = rope_caches,
            uv: list[torch.Tensor] = uv,
            ctx: SimpleNamespace = ctx,
            state: AclGraphExecutionState | None = state,
            return_topk: bool = False,
        ) -> tuple[torch.Tensor, ...]:
            with forward_context(
                ForwardContext(
                    backend,
                    npu_device,
                    SimpleNamespace(is_prefill=False, is_chunked_prefill=False),
                    [],
                    execution_state=state,
                )
            ):
                if quantized:
                    topk = kernels.quant_lightning_indexer(
                        iq, ik, weights, qs, ks, metadata, q_ends, kv_lengths, blocks, 2048, 1
                    )
                else:
                    topk = indexer._select_unquantized(iq, ik, weights, ctx, blocks)
                projected = []
                for layer_id in (0, 1):
                    # Exercise the existing cross-layer contract, not a new cache API.
                    topk = shared._select_topk(
                        q, q, backend, q_rope, q_rope, q_rope, q_rope, (q_rope, q_rope), topk, False
                    )
                    attention = backend._mla_sparse(
                        q, q_rope, caches[layer_id], rope_caches[layer_id], topk, blocks, q_ends, kv_lengths, layer_id
                    )
                    projected.append(kernels.atb_matmul_ein_sum(attention, uv[layer_id]))
                # Do not retain captured TopK as a graph output.
                return (topk, *projected) if return_topk else tuple(projected)

        for _ in range(3):
            run()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            actual = run()
        assert not state.persistent_buffers
        entries.append((iq, ik, q, q_rope, caches, rope_caches, uv, state, run, graph, actual))

    snapshots: dict[int, tuple[torch.Tensor, ...]] = {}
    for entry_id in (0, 1, 0):
        iq, ik, q, q_rope, caches, rope_caches, uv, state, run, graph, actual = entries[entry_id]
        if quantized:
            iq.random_(-8, 9)
            ik.random_(-8, 9)
        else:
            iq.copy_(torch.randn_like(iq))
            ik.copy_(torch.randn_like(ik))
        q.copy_(torch.randn_like(q) / 8)
        for cache in caches:
            cache.copy_(torch.randn_like(cache))
        eager = run(state=None, return_topk=True)
        graph.replay()
        torch.npu.synchronize()
        assert len(actual) == 2
        assert eager[0].shape == (q.shape[0], 1, 2048) and eager[0].dtype == torch.int32
        # Check eager indices directly; captured indices remain local to consumers.
        # topk exceeds every causal prefix: all visible tokens occur once and
        # padding is -1. Neither indexer scores nor another out call is the oracle.
        indices = eager[0].cpu()
        for row in range(q.shape[0]):
            visible = 128 - q.shape[0] + row + 1
            torch.testing.assert_close(
                indices[row, 0][indices[row, 0] >= 0].sort().values,
                torch.arange(visible, dtype=torch.int32),
                rtol=0,
                atol=0,
            )
            assert indices[row, 0][indices[row, 0] < 0].eq(-1).all()
        for layer_id in (0, 1):
            query, query_rope = q.cpu().float(), q_rope.cpu().float()
            keys = caches[layer_id].cpu().float().view(128, 512)
            ropes = rope_caches[layer_id].cpu().float().view(128, 64)
            logits = (
                torch.einsum("thd,kd->thk", query, keys) + torch.einsum("thd,kd->thk", query_rope, ropes)
            ) * backend.scale
            causal = torch.arange(128)[None, :] <= (128 - q.shape[0] + torch.arange(q.shape[0]))[:, None]
            logits.masked_fill_(~causal[:, None, :], float("-inf"))
            attention = torch.einsum("thk,kd->thd", logits.softmax(-1), keys).to(torch.bfloat16).float()
            reference = torch.einsum("thd,hdo->tho", attention, uv[layer_id].cpu().float())
            assert actual[layer_id].shape == (q.shape[0], 4, 128)
            assert actual[layer_id].dtype == torch.bfloat16
            torch.testing.assert_close(actual[layer_id].cpu().float(), reference, rtol=5e-2, atol=5e-2)
            torch.testing.assert_close(actual[layer_id], eager[layer_id + 1], rtol=2e-2, atol=2e-2)
        assert not state.persistent_buffers
        for previous_id, snapshot in snapshots.items():
            if previous_id != entry_id:
                for value, previous in zip(entries[previous_id][-1], snapshot, strict=True):
                    torch.testing.assert_close(value.cpu(), previous, rtol=0, atol=0)
        snapshots[entry_id] = tuple(value.cpu().clone() for value in actual)


@pytest.mark.parametrize("gate_overlap", [False, True])
@torch.inference_mode()
def test_native_shared_expert_fork_join_replays_local_outputs(
    npu_device: torch.device, monkeypatch: pytest.MonkeyPatch, gate_overlap: bool
) -> None:
    monkeypatch.setenv("XLLM_MOE_GATE_OVERLAP", "1" if gate_overlap else "0")
    monkeypatch.setenv("XLLM_MOE_FINE_OVERLAP", "0")
    torch.manual_seed(2404)
    cfg = glm5_2.Glm52Config(
        hidden_size=128, moe_intermediate_size=256, n_routed_experts=8, num_experts_per_tok=2, n_shared_experts=1
    )
    layers = [glm5_2.Glm52MoE(cfg, layer_id, torch.bfloat16, npu_device) for layer_id in (0, 1)]
    for layer in layers:
        layer.gate.weight.copy_(torch.randn_like(layer.gate.weight) / 16)
        layer.allocate_experts_w13_for_loading()
        layer.experts_w13.random_(-3, 4)
        layer.experts_w13_scale.fill_(1 / 16)
        layer.process_experts_w13_after_loading()
        layer.allocate_experts_w2_for_loading()
        layer.experts_w2.random_(-3, 4)
        layer.experts_w2_scale_compute.fill_(1 / 64)
        layer.process_experts_w2_after_loading()
        for projection in (layer.shared_experts.gate_up_proj, layer.shared_experts.down_proj):
            projection.weight.random_(-3, 4)
            projection.weight_scale.fill_(1 / 16)
            projection.weight_offset.zero_()
        layer.shared_experts.process_weights_after_loading()

    entries = []
    for rows in (4, 8):
        hidden = torch.randn(rows, 128, dtype=torch.bfloat16, device=npu_device)
        state = AclGraphExecutionState({})

        def run(
            hidden: torch.Tensor = hidden, state: AclGraphExecutionState | None = state, parallel: bool = True
        ) -> torch.Tensor:
            with forward_context(
                ForwardContext(
                    None,
                    npu_device,
                    SimpleNamespace(is_prefill=False, is_chunked_prefill=False),
                    [],
                    execution_state=state,
                )
            ):
                value = hidden
                for layer in layers:
                    if parallel:
                        value = layer(value)
                    else:
                        value = layer._combine_expert_outputs(
                            layer._run_routed_experts(value), layer._run_shared_experts(value)
                        )
                return value

        for _ in range(3):
            run()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            actual = run()
        assert not state.persistent_buffers
        entries.append((hidden, state, run, graph, actual))

    snapshots: dict[int, torch.Tensor] = {}
    for entry_id in (0, 1, 0):
        hidden, state, run, graph, actual = entries[entry_id]
        hidden.copy_(torch.randn_like(hidden))
        expected = run(state=None, parallel=False)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        assert actual.shape == hidden.shape and actual.dtype == torch.bfloat16
        assert not state.persistent_buffers
        for previous_id, snapshot in snapshots.items():
            if previous_id != entry_id:
                torch.testing.assert_close(entries[previous_id][-1].cpu(), snapshot, rtol=0, atol=0)
        snapshots[entry_id] = actual.cpu().clone()
