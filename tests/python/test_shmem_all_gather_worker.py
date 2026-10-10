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

"""AllGather worker payload, boundary, and profile-state checks."""

import pytest
import torch

from tests.npu.shmem_all_gather_worker import (
    _check_guards,
    _check_profile_output,
    _check_profile_tensor,
    _guarded,
    _payload,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("count", [1, 511, 512, 513, 262143, 262144, 262145, 2478080])
def test_payload_matches_full_position_formula(dtype: torch.dtype, count: int) -> None:
    positions = torch.arange(count, dtype=torch.int64)
    rows = positions // 512
    columns = positions % 512
    for rank, iteration in ((0, -1300), (1, -1), (15, 6)):
        identity = rank + 16 * (iteration + 2)
        expected = ((identity + (2 * iteration + 1) * columns + 31 * rows) % 512 - 256).to(dtype)
        actual = _payload(rank, count, iteration, dtype)
        assert actual.shape == expected.shape
        assert actual.dtype == dtype
        assert actual.is_contiguous()
        assert torch.equal(actual, expected)


def test_payload_changes_between_calls_and_has_independent_storage() -> None:
    first = _payload(1, 262145, -1, torch.bfloat16)
    repeated = _payload(1, 262145, -1, torch.bfloat16)
    assert torch.equal(first, repeated)
    assert not torch.equal(first, _payload(1, 262145, 0, torch.bfloat16))
    assert not torch.equal(first, _payload(2, 262145, -1, torch.bfloat16))
    first.fill_(0)
    assert torch.count_nonzero(repeated) > 0


@pytest.mark.parametrize("count", [1, 63, 64, 65, 262145])
def test_guards_check_every_boundary_without_copying_payload(count: int, monkeypatch: pytest.MonkeyPatch) -> None:
    storage, payload, guard = _guarded(count, torch.bfloat16, torch.device("cpu"))
    payload.fill_(float("nan"))
    copied = []
    original_cpu = torch.Tensor.cpu

    def _record_cpu(tensor: torch.Tensor) -> torch.Tensor:
        copied.append(tensor.numel())
        return original_cpu(tensor)

    monkeypatch.setattr(torch.Tensor, "cpu", _record_cpu)
    _check_guards(storage, count, guard)
    assert copied == [guard, storage.numel() - guard - count]
    for index in range(guard):
        storage[index] = 0
        with pytest.raises(AssertionError, match="Leading guard changed"):
            _check_guards(storage, count, guard)
        storage[index] = -123
    for index in range(guard + count, storage.numel()):
        storage[index] = 0
        with pytest.raises(AssertionError, match="Tail padding or trailing guard changed"):
            _check_guards(storage, count, guard)
        storage[index] = -123


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("executed", [True, False, None])
def test_profile_tensor_checks_full_payload_and_execution_state(dtype: torch.dtype, executed: bool | None) -> None:
    expected = _payload(1, 1025, 0, dtype)
    actual = expected.clone()
    if executed is False:
        with pytest.raises(AssertionError, match="Unexpected retained output state"):
            _check_profile_tensor(actual, expected, executed)
    else:
        assert _check_profile_tensor(actual, expected, executed) == "CORRECT"
    actual.fill_(float("nan"))
    if executed is True:
        with pytest.raises(AssertionError):
            _check_profile_tensor(actual, expected, executed)
    else:
        assert _check_profile_tensor(actual, expected, executed) == "POISONED"
    for index in (0, 511, 512, 1024):
        for value in (float("nan"), float("inf"), float("-inf"), expected[index].item() + 1):
            actual.copy_(expected)
            actual[index] = value
            with pytest.raises(AssertionError):
                _check_profile_tensor(actual, expected, executed)
        actual.fill_(float("nan"))
        actual[index] = expected[index]
        with pytest.raises(AssertionError):
            _check_profile_tensor(actual, expected, executed)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_executed_profile_tensor_skips_poison_scan_and_keeps_failure_diagnostic(
    dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = torch.tensor([0, -1, 255], dtype=dtype)
    actual = expected.clone()
    actual[0] = -0.0

    def _unexpected_poison_scan(tensor: torch.Tensor) -> torch.Tensor:
        pytest.fail("Executed profile output must not scan for poison")

    monkeypatch.setattr(torch, "isnan", _unexpected_poison_scan)
    assert _check_profile_tensor(actual, expected, True) == "CORRECT"
    actual[-1] += 1
    with pytest.raises(AssertionError, match="Greatest absolute difference.*index"):
        _check_profile_tensor(actual, expected, True)


@pytest.mark.parametrize("executed", [True, False, None])
def test_profile_tensor_rejects_metadata_changes(executed: bool | None) -> None:
    expected = torch.zeros(4, dtype=torch.float32)
    # torch.equal alone accepts both a dtype change and these equal values.
    actual = expected.to(torch.float16)
    assert torch.equal(actual, expected)
    with pytest.raises(AssertionError, match="Profile tensor dtype changed"):
        _check_profile_tensor(actual, expected, executed)
    with pytest.raises(AssertionError, match="Profile tensor shape changed"):
        _check_profile_tensor(expected.view(2, 2), expected, executed)
    with pytest.raises(AssertionError, match="Profile tensor device changed"):
        _check_profile_tensor(torch.empty_like(expected, device="meta"), expected, executed)
    with pytest.raises(AssertionError, match="Profile tensor layout changed"):
        _check_profile_tensor(expected.to_sparse(), expected, executed)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("world_size", [2, 8, 16])
def test_profile_output_checks_every_rank_and_phase(dtype: torch.dtype, world_size: int) -> None:
    count = 513
    iterations = (0, -1300, 6)
    positions = torch.arange(count, dtype=torch.int64)
    rows = positions // 512
    columns = positions % 512
    for iteration in iterations:
        actual = torch.cat(
            [
                ((peer + 16 * (iteration + 2) + (2 * iteration + 1) * columns + 31 * rows) % 512 - 256).to(dtype)
                for peer in range(world_size)
            ]
        )
        for executed in (True, None):
            assert _check_profile_output(actual, count, iteration, world_size, dtype, executed) == "CORRECT"
        for peer in range(world_size):
            for index in (0, 511, 512):
                position = peer * count + index
                actual[position] += 1
                with pytest.raises(AssertionError):
                    _check_profile_output(actual, count, iteration, world_size, dtype, True)
                actual[position] -= 1
        with pytest.raises(AssertionError):
            _check_profile_output(actual, count, iteration + 1, world_size, dtype, True)
        with pytest.raises(AssertionError, match="Unexpected retained output state"):
            _check_profile_output(actual, count, iteration, world_size, dtype, False)
        poisoned = torch.full_like(actual, float("nan"))
        for executed in (False, None):
            assert _check_profile_output(poisoned, count, iteration, world_size, dtype, executed) == "POISONED"
        with pytest.raises(AssertionError):
            _check_profile_output(poisoned, count, iteration, world_size, dtype, True)
        poisoned[:count] = actual[:count]
        with pytest.raises(AssertionError, match="partially executed ranks"):
            _check_profile_output(poisoned, count, iteration, world_size, dtype, None)
        with pytest.raises(AssertionError, match="Profile output shape changed"):
            _check_profile_output(actual[:-1], count, iteration, world_size, dtype, True)
