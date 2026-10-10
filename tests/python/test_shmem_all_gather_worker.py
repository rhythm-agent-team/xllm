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

"""Full-element equivalence of the AllGather worker's CPU payload oracle."""

import pytest
import torch

from tests.npu.shmem_all_gather_worker import _check_guards, _guarded, _payload


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
