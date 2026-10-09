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

from tests.npu.shmem_all_gather_worker import _payload


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
