"""Process-lifetime CUDA graph capture resources.

CUDA graph allocations belong to a graph memory pool.  Reusing a pool and its
capture stream for the same static graph shape lets a later model load reclaim
that pool instead of growing device memory on each load/unload cycle.
"""

from __future__ import annotations

from collections.abc import Hashable

import torch

_RESOURCES: dict[tuple[str, str, Hashable], tuple[torch.cuda.Stream, object]] = {}


def get_capture_resources(
    namespace: str, device: str | torch.device, key: Hashable
) -> tuple[torch.cuda.Stream, object]:
    cuda_device = torch.device(device)
    resource_key = (namespace, str(cuda_device), key)
    resources = _RESOURCES.get(resource_key)
    if resources is None:
        resources = (
            torch.cuda.Stream(device=cuda_device),
            torch.cuda.graph_pool_handle(),
        )
        _RESOURCES[resource_key] = resources
    return resources
