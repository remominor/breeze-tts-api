"""CUDA graph capture resources shared while a model is loaded.

CUDA graph allocations belong to a graph memory pool.  Reusing a pool and its
capture stream for the same static graph shape prevents duplicate captures
within a loaded model.  The pools must be released when that model is unloaded
so their VRAM can be returned to the CUDA allocator.
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


def clear_capture_resources() -> None:
    """Release CUDA graph pool handles and capture streams after model unload.

    Call this only after all CUDA graph objects using these pools have been
    discarded and their streams synchronized.
    """
    _RESOURCES.clear()
