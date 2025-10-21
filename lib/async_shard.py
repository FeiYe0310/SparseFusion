from __future__ import annotations

from typing import List, Sequence, Tuple

import jax.numpy as jnp


def _compute_shard_slices(num_params: int, num_nodes: int) -> List[Tuple[int, int]]:
    """Evenly split parameter indices across ``num_nodes`` slices."""
    base = num_params // num_nodes
    remainder = num_params % num_nodes
    slices: List[Tuple[int, int]] = []
    start = 0
    for idx in range(num_nodes):
        extra = 1 if idx < remainder else 0
        end = start + base + extra
        slices.append((start, end))
        start = end
    return slices


class AsyncShardCoordinator:
    """Tracks shard-local parameter updates and periodic global synchronisation."""

    def __init__(
        self,
        num_params: int,
        num_nodes: int,
        sync_interval: int = 10,
        *,
        dtype=jnp.bfloat16,
    ) -> None:
        if num_nodes < 1:
            raise ValueError("async shard coordinator requires at least one node")
        self.num_params = num_params
        self.num_nodes = num_nodes
        self.sync_interval = max(1, sync_interval)
        self.dtype = dtype
        self.shard_slices = _compute_shard_slices(num_params, num_nodes)

        self._global_vector: jnp.ndarray | None = None
        self._pending_slices: List[jnp.ndarray | None] = [None] * num_nodes
        self._unsynced: List[bool] = [False] * num_nodes
        self._iteration: int = 0
        self._last_synced_nodes: List[int] = []

    def bootstrap(self, initial_params: jnp.ndarray) -> None:
        """Initialise the coordinator with a reference parameter vector."""
        initial = jnp.asarray(initial_params, dtype=self.dtype)
        if initial.shape[0] != self.num_params:
            raise ValueError(
                f"expected flattened parameter vector of length {self.num_params}, "
                f"got {initial.shape[0]}"
            )
        self._global_vector = initial
        for idx, (start, end) in enumerate(self.shard_slices):
            self._pending_slices[idx] = initial[start:end]
            self._unsynced[idx] = False
        self._last_synced_nodes = list(range(self.num_nodes))

    def prepare_candidate(
        self, candidate: jnp.ndarray, owner_idx: int
    ) -> jnp.ndarray:
        """Combine a shard-local update with the last globally-synchronised model."""
        if owner_idx < 0 or owner_idx >= self.num_nodes:
            raise ValueError(f"invalid shard owner index: {owner_idx}")

        candidate_arr = jnp.asarray(candidate, dtype=self.dtype)
        if self._global_vector is None:
            self.bootstrap(candidate_arr)

        assert self._global_vector is not None  # for type-checkers
        start, end = self.shard_slices[owner_idx]

        base = self._global_vector
        if self._unsynced[owner_idx] and self._pending_slices[owner_idx] is not None:
            base = base.at[start:end].set(self._pending_slices[owner_idx])

        return base.at[start:end].set(candidate_arr[start:end])

    def commit(self, updated_params: jnp.ndarray, owner_idx: int) -> None:
        """Record a shard-local update and perform a global sync when required."""
        if owner_idx < 0 or owner_idx >= self.num_nodes:
            raise ValueError(f"invalid shard owner index: {owner_idx}")

        updated = jnp.asarray(updated_params, dtype=self.dtype)
        start, end = self.shard_slices[owner_idx]
        self._pending_slices[owner_idx] = updated[start:end]
        self._unsynced[owner_idx] = True

        self._iteration += 1
        self._last_synced_nodes = []

        if self._iteration % self.sync_interval != 0:
            return

        if self._global_vector is None:
            self._global_vector = jnp.zeros(self.num_params, dtype=self.dtype)

        new_global = self._global_vector
        synced_nodes: List[int] = []
        for idx, (slice_start, slice_end) in enumerate(self.shard_slices):
            shard_vals = self._pending_slices[idx]
            if shard_vals is None:
                shard_vals = new_global[slice_start:slice_end]
            else:
                synced_nodes.append(idx)
            new_global = new_global.at[slice_start:slice_end].set(shard_vals)
            self._pending_slices[idx] = shard_vals

        self._global_vector = new_global
        if not synced_nodes:
            synced_nodes = list(range(self.num_nodes))
        self._last_synced_nodes = synced_nodes
        self._unsynced = [False] * self.num_nodes

    def synced(self) -> bool:
        """Return True if the most recent commit triggered a global sync."""
        return bool(self._last_synced_nodes)

    def synced_nodes(self) -> Sequence[int]:
        """Return the shard indices that were refreshed during the last sync."""
        return tuple(self._last_synced_nodes)

    def global_params(self) -> jnp.ndarray | None:
        """Return the last globally-synchronised parameter vector."""
        return self._global_vector

