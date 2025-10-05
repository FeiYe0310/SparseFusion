"""Experimental scaffolding for a sharded archive backend.

This module sketches out how we *could* manage the evolutionary archive in a
distributed fashion using JAX's SPMD primitives without touching the existing
`natural_niches_fn` implementation.  Nothing here is wired into the pipeline
yet; the functions either return placeholders or raise ``NotImplementedError``.

The intention is that future work can experiment inside this file without
breaking the current GPU/CPU archive paths.  Once the implementation is
feature-complete it can be swapped into `natural_niches_fn`.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class ShardedArchiveConfig:
    """Configuration knobs for the experimental sharded archive backend."""

    pop_size: int
    num_params: int
    num_datapoints: int
    axis_name: str = "archive_devices"


@dataclass
class ShardedArchiveState:
    """Container for the per-shard archive / score matrices."""

    archive: jax.Array
    scores: jax.Array


def initialize_state(config: ShardedArchiveConfig) -> ShardedArchiveState:
    """Allocate the archive / score buffers with a population-axis sharding.

    This currently returns device-local zeros using `jnp.zeros`; the caller can
    wrap it in ``pjit`` or place sharding constraints as needed.  The goal is
    to keep the archive logic isolated from the main training loop.
    """

    pop = config.pop_size
    num_params = config.num_params
    num_datapoints = config.num_datapoints

    archive = jnp.zeros((pop, num_params), dtype=jnp.bfloat16)
    scores = jnp.zeros((pop, num_datapoints), dtype=jnp.float32)
    return ShardedArchiveState(archive=archive, scores=scores)


def sample_parents(
    state: ShardedArchiveState,
    rng_key: jax.Array,
    alpha: float,
    use_matchmaker: bool = True,
) -> Tuple[jax.Array, jax.Array]:
    """Placeholder for sharded parent sampling.

    The real implementation needs to combine per-shard fitness statistics and
    broadcast the chosen indices to every device.  For now we only raise to
    signal that this is intentionally incomplete.
    """

    archive = state.archive
    scores = state.scores

    k1, k2 = jax.random.split(rng_key)
    z = scores.sum(axis=0)
    z = jnp.where(z, z, 1) ** alpha
    fitness_matrix = scores / z[None, :]
    fitness = jnp.sum(fitness_matrix, axis=1)
    probs = fitness / jnp.sum(fitness)

    if probs.size == 0:
        raise ValueError("Archive is empty; cannot sample parents.")

    if jnp.any(jnp.isnan(probs)):
        raise ValueError("Encountered NaNs while sampling parents.")

    if use_matchmaker:
        parent_1_idx = jax.random.choice(k1, probs.size, shape=(1,), p=probs)[0]
        match_score = jnp.maximum(0, fitness_matrix - fitness_matrix[parent_1_idx, :]).sum(axis=1)
        match_probs = match_score / jnp.sum(match_score)
        parent_2_idx = jax.random.choice(k2, match_probs.size, shape=(1,), p=match_probs)[0]
    else:
        parent_2_idx, parent_1_idx = jax.random.choice(k1, probs.size, shape=(2,), p=probs)

    return archive[parent_1_idx], archive[parent_2_idx]


def update_state(
    state: ShardedArchiveState,
    new_scores: jax.Array,
    new_params: jax.Array,
    alpha: float,
) -> ShardedArchiveState:
    """Placeholder for a sharded archive update.

    In the eventual version this would:
      * compute fitness locally,
      * participate in a global reduction to find the worst slot,
      * apply the update only on the owning shard.
    """

    archive = state.archive
    scores = state.scores

    ext_scores = jnp.concatenate([scores, new_scores[None, ...]], axis=0)
    z = jnp.sum(ext_scores, axis=0) ** alpha
    z = jnp.where(z, z, 1)
    ext_scores /= z[None, :]
    fitness = jnp.sum(ext_scores, axis=1)

    worst_ix = jnp.asarray(jnp.argmin(fitness), dtype=jnp.int32)
    scores_len = jnp.asarray(scores.shape[0], dtype=jnp.int32)
    update_mask = worst_ix < scores_len

    row_selector = (jnp.arange(archive.shape[0], dtype=jnp.int32) == worst_ix) & update_mask
    row_selector = row_selector[:, None]

    updated_archive = jnp.where(row_selector, new_params[None, :], archive)
    updated_scores = jnp.where(row_selector[: scores.shape[0], :], new_scores[None, :], scores)

    return ShardedArchiveState(archive=updated_archive, scores=updated_scores)


@contextmanager
def with_mesh(mesh_devices, axis_name: str = "archive_devices"):
    """Context manager that yields a JAX mesh over the provided devices."""

    mesh = jax.sharding.Mesh(jnp.array(mesh_devices), (axis_name,))
    with mesh:
        yield mesh


__all__ = [
    "ShardedArchiveConfig",
    "ShardedArchiveState",
    "initialize_state",
    "sample_parents",
    "update_state",
    "with_mesh",
]
