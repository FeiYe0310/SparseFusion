"""
Lightweight package initializer for lib.

Note:
- We intentionally avoid importing heavy submodules (e.g., prune) here to
  prevent side effects or syntax errors from propagating during package import.
  Submodules should be imported explicitly where needed, e.g.:
    from lib.async_shard import AsyncShardCoordinator
"""

__all__: list[str] = []
