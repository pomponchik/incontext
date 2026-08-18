"""Inference-backend contract and its named pristan extension point."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pristan import slot


class Backend(ABC):
    """Minimal contract implemented by exact prompt-token backends."""

    @property
    @abstractmethod
    def source(self) -> str:
        """Return a stable, non-sensitive diagnostic source name."""

        raise NotImplementedError

    @abstractmethod
    def count(
        self,
        request: Dict[str, Any],
        *,
        context_length: int,
    ) -> int:
        """Return the provider-visible prompt size in tokens."""

        raise NotImplementedError

    @abstractmethod
    def clear_cache(self) -> None:
        """Discard backend-local cached data."""

        raise NotImplementedError

    def output_budget_field(self, requested_field: str) -> str:
        """Return the provider-supported wire alias for an output budget."""

        return requested_field

    def coerce_output_budget(self, value: Any) -> Optional[int]:
        """Return a positive caller cap accepted by this provider."""

        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            return None
        return int(value)


@slot(
    entrypoint_group="incontext.backends",
    unique=True,
    explicit_plugin_names=True,
)
def backends() -> List[Backend]:
    """Provide named inference backends discovered through package metadata."""

    return []
