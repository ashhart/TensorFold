"""Keep Mamba states as shared-forward row references until read, letting the next forward consume shared arrays."""

from __future__ import annotations

from typing import Any

from mlx_lm.models.cache import ArraysCache


class RowStateCache(ArraysCache):
    transient = ("ref",)          # a pending row; every read (and every copy or store) resolves it first

    def __init__(self, size: int = 2) -> None:
        self.__dict__["ref"] = None
        super().__init__(size)

    @property
    def cache(self) -> list[Any]:
        ref = self.__dict__.get("ref")
        if ref is not None:
            conv_rows, ssm_rows, row = ref
            self.__dict__["ref"] = None
            self.__dict__["_cache"] = [conv_rows[row:row + 1], ssm_rows[row:row + 1]]
        return self.__dict__["_cache"]

    @cache.setter
    def cache(self, value: list[Any]) -> None:
        self.__dict__["ref"] = None
        self.__dict__["_cache"] = value

    @property
    def ref(self) -> tuple[Any, Any, int] | None:
        """(conv rows [R, KC-1, CD], SSM rows [R, H, DH, DS], row) while the states are a row not yet read."""

        return self.__dict__.get("ref")

    def point(self, conv_rows: Any, ssm_rows: Any, row: int) -> None:
        self.__dict__["ref"] = (conv_rows, ssm_rows, int(row))

    def materialize(self) -> None:
        self.cache  # noqa: B018 - reading resolves the pending row
