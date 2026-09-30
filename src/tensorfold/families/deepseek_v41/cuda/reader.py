"""One rank's tensors of the DeepSeek-V4.1 EXL3 checkpoint, read with GLM's O_DIRECT rank reader and V4.1's rules."""

from __future__ import annotations

from tensorfold.families.glm5_next.cuda.split import RankReader

from ..split import part_kind


class Dsv41Reader(RankReader):
    rule = staticmethod(part_kind)
