"""Top-two diagnostics keep strict token equality separate from a BF16 near-tie label."""
import math


def bf16_spacing(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("nonfinite margin score")
    magnitude = abs(value)
    if magnitude < 2.0 ** -126:
        return 2.0 ** -133
    return 2.0 ** (math.frexp(magnitude)[1] - 1 - 7)


def verdict(native: dict, python: dict, wanted: int, actual: int) -> dict:
    native_margin = native["logits"][0] - native["logits"][1]
    python_margin = python["logits"][0] - python["logits"][1]
    if native_margin < 0 or python_margin < 0:
        raise ValueError("top-two scores are not ordered")
    contenders = {wanted, actual}
    near = (set(native["ids"]) == contenders == set(python["ids"])
            and native_margin <= bf16_spacing(native["logits"][0])
            and python_margin <= bf16_spacing(python["logits"][0]))
    return {"verdict": "near-tie" if near else "bug", "native_margin": native_margin,
            "python_margin": python_margin, "native_top2": native, "python_top2": python,
            "token_equal": False, "near_tie_policy": "both margins <= one BF16 spacing, same contenders"}


def capture(logits, positions) -> list[dict]:
    import mlx.core as mx
    import numpy as np

    rows = np.asarray(logits.astype(mx.float32)).reshape(-1, int(logits.shape[-1]))
    records = []
    for row, position in zip(rows, positions):
        if len(row) < 2 or not np.isfinite(row).all():
            raise ValueError("nonfinite or short oracle logit row")
        first = int(np.argmax(row))
        score = float(row[first])
        row = row.copy()
        row[first] = -np.inf
        second = int(np.argmax(row))
        records.append({"ids": [first, second], "logits": [score, float(row[second])],
                        "position": int(position), "dtype": str(logits.dtype)})
    return records
