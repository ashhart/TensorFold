"""Regenerate synthetic FP32 repetition logits with the pinned Hugging Face processor on CPU."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import struct


TRANSFORMERS_VERSION = "5.20.1"
PROCESSOR = "transformers.generation.logits_process.RepetitionPenaltyLogitsProcessor"
CASES = [
    ("sign_zero_duplicates", [0, 0x80000000, 0x40000000, 0xC0000000, 0x40400000, 0xC0400000],
     [0, 1, 2, 2, 3, 3], 0x3F8CCCCD),
    ("identity", [0x3F800000, 0xBF800000, 0, 0x80000000], [0, 1, 2, 3], 0x3F800000),
    ("factor_below_one", [0x3F800000, 0xBF800000, 0x40000000, 0xC0000000], [0, 1, 2, 3], 0x3F000000),
    ("prompt_and_committed", [0x40800000, 0x40400000, 0x40000000, 0x3F800000, 0xBF800000],
     [0, 2, 4, 2, 3], 0x3F8CCCCD),
    ("greedy_flip", [0x3F800000, 0x3F900000, 0xBF800000], [1], 0x3FC00000),
    ("bf16_rounding_trap", [0x3F800000, 0x3F8D0000], [1], 0x3F8CCCCD),
    ("infinities", [0x7F800000, 0xFF800000, 0x3F800000, 0xBF800000], [0, 1], 0x3F8CCCCD),
    ("boundary_values", [0x3F7F0000, 0x3F810000, 0xBF810000, 0xBF7F0000], [0, 1, 2, 3], 0x3F8CCCCD),
    ("empty_history", [0x3F800000, 0xBF800000], [], 0x3F8CCCCD),
    ("nan_seen_unseen", [0x7FC00000, 0x7FC00000, 0x3F800000, 0xBF800000], [0, 2], 0x3F8CCCCD),
]


def from_bits(value):
    return struct.unpack("<f", struct.pack("<I", value))[0]


def bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def generate():
    import torch
    import transformers
    from transformers import RepetitionPenaltyLogitsProcessor

    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError(f"transformers=={TRANSFORMERS_VERSION} is required; found {transformers.__version__}")
    if RepetitionPenaltyLogitsProcessor.__module__ + "." + RepetitionPenaltyLogitsProcessor.__name__ != PROCESSOR:
        raise RuntimeError("unexpected reference processor")
    source = inspect.getsource(RepetitionPenaltyLogitsProcessor).encode()
    result = []
    for name, logits, history, penalty in CASES:
        scores = torch.tensor([[from_bits(x) for x in logits]], dtype=torch.float32, device="cpu")
        rounded = scores.to(torch.bfloat16).to(torch.float32)
        if any(bits(x) != raw and not torch.isnan(scores[0, i]).item()
               for i, (x, raw) in enumerate(zip(rounded[0].tolist(), logits))):
            raise RuntimeError(f"{name}: synthetic inputs must be exactly representable in BF16")
        ids = torch.tensor([history], dtype=torch.long, device="cpu")
        processor = RepetitionPenaltyLogitsProcessor(penalty=from_bits(penalty))
        expected = processor(ids, scores.clone())
        expected_bits = [bits(x) for x in expected[0].tolist()]
        greedy = 0xFFFFFFFF if torch.isnan(expected).any().item() else torch.argmax(expected[0]).item()
        result.append((name, logits, expected_bits, history, penalty, greedy))
    header = render(result)
    metadata = {
        "schema": "synthetic-repetition-hf-v1",
        "processor": PROCESSOR,
        "transformers": transformers.__version__,
        "processor_source_sha256": hashlib.sha256(source).hexdigest(),
        "torch": torch.__version__,
        "device": "cpu",
        "dtype": "float32",
        "cases": len(result),
        "header_sha256": hashlib.sha256(header).hexdigest(),
        "checkpoint_loaded": False,
    }
    return header, metadata


def render(cases):
    lines = ["#pragma once", "#include <stdint.h>", "",
             "typedef struct { const char *name; const uint32_t *logits_bits; const uint32_t *expected_bits; "
             "const uint32_t *history; uint32_t n, history_n, penalty_bits, expected_greedy; } HfRepetitionCase;"]
    for index, (_, logits, expected, history, _, _) in enumerate(cases):
        for label, values in (("logits_bits", logits), ("expected_bits", expected)):
            content = ", ".join(f"0x{x:08x}u" for x in values)
            lines.append(f"static const uint32_t hf_{index}_{label}[] = {{{content}}};")
        content = ", ".join(f"{x}u" for x in (history or [0]))
        lines.append(f"static const uint32_t hf_{index}_history[] = {{{content}}};")
    lines.append("static const HfRepetitionCase hf_repetition_cases[] = {")
    for index, (name, logits, _, history, penalty, greedy) in enumerate(cases):
        pick = "0xffffffffu" if greedy == 0xFFFFFFFF else f"{greedy}u"
        lines.append(f'  {{"{name}", hf_{index}_logits_bits, hf_{index}_expected_bits, hf_{index}_history, '
                     f"{len(logits)}u, {len(history)}u, 0x{penalty:08x}u, {pick}}},")
    lines += ["};", "static const uint32_t hf_repetition_case_count = "
              "sizeof(hf_repetition_cases)/sizeof(hf_repetition_cases[0]);"]
    return ("\n".join(lines) + "\n").encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", type=Path, help="compare an existing header without modifying it")
    action.add_argument("--out", type=Path, help="write the generated header")
    args = parser.parse_args()
    header, metadata = generate()
    if args.check:
        if args.check.read_bytes() != header:
            raise SystemExit("repetition fixture differs from the pinned CPU reference")
        metadata["header_matches"] = True
    else:
        args.out.write_bytes(header)
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
