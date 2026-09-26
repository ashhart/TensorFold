"""The incremental copy index proposes exactly what the full scan proposes, round after round."""

import random

from tensorfold.families.qwen3_5.cuda.decode import CopyIndex, copy_chain


def test_index_matches_full_scan_as_context_grows():
    rng = random.Random(4)
    base = [rng.randrange(40) for _ in range(300)]
    context = base[:50]
    index = CopyIndex()
    for step in range(400):
        # grow by 1-6 tokens, sometimes copying an earlier stretch so matches exist
        if rng.random() < 0.5 and len(context) > 40:
            at = rng.randrange(len(context) - 20)
            context += context[at:at + rng.randint(1, 6)]
        else:
            context += [rng.randrange(40) for _ in range(rng.randint(1, 6))]
        assert index.propose(context, 31) == copy_chain(context, 31), step
