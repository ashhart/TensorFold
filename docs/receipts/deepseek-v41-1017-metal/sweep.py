"""Cold prefill and four-slot HTTP bursts on frozen synthetic fixtures."""
from argparse import ArgumentParser
import json
from pathlib import Path
import statistics

import benchmark as bench

PROSE = (
    ("coast", "Write an understated opening to a literary story about a ferry mechanic who receives a letter addressed to her late mentor. Set the scene before sunrise and give the mechanic a concrete decision to make. Write at least 450 words of original prose with natural dialogue and no headings."),
    ("orchard", "Write a 450-word magazine feature about a community orchard after an unusually late frost. Begin with an observable detail, move through two points of view, and explain the practical tradeoff of saving a crop without pretending there is one easy answer. Use continuous prose."),
    ("archive", "Write at least 450 words of original fiction. A local archivist discovers that a train timetable has been annotated by three generations of station staff. Make the human stakes emerge from the entries and a conversation, with no supernatural explanation or headings."),
    ("reservoir", "Write a 450-word essay for a general audience about how a small town's reservoir changes its routines during a dry year. Include specific observations and one plausible disagreement. Give the essay a distinct voice and use prose paragraphs without bullet points."),
    ("workshop", "Write at least 450 words of a realistic story set in a public instrument repair workshop. Two strangers disagree over whether an old cello should be made playable or left untouched. Convey their motives through particular objects and actions instead of summarizing their personalities."),
    ("survey", "Write a 450-word travel feature about a rail surveyor returning to the valley where she grew up. Tie two details of the present to things that have changed over a decade. Include a brief exchange with someone she recognizes and keep the tone grounded."),
)


def concurrency(base, pid, out):
    health = bench.before(base, 4)
    reference = {}
    bursts = []
    for i, (name, prompt) in enumerate(PROSE):
        bench.before(base, 4)
        with bench.Memory(pid) as memory:
            row = bench.streaming(base, prompt)
        reference[name] = row
        bursts.append(bench.summarize_burst([dict(name=name, **row)], 1, i, reference, memory.peak_gib))
    write(out, health, reference, bursts)
    print("C1 median", round(statistics.median(b["aggregate_decode_tps"] for b in bursts), 2), flush=True)
    for rep, levels in enumerate(((4, 8, 16), (16, 8, 4), (8, 4, 16))):
        for level in levels:
            bench.before(base, 4)
            items = [PROSE[(i+rep) % len(PROSE)] for i in range(level)]
            with bench.Memory(pid) as memory:
                rows = bench.collect_burst(base, items)
            result = bench.summarize_burst(rows, level, rep, reference, memory.peak_gib)
            bursts.append(result)
            write(out, health, reference, bursts)
            print("C", level, "rep", rep + 1, "decode", round(result["aggregate_decode_tps"], 2),
                  "exact", result["exact_count"], "/", level, flush=True)


def write(out, health, reference, bursts):
    out.write_text(json.dumps(dict(health=health, reference=reference, bursts=bursts), indent=2) + "\n")


def main():
    p = ArgumentParser()
    p.add_argument("phase", choices=("pp", "decode"))
    p.add_argument("base")
    p.add_argument("pid", type=int)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    if a.phase == "pp":
        bench.PP_LENGTHS = (2048, 8192, 16384, 32768)
        bench.prompt_processing(a.base, a.pid, a.out)
    else:
        concurrency(a.base, a.pid, a.out)


if __name__ == "__main__":
    main()
