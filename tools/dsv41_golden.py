"""Collect teacher-forced prompt logprobs from a running vLLM DeepSeek-V4.1 server: goldens for the reference forward.

    python tools/dsv41_golden.py http://127.0.0.1:8888 notes/dsv41/golden.json

Each prompt is tokenized by the server (so ids match its tokenizer), then sent back as token ids with
``prompt_logprobs=5`` and one greedy completion token. The file keeps ids, per-position top-5 and the
logprob of the actual next token.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

AUTH = {"Authorization": "Bearer " + os.environ["VLLM_API_KEY"]} if os.environ.get("VLLM_API_KEY") else {}

PROMPTS = [
    "The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n    if n < 2:\n        return n\n    return",
    ("In 1905, Albert Einstein published four papers that changed physics. The first explained the photoelectric "
     "effect, the second Brownian motion, the third special relativity, and the fourth showed that mass and energy "
     "are equivalent, expressed by the famous equation"),
    ("Q: A train leaves at 3 pm travelling 60 km/h. Another leaves the same station at 4 pm travelling 90 km/h on the "
     "same track. At what time does the second train catch up?\nA: Let t be hours after 3 pm. 60t = 90(t - 1), so"),
    "Die Donau ist der zweitlängste Fluss Europas. Sie entspringt im Schwarzwald und mündet ins",
    ("Once upon a time, in a small village at the edge of a vast forest, there lived an old clockmaker named Elias. "
     "Every morning he opened his shop at dawn, wound every clock on the wall, and listened. One day, one of the "
     "clocks did not tick. He opened its case and found, instead of gears, a tiny folded note that read:"),
    ("The history of the printing press begins in the fifteenth century. Johannes Gutenberg, a goldsmith from Mainz, "
     "combined several existing technologies into a system that made mass production of books possible for the first "
     "time in Europe. His movable metal type was cast from an alloy of lead, tin and antimony, which melted at a low "
     "temperature, cooled quickly, and produced durable letters that could be reused thousands of times. He adapted "
     "the screw press, long used for pressing grapes and olives, to apply even pressure to paper, and he developed an "
     "oil-based ink that adhered to metal far better than the water-based inks used for woodblock printing. "
     "Around 1455 his workshop completed the famous Gutenberg Bible, of which roughly one hundred and eighty copies "
     "were printed, some on paper and some on vellum. Within fifty years, printing shops had spread to more than two "
     "hundred and fifty cities, and an estimated twenty million volumes had been produced. The consequences were "
     "profound. The cost of books fell dramatically, literacy rates began to rise, and scholars could compare "
     "identical copies of texts across great distances. Martin Luther's writings, printed in large numbers, spread "
     "the ideas of the Reformation faster than any authority could suppress them. Scientists such as Copernicus, "
     "Vesalius and later Galileo relied on printed books and diagrams to share observations with colleagues they "
     "would never meet. Standardized spelling and grammar gradually emerged as printers settled on consistent forms "
     "for the words they set in type. Newspapers appeared in the seventeenth century, first in Germany and the Dutch "
     "Republic, and pamphlets became a weapon in every political and religious dispute. Historians sometimes argue "
     "about whether the press caused these changes or merely accelerated trends that were already under way, but few "
     "doubt that it transformed the circulation of knowledge. In summary, the three most important effects of the "
     "printing press were"),
]


def post(url: str, body: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **AUTH})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def main() -> None:
    base, out = sys.argv[1].rstrip("/"), sys.argv[2]
    model = json.loads(urllib.request.urlopen(urllib.request.Request(base + "/v1/models", headers=AUTH)).read())["data"][0]["id"]
    goldens = []
    for text in PROMPTS:
        ids = post(base + "/tokenize", {"model": model, "prompt": text, "add_special_tokens": True})["tokens"]
        r = post(base + "/v1/completions", {"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0,
                                            "prompt_logprobs": 5, "logprobs": 5})
        choice = r["choices"][0]
        rows = choice.get("prompt_logprobs") or []
        top, actual = [], []
        for i, row in enumerate(rows):
            if row is None:
                top.append(None)
                actual.append(None)
                continue
            entries = {int(k): v for k, v in row.items()}
            top.append(sorted(((tid, e["logprob"]) for tid, e in entries.items()), key=lambda t: -t[1])[:5])
            actual.append(entries.get(ids[i], {}).get("logprob"))
        last = choice["logprobs"]["top_logprobs"][0] if choice.get("logprobs") else {}
        goldens.append({"text": text, "ids": ids, "prompt_top5": top, "prompt_actual": actual,
                        "next_text": choice["text"], "next_top5": last})
        print(f"{len(ids):4d} tokens -> {choice['text']!r}", flush=True)
    with open(out, "w") as f:
        json.dump({"model": model, "goldens": goldens}, f)


if __name__ == "__main__":
    main()
