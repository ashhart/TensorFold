# Living Weights test harness and console

Tools for measuring what a Living Weights lesson changes, beyond the fact it teaches. They talk to running
`tensorfold serve` processes over HTTP; nothing here loads a model. Python 3.11+, standard library only.

Every run compares a taught copy against the same checkpoint with greedy decoding (`temperature: 0`). On one engine and
checkpoint greedy replies are byte-identical, so any answer that differs was changed by learning.

## Servers

Teach a copy, never your only folder (see [the Living Weights guide](../../docs/living-weights.md)). For the
cross-checks, also serve an untouched folder of the same checkpoint without `--slide`:

```sh
tensorfold serve COPY --slide --slide-graph facts-graph.json --port 8080 --no-thinking
tensorfold serve ORIGINAL --port 8081 --no-thinking
```

## Fact test (`lw_facts.py`, `facts.json`)

20 fictional facts, so a correct answer can only come from learning. Each has two paraphrased recall questions and a
neighbour question about a real thing of the same kind that must not pick the fact up ("How long is the Golden Gate
Bridge?" next to "The Quillon bridge is 1,312 metres long."), plus 15 unrelated controls.

```sh
python3 lw_facts.py baseline --url http://127.0.0.1:8080
python3 lw_facts.py learn    --url http://127.0.0.1:8080       # --only f14,f18 to teach some; --source chat
python3 lw_facts.py probe    --url http://127.0.0.1:8080       # again after a restart: --label after-restart
```

## General knowledge and rules (`lw_rules.py`, `rules.json`)

60 general-knowledge questions, and four fictional rules (a fee, a pay rate, a made-up unit, a replacement interval)
each used in two problems. `probe` asks the problems from memory and again with the rule stated in the prompt, which
shows whether a lesson damages reasoning the model could already do.

```sh
python3 lw_rules.py probe --url http://127.0.0.1:8080
python3 lw_rules.py learn --url http://127.0.0.1:8080
python3 lw_rules.py probe --url http://127.0.0.1:8080 --label after
```

Each phase writes every question and answer to `results/<phase>-<timestamp>.json` (`--results DIR` to move it).

## Console (`console.py`)

```sh
python3 console.py --learn http://127.0.0.1:8080 --reference http://127.0.0.1:8081 [--model NAME=URL ...]
```

Open `http://127.0.0.1:8090/`. The page is served on loopback with the servers behind one origin, since the server
sends no CORS headers. Only the `--learn` server can be taught; `/v1/slide` paths to any other server are refused.

- Chat with any of the servers. Replies render as markdown (marked + DOMPurify), with mermaid and SVG diagrams.
- Teach a fact, sent as a document or as chat. Before teaching, the reference rates the statement CONTRADICTS,
  CONSISTENT or UNKNOWN from its own knowledge; after teaching, the reference writes questions shaped like the fact
  about real things, and those plus a fixed panel are asked of both servers side by side, with differences marked.
- The fact graph and dashboard links open the learning server's own pages.

The page loads marked, DOMPurify and mermaid from cdn.jsdelivr.net; without them it shows plain text.

## Tests

`python3 -m unittest tools.tests.test_living_weights` from the repository root runs against a fake server.
