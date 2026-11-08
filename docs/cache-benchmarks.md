# Observed cache state in HTTP benchmarks

The maintained `tools/bench_openai.py` and `tools/bench_concurrent.py` retain per-request evidence in their
JSON reports: actual `prompt_tokens`, `cached_tokens`, completion `tokens`, `cache_state`, `complete`,
`token_sha` and `output_sha256`. Prompt counts come from server usage, not text length or requested tokens.

`cache_state: cold` requires a completed SSE stream and explicit, valid zero cached tokens.
`reused` requires a positive count no greater than the reported prompt count. Missing or malformed usage,
booleans, negative counts, string/float counts and truncated streams are `unverified`; none become zero.
Cold here means **no reported retained-prefix reuse**. It does not mean cold filesystem pages, unloaded
weights, an idle GPU, fresh allocator state, or absence of backend caches outside this usage counter.

`token_sha` is TensorFold's token-ID fingerprint when reported. Missing hashes make exactness comparisons
unverified, including when both hashes are absent. Failed or incomplete replies cannot pass those checks.
`output_sha256` fingerprints streamed content and reasoning separately, then hashes their two SHA-256
digests in that order. It is independent of network chunk boundaries and includes reasoning-only output.
Text equivalence does not prove token-ID equivalence. Incomplete streams have no published output hash.
Completed streams with no nonempty text or reasoning also have no output or token fingerprint, even when usage
claims positive completion tokens or the server supplies a token hash. Usage and completion remain recorded.
Missing counts/timings remain null, and failed or insufficiently streamed replies do not enter decode rates.

Tool-call-only replies (`delta.tool_calls` without nonempty text or reasoning) are also outside this
qualification: their fingerprints remain null and the cache client exits 1, including when cache assertions pass.
Tool names, IDs and argument fragments do not contribute to `output_sha256` or text-chunk latency measurements.
A mixed reply with text or reasoning can qualify under the usual checks, but its output fingerprint covers only
that text and reasoning; matching output fingerprints do not establish matching tool calls or arguments.
Use text/reasoning-producing replies for these cache receipts. Tool-call correctness needs a separate check.

## First, repeated and concurrent requests

The standard-library-only cache client runs one first request, identical repeated requests, then a
concurrent group using the same prompt, greedy temperature, seed and reply limit. There is no warm-up:

```sh
python3 tools/bench_cache.py http://127.0.0.1:8080 MODEL \
  --tokens 64 --repeats 2 --streams 4 --require-cold --output cache.json
```

For a server requiring bearer authentication, add `--api-key-env TENSORFOLD_API_KEY` and set that
environment variable before running the client. Credentials are selected explicitly; the client does not
read `OPENAI_API_KEY` by default. The key is sent in the request header; the literal key is redacted from reports.

The phase is the request order; the observed cache state is independent. A first request can reuse an
existing prefix; a repeated request can remain cold. Concurrent requests can observe different states.
The client never clears cache or restarts the server. `--require-cold` asserts only the first request;
`--require-reuse` asserts positive reuse on every repeated request. Use a fresh process and disabled
prefix retention (`--prompt-cache-gib 0`, where supported) for independent cold-prefill cells.

For exact 2K/8K/32K/64K raw prefill, supply a JSON array of IDs from the checkpoint's own tokenizer:

```sh
python3 tools/bench_cache.py http://127.0.0.1:8080 MODEL \
  --prompt-file public-8192-tokens.json --expected-prompt-tokens 8192 \
  --tokens 64 --require-cold --output cache-8192.json
```

The expected count is checked against **every** server-reported prompt count. The built-in public text
fixture has a tokenizer-dependent length and is not advertised as an exact token length.
The client retains only fixture/output hashes, counters, timing, phase and settings in its report;
it omits the endpoint, model alias, file path, raw prompt/output and exception text. Use public fixtures
for public receipts: a hash is a fingerprint, not a promise of anonymity for sensitive input.
The existing single-stream client retains its legacy generated sample; review those reports before sharing.

`qualified` and exit status zero mean requests completed with nonzero reported tokens and observed text or reasoning,
requested cache and token-count assertions passed, and no observed fingerprint comparison differed. A missing token-ID
hash stays an unverified comparison; it is not an exact-token pass. A nonzero exit preserves the evidence.
This is benchmark qualification, not a model certification or proof of a RAM sizing recommendation.
A single buffered text chunk or a reasoning-only reply can qualify without a measurable decode interval. A usage-only
reply cannot qualify in any phase; its receipt is preserved with null fingerprints and exit status 1.

## Latency figures

`tools/bench_metrics.py` derives these figures from the chunk arrivals a client records.
Every time is the client's clock, so each includes HTTP and queueing.

- TTFT: the first content or reasoning chunk's arrival minus the request's send time.
- Prefill tok/s: `prompt_tokens / TTFT`, only for a `cold` request; otherwise null.
  Run the server with `--prompt-cache-gib 0` so no prefix is kept between requests.
- ITL: the gaps between consecutive tokens, the first token excluded. A chunk of k tokens counts as
  k tokens arriving together: its first token takes the gap since the previous chunk, the other k - 1 take 0 ms.
  The client knows only the reply's token count. Of its `tokens - 1` samples, `chunks - 1` are chunk gaps and
  `tokens - chunks` are zeros, so the ITL mean equals the decode span over `tokens - 1`.
- Round gap: the time between consecutive chunks. A drafted round that lands several tokens in one chunk
  makes ITL bursty: zeros, then one long gap. `tokens_per_chunk` is `tokens / chunks`.
- Pooled ITL and round gaps: every concurrent stream's samples in one list.
- Aggregate prefill tok/s: the cold requests' prompt tokens over the time from the first send to the last
  first chunk, for requests sent together; null unless every request in the group is cold.

Percentiles (`_p50`, `_p95`) are nearest-rank. A stream with fewer than two chunks, more chunks than tokens,
an incomplete reply or an unknown token count gives null ITL and round-gap figures.

## Distinct prompts on concurrent streams

Every stream of a default `tools/bench_concurrent.py` cell sends the same prompt; greedy streams also get the same
reply. A mixture-of-experts model then routes every stream to the same experts, which can flatter multi-stream rates.
`--prompts FILE` runs one `distinct` cell instead, where request i sends item i of a JSON list (modulo its length):

```sh
python3 tools/bench_concurrent.py http://127.0.0.1:8080 MODEL \
  --prompts tools/bench_prompts.json --levels 1,4,8 --temperatures 0 --alone --serial --output distinct.json
```

`tools/bench_prompts.json` holds 16 public prompts: code completions, chat code, arithmetic and prose.
Each item has a unique `name`, a `kind` (`chat` or `completion`) and a `prompt`. `--alone` and `--serial` check
each request against its own prompt's solo and `"draft": false` runs. `--prompts` and `--mixed` are exclusive.

## A long prompt beside a running reply

`tools/bench_fill.py` measures what the two cost each other: a reply decoding while a long prompt fills, in both arrival
orders, against each one run alone. Serve with the prompt cache off, so the long prompt stays cold, and two streams or more:

```sh
python3 tools/bench_fill.py http://127.0.0.1:8080 MODEL --corpus FILE --long 65536 --reply 1024 --output fill.json
```

`FILE` is any text long enough for `--long` tokens. The report gives `reply_s` (sent to last token), `long_ttft_s`,
`both_done_s` and the reply's largest gaps for each order, plus `alone` for the sum to compare `both_done_s` with. `exact`
is true only when every reply's `token_sha256` equals its run alone; the tool exits 1 when it isn't, or when `cold` isn't.

## Model-free checks

```sh
python3 -m unittest discover -s tools/tests -p 'test_bench*.py' -v
```

Tests exercise usage validation, missing hashes, errors, incomplete/empty/tool-call-only SSE, Unicode chunking,
concurrent request ordering, assertion failures, receipt allowlisting, and a real loopback HTTP server.
They do not load weights or substitute for checkpoint/device performance and exactness gates.

The maintained decode clients query `/v1/models` once before requests. They omit explicit thinking off only when the selected model advertises `capabilities.thinking_switch: false`. Each chat cell records `thinking`, `thinking_off_requested` and `thinking_switch`; missing discovery preserves the off request and labels it `off requested (unverified)`. Known false uses the server default. `--compare-receipt FILE` adds a warning and `thinking_comparable: false` when matching rows have different thinking labels. Raw completions do not receive a thinking field. Models advertising `thinking_applicable: false` also omit it and report `not applicable`. The single-stream chat prompt remains `gpu-chat-no-think`.
