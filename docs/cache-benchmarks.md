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
Missing counts/timings remain null, and failed or insufficiently streamed replies do not enter decode rates.

## First, repeated and concurrent requests

The standard-library-only cache client runs one first request, identical repeated requests, then a
concurrent group using the same prompt, greedy temperature, seed and reply limit. There is no warm-up:

```sh
python3 tools/bench_cache.py http://127.0.0.1:8080 MODEL \
  --tokens 64 --repeats 2 --streams 4 --require-cold --output cache.json
```

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

`qualified` and exit status zero mean requests completed with nonzero reported output, requested cache
and token-count assertions passed, and no observed fingerprint comparison differed. A missing token-ID
hash stays an unverified comparison; it is not an exact-token pass. A nonzero exit preserves the evidence.
This is benchmark qualification, not a model certification or proof of a RAM sizing recommendation.

## Model-free checks

```sh
python3 -m unittest discover -s tools/tests -p 'test_bench*.py' -v
```

Tests exercise usage validation, missing hashes, errors, incomplete/empty SSE, Unicode chunking,
concurrent request ordering, assertion failures, receipt allowlisting, and a real loopback HTTP server.
They do not load weights or substitute for checkpoint/device performance and exactness gates.
