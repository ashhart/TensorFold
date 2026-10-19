# Native request seeds

A sampled HTTP request uses its explicit `seed`, then a configured default seed, then a fresh 63-bit OS seed.
Default seeds cover injected `Server.Config.default_sampling`; no stock serve flag or generation config sets one.
An omitted or null request seed uses this fallback. Sampled defaults without a seed still draw fresh entropy.
An explicit seed keeps its existing bit conversion and sampling behavior. Greedy requests do not draw a seed.
If the secure random source fails, the request fails rather than falling back to prompt keying.

Every sampled done line appends `seed=<n>` after `ttft`, `decode`, `elapsed`, `queued` and `tok/s`.
Copy that number into the next request's `seed` field to reproduce the draw at identical sampling settings.
A fresh seed permits a different draw; it does not guarantee a different token or reply.

Native `serve --prompt-seed` restores historical prompt keying when neither request nor defaults supplies a seed:
the key comes from prompt IDs and `TENSORFOLD_SEED_SALT`. Request seeds still override configured defaults.
The standalone CLI and speech retain their prompt keying. This flag concerns the native server.

Exactness checks must use identical explicit seeds or start the native server with `--prompt-seed`.
The release-cell harness pins missing keys to 1234 and preserves every supplied key. The shared-prefix load tool
uses 1234 plus item ID by default. `bench_openai --seed-from-prompt` requires native `serve --prompt-seed`.
No output comparison or acceptance threshold is weakened by fresh default seeds.

Requested by @tpischke in #1026.
