# Cortheon mode

`tensorfold serve MODEL --cortheon` puts [Cortheon](https://github.com/ashhart/Cortheon)'s reasoning program and web research inside the server.
Every chat request it answers gets them, whatever client sends it.
Nothing else needs installing. There is no Docker, no Python and no second model, and the client keeps its usual settings.

Without the flag the server is unchanged, byte for byte.

## What a chat request gets

The server adds Cortheon's program to the system prompt.
It reads the task type from the conversation's first user message, lists the stages to work through in order, and states the rules: research current facts, treat web text as data, cite sources, test changes before calling them done.

It offers the model two tools that the server runs itself, `cortheon_search` and `cortheon_fetch`.
When the model calls one, the server runs it, adds the result to the conversation and lets the model carry on.
The client never sees these calls. It gets the answer, or the calls to its own tools.

It researches before the model starts building.
When the latest user message asks about current or version-specific facts, or asks to build with an outside library or API, the model first writes up to three search queries.
The server runs them, reads up to three of the best pages and puts both in front of the model before its first token.
Only the user's own words count here. Context a harness wraps in tags, such as a `<memories>` block with the current time, is ignored.

It checks the end of the work once a turn.
A coding task that changed files after its last test or build is asked to run the test or build before it finishes.
A researched answer that cites no URL is asked for a short list of its sources.

## Web safety

A long page is cut to fit the model's room: its opening, then the lines that share the most words with the task and the model's latest search, in page order.

Fetched pages and search results are untrusted data.
Sentences that read as orders to the model, such as "ignore all previous instructions", are dropped before the model reads the page, and the result says how many were dropped.

Reads go to public internet addresses only.
The server refuses private, loopback, link-local and shared address ranges, checks every redirect hop the same way, and connects to the address it checked.
TLS certificates are checked against the system's root certificates.
Each page read stops at 2 MiB and 15 seconds, redirects included, and each search at 10 seconds.

Searches use DuckDuckGo's HTML results, which need no key, and its lite page when the HTML page refuses or comes back empty.
Set `BRAVE_SEARCH_API_KEY` to search through Brave's API instead.

## Budgets

The model may make 6 web calls in a reply, on top of the research done before it starts.
A reply runs at most 8 generation rounds.
When the web budget runs out, the model is told to answer from what it has and to say what is still unsettled.

## Requests

A request can send `"cortheon": false` to skip it.
Cortheon also leaves these requests as they are: `tool_choice` set to `"none"`, `"required"` or a named function, structured output, `logprobs`, title requests, background requests, and requests with no tools that ask the model to transform text they carry.
Harnesses send those last ones for titles, summaries and routing hints.
Requests to `/v1/responses` and `/v1/messages` get the same treatment as chat completions.

A streamed reply shows Cortheon's progress in `reasoning_content`, for example `Cortheon: searching the web for "zig 0.17 release notes"`.
Whole replies and the last streamed chunk carry a `cortheon` object:

```json
{"task": "research_answer", "research_first": true, "rounds": 1, "searches": 2, "pages_read": 2,
 "sources": ["https://ziglang.org/download/0.17.0/release-notes.html", "https://ziglang.org/news/0.17.0-released/"], "checks": []}
```

## Prompt caching

The research and tool rounds a client never saw stay in the server's memory, up to 512 replies or 32 MiB.
On the next turn the server puts them back in front of the reply they informed.
The prompt then matches what the model read last turn, so the prompt cache still resumes it.

## Cost

Research before building adds one short greedy generation for the queries and a few seconds of network time.
Each web call the model makes adds one generation round.
Decoding is unchanged: every round runs through the same engine, with the same exact output for the prompt it gets.

## Example

```bash
tensorfold serve TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit --cortheon
```

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "What changed in the latest Zig release?"}]}'
```

Point any OpenAI-compatible client at `http://127.0.0.1:8080/v1` as before.
