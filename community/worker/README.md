# Community benchmark receiver

This Worker adds the upload API and `/benchmarks` to TensorFold's existing site.
The default `src/durable.js` deployment uses the `BENCHMARKS_STORE` SQLite Durable Object and forwards other requests to `ASSETS`. The same receiver can alternatively use a `BENCHMARKS_DB` D1 binding.
The exported `handleRequest(request, env)` returns `null` for site routes, so an existing Worker can call it before its asset handler.

## Review and publish

Uploads require scoped contributor tokens, with a default limit of 20 accepted runs per UTC day.
The owner issues a token with a public display name.
Names are self-chosen pseudonyms and do not prove a GitHub identity.
Only hashes of contributor tokens enter storage, and tokens can be revoked.
No upload credential is shipped in the CLI or site.

Every accepted upload starts as `pending` and is invisible to the public API.
The owner must inspect the receipt and approve it.
All displayed results remain self-reported after approval.
The server recomputes medians and ranges from samples, rejects unknown fields and inconsistent delivery arithmetic, and checks model IDs against Hugging Face without authentication before storing new receipts.
Private, inaccessible or redirected checkpoint lookups fail the upload.
The 1 MiB JSON limit bounds both request parsing and stored receipts.
Generated text, user prompts, local paths, arbitrary error messages and credentials are not receipt fields.
R2 is not required for this first version.

The board shows complete standard runs using the chat fixture at temperature 0.
Its main rate is **stream delivery tok/s**, which measures SSE arrival at the client.
Server-reported rate stays separate in receipt details.
`protocol_complete` requires the pinned suite, 256 output tokens, five successful seeded repeats in all four cells, a managed run with one active rank, a pinned model revision and config hash, and GPU inventory evidence.
That flag checks the submitted protocol and does not verify speed.
Attached runs, partial runs, custom settings and multiple ranks remain unranked.
GPU inventory is never treated as the number of devices used by a run.

Withdrawal clears the stored receipt and summary, removes any approved result immediately, and retains the run ID and hash for retry protection.
The quota includes withdrawn or administrator-deleted uploads, so deletion cannot bypass the daily limit.
Revoking a publisher token blocks future authenticated requests, but does not automatically remove previously approved results.
Delete those submissions separately when removal is needed.

## Local checks

The tests need Node 22.13 or newer with `node:sqlite` enabled.
They run the actual migration in an in-memory SQLite database and execute each route's real SQL.

```sh
cd community/worker
npm test
```

Tests cover authentication, moderation, owner-only withdrawal, immutable retries, quotas, revocation, public checkpoint lookup, strict validation, aggregation, pagination and asset fallback.
Test receipts use fictional hardware and model labels and mock checkpoint lookup.
They do not measure TensorFold inference performance.

After changing `src/protocol.js`, run `npm run build:browser-validator` to refresh the shipped browser validation module.
The test suite checks that browser and server validators match.
The browser checks all nested fields and previews the parsed receipt before an explicit publication confirmation.

## Connect to the site

Use `src/durable.js` as the site's main Worker and add the `BENCHMARKS_STORE` Durable Object binding and the `benchmark-v1` SQLite class migration from `wrangler.jsonc`. The object's first use initializes `migrations/0001.sql` transactionally. This path uses the existing Worker permissions rather than requiring a separate D1 permission grant.

Alternatively create a D1 database dedicated to community benchmarks, apply `migrations/0001.sql`, and bind it as `BENCHMARKS_DB` with `src/index.js` as the main Worker.
With Workers Static Assets, configure `run_worker_first` for `/api/benchmarks/*`, `/benchmarks` and `/benchmarks/*` so static fallback cannot swallow API routes.
Keep the existing site's static assets in its `assets.directory`.

Set `ADMIN_TOKEN` through Wrangler secrets, with at least 32 characters of random material.
Do not place it in a configuration file, Git, shell command arguments or logs.
Serve the API over HTTPS and keep request-body and Authorization-header logging disabled.
The API accepts same-origin browser requests and command-line requests without an Origin header.

These are the current Cloudflare references used for this implementation:

- [SQLite Durable Object storage](https://developers.cloudflare.com/durable-objects/api/sqlite-storage-api/)
- [D1 prepared statements](https://developers.cloudflare.com/d1/worker-api/prepared-statements/)
- [D1 limits](https://developers.cloudflare.com/d1/platform/limits/)
- [Workers Static Assets routing](https://developers.cloudflare.com/workers/static-assets/routing/worker-script/)

## API

All routes use `/api/benchmarks/v1`.
JSON responses disable caching.
Use `Authorization: Bearer TOKEN` for contributor or administrator routes.

| Method | Path | Access | Result |
| --- | --- | --- | --- |
| GET | `/suites` | Public | Pinned manifest and SHA-256 |
| GET | `/me` | Contributor | Publisher ID and public display name |
| POST | `/submissions` | Contributor | Submit the receipt directly as JSON |
| GET | `/submissions/ID` | Owner | Submission status and content hash |
| DELETE | `/submissions/ID` | Owner | Withdraw and erase receipt data |
| GET | `/results` | Public | Approved runs, newest first |
| GET | `/results/ID` | Public | One approved run with receipt and computed summary |
| POST | `/admin/publishers` | Administrator | Issue a contributor token |
| DELETE | `/admin/publishers/ID` | Administrator | Revoke a contributor token |
| GET | `/admin/submissions` | Administrator | First 50 pending receipts, oldest first |
| GET | `/admin/submissions/ID` | Administrator | Inspect a stored receipt |
| POST | `/admin/submissions/ID/approve` | Administrator | Publish a pending receipt |
| DELETE | `/admin/submissions/ID` | Administrator | Erase a submission |

Publisher creation accepts `{"display_name":"example-runner","daily_quota":20}`.
The token appears only in that creation response, so store it securely at issuance.
Display names contain 1 to 40 letters, numbers, spaces, dots, underscores or hyphens.

New submissions return HTTP 201 and `status: "pending"`.
Retrying the same publisher and run ID with the same content returns HTTP 200 and the original submission.
Changing that content returns HTTP 409.
The server's `content_sha256` hashes its sorted JSON representation, with JSON number normalization.
It can differ from a Python file digest that preserves integral floats such as `1.0`.

`/results` accepts exact `model`, `backend` and `gpu` filters, `limit` from 1 to 50, and the opaque `cursor` returned by a prior page.
The GPU filter matches the deduplicated GPU inventory label joined with ` + `.
The public response contains `results` and `next_cursor`.
Unapproved receipts never appear there.

The browser upload form keeps its pasted token only in the current document.
It clears the field after a successful submission and never writes tokens to local storage or URLs.
The form accepts CLI receipts rather than arbitrary benchmark text.

## Branch preview

The benchmark CLI is introduced on `codex/community-benchmark-cli`; until it is merged into a release, install that branch explicitly to try it:

```sh
python -m pip install git+https://github.com/ashhart/TensorFold.git@codex/community-benchmark-cli
```

The site's normal install script continues installing public main. Local Worker smoke tests should use an administrator secret in a private `.dev.vars` file outside the publication checkout and an isolated local storage directory. Never put a live token in commands, screenshots or published receipts.
