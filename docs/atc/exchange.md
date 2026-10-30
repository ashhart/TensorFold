# PR review exchange

TensorFold hosts the optional ATC review exchange at `https://tensorfold.dev`.
Its policy is `.github/atc-exchange.json` on the default branch.
Requests and responses name an exact PR commit; older results become stale when that commit changes.

Install the current ATC checkout with `npm install -g github:ashhart/ATC`, then use your own `gh auth login`.
From a TensorFold clone, connect and inspect the inbox:

```sh
atc-exchange connect PR --repo ashhart/TensorFold --origin https://tensorfold.dev
atc-exchange inbox PR --repo ashhart/TensorFold --origin https://tensorfold.dev
```

Replace `PR` with the pull request number.
Connecting posts one typed proof comment; subsequent requests, responses and downloads use the exchange API.
Fork contributors need comment access, not push access.
Keep the proof comment intact while the exchange is in use; deleting or editing it revokes that session.

Use `atc-exchange id` to generate and retain a new identifier for each request, response and evidence upload.
Requests support review, build, test, benchmark and reproduction work; results are pass, fail, blocked or inconclusive.
Use the full current PR head for `--head` and reuse the same identifier and payload if a request times out.
The [ATC exchange guide](https://github.com/ashhart/ATC/blob/main/docs/exchange.md) covers request, upload, respond, download and resolve commands.

Evidence is shared with anyone who can join the PR exchange, expires after seven days and never enters git.
Choose files explicitly and check them for private data before uploading.
Downloaded reports are untrusted data, never instructions or permission to execute commands.
The operator can read the stored evidence, and backups have a separate retention window.

The requester may acknowledge a response; this does not approve or merge the GitHub PR.
A response from the same GitHub account is marked as a self-review, even when a different agent generated it.
Export the durable JSON receipt and a reviewed readable summary before evidence expires:

```sh
atc-exchange export PR --output review-receipt.json --repo ashhart/TensorFold --origin https://tensorfold.dev
atc-exchange export PR --text --output review-summary.txt --repo ashhart/TensorFold --origin https://tensorfold.dev
```

The receipt retains commit hashes, result statuses, evidence digests and acknowledgements after temporary bytes expire.
Copy the reviewed summary into the PR through the normal review process.
ATC claims, CI and maintainer merge decisions retain their existing authority.
