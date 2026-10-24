# Notes storage API

This first slice of #1012 adds a single-owner, journal-backed Notes store. It does
not add note retrieval, prompt injection, live fetching or promotion. Those are
separate dependent changes. Ordinary generation does not read the store.

Start the native server with `--notes snapshot.json`. The snapshot must use the
served tokenizer's vocabulary and, when present, its `tokenizer.json` SHA-256.
The journal is stored beside the snapshot as `snapshot.json.journal`. A second
writer is refused from startup, including before the first write. The journal
uses format 3 for stable identities and retry work; older formats are upgraded
under the writer lock, and older binaries refuse the new format. Without `--notes`, `/v1/notes` routes return 404.

The routes use the server's existing authentication:

| Request | Behavior |
| --- | --- |
| `GET /v1/notes` | List current notes. |
| `GET /v1/notes?history=true` | Include superseded and withdrawn notes. |
| `GET /v1/notes/ID` | Read one note, including its state and source. |
| `POST /v1/notes` with `{"text":"My favourite colour is green."}` | Add a durable write. |
| `POST /v1/notes/ID` with `{"text":"My favourite colour is blue now."}` | Change a note. |
| `DELETE /v1/notes/ID` | Record a take-back; its history remains. |
| `DELETE /v1/notes/ID?erase=true` | Rewrite the journal and replay the snapshot without the erased write's closure. |

An erase refuses snapshot-owned notes: change the source snapshot and export it
again. It also refuses a snapshot file that changed since startup. Source numbers and numeric note IDs
are never reused, including after erase and restart. Surviving IDs stay unchanged;
`renumbered` is retained as an empty compatibility field. Invalid or repeated
`erase` query values are refused before either delete mode runs.

The erase reply reports the erased ID, removed sources, per-place status and
`complete`. It asks the engine thread to remove matching prompt states, clears
tokenizer memoized encodes, cancels matching pending conversation writes and
removes matching persisted conversation states and their unreferenced row blocks.
Unrelated states remain reusable. Failed file removal or directory sync keeps the
result incomplete. A journal rename followed by a failed directory sync returns
HTTP 500 with `erased` and the journal's unresolved status.

`complete: false` means the store change applied but other copies may remain.
Retry the same DELETE to resume pending cleanup, including after a restart. The
journal atomically records the target identity and token spans before engine
cleanup. These spans can reveal the erased text and remain protected like the
notes journal until cleanup completes; the response names the journal while
that record remains. No removed note plaintext is retained. A completed erase
subsequently returns 404 and can never target a different note.

The reply names unresolved places, including Recall prompt/reply records,
engine token traces, running requests, stored Responses,
compaction files, request logs and learned weights. Learned-folder diagnostics
cover only the current configured root; they do not prove that earlier roots or
another server's files are clear. Enabling a learned writer therefore remains
conservative even after the current conversation states have been removed.

Erase matching is linear for engine requests of at most 64k tokens. Sibling
inspection stops at 16 folders and refuses indexes over 8 MiB, reporting an
unresolved scan budget. Erase still waits for checked disk writes and syncs;
these checks do not establish a frame-latency bound on slow storage.

Host regression checks use fake engine states and real journal/state files:

```sh
zig build test-server-cpu test-lane-api-host test-cuda-host
zig build test-kernel-contract test-golden test-server-security test-pull-security fuzz-cortheon
zig build
zig build native -Dcpu=apple_m1
```

These checks do not establish real-model Metal/CUDA qualification. The final
stack still requires the exactness, accuracy and erasure receipts agreed in #1012.
