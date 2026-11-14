# Evidence index

The evidence index is a machine-readable map of what this checkout has measured. For every family, backend and
capability it lists the evidence (the file, the commit when the source records one, the date, and what was
measured), the grade of that measurement, and whether the evidence is stale or failed. It is a read-only walk of the
sources on the checkout; it changes nothing and needs no GPU.

The tool is `tools/evidence_index.py`, standard library only. Run it from the checkout root:

```sh
python3 tools/evidence_index.py --output out/evidence-index.json
```

It prints the path it wrote and a one-line summary (the entry count, the count per grade, and the stale, failed and
unknown-grade counts). Without `--output` it prints the JSON to standard output.

## The sources

The index walks the qualification-evidence sources on the checkout:

- `docs/receipts/`: the device and host receipts, one directory per run, each a free-form `README.md` (and sometimes a
  JSON sidecar). The directory names the family and backend it covers; a directory the index does not list is indexed
  with its family and backend left unknown rather than guessed.
- `docs/optimization-ledger.md`: the family by optimization status table. A cell states a status (done, failed, open,
  yes, no or ?), not a kind of measurement, so its grade is always unknown. The status and the PR, when the cell
  names one, go in the `measured` field.
- `tools/conformance_profiles/`: the served-exactness gate profiles, the inputs to the conformance kit. Each is a
  `served_exactness` record naming its gates.
- `docs/recipes/lfm3-native.md`: the LFM3 evidence ledger, the table of what was run and what it showed.
- `docs/known-issues.md`: the open-issue list. Each bullet is an open limitation for a family, indexed as a
  `known_issue` capability and marked failed, because an open issue is a requirement not yet met.
- A capabilities file, if given with `--capabilities`. The capabilities come from a command,
  `tensorfold-native capabilities --json`, not a file in the checkout, so the index records the source as unavailable
  (zero records and a note) when no file is supplied. Pass a saved file to include it.

The capabilities reader accepts native schema 1, whose `families` object maps family names to format arrays.
Each advertised format becomes a `format:NAME` capability for that family with no inferred backend. Root
`backends` become separate `backend` records with no family, and `chip` becomes a `chip:NAME` record. These
facts always have `unknown` measurement grade and no measured commit, result or staleness. A root backend
does not prove that every listed family runs on it. Native admission remains authoritative.
An absent or `null` chip means unknown device identity and emits no chip fact. Bounded chip labels also allow
spaces and parentheses, as in the native XPU label `Intel Xe2 (Level Zero)`; other names keep the token rules below.

The earlier family-list input remains supported, either as a root array or as a `families` array containing
objects with `family` or `name`, optional `backend`, and `capabilities` arrays of named objects. A valid empty
input records an available source with zero facts. Missing, unreadable, malformed or unsupported supplied
input exits nonzero with a bounded diagnostic and does not write an output index. Input is limited to 4 MiB,
256 families, 64 formats or capabilities per family, 64 backends and 4096 facts. Names are 1 to 128 ASCII letters,
digits, dots, underscores, pluses or hyphens, starting with a letter or digit. Duplicate JSON keys are rejected.
Source paths inside the checkout are relative; external inputs use a fixed label. Unrelated native metadata,
including environment and library installation information, is not copied into the index.

## The grades

Each piece of evidence carries one of four grades, or `unknown` when the source does not state which kind of
measurement it is. A grade is never inferred: a source that says only that something ran leaves the grade `unknown`.

- `host_test`: a no-device host step, `zig build test-*-host` and the host checks it runs.
- `kernel_contract`: the kernel contract runner, the gates and their planted controls.
- `served_exactness`: served replies compared by full token hash.
- `served_speed`: served timing, time to first token, tokens per second, prefill and decode rates.

A source can state more than one grade. A receipt that records both a token-hash comparison and a timing is one
record per grade, so each capability entry carries its own grade.

## Stale, failed, unknown

- `stale` is computed with git. A record is stale when the commit it was measured at is an ancestor of the current
  head and a file it covers changed after that commit. The files it covers are the family's own source directory,
  plus the CUDA kernel tree for a CUDA backend. A record whose commit is not in this history, or that covers no
  files, is `null` (unknown). Changes to the shared core are not captured here; that is a known limitation.
- `failed` is read from the source's own words, conservatively. A record is failed only when the source states an
  unambiguous result failure (a gate failed, a regression confirmed, no performance-gate pass). A gate out on the
  good side, or a candidate excluded from a comparison, is not a failure. A record with no stated result is `null`.
- `unknown` is the honest default. A grade the source does not state stays `unknown`. A commit the index cannot place
  in this history stays unknown for staleness. An open issue is a failed, not unknown, requirement.

The summary at the top of the index counts the entries, the entries per grade, the entries per family and per
backend, and the stale, failed and unknown-grade totals. The optimization ledger contributes most of the
unknown-grade entries, because it is a status tracker rather than a measurement record.

## Reading an entry

Each entry is one family, backend and capability:

```json
{
  "family": "gemma4",
  "backend": "cuda",
  "capability": "served_exactness",
  "grade": "served_exactness",
  "evidence": [
    {
      "source": "receipts",
      "file": "docs/receipts/prompt-tail-gemma-nemo/README.md",
      "commit": "5a45539af031726f68f6dacdf713dbdff8f55b60",
      "date": null,
      "measured": "Prompt chunks nothing reads: Gemma 4 and Nemotron on CUDA",
      "failed": null,
      "stale": true
    }
  ],
  "stale": true,
  "failed": null
}
```

The `evidence` list holds every source record that supports this capability; the entry's `grade`, `stale` and
`failed` are the aggregate over that list. A capability with several sources (a conformance profile and the receipt
that ran it) groups them under one entry.
