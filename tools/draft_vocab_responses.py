"""Build a proposal artifact from declared committed response IDs, without tokenization or model execution."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile

SPLITS = ("calibration", "selection", "heldout")
MAX_BYTES = 16 << 20


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def ids_sha(ids):
    h = hashlib.sha256()
    for token in ids:
        h.update(struct.pack("<I", token))
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def keys(value, names):
    require(type(value) is dict and set(value) == set(names.split()), "missing or unknown fields")


def text(value):
    require(type(value) is str and 0 < len(value.encode()) <= 4096 and "\0" not in value, "invalid text")


def digest(value):
    require(type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value), "invalid digest")


def uint(value, maximum=(1 << 32) - 1):
    require(type(value) is int and 0 <= value <= maximum, "invalid integer")


def binding(value):
    keys(value, "name revision sha256")
    text(value["name"])
    text(value["revision"])
    digest(value["sha256"])


def identity(value):
    keys(value, "target drafter head tokenizer recipe vocab hidden format group scale_recipe")
    for field in ("target", "drafter", "head", "tokenizer", "recipe"):
        binding(value[field])
    uint(value["vocab"], 1 << 24)
    uint(value["hidden"], 1 << 20)
    uint(value["group"], 1 << 20)
    require(value["vocab"] > 0 and value["hidden"] > 0, "empty head geometry")
    text(value["format"])
    text(value["scale_recipe"])


def token_ids(ids, vocab, ordered=False):
    require(type(ids) is list, "invalid ID list")
    if ordered:
        require(len(ids) <= vocab, "ID list exceeds vocabulary")
    for token in ids:
        uint(token)
        require(token < vocab, "ID outside vocabulary")
    if ordered:
        require(all(a < b for a, b in zip(ids, ids[1:])), "IDs must be unique and ascending")


def plan(manifest):
    keys(manifest, "version identity mandatory_ids mandatory_source collector settings_sha256 corpus records buckets")
    require(type(manifest["version"]) is int and manifest["version"] == 1, "unsupported manifest version")
    identity(manifest["identity"])
    binding(manifest["collector"])
    binding(manifest["corpus"])
    digest(manifest["settings_sha256"])
    require(manifest["mandatory_source"] == manifest["identity"]["recipe"]["name"], "mandatory recipe mismatch")
    token_ids(manifest["mandatory_ids"], manifest["identity"]["vocab"], ordered=True)
    buckets = manifest["buckets"]
    require(type(buckets) is list and 0 < len(buckets) <= 1000 and len(set(buckets)) == len(buckets), "invalid buckets")
    for bucket in buckets:
        text(bucket)
        require(bucket != "*", "reserved total bucket")
    records = manifest["records"]
    require(type(records) is list and 0 < len(records) <= 100000, "invalid record plan")
    by_id, groups, cells = {}, {}, set()
    for row in records:
        keys(row, "id split bucket prompt_sha256 prompt_group template_group source_group")
        for field in ("id", "prompt_group", "template_group", "source_group"):
            text(row[field])
        digest(row["prompt_sha256"])
        require(row["split"] in SPLITS and row["bucket"] in buckets, "unplanned split or bucket")
        require(row["id"] not in by_id, "duplicate planned record")
        by_id[row["id"]] = row
        cells.add((row["split"], row["bucket"]))
        for field in ("prompt_group", "template_group", "source_group", "prompt_sha256"):
            key = (field, row[field])
            require(key not in groups or groups[key] == row["split"], "prompt/template/source split leakage")
            groups[key] = row["split"]
    require(cells == {(s, b) for s in SPLITS for b in buckets}, "every split needs every declared bucket")
    return by_id


def construct(manifest, receipts, size):
    by_id = plan(manifest)
    vocab = manifest["identity"]["vocab"]
    uint(size, vocab)
    require(size > 0, "empty candidate size")
    mandatory = manifest["mandatory_ids"]
    require(len(mandatory) <= size, "mandatory IDs exceed candidate size")
    expected_identity = sha(manifest["identity"])
    counts = {split: collections.Counter() for split in SPLITS}
    bucket_counts = {(s, b): collections.Counter() for s in SPLITS for b in manifest["buckets"]}
    seen, recorded = set(), []
    for receipt in receipts:
        keys(receipt, "id prompt_sha256 prompt_ids_sha256 ids ids_sha256 identity_sha256 settings_sha256 collector_sha256 kind committed_only unrestricted_target")
        rid = receipt["id"]
        require(type(rid) is str and rid in by_id and rid not in seen, "unknown or duplicate receipt")
        row = by_id[rid]
        require(receipt["kind"] == "committed_response_ids" and receipt["committed_only"] is True and
                receipt["unrestricted_target"] is True, "requires unrestricted committed response IDs")
        require(receipt["identity_sha256"] == expected_identity, "receipt checkpoint/tokenizer/head identity mismatch")
        require(receipt["settings_sha256"] == manifest["settings_sha256"], "receipt settings mismatch")
        require(receipt["collector_sha256"] == manifest["collector"]["sha256"], "receipt collector mismatch")
        require(receipt["prompt_sha256"] == row["prompt_sha256"], "receipt prompt mismatch")
        digest(receipt["prompt_ids_sha256"])
        ids = receipt["ids"]
        token_ids(ids, vocab)
        require(0 < len(ids) <= 1000000, "empty or oversized response")
        require(receipt["ids_sha256"] == ids_sha(ids), "response ID checksum mismatch")
        seen.add(rid)
        counts[row["split"]].update(ids)
        bucket_counts[(row["split"], row["bucket"])].update(ids)
        recorded.append((row, receipt))
    require(seen == set(by_id), "missing planned receipts")
    keep = set(mandatory)
    ranked = sorted(counts["calibration"], key=lambda token: (-counts["calibration"][token], token))
    for token in ranked:
        if len(keep) == size:
            break
        keep.add(token)
    require(len(keep) == size, "not enough observed calibration IDs; no low-ID fill")
    selected = sorted(keep)
    coverage = []
    for split in SPLITS:
        for bucket in ["*"] + sorted(manifest["buckets"]):
            counter = counts[split] if bucket == "*" else bucket_counts[(split, bucket)]
            coverage.append(dict(split=split, bucket=bucket, tokens=sum(counter.values()),
                                 covered=sum(n for token, n in counter.items() if token in keep)))
    recorded.sort(key=lambda pair: pair[0]["id"])
    provenance = dict(corpus_sha256=sha(manifest), settings_sha256=manifest["settings_sha256"],
                      prompts_sha256=sha([(r["id"], v["prompt_sha256"], v["prompt_ids_sha256"]) for r, v in recorded]),
                      responses_sha256=sha([(r["id"], v["ids_sha256"]) for r, v in recorded]),
                      collector=manifest["collector"], mandatory_source=manifest["mandatory_source"],
                      policy_version=1, source_kind="committed_response_ids")
    for split in SPLITS:
        provenance[split + "_sha256"] = sha([(r, v["ids_sha256"]) for r, v in recorded if r["split"] == split])
    artifact = dict(version=1, identity=manifest["identity"], ids=selected, ids_sha256=ids_sha(selected),
                    candidate_size=len(selected), mandatory_ids=mandatory, provenance=provenance, coverage=coverage)
    require(len(encoded(artifact)) <= MAX_BYTES, "artifact exceeds parser bound")
    return artifact


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode()


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def loads(data):
    def bad_constant(_):
        raise ValueError("nonfinite JSON number")
    return json.loads(data, object_pairs_hook=unique_pairs, parse_constant=bad_constant)


def read_manifest(path):
    with open(path, "rb") as handle:
        data = handle.read(MAX_BYTES + 1)
    require(len(data) <= MAX_BYTES, "manifest exceeds bound")
    return loads(data)


def read_receipts(path):
    total = count = 0
    with open(path, "rb") as handle:
        while True:
            line = handle.readline(MAX_BYTES + 1)
            if not line:
                break
            total += len(line)
            count += 1
            require(len(line) <= MAX_BYTES and total <= 64 << 20 and count <= 100000, "receipts exceed bound")
            yield loads(line)


def write_artifact(path, artifact):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".proposal-vocab-", delete=False) as handle:
            temporary = handle.name
            handle.write(encoded(artifact))
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            os.unlink(temporary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("receipts")
    parser.add_argument("out")
    parser.add_argument("--size", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        artifact = construct(read_manifest(args.manifest), read_receipts(args.receipts), args.size)
        write_artifact(args.out, artifact)
    except (ValueError, OSError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(dict(candidate_size=artifact["candidate_size"], ids_sha256=artifact["ids_sha256"],
                          artifact_sha256=hashlib.sha256(encoded(artifact)).hexdigest(),
                          ranking_split="calibration", coverage=artifact["coverage"]), sort_keys=True))


if __name__ == "__main__":
    main()
