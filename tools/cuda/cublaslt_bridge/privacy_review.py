"""Scan every readable span using generic rules, a private digest denylist and a secret-scanner corpus."""
import base64
import hashlib
import json
import io
from pathlib import Path
import re

NORMALIZATION = "casefold-alnum-v1"
RULES = {
    "home-path": re.compile(r"(?:/(?:Users|home)/|[A-Za-z]:[/\\]+Users[/\\]+)[^/\\\s]+", re.I),
    "private-ip": re.compile(r"\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|"
                             r"172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2}|100\."
                             r"(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])(?:\.\d{1,3}){2})\b"),
    "private-key": re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----"),
    "credential-url": re.compile(r"[a-z][a-z0-9+.-]+://[^/\s:@]+:[^/\s@]+@", re.I),
    "email": re.compile(r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,24}\b"),
}
SPAN = re.compile(rb"[\x20-\x7e\t\r\n]{4,}")
WIDE = (
    ("utf-16le", re.compile(rb"(?:[\x20-\x7e]\x00){4,}")),
    ("utf-16be", re.compile(rb"(?:\x00[\x20-\x7e]){4,}")),
)
WORD = re.compile(r"\b[\w]+(?:[-_.][\w]+)*\b")
B64 = re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{80,}={0,2}(?![A-Za-z0-9+/=])")


def normalized(value):
    return "".join(char for char in value.casefold() if char.isalnum())


def word_hash(value):
    return hashlib.sha256(normalized(value).encode()).hexdigest()


class Scanner:
    def __init__(self, context_file, policy, audit):
        context = json.loads(Path(context_file).read_text(encoding="utf-8"))
        if context.get("schema") != 1 or context.get("normalization") != NORMALIZATION:
            raise ValueError("private digest context schema")
        values = context.get("sha256")
        if not isinstance(values, list) or not values or len(values) > 4096:
            raise ValueError("private digest context is empty or excessive")
        if any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in values):
            raise ValueError("private context must contain digests only")
        if set(context) != {"schema", "normalization", "sha256"}:
            raise ValueError("private context must contain no plaintext identifiers")
        self.deny = set(values)
        self.context_sha256 = hashlib.sha256(Path(context_file).read_bytes()).hexdigest()
        self.allowed = {(row["category"], row["sha256"]) for row in policy["public_readable_values"]}
        self.audit = audit
        self.corpus = (audit / "readable-spans.txt").open("w", encoding="utf-8", newline="\n")
        self.findings = []
        self.seen = set()
        self.decoded_seen = set()
        self.counts = {"spans": 0, "unique_spans": 0, "base64_decodes": 0, "text_bytes": 0}
        self.decode_bytes = 0

    def text(self, value, origin, offset, encoding, depth=0, preceding=()):
        if len(value) > 16 << 20:
            raise ValueError("readable span exceeds inspection bound")
        self.counts["spans"] += 1
        value_hash = hashlib.sha256(value.encode()).hexdigest()
        if value_hash in self.seen:
            first = WORD.findall(value[:1024])[:2]
            border = list(preceding) + first
            if any(word_hash(" ".join(border[i:i + width])) in self.deny
                   for width in (1, 2, 3) for i in range(len(border) - width + 1)):
                self.findings.append({"category": "private-context", "origin": origin,
                                      "offset": offset, "encoding": encoding, "span_sha256": value_hash})
            return
        self.seen.add(value_hash)
        self.counts["unique_spans"] += 1
        self.counts["text_bytes"] += len(value.encode())
        if self.counts["text_bytes"] > 2 << 30:
            raise ValueError("readable corpus limit reached")
        self.corpus.write(value + "\n")
        words = list(preceding) + WORD.findall(value)
        candidates = set(words)
        for width in (2, 3):
            candidates.update(" ".join(words[i:i + width]) for i in range(len(words) - width + 1))
        if any(word_hash(candidate) in self.deny for candidate in candidates):
            self.findings.append({"category": "private-context", "origin": origin,
                                  "offset": offset, "encoding": encoding, "span_sha256": value_hash})
        for category, rule in RULES.items():
            for hit in rule.finditer(value):
                digest = hashlib.sha256(hit.group().encode()).hexdigest()
                if (category, digest) not in self.allowed:
                    self.findings.append({"category": category, "origin": origin,
                                          "offset": offset + hit.start(), "encoding": encoding,
                                          "match_sha256": digest, "match_bytes": len(hit.group().encode(encoding))})
        if depth >= 4:
            if B64.search(value):
                raise ValueError("base64 recursion limit reached")
            return
        for hit in B64.finditer(value):
            token = hit.group()
            if len(token) > 16 << 20:
                raise ValueError("base64 candidate limit reached")
            try:
                decoded = base64.b64decode(token + "=" * (-len(token) % 4), validate=True)
            except ValueError:
                continue
            digest = hashlib.sha256(decoded).hexdigest()
            if digest in self.decoded_seen:
                continue
            self.decoded_seen.add(digest)
            self.decode_bytes += len(decoded)
            if self.decode_bytes > 128 << 20:
                raise ValueError("base64 total decode limit reached")
            self.counts["base64_decodes"] += 1
            magic = (b"PK\x03\x04", b"\x1f\x8b\x08", b"\xfd7zXZ\x00", b"\x28\xb5\x2f\xfd",
                     b"\x7fELF", b"\x89PNG\r\n\x1a\n", b"%PDF", b"\xd0\xcf\x11\xe0",
                     b"\x00asm", b"7z\xbc\xaf\x27\x1c", b"Rar!")
            pe_at = int.from_bytes(decoded[60:64], "little") if len(decoded) >= 64 else 0
            recognized_pe = decoded[:2] == b"MZ" and 64 <= pe_at <= len(decoded) - 4
            if decoded.startswith(magic) or (recognized_pe and decoded[pe_at:pe_at + 4] == b"PE\x00\x00"):
                raise ValueError("nested encoded binary payload requires an explicit reviewed decoder")
            self.scan(decoded, origin + "-base64-" + digest[:12], depth + 1)

    def scan(self, data, origin, depth=0):
        for match in SPAN.finditer(data):
            position, preceding = match.start(), []
            for line in io.BytesIO(match.group()):
                value = line.decode("ascii")
                self.text(value, origin, position, "ascii", depth, preceding)
                preceding = WORD.findall(value[-1024:])[-2:]
                position += len(line)
        for encoding, rule in WIDE:
            for match in rule.finditer(data):
                self.text(match.group().decode(encoding), origin, match.start(), encoding, depth)

    def close(self):
        self.corpus.close()
        corpus_path = self.audit / "readable-spans.txt"
        if corpus_path.stat().st_size > 2 << 30:
            raise ValueError("actual readable corpus exceeds inspection bound")
        with corpus_path.open("rb") as stream:
            corpus_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        report = {"schema": 1, "context_sha256": self.context_sha256, "counts": self.counts,
                  "findings": self.findings, "values_printed": False,
                  "corpus_sha256": corpus_sha256, "corpus_bytes": corpus_path.stat().st_size}
        (self.audit / "context-findings.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
