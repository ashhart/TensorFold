# Attested HTTP voices

`tensorfold-native serve MODEL --voices DIR --default-voice NAME` loads the catalog at startup. Without these flags the built-in voice and speech request lowering are unchanged. `--default-voice` defaults to `default`; a configured name must exist. A missing, malformed or unattested catalog file refuses startup rather than silently substituting a voice. Restart to reload files.

`POST /v1/audio/speech` selects a catalog entry with `voice` (or the existing `speaker` alias). An absent, empty or case-insensitive `default` selects the configured default. Other names match exactly. Request-supplied reference arrays and paths remain unsupported. `GET /v1/voices` returns `{"voices": ["default", ...]}` with catalog names sorted, without transcripts, codes or attestation notes. Non-speech models return the existing unknown-route response.

Each regular `NAME.json` file contains:

```json
{
  "format": "drama3-http-voice/1",
  "source": {"kind": "synthetic", "note": "Generated integer codes for a parser example, not recorded speech."},
  "references": [{"text": "A small cloud passes.", "codes": [[7,1,2,3,4,5,6,7,8,9]]}]
}
```

The example codes are synthetic, not an acoustic-quality fixture. Names contain ASCII letters, digits, hyphens and underscores, begin with a letter or digit, and cannot be `default` in any case. Names are bounded by the loader. Symlink catalog entries are refused.

The mandatory `source.kind` is `synthetic` (generated audio), `self` (the operator's own voice), or `consented` (explicit permission from the speaker). `source.note` must be nonempty UTF-8 describing the provenance or consent. This is an operator attestation, not proof of identity or consent; do not use someone else's voice without permission. Notes are not returned by the API.

Reference `text` is the exact transcript. Codes are frame-major signed integers, with the semantic code first and residual codebooks following, in the same order as upstream `drama3-generate --encode` emits. Encode a synthetic reference through upstream's codec using `--codec`, `--ref-audio` and `--encode`; decode its native signed integer binary output into frame rows when preparing the JSON. This PR does not provide voice-file writer tooling or an encoder implementation. Do not substitute codes from an unrelated codec.

The loader enforces the family's reference count, transcript length, frame count and codebook ranges before serving. It also rejects empty references, structural prompt tokens, unsupported speaker tags, oversized files and oversized catalogs. The assembled prompt must still fit the engine's advertised prompt budget. Reference rows go through the same family `prompt.build` used by the generation CLI; only the tokenizer interface is adapted.
