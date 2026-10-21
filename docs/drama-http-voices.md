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

The mandatory `source.kind` is `synthetic` (generated audio), `self` (the operator's own voice), or `consented` (explicit permission from the speaker). `source.note` must be nonempty UTF-8 describing the provenance or consent. This is an operator attestation, not proof of identity or consent. Notes are not returned by the API.

The model's licence, the [Fish Audio Community License Agreement](https://huggingface.co/fishaudio/drama-3/blob/main/LICENSE) v1.0, section IV(b)(i), forbids using it to "generate speech that imitates the voice of an identifiable natural person without that person's prior consent (or, for a minor, the consent of a parent or legal guardian, and for a deceased person, the consent of the holder of the relevant rights) or another legal right to do so". The attestation records the operator's claim; it does not grant that consent or right.

Reference `text` is the exact transcript. Codes are frame-major signed integers, with the semantic code first and residual codebooks following, in the same order as upstream `drama3-generate --encode` emits. Do not substitute codes from an unrelated codec.

The loader enforces the family's reference count, transcript length, frame count and codebook ranges before serving. It also rejects empty references, structural prompt tokens, unsupported speaker tags, oversized files and oversized catalogs. The assembled prompt must still fit the engine's advertised prompt budget. Reference rows go through the same family `prompt.build` used by the generation CLI; only the tokenizer interface is adapted.

## Writing a voice file

`zig build drama3-voice` builds the writer next to `drama3-generate`, which it runs once per reference to encode the audio:

```sh
drama3-voice --codec MODEL/codec.pth --ref-audio tone.wav --ref-text "A small cloud passes." \
  --source-kind synthetic --source-note "A generated test tone." --out voices/tone.json
```

Repeat `--ref-audio` and `--ref-text` in pairs for up to four references. `--generate PATH` names the encoder when `drama3-generate` is not on `PATH`. The writer refuses, before encoding, a missing or invalid attestation, transcripts the speech request would reject, and an output name the catalog cannot load. The encoder writes into a fresh private directory beside the output. The writer replaces `NAME.json` by an atomic rename only after the catalog's own parser accepts the new file.

The operator's encoder binary and processes with the same UID are trusted. The safety boundary covers other users (including shared sticky directories), stale files or symlinks, encoder failures and concurrent tool runs; it does not cover a malicious encoder or same-UID processes renaming the operator's directories during a run. The output parent must not allow other users to rename the operator's entries; a shared sticky directory is suitable.

Each run exclusively creates a randomly named private directory with owner-only access beside the output, retrying name collisions. The encoder receives an ordinary path with a fresh output name for each reference; writing nothing is an error. Directory-relative reads reject code-file symlinks. The validated final file is staged in that directory and renamed over `NAME.json`, replacing a destination symlink rather than following it. No pre-existing file is touched before publication.

Cleanup restores owner access on the private directory and removes its contents and the directory. Before publication, cleanup failures are reported alongside the primary error. After publication, a cleanup failure is a warning on stderr naming the leftover directory, and the command succeeds because the output is correct. Abrupt termination such as SIGKILL can also leave a private directory behind.

Cleanup responsibility starts immediately after directory creation, before allocating its encoder pathname, so allocation failure also removes the private directory.
