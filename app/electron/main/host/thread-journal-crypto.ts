import { encryptWithKey, decryptWithKey, resolveLocalSecretKeySync } from '../../local-crypto'
import { vontraPath } from '../../store'

// Per-thread NDJSON journals hold the FULL transcript (messages, tool args, tool output,
// diffs) — the most sensitive data in the app. threads.json is encrypted at rest via
// writeProtectedJson, so writing the journal as plaintext NDJSON defeats the same
// at-rest posture (local-crypto.ts: casual disk reads / accidental settings-dir sharing).
// We encrypt each journal event with the same app key, keeping the {seq, ts} envelope in
// the clear so compaction (which only reads seq) keeps working without a key.
const JOURNAL_ENC_PREFIX = 'enc:v1:'

// Lazily resolve the same primary/fallback keys as store.ts. Failure disables the
// best-effort journal; it must never downgrade a full transcript to plaintext.
let cachedJournalKey: Buffer | null = null
export function journalKey(): Buffer | null {
  if (cachedJournalKey) return cachedJournalKey
  try {
    cachedJournalKey = resolveLocalSecretKeySync(vontraPath('.secretkey')).key
    return cachedJournalKey
  } catch {
    return null
  }
}

// Wrap an AgentEvent for the journal: encrypt its JSON with the app key when available,
// emitting { e: 'enc:v1:<ciphertext>' }. If crypto is unavailable, return null so
// the caller skips this optional journal rather than leaking the event verbatim.
export function encodeJournalEvent(event: unknown, key: Buffer | null): unknown | null {
  if (!key) return null
  try {
    return { e: JOURNAL_ENC_PREFIX + encryptWithKey(JSON.stringify(event), key) }
  } catch {
    return null
  }
}

// Inverse of encodeJournalEvent: decrypt an encrypted journal envelope back to the
// original event, tolerating legacy/plaintext lines written before encryption existed.
export function decodeJournalEvent(stored: unknown, key: Buffer | null): unknown {
  if (stored && typeof stored === 'object' && typeof (stored as { e?: unknown }).e === 'string') {
    const payload = (stored as { e: string }).e
    if (payload.startsWith(JOURNAL_ENC_PREFIX)) {
      if (!key) return null // cannot decrypt without the key
      try {
        return JSON.parse(decryptWithKey(payload.slice(JOURNAL_ENC_PREFIX.length), key))
      } catch {
        return null // wrong key / tampered / corrupt
      }
    }
  }
  return stored // legacy plaintext event written before journal encryption
}
