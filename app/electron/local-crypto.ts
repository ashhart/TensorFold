import { randomBytes, createCipheriv, createDecipheriv } from 'node:crypto'
import { readFileSync, writeFileSync, mkdirSync, linkSync, unlinkSync } from 'node:fs'
import path from 'node:path'

// App-managed secret encryption — the alternative to Electron's safeStorage (which
// keys off the macOS Keychain and triggers a "wants to use your confidential
// information" prompt on first use). We keep a random key in the app's private data
// dir (0600, owner-only) and AES-256-GCM the secrets with it. Same posture as a
// Chromium/Electron app profile: encrypted at rest, no OS keychain, no prompt.
//
// This is NOT defense against a determined attacker already running as the user
// (the key sits beside the data) — it's protection against casual disk reads and
// accidental sharing of settings.json, which is what local-first tools need.

const KEY_BYTES = 32 // AES-256
const IV_BYTES = 12 // GCM standard nonce
const TAG_BYTES = 16 // GCM auth tag

export const VONTRA_SECRET_KEY_ENV = 'VONTRA_SECRET_KEY_BASE64'
export const VONTRA_REQUIRE_EXTERNAL_SECRET_KEY_ENV = 'VONTRA_REQUIRE_EXTERNAL_SECRET_KEY'
export type LocalSecretKeyMode = 'profile-key' | 'external-env-key'

// Capture once at module load, then remove the credential from process.env before
// any shell, terminal, model CLI, Git hook, connector helper, or other child can
// inherit it. Standalone crypto consumers import this module too, so they get the
// same confinement without every spawn site having to remember a denylist.
const capturedExternalKeyValue = process.env[VONTRA_SECRET_KEY_ENV]?.trim() || undefined
if (capturedExternalKeyValue) delete process.env[VONTRA_SECRET_KEY_ENV]

export function capturedExternalSecretKeyValue(): string | undefined {
  return capturedExternalKeyValue
}

export function externalSecretKeyRequired(
  value = process.env[VONTRA_REQUIRE_EXTERNAL_SECRET_KEY_ENV],
): boolean {
  return /^(1|true|yes|on)$/i.test(value?.trim() ?? '')
}

/**
 * Parse an independently-held deployment key. A configured-but-invalid value is
 * a startup error: silently falling back to the profile key would make operators
 * believe copied-profile protection was active when it was not.
 */
export function parseExternalSecretKey(value: string | undefined): Buffer | null {
  const encoded = value?.trim() ?? ''
  if (!encoded) return null
  if (!/^[A-Za-z0-9+/]{43}=$/.test(encoded)) {
    throw new Error(`${VONTRA_SECRET_KEY_ENV} must be canonical base64 for exactly ${KEY_BYTES} bytes`)
  }
  const key = Buffer.from(encoded, 'base64')
  if (key.length !== KEY_BYTES || key.toString('base64') !== encoded) {
    throw new Error(`${VONTRA_SECRET_KEY_ENV} must be canonical base64 for exactly ${KEY_BYTES} bytes`)
  }
  return key
}

export function configuredLocalSecretKeyMode(value = capturedExternalKeyValue): LocalSecretKeyMode {
  return parseExternalSecretKey(value) ? 'external-env-key' : 'profile-key'
}

export function loadExistingKeySync(keyPath: string): Buffer | null {
  try {
    const key = readFileSync(keyPath)
    return key.length === KEY_BYTES ? key : null
  } catch {
    return null
  }
}

export function resolveLocalSecretKeySync(
  keyPath: string,
  externalValue = capturedExternalKeyValue,
  requireExternal = externalSecretKeyRequired(),
): { key: Buffer; mode: LocalSecretKeyMode; profileFallbackKey: Buffer | null } {
  const external = parseExternalSecretKey(externalValue)
  if (external) {
    // Retain a read-only fallback for a seamless one-time migration from the
    // profile key. Never create a new profile key in external-key mode.
    return { key: external, mode: 'external-env-key', profileFallbackKey: loadExistingKeySync(keyPath) }
  }
  if (requireExternal) {
    throw new Error(`${VONTRA_REQUIRE_EXTERNAL_SECRET_KEY_ENV} is enabled but ${VONTRA_SECRET_KEY_ENV} is missing`)
  }
  return { key: loadOrCreateKeySync(keyPath), mode: 'profile-key', profileFallbackKey: null }
}

// base64( iv | tag | ciphertext )
export function encryptWithKey(plaintext: string, key: Buffer): string {
  if (key.length !== KEY_BYTES) throw new Error(`local-crypto: key must be ${KEY_BYTES} bytes`)
  const iv = randomBytes(IV_BYTES)
  const cipher = createCipheriv('aes-256-gcm', key, iv)
  const ct = Buffer.concat([cipher.update(Buffer.from(plaintext, 'utf8')), cipher.final()])
  const tag = cipher.getAuthTag()
  return Buffer.concat([iv, tag, ct]).toString('base64')
}

export function decryptWithKey(payload: string, key: Buffer): string {
  if (key.length !== KEY_BYTES) throw new Error(`local-crypto: key must be ${KEY_BYTES} bytes`)
  const raw = Buffer.from(payload, 'base64')
  if (raw.length < IV_BYTES + TAG_BYTES) throw new Error('local-crypto: payload too short')
  const iv = raw.subarray(0, IV_BYTES)
  const tag = raw.subarray(IV_BYTES, IV_BYTES + TAG_BYTES)
  const ct = raw.subarray(IV_BYTES + TAG_BYTES)
  const decipher = createDecipheriv('aes-256-gcm', key, iv)
  decipher.setAuthTag(tag)
  // final() throws if the tag doesn't verify (wrong key, tampered, or a foreign
  // blob such as a legacy safeStorage ciphertext) — callers treat that as "unset".
  return Buffer.concat([decipher.update(ct), decipher.final()]).toString('utf8')
}

// Read the persisted key, creating it (and its parent dir) on first use. Sync so the
// store's synchronous protect/reveal helpers can call it without going async.
//
// After creating a key we re-read it to confirm it actually landed on disk. If the
// write silently failed (or another process raced in), we adopt whatever is on disk
// rather than encrypting with an ephemeral key that won't survive a restart — that
// would make every file written with it unreadable on the next launch.
export function loadOrCreateKeySync(keyPath: string): Buffer {
  try {
    const existing = readFileSync(keyPath)
    if (existing.length === KEY_BYTES) return existing
  } catch {
    /* missing or unreadable — fall through and create */
  }
  const key = randomBytes(KEY_BYTES)
  mkdirSync(path.dirname(keyPath), { recursive: true })
  // Write the full key to a private temp first, then ATOMICALLY link it into place. link()
  // fails with EEXIST if another concurrently-booting process already won — and because the
  // temp is fully written before linking, any visible keyPath is always the complete 32 bytes
  // (no 0-byte window where a racing reader sees a half-created file and bails).
  const tmp = `${keyPath}.tmp-${process.pid}-${randomBytes(4).toString('hex')}`
  try {
    writeFileSync(tmp, key, { mode: 0o600 })
    try {
      linkSync(tmp, keyPath)
      return key // we won the race
    } catch (e: any) {
      if (e?.code !== 'EEXIST') {
        throw new Error(`local-crypto: failed to persist secret key to ${keyPath}: ${String(e?.message ?? e)}`)
      }
      // Another process linked first — adopt its (fully-written) key.
      const persisted = readFileSync(keyPath)
      if (persisted.length === KEY_BYTES) return persisted
      throw new Error(`local-crypto: secret key at ${keyPath} exists but is not ${KEY_BYTES} bytes`)
    }
  } finally {
    try { unlinkSync(tmp) } catch { /* temp already gone or never created */ }
  }
}
