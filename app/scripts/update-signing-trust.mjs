import { createHash, createPublicKey, timingSafeEqual } from 'node:crypto'
import { readFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const projectRoot = fileURLToPath(new URL('..', import.meta.url))
const RELEASE_TRUST_PATH = 'build/release-trust.json'

// SHA-256 of the canonical DER SPKI documented as the orphaned development key. This immutable
// byte fingerprint is independent of the mutable ID/readiness metadata in release-trust.json.
export const ORPHANED_DEVELOPMENT_UPDATE_TRUST_SPKI_SHA256 = '8cbd0edad76180f40f2daa086772f3ab12a9b255084e8db459b15778e2611732'

export async function readReleaseTrustConfig(root = projectRoot) {
  const filename = path.join(root, RELEASE_TRUST_PATH)
  const source = await readFile(filename, 'utf8')
  let parsed
  try {
    parsed = JSON.parse(source)
  } catch {
    throw new Error(`${RELEASE_TRUST_PATH} must be strict JSON without comments or trailing content.`)
  }
  assertRecord(parsed, 'release trust configuration')
  assertExactKeys(parsed, ['schemaVersion', 'update', 'macos'], 'release trust configuration')
  if (parsed.schemaVersion !== 1) throw new Error(`${RELEASE_TRUST_PATH} schemaVersion must be 1.`)

  assertRecord(parsed.update, 'update trust configuration')
  assertExactKeys(parsed.update, ['id', 'productionReady', 'ed25519PublicKey'], 'update trust configuration')
  const id = requiredString(parsed.update.id, 'update.id', 160)
  if (!/^[a-z0-9][a-z0-9._-]*$/i.test(id)) throw new Error(`${RELEASE_TRUST_PATH} update.id contains unsupported characters.`)
  if (typeof parsed.update.productionReady !== 'boolean') throw new Error(`${RELEASE_TRUST_PATH} update.productionReady must be boolean.`)
  const ed25519PublicKey = requiredString(parsed.update.ed25519PublicKey, 'update.ed25519PublicKey', 4096)
  // Parse eagerly so malformed/non-Ed25519 roots fail before any release operation.
  publicSpkiDer(ed25519PublicKey)

  assertRecord(parsed.macos, 'macOS release signer configuration')
  assertExactKeys(parsed.macos, ['identity', 'teamId', 'productionReady'], 'macOS release signer configuration')
  const identity = requiredString(parsed.macos.identity, 'macos.identity', 320)
  if (/[/\r\n\u2028\u2029]/.test(identity)) throw new Error(`${RELEASE_TRUST_PATH} macos.identity contains unsupported characters.`)
  const teamId = requiredString(parsed.macos.teamId, 'macos.teamId', 10)
  if (!/^[A-Z0-9]{10}$/.test(teamId)) throw new Error(`${RELEASE_TRUST_PATH} macos.teamId must contain exactly 10 uppercase letters or digits.`)
  const identityMatch = /^Developer ID Application: .+ \(([A-Z0-9]{10})\)$/.exec(identity)
  if (!identityMatch) throw new Error(`${RELEASE_TRUST_PATH} macos.identity must be an exact Developer ID Application identity.`)
  if (identityMatch[1] !== teamId) throw new Error(`${RELEASE_TRUST_PATH} macos.identity and macos.teamId must identify the same Apple team.`)
  if (typeof parsed.macos.productionReady !== 'boolean') throw new Error(`${RELEASE_TRUST_PATH} macos.productionReady must be boolean.`)

  // Re-serialize a normalized, fixed-order schema and require byte equality. Besides making review
  // diffs deterministic, this rejects duplicate JSON keys: JSON.parse would otherwise silently let
  // the last duplicate win, which is too ambiguous for release trust material.
  const normalized = {
    schemaVersion: 1,
    update: { id, productionReady: parsed.update.productionReady, ed25519PublicKey },
    macos: { identity, teamId, productionReady: parsed.macos.productionReady },
  }
  if (source !== `${JSON.stringify(normalized, null, 2)}\n`) {
    throw new Error(`${RELEASE_TRUST_PATH} must use the canonical strict-JSON serialization (duplicate keys, reordered fields, comments, and hidden decoys are refused).`)
  }
  return normalized
}

export async function embeddedUpdateTrustRoot(root = projectRoot) {
  return (await readReleaseTrustConfig(root)).update.ed25519PublicKey
}

export async function embeddedUpdateTrustMetadata(root = projectRoot) {
  const { id, productionReady } = (await readReleaseTrustConfig(root)).update
  return { id, productionReady }
}

export async function assertProductionUpdateTrustRoot(root = projectRoot, config = null) {
  const trust = config ?? await readReleaseTrustConfig(root)
  const metadata = { id: trust.update.id, productionReady: trust.update.productionReady }
  const fingerprint = publicSpkiSha256(trust.update.ed25519PublicKey)
  const reasons = []
  if (!metadata.productionReady) reasons.push('VONTRA_UPDATE_TRUST_ROOT_PRODUCTION_READY is false')
  if (/(?:placeholder|orphan|development|dev(?:elopment)?-only|replace|example|unknown|tbd)/i.test(metadata.id)) reasons.push(`trust-root ID ${metadata.id} is marked placeholder/orphan/development-only`)
  if (fingerprint === ORPHANED_DEVELOPMENT_UPDATE_TRUST_SPKI_SHA256) reasons.push(`public-key SPKI SHA-256 ${fingerprint} is the known orphaned development key`)
  if (reasons.length) {
    throw new Error(`Production packaging is blocked by the embedded update trust root: ${reasons.join('; ')}. Replace it with the organization-controlled release public key before distribution.`)
  }
  return { ...metadata, fingerprint }
}

export async function assertProductionMacSigner(root = projectRoot, config = null) {
  const trust = config ?? await readReleaseTrustConfig(root)
  const { identity, teamId, productionReady } = trust.macos
  const reasons = []
  if (!productionReady) reasons.push('MACOS_RELEASE_IDENTITY_PRODUCTION_READY is false')
  if (/(?:replace|placeholder|example|unknown|tbd)/i.test(identity) || /^X{10}$/.test(teamId)) {
    reasons.push(`macOS release identity ${identity} / team ${teamId} is still a placeholder`)
  }
  if (reasons.length) {
    throw new Error(`Production packaging is blocked by the repository-pinned macOS signer: ${reasons.join('; ')}. Replace it with the organization's exact Developer ID legal name and Team ID before distribution.`)
  }
  return {
    identity,
    teamId,
    productionReady,
  }
}

export function publicSpkiDer(key) {
  const publicKey = key?.type === 'public' ? key : createPublicKey(key)
  if (publicKey.asymmetricKeyType !== 'ed25519') throw new Error('Update trust roots must be Ed25519 public keys.')
  return Buffer.from(publicKey.export({ type: 'spki', format: 'der' }))
}

export function publicSpkiSha256(key) {
  return createHash('sha256').update(publicSpkiDer(key)).digest('hex')
}

export function assertSigningKeyMatchesTrustRoot(privateKey, trustedPublicKey) {
  if (privateKey.asymmetricKeyType !== 'ed25519') throw new Error('--private-key must contain an Ed25519 private key.')
  const derived = publicSpkiDer(privateKey)
  const trusted = publicSpkiDer(trustedPublicKey)
  if (derived.byteLength !== trusted.byteLength || !timingSafeEqual(derived, trusted)) {
    throw new Error('The supplied private key does not match the embedded TensorFold update trust root.')
  }
}

function assertRecord(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value) || Object.getPrototypeOf(value) !== Object.prototype) {
    throw new Error(`${RELEASE_TRUST_PATH} ${label} must be an object.`)
  }
}

function assertExactKeys(value, expected, label) {
  const actual = Object.keys(value).sort()
  const wanted = [...expected].sort()
  if (actual.length !== wanted.length || actual.some((key, index) => key !== wanted[index])) {
    throw new Error(`${RELEASE_TRUST_PATH} ${label} must contain exactly: ${expected.join(', ')}.`)
  }
}

function requiredString(value, field, maxLength) {
  if (typeof value !== 'string' || value !== value.trim() || !value || value.length > maxLength || /\0/.test(value)) {
    throw new Error(`${RELEASE_TRUST_PATH} ${field} must be a non-empty, trimmed string of at most ${maxLength} characters.`)
  }
  return value
}
