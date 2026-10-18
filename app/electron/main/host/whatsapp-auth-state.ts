import { promises as fs } from 'node:fs'
import path from 'node:path'
import { overwriteAndUnlink, readProtectedJson, writeProtectedJson } from '../../store'

// Baileys deliberately leaves persistence to the embedding application. Its sample
// useMultiFileAuthState writes creds.json and every Signal key as plaintext. This adapter
// keeps the same AuthenticationState contract while persisting one app-key protected JSON
// envelope with private file/directory modes.

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type BaileysApi = any

interface StoredWhatsAppAuthState {
  version: 1
  // BufferJSON-compatible JSON values (Buffers become { type: 'Buffer', data: base64 }).
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  creds: any
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  keys: Record<string, Record<string, any>>
  credentialUpdatedAt: number | null
}

export interface ProtectedWhatsAppAuthState {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  state: any
  /** Persist the current mutable creds object; returns the authoritative rotation time. */
  saveCreds(): Promise<number | null>
  credentialUpdatedAt(): number | null
}

interface AuthStateDeps {
  readProtected<T>(file: string, fallback: T, migratePlaintext?: boolean): Promise<T>
  writeProtected(file: string, value: unknown): Promise<void>
  shredFile(file: string): Promise<void>
  now(): number
}

const defaultDeps: AuthStateDeps = {
  readProtected: readProtectedJson,
  writeProtected: writeProtectedJson,
  shredFile: overwriteAndUnlink,
  now: Date.now,
}

const AUTH_FILE = 'auth-state.json'
const AUTH_VERSION = 1
const MAX_LEGACY_FILES = 20_000
const MAX_LEGACY_FILE_BYTES = 8 * 1024 * 1024
const MAX_LEGACY_TOTAL_BYTES = 128 * 1024 * 1024
const pendingAuthWrites = new Map<string, Set<Promise<unknown>>>()
const activeAuthGenerations = new Map<string, number>()
const wipingAuthFiles = new Set<string>()
let globalAuthWipeDepth = 0
let authGenerationCounter = 0

/** Startup never creates or migrates credentials; only an existing protected paired envelope resumes. */
export async function hasProtectedRegisteredWhatsAppAuthState(baseDir: string, sourceId: string): Promise<boolean> {
  const file = protectedAuthFileFor(baseDir, sourceId)
  if (globalAuthWipeDepth > 0 || wipingAuthFiles.has(file)) return false
  if (!await protectedFileExists(file)) return false
  await assertProtectedEnvelope(file)
  const stored = await readProtectedJson<StoredWhatsAppAuthState | null>(file, null, false)
  return stored?.version === AUTH_VERSION && stored.creds?.registered === true && typeof stored.keys === 'object'
}

class AuthStateInvalidatedError extends Error {
  constructor() { super('WhatsApp credential state is no longer active; refusing a late write.') }
}

function beginAuthGeneration(file: string): number {
  const generation = ++authGenerationCounter
  activeAuthGenerations.set(file, generation)
  return generation
}

function assertAuthGeneration(file: string, generation: number): void {
  if (activeAuthGenerations.get(file) !== generation) throw new AuthStateInvalidatedError()
}

function trackAuthWrite<T>(file: string, operation: Promise<T>): Promise<T> {
  const pending = pendingAuthWrites.get(file) ?? new Set<Promise<unknown>>()
  pending.add(operation)
  pendingAuthWrites.set(file, pending)
  void operation.finally(() => {
    pending.delete(operation)
    if (!pending.size) pendingAuthWrites.delete(file)
  }).catch(() => {})
  return operation
}

// These are the complete SignalDataTypeMap keys in the pinned Baileys version. Sort longest
// first because several share prefixes (sender-key-memory / sender-key).
const LEGACY_KEY_TYPES = [
  'app-state-sync-version',
  'app-state-sync-key',
  'sender-key-memory',
  'identity-key',
  'device-list',
  'lid-mapping',
  'sender-key',
  'pre-key',
  'session',
  'tctoken',
] as const

function assertSourceId(sourceId: string): string {
  const id = String(sourceId || '')
  if (!/^wa_[A-Za-z0-9_-]{1,80}$/.test(id)) {
    throw new Error('WhatsApp bridge source id is invalid; refusing to access credential storage.')
  }
  return id
}

export function authFolderFor(baseDir: string, sourceId: string): string {
  return path.join(baseDir, 'whatsapp-auth', assertSourceId(sourceId))
}

export function protectedAuthFileFor(baseDir: string, sourceId: string): string {
  return path.join(authFolderFor(baseDir, sourceId), AUTH_FILE)
}

async function ensurePrivateDirectory(directory: string): Promise<void> {
  const existing = await fs.lstat(directory).catch((error: NodeJS.ErrnoException) => {
    if (error?.code === 'ENOENT') return null
    throw error
  })
  if (existing?.isSymbolicLink() || (existing && !existing.isDirectory())) {
    throw new Error(`Refusing unsafe WhatsApp credential path: ${directory}`)
  }
  if (!existing) await fs.mkdir(directory, { recursive: true, mode: 0o700 })
  const verified = await fs.lstat(directory)
  if (verified.isSymbolicLink() || !verified.isDirectory()) {
    throw new Error(`Refusing unsafe WhatsApp credential path: ${directory}`)
  }
  await fs.chmod(directory, 0o700).catch(() => {})
}

function encodeBufferJson(value: unknown, B: BaileysApi): unknown {
  return JSON.parse(JSON.stringify(value, B.BufferJSON.replacer))
}

function decodeBufferJson<T>(value: unknown, B: BaileysApi): T {
  return JSON.parse(JSON.stringify(value), B.BufferJSON.reviver) as T
}

function emptyState(B: BaileysApi): StoredWhatsAppAuthState {
  return {
    version: AUTH_VERSION,
    creds: encodeBufferJson(B.initAuthCreds(), B),
    keys: {},
    credentialUpdatedAt: null,
  }
}

function legacyKeyName(fileName: string): { type: string; id: string } | null {
  if (!fileName.endsWith('.json') || fileName === 'creds.json' || fileName === AUTH_FILE) return null
  const stem = fileName.slice(0, -'.json'.length)
  for (const type of LEGACY_KEY_TYPES) {
    const prefix = `${type}-`
    if (stem.startsWith(prefix) && stem.length > prefix.length) return { type, id: stem.slice(prefix.length) }
  }
  return null
}

function legacySafeId(id: string): string {
  return id.replace(/\//g, '__').replace(/:/g, '-')
}

async function readLegacyState(folder: string, B: BaileysApi): Promise<{ state: StoredWhatsAppAuthState; files: string[] } | null> {
  const names = await fs.readdir(folder)
  const legacyNames = names.filter((name) => name !== AUTH_FILE)
  if (!legacyNames.length) return null
  if (legacyNames.length > MAX_LEGACY_FILES) throw new Error('Legacy WhatsApp credential folder has too many files to migrate safely.')
  if (!legacyNames.includes('creds.json')) {
    throw new Error('Legacy WhatsApp credential folder is incomplete; plaintext auth remains disabled.')
  }

  let totalBytes = 0
  const parsed = new Map<string, unknown>()
  const files: string[] = []
  let credsModifiedAt = 0
  for (const name of legacyNames) {
    if (name !== 'creds.json' && !legacyKeyName(name)) {
      throw new Error(`Legacy WhatsApp credential file is not recognized (${name}); plaintext auth remains disabled.`)
    }
    const file = path.join(folder, name)
    const stat = await fs.lstat(file)
    if (stat.isSymbolicLink() || !stat.isFile()) {
      throw new Error(`Legacy WhatsApp credential entry is unsafe (${name}); plaintext auth remains disabled.`)
    }
    if (stat.size > MAX_LEGACY_FILE_BYTES) throw new Error(`Legacy WhatsApp credential file is oversized (${name}).`)
    totalBytes += stat.size
    if (totalBytes > MAX_LEGACY_TOTAL_BYTES) throw new Error('Legacy WhatsApp credential state is oversized.')
    const raw = await fs.readFile(file, 'utf8')
    try {
      parsed.set(name, JSON.parse(raw, B.BufferJSON.reviver))
    } catch {
      throw new Error(`Legacy WhatsApp credential file is invalid JSON (${name}); plaintext auth remains disabled.`)
    }
    if (name === 'creds.json') credsModifiedAt = stat.mtimeMs
    files.push(file)
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const creds = parsed.get('creds.json') as any
  if (!creds || typeof creds !== 'object' || Array.isArray(creds)) {
    throw new Error('Legacy WhatsApp creds.json has an invalid shape; plaintext auth remains disabled.')
  }
  const keys: StoredWhatsAppAuthState['keys'] = {}
  for (const [name, value] of parsed) {
    const key = legacyKeyName(name)
    if (!key) continue
    ;(keys[key.type] ??= {})[key.id] = encodeBufferJson(value, B)
  }
  return {
    state: {
      version: AUTH_VERSION,
      creds: encodeBufferJson(creds, B),
      keys,
      credentialUpdatedAt: creds.registered || creds.me ? Math.floor(credsModifiedAt) || null : null,
    },
    files,
  }
}

async function protectedFileExists(file: string): Promise<boolean> {
  const stat = await fs.lstat(file).catch((error: NodeJS.ErrnoException) => {
    if (error?.code === 'ENOENT') return null
    throw error
  })
  if (!stat) return false
  if (stat.isSymbolicLink() || !stat.isFile()) throw new Error('WhatsApp protected credential file is unsafe.')
  return true
}

async function assertProtectedEnvelope(file: string): Promise<void> {
  const raw = await fs.readFile(file, 'utf8')
  let parsed: unknown
  try { parsed = JSON.parse(raw) } catch { throw new Error('WhatsApp protected credential file is invalid.') }
  const encrypted = (parsed as { encrypted?: unknown } | null)?.encrypted
  if (typeof encrypted !== 'string' || !encrypted.startsWith('safe-json:v1:')) {
    throw new Error('WhatsApp credential state is not app-key protected; plaintext auth remains disabled.')
  }
}

/**
 * Create a Baileys AuthenticationState backed by TensorFold's app-key protected JSON.
 * Legacy multi-file state is migrated only after every entry has been bounded, recognized,
 * parsed, and written successfully; then each plaintext file is overwritten and unlinked.
 */
export async function createProtectedWhatsAppAuthState(
  B: BaileysApi,
  baseDir: string,
  sourceId: string,
  deps: AuthStateDeps = defaultDeps,
): Promise<ProtectedWhatsAppAuthState> {
  if (!B?.BufferJSON?.replacer || !B?.BufferJSON?.reviver || typeof B?.initAuthCreds !== 'function') {
    throw new Error('WhatsApp bridge library lacks the protected-auth primitives required by this build.')
  }
  const root = path.join(baseDir, 'whatsapp-auth')
  const folder = authFolderFor(baseDir, sourceId)
  const file = protectedAuthFileFor(baseDir, sourceId)
  if (globalAuthWipeDepth > 0 || wipingAuthFiles.has(file)) throw new AuthStateInvalidatedError()
  // Claim the new generation synchronously, before the first await. A reconnect replaces the
  // prior socket's writer; draining below settles any write it had already entered while every
  // not-yet-started old update now fails its generation check.
  const generation = beginAuthGeneration(file)
  await quiesceWhatsAppAuthStateWrites(file)
  assertAuthGeneration(file, generation)
  await ensurePrivateDirectory(root)
  assertAuthGeneration(file, generation)
  await ensurePrivateDirectory(folder)
  assertAuthGeneration(file, generation)

  const guardedWrite = async (value: unknown): Promise<void> => {
    assertAuthGeneration(file, generation)
    // Register before yielding so a wipe sees and drains an initialization/migration write that
    // passed its generation check immediately before invalidation.
    await trackAuthWrite(file, deps.writeProtected(file, value))
    assertAuthGeneration(file, generation)
  }

  let stored: StoredWhatsAppAuthState
  if (await protectedFileExists(file)) {
    await assertProtectedEnvelope(file)
    const unreadable = emptyState(B)
    stored = await deps.readProtected<StoredWhatsAppAuthState>(file, unreadable, false)
    if (stored === unreadable) throw new Error('WhatsApp protected credential state could not be decrypted; re-pairing is required.')
    if (stored.version !== AUTH_VERSION || !stored.creds || typeof stored.keys !== 'object') {
      throw new Error('WhatsApp protected credential state has an unsupported shape.')
    }
  } else {
    const legacy = await readLegacyState(folder, B)
    stored = legacy?.state ?? emptyState(B)
    await guardedWrite(stored)
    await assertProtectedEnvelope(file)
    // Only shred after a durable protected replacement exists and can be decoded.
    const verified = await deps.readProtected<StoredWhatsAppAuthState | null>(file, null, false)
    if (!verified || verified.version !== AUTH_VERSION) throw new Error('WhatsApp credential migration could not verify its protected replacement.')
    assertAuthGeneration(file, generation)
    for (const legacyFile of legacy?.files ?? []) await deps.shredFile(legacyFile)
  }
  assertAuthGeneration(file, generation)
  await fs.chmod(file, 0o600).catch(() => {})

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const creds = decodeBufferJson<any>(stored.creds, B)
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const keyData = decodeBufferJson<Record<string, Record<string, any>>>(stored.keys ?? {}, B)
  let credentialUpdatedAt = Number.isFinite(stored.credentialUpdatedAt) ? Number(stored.credentialUpdatedAt) : null
  let queue = Promise.resolve()

  const persist = (): Promise<void> => {
    assertAuthGeneration(file, generation)
    const snapshot: StoredWhatsAppAuthState = {
      version: AUTH_VERSION,
      creds: encodeBufferJson(creds, B),
      keys: encodeBufferJson(keyData, B) as StoredWhatsAppAuthState['keys'],
      credentialUpdatedAt,
    }
    return guardedWrite(snapshot)
  }
  const mutate = <T>(work: () => Promise<T> | T): Promise<T> => {
    const result = queue.then(() => {
      assertAuthGeneration(file, generation)
      return work()
    })
    queue = result.then(() => undefined, () => undefined)
    return trackAuthWrite(file, result)
  }

  return {
    state: {
      creds,
      keys: {
        get: async (type: string, ids: string[]) => {
          await queue
          assertAuthGeneration(file, generation)
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const result: Record<string, any> = {}
          for (const id of ids) {
            let value = keyData[type]?.[id]
            // Legacy multi-file names replace ':' and '/' and cannot be reversed reliably.
            // Exact keys win; this compatibility lookup preserves migrated sessions.
            if (value == null) value = keyData[type]?.[legacySafeId(id)]
            if (type === 'app-state-sync-key' && value) value = B.proto.Message.AppStateSyncKeyData.fromObject(value)
            result[id] = value
          }
          return result
        },
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        set: (data: Record<string, Record<string, any>>) => mutate(async () => {
          for (const [type, values] of Object.entries(data ?? {})) {
            const bucket = (keyData[type] ??= {})
            for (const [id, value] of Object.entries(values ?? {})) {
              if (value == null) {
                delete bucket[id]
                delete bucket[legacySafeId(id)]
              } else {
                bucket[id] = value
                if (legacySafeId(id) !== id) delete bucket[legacySafeId(id)]
              }
            }
            if (!Object.keys(bucket).length) delete keyData[type]
          }
          await persist()
        }),
        clear: () => mutate(async () => {
          for (const type of Object.keys(keyData)) delete keyData[type]
          await persist()
        }),
      },
    },
    saveCreds: () => mutate(async () => {
      // Pre-pair init creates random keys too, but only registered/me-bearing credentials are
      // a durable paired credential whose age should enter the 90-day readiness policy.
      if (creds.registered || creds.me) credentialUpdatedAt = deps.now()
      await persist()
      return credentialUpdatedAt
    }),
    credentialUpdatedAt: () => credentialUpdatedAt,
  }
}

async function shredTree(target: string, shredFile: (file: string) => Promise<void>): Promise<void> {
  const stat = await fs.lstat(target).catch((error: NodeJS.ErrnoException) => {
    if (error?.code === 'ENOENT') return null
    throw error
  })
  if (!stat) return
  if (stat.isSymbolicLink()) {
    await fs.unlink(target)
    return
  }
  if (stat.isFile()) {
    await shredFile(target)
    return
  }
  if (!stat.isDirectory()) throw new Error(`Refusing to shred non-file WhatsApp auth entry: ${target}`)
  for (const name of await fs.readdir(target)) await shredTree(path.join(target, name), shredFile)
  await fs.rmdir(target).catch((error: NodeJS.ErrnoException) => { if (error?.code !== 'ENOENT') throw error })
}

/** Drain in-flight Signal/credential writes before a source removal or global purge. */
export async function quiesceWhatsAppAuthStateWrites(file?: string): Promise<void> {
  while (file ? (pendingAuthWrites.get(file)?.size ?? 0) > 0 : pendingAuthWrites.size > 0) {
    const pending = file
      ? [...(pendingAuthWrites.get(file) ?? [])]
      : [...pendingAuthWrites.values()].flatMap((items) => [...items])
    const settled = await Promise.allSettled(pending)
    const failed = settled.find((item): item is PromiseRejectedResult => item.status === 'rejected' && !(item.reason instanceof AuthStateInvalidatedError))
    if (failed) throw failed.reason
  }
}

/**
 * Global credential-wipe barrier used by local-state purge. Begin is synchronous: it blocks
 * new auth handles and invalidates every existing handle before purge reaches its first await.
 * The invalidation remains permanent after end; only a newly-created post-purge handle can write.
 */
export function beginWhatsAppAuthStatePurge(): void {
  globalAuthWipeDepth++
  activeAuthGenerations.clear()
}

export function endWhatsAppAuthStatePurge(): void {
  globalAuthWipeDepth = Math.max(0, globalAuthWipeDepth - 1)
}

/** Symlink-safe credential removal for source deletion. Symlinks are unlinked, never followed. */
export async function shredWhatsAppAuthState(baseDir: string, sourceId: string, shredFile = overwriteAndUnlink): Promise<void> {
  const file = protectedAuthFileFor(baseDir, sourceId)
  // Invalidate BEFORE draining: a creds.update already queued inside a stopped Baileys socket may
  // arrive after the first empty-queue check. Its captured saveCreds closure must never recreate
  // auth-state.json after this wipe begins.
  wipingAuthFiles.add(file)
  activeAuthGenerations.delete(file)
  try {
    await quiesceWhatsAppAuthStateWrites(file)
    await shredTree(authFolderFor(baseDir, sourceId), shredFile)
  } finally {
    wipingAuthFiles.delete(file)
  }
}
