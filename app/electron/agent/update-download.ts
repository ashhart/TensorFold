import { createHash, randomUUID } from 'node:crypto'
import { constants as fsConstants, promises as fs } from 'node:fs'
import path from 'node:path'
import { discardResponseBody, pinnedFetch, type PinnedHostResolver } from './pinned-fetch'
import { resolveFetchableUrl } from './tool-web'
import type { AppUpdateChecksum } from './types'

export const MAX_UPDATE_ARTIFACT_BYTES = 1024 * 1024 * 1024
const MAX_UPDATE_REDIRECTS = 5

export interface VerifiedUpdateDownloadInput {
  downloadUrl: string
  checksum: AppUpdateChecksum
  manifestSignatureVerified: boolean
  downloadRoot: string
  signal?: AbortSignal
  onProgress?: (progress: { receivedBytes: number; totalBytes: number | null }) => void
}

export interface VerifiedUpdateDownloadResult {
  path: string
  finalUrl: string
  sha256: string
  bytes: number
  redirects: number
  fileIdentity: { device: string; inode: string }
}

export interface UpdateDownloadDependencies {
  maxBytes?: number
  maxRedirects?: number
  lookup?: PinnedHostResolver
  fetchImpl?: typeof fetch
  randomId?: () => string
}

function safeArtifactName(url: URL): string {
  let basename = ''
  try { basename = decodeURIComponent(path.posix.basename(url.pathname)) } catch { basename = path.posix.basename(url.pathname) }
  const clean = basename.replace(/[^a-zA-Z0-9._-]+/g, '-').replace(/^[-.]+/, '').slice(0, 160)
  return clean || 'TensorFold-update.bin'
}

function expectedSha256(checksum: AppUpdateChecksum): string {
  if (checksum.algorithm !== 'sha256' || !/^[a-f0-9]{64}$/i.test(checksum.value)) {
    throw new Error('Update download requires a valid SHA-256 checksum from the signed manifest.')
  }
  return checksum.value.toLowerCase()
}

function isWithin(root: string, candidate: string): boolean {
  const relative = path.relative(root, candidate)
  return relative === '' || (!relative.startsWith(`..${path.sep}`) && relative !== '..' && !path.isAbsolute(relative))
}

function sameIdentity(
  left: { dev: bigint; ino: bigint },
  right: { dev: bigint; ino: bigint },
): boolean {
  return left.dev === right.dev && left.ino === right.ino
}

function declaredLength(response: Response, maxBytes: number): number | null {
  const raw = response.headers.get('content-length')
  if (raw == null || raw.trim() === '') return null
  if (!/^\d+$/.test(raw.trim())) throw new Error('Update server returned an invalid Content-Length header.')
  const value = Number(raw)
  if (!Number.isSafeInteger(value)) throw new Error('Update server returned an invalid Content-Length header.')
  if (value > maxBytes) throw new Error(`Update artifact exceeds the ${maxBytes}-byte download limit.`)
  return value
}

async function writeResponseBody(
  response: Response,
  filePath: string,
  maxBytes: number,
  onProgress?: VerifiedUpdateDownloadInput['onProgress'],
): Promise<{ bytes: number; sha256: string }> {
  if (!response.body) throw new Error('Update server returned an empty response body.')
  const totalBytes = declaredLength(response, maxBytes)
  onProgress?.({ receivedBytes: 0, totalBytes })
  const handle = await fs.open(filePath, 'wx', 0o600)
  const hash = createHash('sha256')
  const reader = response.body.getReader()
  let bytes = 0
  try {
    for (;;) {
      const chunk = await reader.read()
      if (chunk.done) break
      const value = Buffer.from(chunk.value)
      bytes += value.byteLength
      if (bytes > maxBytes) {
        await reader.cancel('update artifact exceeded size limit').catch(() => {})
        throw new Error(`Update artifact exceeds the ${maxBytes}-byte download limit.`)
      }
      hash.update(value)
      await handle.write(value)
      onProgress?.({ receivedBytes: bytes, totalBytes })
    }
    await handle.sync()
  } finally {
    await handle.close().catch(() => {})
  }
  return { bytes, sha256: hash.digest('hex') }
}

/**
 * Download one artifact whose URL/checksum are bound by a verified manifest signature.
 * Every redirect hop is re-resolved, checked for globally-routable addresses, and socket-pinned.
 * Bytes land in a unique `.part` file and are atomically renamed only after SHA-256 verification.
 * Any failure removes the whole unique download directory, including partial/final files.
 */
export async function downloadVerifiedUpdate(
  input: VerifiedUpdateDownloadInput,
  deps: UpdateDownloadDependencies = {},
): Promise<VerifiedUpdateDownloadResult> {
  if (!input.manifestSignatureVerified) {
    throw new Error('Refusing update download because the manifest signature is not verified.')
  }
  const expected = expectedSha256(input.checksum)
  const maxBytes = Math.max(1, Math.floor(deps.maxBytes ?? MAX_UPDATE_ARTIFACT_BYTES))
  const maxRedirects = Math.max(0, Math.floor(deps.maxRedirects ?? MAX_UPDATE_REDIRECTS))
  await fs.mkdir(input.downloadRoot, { recursive: true, mode: 0o700 })
  await fs.chmod(input.downloadRoot, 0o700)
  const uniqueDir = await fs.mkdtemp(path.join(input.downloadRoot, 'vontra-update-'))
  await fs.chmod(uniqueDir, 0o700)
  const id = (deps.randomId ?? randomUUID)().replace(/[^a-zA-Z0-9_-]/g, '').slice(0, 48) || 'download'
  let current = new URL(input.downloadUrl)
  let response: Response | null = null
  let partialPath: string | null = null
  try {
    for (let redirects = 0; ; redirects++) {
      const target = await resolveFetchableUrl(current.toString(), deps.lookup)
      response = await pinnedFetch(target, {
        redirect: 'manual',
        signal: input.signal,
        headers: { Accept: 'application/octet-stream, application/zip, application/x-apple-diskimage, */*' },
      }, deps.fetchImpl ?? fetch)
      const location = response.headers.get('location')
      if (response.status >= 300 && response.status < 400) {
        if (!location) throw new Error(`Update redirect returned HTTP ${response.status} without a Location header.`)
        if (redirects >= maxRedirects) throw new Error('Update download exceeded the redirect limit.')
        const next = new URL(location, current)
        await discardResponseBody(response)
        response = null
        // The next loop's resolveFetchableUrl is the authoritative string + DNS + pin check.
        current = next
        continue
      }
      if (!response.ok) throw new Error(`Update download returned HTTP ${response.status}.`)

      const filename = safeArtifactName(current)
      partialPath = path.join(uniqueDir, `.${filename}.${id}.part`)
      const finalPath = path.join(uniqueDir, filename)
      const written = await writeResponseBody(response, partialPath, maxBytes, input.onProgress)
      response = null
      if (written.sha256 !== expected) {
        throw new Error(`Update artifact SHA-256 mismatch (expected ${expected}, received ${written.sha256}).`)
      }
      await fs.rename(partialPath, finalPath)
      await fs.chmod(finalPath, 0o600)
      partialPath = null
      const finalStat = await fs.lstat(finalPath, { bigint: true })
      if (!finalStat.isFile() || finalStat.isSymbolicLink()) throw new Error('Downloaded update did not become a regular file.')
      return {
        path: finalPath,
        finalUrl: current.toString(),
        sha256: written.sha256,
        bytes: written.bytes,
        redirects,
        fileIdentity: { device: String(finalStat.dev), inode: String(finalStat.ino) },
      }
    }
  } catch (error) {
    if (response) await discardResponseBody(response)
    if (partialPath) await fs.rm(partialPath, { force: true }).catch(() => {})
    await fs.rm(uniqueDir, { recursive: true, force: true }).catch(() => {})
    throw error
  }
}

/**
 * Re-open and hash the final artifact immediately before handing its path to the OS.
 * The path must still be a non-symlink regular file under the private managed root,
 * and its inode/device identity must remain unchanged throughout verification.
 */
export async function reverifyDownloadedUpdate(
  download: VerifiedUpdateDownloadResult,
  downloadRoot: string,
  maxBytes = MAX_UPDATE_ARTIFACT_BYTES,
): Promise<VerifiedUpdateDownloadResult> {
  const limit = Math.max(1, Math.floor(maxBytes))
  const expected = expectedSha256({ algorithm: 'sha256', value: download.sha256 })
  const configuredRoot = path.resolve(downloadRoot)
  const root = await fs.realpath(configuredRoot)
  const candidate = path.resolve(download.path)
  if (!isWithin(configuredRoot, candidate) || candidate === configuredRoot) {
    throw new Error('Verified update path escaped the managed update directory.')
  }

  const before = await fs.lstat(candidate, { bigint: true })
  if (before.isSymbolicLink() || !before.isFile()) {
    throw new Error('Verified update path is no longer a regular file.')
  }
  if (String(before.dev) !== download.fileIdentity.device || String(before.ino) !== download.fileIdentity.inode) {
    throw new Error('Verified update file identity changed after download.')
  }
  const resolved = await fs.realpath(candidate)
  const expectedResolved = path.join(root, path.relative(configuredRoot, candidate))
  if (!isWithin(root, resolved) || resolved !== expectedResolved) {
    throw new Error('Verified update path changed identity or escaped the managed update directory.')
  }

  const noFollow = typeof fsConstants.O_NOFOLLOW === 'number' ? fsConstants.O_NOFOLLOW : 0
  const handle = await fs.open(candidate, fsConstants.O_RDONLY | noFollow)
  const hash = createHash('sha256')
  let bytes = 0
  try {
    const opened = await handle.stat({ bigint: true })
    if (!opened.isFile() || !sameIdentity(before, opened)) {
      throw new Error('Verified update file identity changed before re-verification.')
    }
    const buffer = Buffer.allocUnsafe(64 * 1024)
    for (;;) {
      const { bytesRead } = await handle.read(buffer, 0, buffer.byteLength, null)
      if (bytesRead === 0) break
      bytes += bytesRead
      if (bytes > limit) throw new Error(`Update artifact exceeds the ${limit}-byte download limit.`)
      hash.update(buffer.subarray(0, bytesRead))
    }
    const afterRead = await handle.stat({ bigint: true })
    if (!sameIdentity(opened, afterRead) || afterRead.size !== BigInt(bytes)) {
      throw new Error('Verified update file changed during re-verification.')
    }
  } finally {
    await handle.close().catch(() => {})
  }

  const after = await fs.lstat(candidate, { bigint: true })
  if (after.isSymbolicLink() || !after.isFile() || !sameIdentity(before, after)) {
    throw new Error('Verified update file identity changed during re-verification.')
  }
  const sha256 = hash.digest('hex')
  if (bytes !== download.bytes || sha256 !== expected) {
    throw new Error('Verified update artifact no longer matches the downloaded bytes and SHA-256.')
  }
  await fs.chmod(candidate, 0o600)
  return { ...download, path: candidate, sha256, bytes }
}
