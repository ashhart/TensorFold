import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process'
import { lstat } from 'node:fs/promises'
import os from 'node:os'
import type { AttachmentTextCapability, AttachmentTextRequest, AttachmentTextResult } from '../../agent/types'

export const ATTACHMENT_TEXT_MAX_BYTES = 20 * 1024 * 1024
export const ATTACHMENT_TEXT_MAX_PAGES = 8
export const ATTACHMENT_TEXT_TIMEOUT_MS = 45_000
const MAX_OUTPUT_BYTES = 1024 * 1024
const MAX_TEXT_CHARS = 199_800
const SUPPORTED_MIME = /^(application\/pdf|image\/(jpeg|jpg|png|gif|webp|bmp|x-ms-bmp|heic|heif|tiff))$/
const REVIEW_WARNING = 'Review every line against the original, especially handwriting, names and numbers. OCR may omit or misread text even when its confidence score is high.'

export function validateAttachmentTextRequest(input: unknown): { name: string; mime: string; bytes: Buffer } {
  if (!input || typeof input !== 'object' || Array.isArray(input)) throw new Error('Select an image or PDF file for text extraction.')
  const value = input as Record<string, unknown>
  if ('path' in value || 'filePath' in value || 'url' in value) throw new Error('Text extraction accepts selected file bytes, never filesystem paths or URLs.')
  const name = typeof value.name === 'string' ? value.name.replace(/[\u0000-\u001f]/g, ' ').slice(0, 240) : 'file'
  const mime = typeof value.mime === 'string' ? value.mime.toLowerCase() : ''
  if (!SUPPORTED_MIME.test(mime)) throw new Error('Text extraction supports images and PDF files only.')
  const bytes = value.bytes
  if (!(bytes instanceof Uint8Array) && !Array.isArray(bytes)) throw new Error('Select the file again; its bytes are unavailable.')
  if (bytes.length === 0 || bytes.length > ATTACHMENT_TEXT_MAX_BYTES) throw new Error('Text extraction accepts nonempty files up to 20 MiB.')
  if (Array.isArray(bytes) && !bytes.every((item) => Number.isInteger(item) && item >= 0 && item <= 255)) throw new Error('The selected file bytes are invalid.')
  return { name, mime, bytes: Buffer.from(bytes) }
}

/** Reject malformed or incomplete helper output instead of presenting a guessed transcript. */
export function normalizeAttachmentTextResult(input: unknown): AttachmentTextResult {
  if (!input || typeof input !== 'object' || Array.isArray(input)) throw new Error('The local text extractor returned an invalid result.')
  const value = input as Record<string, unknown>
  if (value.ok !== true) return { ok: false, error: typeof value.error === 'string' ? value.error.slice(0, 500) : 'Text extraction failed. Choose a clearer image or type a transcript.' }
  if (typeof value.text !== 'string' || value.text.length > MAX_TEXT_CHARS) throw new Error('The local text extractor returned oversized or missing text.')
  if (!Number.isInteger(value.pageCount) || Number(value.pageCount) < 1 || !Number.isInteger(value.processedPages) || Number(value.processedPages) < 1 || Number(value.processedPages) > ATTACHMENT_TEXT_MAX_PAGES || Number(value.processedPages) > Number(value.pageCount)) throw new Error('The local text extractor returned invalid page counts.')
  if (value.confidence !== null && (typeof value.confidence !== 'number' || !Number.isFinite(value.confidence) || value.confidence < 0 || value.confidence > 1)) throw new Error('The local text extractor returned an invalid confidence score.')
  if (!Array.isArray(value.warnings) || !value.warnings.every((item) => typeof item === 'string')) throw new Error('The local text extractor returned invalid review warnings.')
  const warnings = [...new Set([REVIEW_WARNING, ...value.warnings.map((item) => item.slice(0, 500))])].slice(0, 20)
  return { ok: true, text: value.text, confidence: value.confidence as number | null, warnings, pageCount: Number(value.pageCount), processedPages: Number(value.processedPages), truncated: value.truncated === true }
}

export interface AttachmentTextServiceOptions {
  helperPath: () => string
  platform?: NodeJS.Platform
  darwinRelease?: string
  timeoutMs?: number
  spawn?: (file: string, args: string[], options: { env: NodeJS.ProcessEnv; stdio: ['pipe', 'pipe', 'pipe'] }) => ChildProcessWithoutNullStreams
}

export function createAttachmentTextService(options: AttachmentTextServiceOptions) {
  let capabilityPromise: Promise<AttachmentTextCapability> | null = null
  const children = new Set<ChildProcessWithoutNullStreams>()
  let closed = false
  let active = 0
  const capabilityBase = { method: 'apple-vision' as const, maxBytes: ATTACHMENT_TEXT_MAX_BYTES, maxPages: ATTACHMENT_TEXT_MAX_PAGES }
  const failure = (reason: string): AttachmentTextCapability => ({ ...capabilityBase, available: false, reason })
  const launch = options.spawn ?? ((file, args, config) => spawn(file, args, config))

  async function execute(args: string[], input?: Buffer, timeoutMs = options.timeoutMs ?? ATTACHMENT_TEXT_TIMEOUT_MS): Promise<unknown> {
    if (closed) throw new Error('Text extraction is shutting down. Reopen the app and try again.')
    return await new Promise((resolve, reject) => {
      // No shell, arbitrary environment, user-provided paths, temporary content files or cloud requests.
      const child = launch(options.helperPath(), args, { env: { PATH: '/usr/bin:/bin:/usr/sbin:/sbin', LANG: 'en_US.UTF-8', TMPDIR: os.tmpdir() }, stdio: ['pipe', 'pipe', 'pipe'] })
      children.add(child)
      const chunks: Buffer[] = []
      let length = 0
      let settled = false
      const finish = (error?: Error, result?: unknown) => {
        if (settled) return
        settled = true
        clearTimeout(timer)
        children.delete(child)
        if (error) { child.kill('SIGKILL'); reject(error) } else resolve(result)
      }
      const timer = setTimeout(() => finish(new Error('Text extraction timed out. Try a clearer image or a smaller PDF.')), timeoutMs)
      timer.unref?.()
      child.stdout.on('data', (chunk: Buffer) => {
        length += chunk.length
        if (length > MAX_OUTPUT_BYTES) { finish(new Error('Text extraction exceeded its output safety limit. Choose fewer pages.')); return }
        chunks.push(chunk)
      })
      // Drain stderr without retaining potentially sensitive framework diagnostics or paths.
      child.stderr.on('data', () => {})
      child.on('error', () => finish(new Error('The local text extractor could not start. Reinstall the app or rebuild its native helper.')))
      child.stdin.on('error', () => {})
      child.on('close', () => {
        if (settled) return
        try { finish(undefined, JSON.parse(Buffer.concat(chunks).toString('utf8').trim())) }
        catch { finish(new Error('The local text extractor returned no usable result. Reinstall the app or rebuild its native helper.')) }
      })
      child.stdin.end(input)
    })
  }

  async function getCapability(): Promise<AttachmentTextCapability> {
    if (closed) return failure('Text extraction is shutting down.')
    if ((options.platform ?? process.platform) !== 'darwin') return failure('Local text extraction requires the macOS app with Apple Vision. A text-only transcript can be attached on other platforms.')
    if (Number.parseInt(options.darwinRelease ?? os.release(), 10) < 22) return failure('Local text extraction requires macOS 13 or later; this app supports macOS 13.3 and later.')
    capabilityPromise ??= (async () => {
      try {
        const info = await lstat(options.helperPath())
        if (!info.isFile() || info.isSymbolicLink() || !(info.mode & 0o111)) return failure('The native text-extraction helper is missing. Reinstall the app, or run npm run computer:build in a development checkout.')
        const raw = await execute(['--attachment-text-capability'], undefined, Math.min(options.timeoutMs ?? 5_000, 5_000)) as Record<string, unknown>
        if (raw.available !== true || raw.method !== 'apple-vision' || raw.maxBytes !== ATTACHMENT_TEXT_MAX_BYTES || raw.maxPages !== ATTACHMENT_TEXT_MAX_PAGES) return failure('The installed native helper does not support text extraction. Reinstall the app or rebuild its native helper.')
        return { ...capabilityBase, available: true }
      } catch { return failure('The native text-extraction helper is unavailable. Reinstall the app or rebuild its native helper.') }
    })()
    return capabilityPromise
  }

  async function extract(input: AttachmentTextRequest | unknown): Promise<AttachmentTextResult> {
    try {
      const request = validateAttachmentTextRequest(input)
      const capability = await getCapability()
      if (!capability.available) return { ok: false, error: capability.reason }
      if (closed) return { ok: false, error: 'Text extraction is shutting down.' }
      if (active >= 2) return { ok: false, error: 'Two files are being read already. Wait for one to finish, then try again.' }
      active++
      try { return normalizeAttachmentTextResult(await execute(['--extract-attachment-text', '--mime', request.mime], request.bytes)) }
      finally { active-- }
    } catch (error) { return { ok: false, error: error instanceof Error ? error.message : 'Text extraction failed. Select the file again.' } }
  }

  function shutdown(): void {
    closed = true
    for (const child of children) child.kill('SIGKILL')
    children.clear()
  }

  return { getCapability, extract, shutdown }
}
