import { createHash } from 'node:crypto'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import type { EmailReadResult, EmailSourceConfig } from './types'

export const MAX_FILE_BYTES = 2_000_000

export const MAX_SCAN_FILES = 1_500

export const EMAIL_EXTENSIONS = new Set(['.eml', '.emlx'])

export function normalizeMessageHeaderId(value: string | undefined): string {
  return String(value || '').trim().replace(/^<|>$/g, '').toLowerCase()
}

export async function collectEmailFiles(root: string, kind: EmailSourceConfig['kind']): Promise<string[]> {
  const stat = await fs.stat(root)
  if (!stat.isDirectory()) throw new Error('Email source is not a folder.')
  const files: string[] = []
  const roots = kind === 'maildir'
    ? [path.join(root, 'new'), path.join(root, 'cur'), root]
    : [root]

  const seen = new Set<string>()
  const walk = async (dir: string, depth: number) => {
    if (depth > 8 || files.length >= MAX_SCAN_FILES) return
    let entries: any[] = []
    try { entries = await fs.readdir(dir, { withFileTypes: true }) } catch { return }
    for (const entry of entries.sort((a, b) => a.name.localeCompare(b.name))) {
      if (files.length >= MAX_SCAN_FILES) break
      const full = path.join(dir, entry.name)
      if (seen.has(full)) continue
      seen.add(full)
      if (entry.isDirectory()) {
        if (entry.name === 'tmp' || entry.name === '.git' || entry.name === 'node_modules') continue
        await walk(full, depth + 1)
        continue
      }
      if (!entry.isFile()) continue
      const ext = path.extname(entry.name).toLowerCase()
      const inMaildirBucket = /[/\\](cur|new)[/\\]/.test(full)
      if (EMAIL_EXTENSIONS.has(ext) || inMaildirBucket) files.push(full)
    }
  }

  for (const rootDir of roots) await walk(rootDir, 0)
  return [...new Set(files)]
}

export async function readEmailFile(source: EmailSourceConfig, file: string, includeBody = false): Promise<EmailReadResult | null> {
  const stat = await fs.stat(file)
  if (!stat.isFile() || stat.size > MAX_FILE_BYTES) return null
  let raw = await fs.readFile(file, 'utf8')
  raw = normalizeRawEmail(raw, file)
  const parsed = parseEmail(raw)
  const messageId = parsed.headers['message-id'] || `${file}:${stat.mtimeMs}:${stat.size}`
  const id = createHash('sha1').update(`${source.id}:${messageId}:${file}`).digest('hex').slice(0, 20)
  const date = parsed.headers.date ? Date.parse(parsed.headers.date) : stat.mtimeMs
  const message: EmailReadResult = {
    id,
    sourceId: source.id,
    sourceLabel: source.label,
    path: file,
    subject: decodeHeader(parsed.headers.subject || ''),
    from: decodeHeader(parsed.headers.from || ''),
    to: decodeHeader(parsed.headers.to || ''),
    date: Number.isFinite(date) ? date : stat.mtimeMs,
    unread: isUnread(file),
    snippet: snippet(parsed.body, 280),
    size: stat.size,

    messageId: normalizeMessageHeaderId(parsed.headers['message-id']) || null,
    inReplyTo: normalizeMessageHeaderId(parsed.headers['in-reply-to']) || null,
    headers: includeBody ? parsed.headers : {},
    body: includeBody ? parsed.body.slice(0, 20_000) : '',
  }
  return message
}

export function parseEmail(raw: string): { headers: Record<string, string>; body: string } {
  const normalized = raw.replace(/\r\n/g, '\n')
  const split = normalized.search(/\n\s*\n/)
  const headerText = split >= 0 ? normalized.slice(0, split) : ''
  const bodyText = split >= 0 ? normalized.slice(split).trim() : normalized
  const headers = parseHeaders(headerText)
  return { headers, body: extractReadableBody(headers, bodyText) }
}

export function parseHeaders(headerText: string): Record<string, string> {
  const headers: Record<string, string> = {}
  const unfolded = headerText.replace(/\n[ \t]+/g, ' ')
  for (const line of unfolded.split('\n')) {
    const idx = line.indexOf(':')
    if (idx <= 0) continue
    const key = line.slice(0, idx).trim().toLowerCase()
    const value = line.slice(idx + 1).trim()
    headers[key] = headers[key] ? `${headers[key]}, ${value}` : value
  }
  return headers
}

export

  interface MimeLeaf { contentType: string; encoding: string; body: string }

export

  function selectTextLeaf(contentType: string, bodyText: string): MimeLeaf | null {
  const boundary = contentType.match(/boundary="?([^";]+)"?/i)?.[1]
  if (!boundary) return null
  let plain: MimeLeaf | null = null
  let html: MimeLeaf | null = null

  for (const segment of bodyText.split(`--${boundary}`)) {
    const part = segment.replace(/^--/, '') // tolerate the closing delimiter's trailing --
    const split = part.search(/\n\s*\n/)
    if (split < 0) continue
    const partHeaders = parseHeaders(part.slice(0, split))
    const partType = partHeaders['content-type'] || ''
    if (!partType) continue
    if (/^multipart\//i.test(partType)) {
      const nested = selectTextLeaf(partType, part.slice(split))
      if (nested) {
        if (/text\/plain/i.test(nested.contentType) && !plain) plain = nested
        else if (/text\/html/i.test(nested.contentType) && !html) html = nested
      }
      continue
    }
    const leaf: MimeLeaf = { contentType: partType, encoding: partHeaders['content-transfer-encoding'] || '', body: part.slice(split).trim() }
    if (/text\/plain/i.test(partType) && !plain) plain = leaf
    else if (/text\/html/i.test(partType) && !html) html = leaf
  }
  return plain ?? html
}

export function extractReadableBody(headers: Record<string, string>, bodyText: string): string {
  const contentType = headers['content-type'] || ''
  let body = bodyText
  let partContentType = contentType

  let partEncoding: string | undefined
  const leaf = selectTextLeaf(contentType, bodyText)
  if (leaf) {
    body = leaf.body
    partEncoding = leaf.encoding
    partContentType = leaf.contentType
  }
  body = decodeTransfer(body, partEncoding || headers['content-transfer-encoding'] || '', charsetOf(partContentType))
  if (/text\/html/i.test(partContentType) || /<html|<body|<p[>\s]/i.test(body)) body = stripHtml(body)
  return body.replace(/\n{3,}/g, '\n\n').trim()
}

export

  function charsetOf(contentType: string): string {
  return contentType.match(/charset="?([^";]+)"?/i)?.[1]?.trim() || 'utf-8'
}

export

  function decodeBytes(bytes: Buffer, charset: string): string {
  const cs = (charset || 'utf-8').toLowerCase().trim()
  if (cs === 'utf-8' || cs === 'utf8' || cs === 'us-ascii' || cs === 'ascii') return bytes.toString('utf8')
  try {
    return new TextDecoder(cs).decode(bytes)
  } catch {
    try { return bytes.toString('utf8') } catch { return bytes.toString('latin1') }
  }
}

export function decodeTransfer(value: string, encoding: string, charset = 'utf-8'): string {
  if (/base64/i.test(encoding)) {
    try { return decodeBytes(Buffer.from(value.replace(/\s+/g, ''), 'base64'), charset) } catch { return value }
  }
  if (/quoted-printable/i.test(encoding) || /=\r?\n|=[0-9A-F]{2}/i.test(value)) return decodeQuotedPrintable(value, charset)
  return value
}

export function decodeQuotedPrintable(value: string, charset = 'utf-8'): string {
  const text = value.replace(/=\r?\n/g, '')
  const bytes: number[] = []
  for (let i = 0; i < text.length; i++) {
    const ch = text[i]
    if (ch === '=' && /^[0-9A-Fa-f]{2}$/.test(text.slice(i + 1, i + 3))) {
      bytes.push(parseInt(text.slice(i + 1, i + 3), 16))
      i += 2
    } else {

      for (const b of Buffer.from(ch, 'utf8')) bytes.push(b)
    }
  }
  return decodeBytes(Buffer.from(bytes), charset)
}

export function decodeHeader(value: string): string {
  return value.replace(/=\?([^?]+)\?([BQbq])\?([^?]+)\?=/g, (_m, charset, enc, data) => {
    try {
      if (/b/i.test(enc)) return decodeBytes(Buffer.from(data, 'base64'), charset)
      return decodeQuotedPrintable(data.replace(/_/g, ' '), charset)
    } catch {
      return data
    }
  }).replace(/\s+/g, ' ').trim()
}

export function stripHtml(value: string): string {
  return value
    .replace(/<style[\s\S]*?<\/style>/gi, ' ')
    .replace(/<script[\s\S]*?<\/script>/gi, ' ')
    .replace(/<br\s*\/?>/gi, '\n')
    .replace(/<\/p>/gi, '\n')
    .replace(/<[^>]+>/g, ' ')
    .replace(/&nbsp;/g, ' ')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
}

export function normalizeRawEmail(raw: string, file: string): string {
  if (path.extname(file).toLowerCase() !== '.emlx') return raw
  const firstLine = raw.indexOf('\n')
  if (firstLine > 0 && /^\d+$/.test(raw.slice(0, firstLine).trim())) raw = raw.slice(firstLine + 1)
  const plist = raw.indexOf('<?xml')
  return plist > 0 ? raw.slice(0, plist) : raw
}

export function snippet(value: string, max: number): string {
  return value.replace(/\s+/g, ' ').trim().slice(0, max)
}

export function looksLikeMaildirPath(dir: string): boolean {
  return /[/\\](cur|new)$/.test(dir) || /Maildir/i.test(dir)
}

export function isUnread(file: string): boolean {
  if (/[/\\]new[/\\]/.test(file)) return true
  const flags = file.match(/:2,([A-Z]*)$/)?.[1]
  return flags ? !flags.includes('S') : false
}
