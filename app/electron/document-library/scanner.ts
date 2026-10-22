import { createHash } from 'node:crypto'
import { promises as fs, type Dirent } from 'node:fs'
import path from 'node:path'
import type { DocumentConnectorSettings, DocumentReadResult, DocumentSourceConfig } from '../agent/types'
import type { DocumentFileOutcome, DocumentScanSummary } from '../../shared/document-library'
import { excerpt, isInside, looksLikeSupportedDocument, normalizeExt, readFileBytesBounded, readSupportedDocument, stableDocumentId, titleFromContent } from '../agent/document-parser'
import { documentSections } from './locations'
import { blockedReadReason } from '../agent/command-policy'
import { extractWorkbook } from './xlsx'

export const DOCUMENT_LIBRARY_BOUNDS = { maxFiles: 1_500, maxCharacters: 32_000_000, maxPassages: 50_000 }
const MAX_DEPTH = 8
const PRIVATE_NAMES = /^(?:\.env(?:\..*)?|\.npmrc|\.netrc|id_rsa|id_ed25519)$/i
const EXCLUDED_FOLDERS = new Set(['.git', 'node_modules', '.vontra', '.tensorfold'])
export interface DocumentScanSnapshot { documents: DocumentReadResult[]; sources: DocumentSourceConfig[]; scannedAt: number }

export async function scanDocumentSnapshot(settings: DocumentConnectorSettings): Promise<DocumentScanSnapshot> {
  const now = Date.now()
  const sources = settings.sources.map(source => ({ ...source }))
  const documents: DocumentReadResult[] = []
  const maxBytes = documentByteLimit(settings)
  const maxDocuments = Math.max(1, Math.min(settings.maxDocuments || 500, 5_000))
  let characters = 0
  let passages = 0
  let seenFiles = 0
  let seenEntries = 0
  for (const source of sources) {
    const outcomes: DocumentFileOutcome[] = []
    let truncated = false
    if (!source.enabled) {
      source.lastScan = summary([], now, false, 'disabled')
      continue
    }
    if (seenFiles >= DOCUMENT_LIBRARY_BOUNDS.maxFiles || seenEntries >= DOCUMENT_LIBRARY_BOUNDS.maxFiles * 2) {
      source.lastScan = summary([], now, true, 'error')
      source.lastFileOutcomes = []
      source.lastError = 'Scan file or traversal limit reached; this source was not examined.'
      source.updatedAt = now
      continue
    }
    try {
      const inventory = await collectCandidates(source, now, DOCUMENT_LIBRARY_BOUNDS.maxFiles - seenFiles, DOCUMENT_LIBRARY_BOUNDS.maxFiles * 2 - seenEntries)
      seenEntries += inventory.visited
      outcomes.push(...inventory.outcomes)
      seenFiles += inventory.outcomes.length + inventory.files.length
      truncated = inventory.truncated
      for (const file of inventory.files) {
        const relativePath = relativeDocumentPath(source, file)
        try {
          if (documents.length >= maxDocuments) { truncated = true; outcomes.push(outcome(source, relativePath, 'skipped', now, 'Configured document count limit reached; increase the limit and retry.')); continue }
          const item = await readDocumentFile(source, file, maxBytes)
          if (!item) { outcomes.push(outcome(source, relativePath, 'failed', now, 'No readable text was extracted.')); continue }
          if (characters + item.content.length > DOCUMENT_LIBRARY_BOUNDS.maxCharacters || passages + (item.sections?.length ?? 0) > DOCUMENT_LIBRARY_BOUNDS.maxPassages) {
            truncated = true
            outcomes.push(outcome(source, relativePath, 'skipped', now, 'Managed library content limit reached; narrow this collection and retry.'))
            continue
          }
          documents.push(item)
          characters += item.content.length
          passages += item.sections?.length ?? 0
          outcomes.push({ ...outcome(source, relativePath, 'ready', now), revision: item.revision })
        } catch (error) { outcomes.push(outcome(source, relativePath, 'failed', now, message(error))) }
      }
      source.lastScan = summary(outcomes, now, truncated)
      source.lastSyncAt = now
      source.lastError = source.lastScan.status === 'partial' || source.lastScan.status === 'error'
        ? `${source.lastScan.ready} ready; ${source.lastScan.failed} failed; ${source.lastScan.unsupported} unsupported; ${source.lastScan.skipped} skipped.${truncated ? ' Scan limits reached; some paths were not examined.' : ''}`
        : null
    } catch (error) {
      seenFiles++; seenEntries++
      outcomes.push(outcome(source, path.basename(source.path), 'failed', now, message(error)))
      source.lastScan = summary(outcomes, now, false, 'error')
      source.lastError = message(error)
    }
    source.lastFileOutcomes = outcomes
    source.updatedAt = now
  }
  return { documents, sources, scannedAt: now }
}

export function documentByteLimit(settings: DocumentConnectorSettings): number { return Math.max(8_192, Math.min(settings.maxDocumentBytes || 1_000_000, 8_000_000)) }
export function relativeDocumentPath(source: DocumentSourceConfig, file: string): string {
  return path.relative(source.kind === 'file' ? path.dirname(source.path) : source.path, file) || path.basename(file)
}

export async function collectDocumentFiles(root: string, kind: DocumentSourceConfig['kind']): Promise<string[]> {
  const source = { id: '', path: root, kind } as DocumentSourceConfig
  return (await collectCandidates(source, Date.now(), DOCUMENT_LIBRARY_BOUNDS.maxFiles)).files
}

export async function collectCandidates(source: DocumentSourceConfig, now: number, limit: number, workLimit = DOCUMENT_LIBRARY_BOUNDS.maxFiles * 2): Promise<{ files: string[]; outcomes: DocumentFileOutcome[]; truncated: boolean; visited: number }> {
  const root = path.resolve(source.path)
  const refusal = blockedReadReason(root) || (root.split(path.sep).some(part => EXCLUDED_FOLDERS.has(part)) ? 'Private or generated source roots are excluded.' : null)
  if (refusal) throw new Error(refusal)
  const rootStat = await fs.lstat(root)
  if (rootStat.isSymbolicLink()) throw new Error('Symbolic-link document roots are not allowed; choose the original folder or file.')
  const files: string[] = []
  const outcomes: DocumentFileOutcome[] = []
  let truncated = false
  let visited = 0
  const available = () => files.length + outcomes.length < limit
  const add = (file: string, status: DocumentFileOutcome['status'], error: string) => outcomes.push(outcome(source, relativeDocumentPath(source, file), status, now, error))
  const accept = (file: string) => {
    if (!available()) { truncated = true; return }
    const refusal = blockedReadReason(file)
    if (refusal) add(file, 'skipped', refusal)
    else if (PRIVATE_NAMES.test(path.basename(file))) add(file, 'skipped', 'Secret-file names are excluded from document ingestion.')
    else if (!looksLikeSupportedDocument(file)) add(file, 'unsupported', 'This file format is not supported by local document extraction.')
    else files.push(file)
  }
  if (source.kind === 'file' || rootStat.isFile()) { visited = 1; accept(root); return { files, outcomes, truncated, visited } }
  if (!rootStat.isDirectory()) throw new Error('Document source is not a folder.')
  const rootReal = await fs.realpath(root)
  const walk = async (dir: string, depth: number): Promise<void> => {
    if (!available()) { truncated = true; return }
    if (depth > MAX_DEPTH) { truncated = true; add(dir, 'skipped', 'Folder depth limit reached; this subtree was not examined.'); return }
    let directory
    try { directory = await fs.opendir(dir) }
    catch (error) { add(dir, 'failed', `Folder could not be read: ${message(error)}`); return }
    const entries: Dirent[] = []
    for await (const entry of directory) {
      if (visited + entries.length >= workLimit) { truncated = true; break }
      entries.push(entry)
    }
    // Name order, so every filesystem walks a folder in the same sequence.
    entries.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0))
    for (const entry of entries) {
      visited++
      if (!available() || visited > workLimit) { truncated = true; break }
      const full = path.join(dir, entry.name)
      if (entry.isSymbolicLink()) { add(full, 'skipped', 'Symbolic links are excluded from document ingestion.'); continue }
      if (entry.isDirectory()) {
        if (EXCLUDED_FOLDERS.has(entry.name) || (entry.name.startsWith('.') && entry.name !== '.github')) { add(full, 'skipped', 'Private or generated folder excluded; subtree was not examined.'); continue }
        await walk(full, depth + 1)
      } else if (entry.isFile()) {
        const real = await fs.realpath(full).catch(() => null)
        if (!real || !isInside(rootReal, real)) add(full, 'skipped', 'File is unavailable or outside its configured source.')
        else accept(full)
      } else add(full, 'skipped', 'Only regular local files are supported.')
    }
  }
  await walk(root, 0)
  return { files, outcomes, truncated, visited }
}

/** Called again for evidence, so a changed path or symlink never trusts cached text. */
export async function assertDocumentPath(source: DocumentSourceConfig, file: string): Promise<void> {
  if (!source.enabled) throw new Error('Document source is disabled.')
  const root = path.resolve(source.path)
  const target = path.resolve(file)
  const refusal = blockedReadReason(target) || (root.split(path.sep).some(part => EXCLUDED_FOLDERS.has(part)) ? 'Private or generated source roots are excluded.' : null)
  if (refusal) throw new Error(refusal)
  const rootStat = await fs.lstat(root)
  if (rootStat.isSymbolicLink()) throw new Error('Symbolic-link document roots are not allowed.')
  if (source.kind === 'file' || rootStat.isFile()) {
    if (target !== root) throw new Error('Document path is outside its configured source.')
  } else {
    if (!isInside(root, target)) throw new Error('Document path is outside its configured source.')
    const pieces = path.relative(root, target).split(path.sep)
    let candidate = root
    for (const part of pieces) {
      if (!part || part === '..' || PRIVATE_NAMES.test(part) || EXCLUDED_FOLDERS.has(part) || (part.startsWith('.') && part !== '.github')) throw new Error('Private document path is excluded.')
      candidate = path.join(candidate, part)
      if ((await fs.lstat(candidate)).isSymbolicLink()) throw new Error('Symbolic-link document paths are excluded.')
    }
    if (!isInside(await fs.realpath(root), await fs.realpath(target))) throw new Error('Document path is outside its configured source.')
  }
  if (PRIVATE_NAMES.test(path.basename(target)) || !looksLikeSupportedDocument(target)) throw new Error('Document file is excluded or unsupported.')
}

export async function readDocumentFile(source: DocumentSourceConfig, file: string, maxBytes: number): Promise<DocumentReadResult | null> {
  await assertDocumentPath(source, file)
  const bytes = await readFileBytesBounded(file, maxBytes)
  const revision = createHash('sha256').update(bytes).digest('hex')
  const extension = normalizeExt(path.extname(file))
  const workbook = extension === '.xlsx' ? extractWorkbook(bytes, maxBytes) : null
  const raw = workbook?.content ?? await readSupportedDocument(file, maxBytes, bytes)
  if (!raw?.trim()) throw new Error('No readable text was extracted; this document may need an unsupported extraction method.')
  if (raw.includes('\u0000')) throw new Error('Binary content is not supported as a text document.')
  if (Buffer.byteLength(raw, 'utf8') > maxBytes) throw new Error('Extracted document exceeds its configured byte limit.')
  const content = workbook ? raw : raw.replace(/\r\n/g, '\n')
  const relativePath = relativeDocumentPath(source, file)
  const stat = await fs.stat(file)
  return {
    id: stableDocumentId(source.id, relativePath), sourceId: source.id, sourceLabel: source.label, path: file, relativePath,
    title: titleFromContent(content) || path.basename(file), excerpt: excerpt(content), size: bytes.length, modifiedAt: stat.mtimeMs,
    extension, content, revision, sections: workbook?.sections ?? documentSections(content, extension),
    locationNote: extension === '.pdf' ? 'Extracted PDF text only; page mapping and OCR are unavailable.' : undefined,
  }
}
function outcome(source: DocumentSourceConfig, relativePath: string, status: DocumentFileOutcome['status'], now: number, error?: string): DocumentFileOutcome {
  return { id: stableDocumentId(source.id, relativePath), sourceId: source.id, relativePath, status, error, retryable: status === 'failed' || status === 'skipped', scannedAt: now }
}
function summary(files: DocumentFileOutcome[], scannedAt: number, truncated: boolean, forced?: DocumentScanSummary['status']): DocumentScanSummary {
  const count = (status: DocumentFileOutcome['status']) => files.filter(file => file.status === status).length
  const ready = count('ready'), failed = count('failed'), unsupported = count('unsupported'), skipped = count('skipped')
  return { ready, failed, unsupported, skipped, truncated, scannedAt, status: forced ?? (failed || unsupported || skipped || truncated ? ready ? 'partial' : 'error' : ready ? 'ready' : 'empty') }
}
function message(error: unknown): string { return String((error as Error)?.message ?? error).slice(0, 300) }
