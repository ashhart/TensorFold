// First-class artifacts (VST-011): documents, PDFs, spreadsheets, presentations, images,
// generated files, reports, screenshots, logs, and run exports become typed records with
// provenance (which thread/run/turn/tool produced them), a redaction status, and a bounded
// preview — instead of loose strings derived from the timeline. Pure: file I/O stays with
// the caller; inline content is bounded so a record can never balloon the store.

import { aggregate, redactText } from './export-redaction'
import type { Artifact, ArtifactKind, ArtifactPreview, ArtifactProvenance, WorkspaceLocation } from './types'

export const MAX_INLINE_ARTIFACT_CHARS = 64_000
export const ARTIFACT_PREVIEW_CHARS = 1_200

export type ArtifactPathAccessKind = 'local' | 'ssh' | 'unknown'

function normalizeArtifactWorkspace(raw: unknown): WorkspaceLocation | undefined {
  if (!raw || typeof raw !== 'object') return undefined
  const workspace = raw as Partial<WorkspaceLocation>
  if ((workspace.kind !== 'local' && workspace.kind !== 'ssh') || typeof workspace.path !== 'string' || !workspace.path.trim()) return undefined
  if (workspace.kind === 'ssh' && (typeof workspace.hostId !== 'string' || !workspace.hostId.trim())) return undefined
  return {
    kind: workspace.kind,
    path: workspace.path,
    hostId: workspace.kind === 'ssh' ? workspace.hostId!.trim() : null,
  }
}

/**
 * Decide which filesystem may satisfy a path-backed artifact. New records carry an
 * immutable workspace snapshot. Legacy records cannot safely borrow their source
 * thread's current workspace because that ownership is mutable; treating an old path
 * as local could expose unrelated Mac data for an artifact registered over SSH.
 */
export function artifactPathAccessKind(
  artifact: Pick<Artifact, 'provenance'>,
): ArtifactPathAccessKind {
  const captured = normalizeArtifactWorkspace(artifact.provenance.workspace)
  if (captured) return captured.kind
  // A legacy thread's *current* workspace is not provenance: threads can be
  // reassigned after registration. Never use mutable ownership to guess that an
  // old absolute path is local.
  return 'unknown'
}

const KIND_BY_EXTENSION: Record<string, ArtifactKind> = {
  '.pdf': 'pdf',
  '.doc': 'document', '.docx': 'document', '.md': 'document', '.txt': 'document', '.rtf': 'document',
  '.xls': 'spreadsheet', '.xlsx': 'spreadsheet', '.csv': 'spreadsheet', '.tsv': 'spreadsheet',
  '.ppt': 'presentation', '.pptx': 'presentation', '.key': 'presentation',
  '.png': 'image', '.jpg': 'image', '.jpeg': 'image', '.gif': 'image', '.webp': 'image', '.svg': 'image',
  '.log': 'log',
  '.json': 'file', '.yaml': 'file', '.yml': 'file', '.html': 'file',
}

export function inferArtifactKind(input: { kind?: string | null; path?: string | null; mimeType?: string | null }): ArtifactKind {
  const explicit = String(input.kind ?? '').trim() as ArtifactKind
  if (['document', 'pdf', 'spreadsheet', 'presentation', 'image', 'file', 'report', 'screenshot', 'log', 'export'].includes(explicit)) return explicit
  const mime = String(input.mimeType ?? '').toLowerCase()
  if (mime.startsWith('image/')) return 'image'
  if (mime === 'application/pdf') return 'pdf'
  if (mime.includes('spreadsheet') || mime === 'text/csv') return 'spreadsheet'
  if (mime.includes('presentation')) return 'presentation'
  const path = String(input.path ?? '').toLowerCase()
  const dot = path.lastIndexOf('.')
  if (dot >= 0) {
    const kind = KIND_BY_EXTENSION[path.slice(dot)]
    if (kind) return kind
  }
  return 'file'
}

export interface CreateArtifactInput {
  id?: string
  title: string
  kind?: ArtifactKind | string | null
  // Exactly one source: a file path (content stays on disk) or bounded inline text.
  path?: string | null
  inlineText?: string | null
  mimeType?: string | null
  sizeBytes?: number | null
  provenance: ArtifactProvenance
}

export function createArtifactRecord(input: CreateArtifactInput, now = Date.now(), id = `art_${Math.random().toString(36).slice(2, 9)}`): Artifact {
  const title = String(input.title ?? '').replace(/\s+/g, ' ').trim()
  if (!title) throw new Error('artifact requires a title')
  const path = input.path?.trim() || null
  const inline = typeof input.inlineText === 'string' && input.inlineText.length ? input.inlineText : null
  if (!path && !inline) throw new Error('artifact requires a file path or inline content')
  if (path && inline) throw new Error('artifact takes a path OR inline content, not both')
  if (!input.provenance?.threadId?.trim()) throw new Error('artifact requires provenance (threadId)')
  return {
    id: input.id ?? id,
    title: title.slice(0, 160),
    kind: inferArtifactKind({ kind: input.kind, path, mimeType: input.mimeType }),
    path,
    inlineText: inline ? inline.slice(0, MAX_INLINE_ARTIFACT_CHARS) : null,
    inlineTruncated: !!inline && inline.length > MAX_INLINE_ARTIFACT_CHARS,
    mimeType: input.mimeType?.trim() || null,
    sizeBytes: typeof input.sizeBytes === 'number' && input.sizeBytes >= 0 ? Math.floor(input.sizeBytes) : inline ? inline.length : null,
    provenance: {
      threadId: input.provenance.threadId,
      runId: input.provenance.runId ?? null,
      turn: typeof input.provenance.turn === 'number' && input.provenance.turn >= 0 ? input.provenance.turn : null,
      tool: input.provenance.tool ?? null,
      ...(normalizeArtifactWorkspace(input.provenance.workspace)
        ? { workspace: normalizeArtifactWorkspace(input.provenance.workspace) }
        : {}),
      sourceRef: input.provenance.sourceRef ?? null,
    },
    redaction: 'unredacted',
    redactionReport: [],
    createdAt: now,
    updatedAt: now,
  }
}

// Load defensiveness: a corrupt record degrades to null rather than poisoning the list.
export function normalizeArtifactRecord(raw: unknown): Artifact | null {
  const record = raw as Partial<Artifact> | null
  if (!record || typeof record !== 'object') return null
  if (typeof record.id !== 'string' || !record.id) return null
  if (typeof record.title !== 'string' || !record.title) return null
  if (!record.provenance || typeof record.provenance.threadId !== 'string' || !record.provenance.threadId) return null
  if (!record.path && !record.inlineText) return null
  return {
    id: record.id,
    title: record.title,
    kind: inferArtifactKind({ kind: record.kind, path: record.path ?? null, mimeType: record.mimeType ?? null }),
    path: record.path ?? null,
    inlineText: record.inlineText ?? null,
    inlineTruncated: record.inlineTruncated === true,
    mimeType: record.mimeType ?? null,
    sizeBytes: typeof record.sizeBytes === 'number' ? record.sizeBytes : null,
    provenance: {
      threadId: record.provenance.threadId,
      runId: record.provenance.runId ?? null,
      turn: typeof record.provenance.turn === 'number' ? record.provenance.turn : null,
      tool: record.provenance.tool ?? null,
      ...(normalizeArtifactWorkspace(record.provenance.workspace)
        ? { workspace: normalizeArtifactWorkspace(record.provenance.workspace) }
        : {}),
      sourceRef: record.provenance.sourceRef ?? null,
    },
    redaction: record.redaction === 'redacted' ? 'redacted' : record.redaction === 'not_required' ? 'not_required' : 'unredacted',
    redactionReport: Array.isArray(record.redactionReport) ? record.redactionReport : [],
    createdAt: typeof record.createdAt === 'number' ? record.createdAt : 0,
    updatedAt: typeof record.updatedAt === 'number' ? record.updatedAt : 0,
  }
}

const TEXTUAL_KINDS = new Set<ArtifactKind>(['document', 'report', 'log', 'file', 'spreadsheet', 'export'])

/**
 * Bounded preview. `content` is the artifact's text when the caller has it (inline records
 * carry their own); binary kinds (image/pdf/presentation/screenshot) never inline bytes —
 * their preview is the metadata line.
 */
export function artifactPreview(artifact: Artifact, content?: string | null): ArtifactPreview {
  const size = artifact.sizeBytes != null ? `${formatBytes(artifact.sizeBytes)}` : 'size unknown'
  const source = artifact.path ? artifact.path : 'inline'
  const meta = [artifact.kind, size, source, artifact.redaction === 'redacted' ? 'redacted' : null].filter(Boolean).join(' · ')
  const text = content ?? artifact.inlineText
  if (!TEXTUAL_KINDS.has(artifact.kind) || text == null) {
    return { kind: artifact.kind, title: artifact.title, excerpt: null, truncated: false, meta }
  }
  const excerpt = text.slice(0, ARTIFACT_PREVIEW_CHARS)
  return { kind: artifact.kind, title: artifact.title, excerpt, truncated: text.length > excerpt.length, meta }
}

/**
 * Redact an artifact's textual content in place (VST-011): secrets/emails/tokens are
 * masked via the shared export redaction rules, the record carries the per-kind report,
 * and the status flips so a receipt can show what shipped clean. Binary artifacts can't be
 * text-redacted — they flip to 'not_required' with an empty report (visible, not implied).
 */
export function redactArtifactRecord(artifact: Artifact, content?: string | null, now = Date.now()): { artifact: Artifact; content: string | null } {
  const text = content ?? artifact.inlineText
  if (!TEXTUAL_KINDS.has(artifact.kind) || text == null) {
    return { artifact: { ...artifact, redaction: 'not_required', redactionReport: [], updatedAt: now }, content: null }
  }
  const result = redactText(text)
  return {
    artifact: {
      ...artifact,
      redaction: 'redacted',
      redactionReport: aggregate(result.hits),
      ...(artifact.inlineText != null ? { inlineText: result.text.slice(0, MAX_INLINE_ARTIFACT_CHARS) } : {}),
      updatedAt: now,
    },
    content: result.text,
  }
}

export function artifactsForThread(artifacts: Artifact[], threadId: string): Artifact[] {
  return artifacts.filter((artifact) => artifact.provenance.threadId === threadId)
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}
