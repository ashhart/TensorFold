import type { Thread } from './types'

export const COMPOSER_INSERT_EVENT = 'vontra:composer-insert'

export type BrowserPreviewCandidate = {
  url: string
  label: string
  source: 'thread' | 'default'
}

export type BrowserInspection = {
  domReadable: boolean
  title?: string
  activeElement?: string
  interactiveCount?: number
  diagnostics: string[]
}

const LOCAL_URL_RE = /\b(?:https?:\/\/)?(?:localhost|127(?:\.\d{1,3}){3}|\[::1\]|::1)(?::\d{2,5})?(?:\/[^\s"'<>)]*)?/gi
export const ALLOWED_PREVIEW_PORTS = [5173, 3000, 4173] as const

function basename(path: string): string {
  return path.split(/[\\/]/).filter(Boolean).pop() || path
}

export function normalizeLocalPreviewUrl(raw: string): string | null {
  let value = String(raw || '').trim().replace(/[),.;]+$/g, '')
  if (!value) return null
  if (!/^https?:\/\//i.test(value)) value = `http://${value}`
  try {
    const parsed = new URL(value)
    const host = parsed.hostname.replace(/^\[|\]$/g, '')
    const isLoopback = host === 'localhost' || host === '::1' || /^127(?:\.\d{1,3}){3}$/.test(host)
    const port = parsed.port ? Number(parsed.port) : parsed.protocol === 'https:' ? 443 : 80
    if (!['http:', 'https:'].includes(parsed.protocol) || !isLoopback || !ALLOWED_PREVIEW_PORTS.includes(port as any)) return null
    parsed.username = ''
    parsed.password = ''
    return parsed.href
  } catch {
    return null
  }
}

function extractLocalPreviewUrls(value: string): string[] {
  const urls: string[] = []
  for (const match of String(value || '').matchAll(LOCAL_URL_RE)) {
    const url = normalizeLocalPreviewUrl(match[0])
    if (url && !urls.includes(url)) urls.push(url)
  }
  return urls
}

export function localPreviewCandidates(thread: Pick<Thread, 'cwd' | 'timeline'> | null | undefined): BrowserPreviewCandidate[] {
  const candidates: BrowserPreviewCandidate[] = []
  const add = (url: string | null, label: string, source: BrowserPreviewCandidate['source']) => {
    if (!url || candidates.some((item) => item.url === url)) return
    candidates.push({ url, label, source })
  }

  for (const item of thread?.timeline ?? []) {
    const haystack = [item.text, item.output, item.command, item.meta].filter(Boolean).join('\n')
    for (const url of extractLocalPreviewUrls(haystack)) add(url, 'Thread URL', 'thread')
  }

  const projectLabel = thread?.cwd ? basename(thread.cwd) : 'Local app'
  for (const port of ALLOWED_PREVIEW_PORTS) add(`http://localhost:${port}/`, `${projectLabel} :${port}`, 'default')
  return candidates
}

export function browserClickDiagnostics(input: { domReadable: boolean; note?: string; interactiveCount?: number; error?: string }): string[] {
  const diagnostics: string[] = []
  const note = input.note?.trim()
  if (note) diagnostics.push(`User note: ${note}`)
  if (!input.domReadable) {
    diagnostics.push('Preview could not inspect DOM hit targets from the sandboxed local iframe.')
    diagnostics.push('Attach Controller foreground-window context when a click target is missing, covered, disabled, or rendered inside canvas/WebGL.')
    if (input.error) diagnostics.push(`DOM access: ${input.error.slice(0, 180)}`)
    return diagnostics
  }
  diagnostics.push(`${input.interactiveCount ?? 0} interactive element${input.interactiveCount === 1 ? '' : 's'} visible to DOM inspection.`)
  diagnostics.push('If a click failed, compare the target text with the active element and check for overlays, disabled controls, or a stale preview URL.')
  return diagnostics
}

export function formatBrowserEvidence(input: {
  url: string
  capturedAt: number
  loadState: 'loaded' | 'loading' | 'error'
  cwd?: string
  inspection: BrowserInspection
}): string {
  const lines = [
    '<browser_evidence>',
    `Captured: ${new Date(input.capturedAt).toISOString()}`,
    `URL: ${input.url}`,
    `Load state: ${input.loadState}`,
    'Source: TensorFold local Preview (explicit user attachment)',
  ]
  if (input.cwd) lines.push(`Workspace: ${input.cwd}`)
  if (input.inspection.title) lines.push(`Title: ${input.inspection.title}`)
  if (input.inspection.activeElement) lines.push(`Active element: ${input.inspection.activeElement}`)
  if (typeof input.inspection.interactiveCount === 'number') lines.push(`Interactive elements: ${input.inspection.interactiveCount}`)
  lines.push('', 'Click diagnostics:')
  lines.push(...input.inspection.diagnostics.map((item) => `- ${item}`))
  lines.push('</browser_evidence>')
  return lines.join('\n')
}
