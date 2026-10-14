import { useEffect, useLayoutEffect, useMemo, useRef } from 'react'
import { useStore } from '../lib/store'
import { Icon } from '../lib/icons'
import { Button } from './ui'
import { UpdateActions } from './UpdateActions'
import { updateEmptyMessage } from '../lib/update-status'

// Release-notes → structured changelog. The feed (update-checker.ts) normalizes
// many manifest shapes (GitHub releases, Sparkle appcast, JSON) into a single
// `releaseNotes` string. Here we split it into titled sections and bullet lines
// so the modal reads like a real changelog rather than a wall of text.

interface ChangelogSection {
  title: string | null
  lines: string[]
}

const SECTION_RE = /^\s*(?:#{1,6}\s*|[-=*]{3,}\s*)?$/

function parseChangelog(notes: string | null): ChangelogSection[] {
  if (!notes || !notes.trim()) return []
  const raw = notes.replace(/\r\n/g, '\n').split('\n')

  // Group lines under headings. A heading is a markdown header, a short line
  // ending in ':', or an ALL CAPS / Title Case label not starting with a bullet.
  const sections: ChangelogSection[] = []
  let current: ChangelogSection = { title: null, lines: [] }

  const flush = () => {
    if (current.lines.length || current.title) sections.push(current)
    current = { title: null, lines: [] }
  }

  for (const original of raw) {
    const line = original.trim()
    if (!line || SECTION_RE.test(original)) continue

    const isMdHeading = /^#{1,6}\s+/.test(original)
    const looksLikeHeading =
      isMdHeading ||
      (/^[A-Z][\w &'/+-]{1,40}$/.test(line) && line.endsWith(':')) ||
      (/^[A-Z][A-Z0-9 _/-]{2,40}$/.test(line) && !line.includes('.'))

    if (looksLikeHeading && current.lines.length) flush()
    if (isMdHeading || looksLikeHeading) {
      if (current.lines.length) flush()
      current.title = line.replace(/^#{1,6}\s+/, '').replace(/:$/, '').trim()
      continue
    }

    // Normalize bullets: -, *, •, digit. or nothing → plain text.
    const bullet = line.replace(/^[-*•]\s+/, '').replace(/^\d+[.)]\s+/, '')
    current.lines.push(bullet)
  }
  flush()

  // If everything landed in one untitled bucket, keep it but drop the null title.
  return sections.filter((s) => s.lines.length)
}

function formatDate(ms: number | null): string | null {
  if (!ms) return null
  try {
    return new Date(ms).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' })
  } catch {
    return null
  }
}

export function UpdateModal() {
  const open = useStore((s) => s.updateModalOpen)
  const status = useStore((s) => s.updateStatus)
  const close = useStore((s) => s.closeUpdateModal)
  const installPending = useStore((s) => s.updateInstallPending)
  const installing = installPending || status?.state === 'installing'
  const dialog = useRef<HTMLDivElement>(null)

  // Esc closes, like every other modal in the app.
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape' && !installing) close() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, close, installing])

  // Pending preflight must block keyboard shortcuts and previously focused
  // background editors as well as pointer clicks on the modal overlay.
  useLayoutEffect(() => {
    if (!open || !installing) return
    const element = dialog.current
    if (!element) return
    element.focus()
    const onFocus = (event: FocusEvent) => {
      const state = useStore.getState()
      if (!state.updateInstallPending && state.updateStatus?.state !== 'installing') return
      if (!element.contains(event.target as Node)) element.focus()
    }
    const onKey = (event: KeyboardEvent) => {
      const state = useStore.getState()
      if (!state.updateInstallPending && state.updateStatus?.state !== 'installing') return
      event.stopPropagation()
      if (event.key === 'Escape' || event.key === 'Tab') {
        event.preventDefault()
        element.focus()
      }
    }
    window.addEventListener('focusin', onFocus, true)
    window.addEventListener('keydown', onKey, true)
    return () => {
      window.removeEventListener('focusin', onFocus, true)
      window.removeEventListener('keydown', onKey, true)
    }
  }, [open, installing])

  const sections = useMemo(() => parseChangelog(status?.releaseNotes ?? null), [status?.releaseNotes])

  if (!open || !status) return null

  const available = ['available', 'downloading', 'ready', 'installing'].includes(status.state) && status.latestVersion
  const date = formatDate(status.checkedAt)

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-6 fade-in" onClick={() => { if (!installing) close() }}>
      <div className="absolute inset-0 bg-black/40 backdrop-blur-[2px]" />
      <div
        onClick={(e) => e.stopPropagation()}
        ref={dialog}
        tabIndex={-1}
        aria-busy={installing}
        role="dialog"
        aria-modal="true"
        aria-labelledby="update-modal-title"
        className="relative flex max-h-[85vh] w-full max-w-[480px] flex-col overflow-hidden rounded-2xl border border-line bg-card shadow-pop"
      >
        {/* Header */}
        <div className="flex items-start justify-between gap-3 border-b border-line px-5 py-4">
          <div className="flex min-w-0 items-center gap-3">
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-accent/12 text-accent">
              <Icon name="spark" size={18} />
            </div>
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <h2 id="update-modal-title" className="truncate text-[15px] font-semibold tracking-[-0.01em] text-ink">
                  {available ? 'Update available' : 'App updates'}
                </h2>
                {available && (
                  <span className="shrink-0 rounded-full bg-accent/12 px-2 py-0.5 text-[12px] font-semibold text-accent">
                    v{status.latestVersion}
                  </span>
                )}
              </div>
              <div className="mt-0.5 flex items-center gap-2 text-[13px] text-ink-2">
                <span>Current v{status.currentVersion}</span>
                {status.latestVersion && (
                  <>
                    <span className="text-ink-5">→</span>
                    <span className="font-medium text-ink-2">v{status.latestVersion}</span>
                  </>
                )}
                {date && <><span className="text-ink-5">·</span><span>{date}</span></>}
              </div>
            </div>
          </div>
          <button onClick={close} disabled={installing} aria-label="Close" className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg text-ink-3 transition-colors hover:bg-ink-5 hover:text-ink disabled:opacity-40">
            <Icon name="x" size={15} />
          </button>
        </div>

        {/* Body: changelog */}
        <div className="scroll-thin min-h-0 flex-1 overflow-y-auto px-5 py-4">
          {status.releaseName && (
            <div className="mb-3 text-[13px] font-medium text-ink-2">{status.releaseName}</div>
          )}

          {sections.length === 0 ? status.error ? null : (
            <div className="rounded-xl border border-line bg-card-2 px-3.5 py-3 text-[13px] leading-relaxed text-ink-3">
              {updateEmptyMessage(status)}
            </div>
          ) : (
            <div className="space-y-4">
              {sections.map((section, i) => (
                <div key={i}>
                  {section.title && (
                    <div className="mb-1.5 text-[12px] font-semibold uppercase tracking-[0.06em] text-ink-3">{section.title}</div>
                  )}
                  <ul className="space-y-1.5">
                    {section.lines.map((line, j) => (
                      <li key={j} className="flex gap-2 text-[13px] leading-relaxed text-ink-2">
                        <span className="mt-[7px] h-1 w-1 shrink-0 rounded-full bg-ink-3" />
                        <span className="min-w-0 break-words">{line}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          )}

          {status.error && (
            <div className="mt-3 rounded-lg border border-danger/30 bg-danger/8 px-3 py-2 text-[13px] text-danger">
              {status.error}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="border-t border-line px-5 py-3.5">
          {available && (
            <div className="mb-2.5 flex items-center gap-2 text-[12px] text-ink-2">
              <Icon name="shield" size={12} />
              <span>{status.integrity === 'artifact_verified' ? 'Downloaded update verified.' : 'The signed release is verified again after download.'}</span>
            </div>
          )}
          <UpdateActions status={status} />
          <div className="mt-2 flex justify-end"><Button variant="ghost" size="sm" onClick={close} disabled={installing}>Close</Button></div>
        </div>
      </div>
    </div>
  )
}
