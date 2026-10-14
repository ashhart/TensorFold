import { useEffect, useMemo, useRef, useState } from 'react'
import { Icon } from '../lib/icons'
import { api } from '../lib/ipc'
import { useStore } from '../lib/store'
import {
  COMPOSER_INSERT_EVENT,
  browserClickDiagnostics,
  formatBrowserEvidence,
  localPreviewCandidates,
  normalizeLocalPreviewUrl,
  type BrowserInspection,
} from '../lib/browser-preview'

function describeElement(el: Element | null): string {
  if (!el) return ''
  return [
    el.tagName.toLowerCase(),
    el.getAttribute('role') ? `[role="${el.getAttribute('role')}"]` : '',
    el.getAttribute('aria-label') ? `[aria-label="${el.getAttribute('aria-label')}"]` : '',
    el.textContent?.trim() ? ` "${el.textContent.trim().replace(/\s+/g, ' ').slice(0, 80)}"` : '',
  ].join('').slice(0, 160)
}

function inspectPreviewFrame(frame: HTMLIFrameElement | null, note: string): BrowserInspection {
  try {
    const doc = frame?.contentDocument ?? frame?.contentWindow?.document
    if (!doc) throw new Error('iframe document unavailable')
    const interactiveCount = doc.querySelectorAll('a,button,input,select,textarea,[role="button"],[role="link"],[onclick],summary').length
    const activeElement = describeElement(doc.activeElement)
    return {
      domReadable: true,
      title: doc.title || undefined,
      activeElement: activeElement || undefined,
      interactiveCount,
      diagnostics: browserClickDiagnostics({ domReadable: true, note, interactiveCount }),
    }
  } catch (e: any) {
    return {
      domReadable: false,
      diagnostics: browserClickDiagnostics({ domReadable: false, note, error: String(e?.message || e || 'blocked') }),
    }
  }
}

// Local preview browser: loads localhost URLs and can attach explicit QA evidence.
export function Preview({ onClose }: { onClose: () => void }) {
  const view = useStore((s) => s.view)
  const thread = useStore((s) => view.name === 'thread' ? s.threads.find((t) => t.id === view.id) ?? null : null)
  const candidates = useMemo(() => localPreviewCandidates(thread), [thread?.id, thread?.cwd, thread?.updatedAt])
  const defaultUrl = candidates[0]?.url ?? 'http://localhost:5173/'
  const frameRef = useRef<HTMLIFrameElement>(null)
  const lastDefaultUrl = useRef(defaultUrl)
  const [url, setUrl] = useState(defaultUrl)
  const [src, setSrc] = useState(defaultUrl)
  const [nonce, setNonce] = useState(0)
  const [error, setError] = useState('')
  const [loadState, setLoadState] = useState<'loaded' | 'loading' | 'error'>('loading')
  const [note, setNote] = useState('')
  const [attached, setAttached] = useState(false)
  const [saved, setSaved] = useState(false)

  useEffect(() => {
    if (src === lastDefaultUrl.current && url === lastDefaultUrl.current) {
      setUrl(defaultUrl)
      setSrc(defaultUrl)
      setLoadState('loading')
    }
    lastDefaultUrl.current = defaultUrl
  }, [defaultUrl]) // eslint-disable-line react-hooks/exhaustive-deps

  const go = () => {
    const u = normalizeLocalPreviewUrl(url)
    if (!u) {
      setError('Enter a loopback URL on preview port 3000, 4173, or 5173.')
      return
    }
    setError('')
    setLoadState('loading')
    setUrl(u); setSrc(u); setNonce((n) => n + 1)
  }

  const attachEvidence = () => {
    const inspection = inspectPreviewFrame(frameRef.current, note)
    const evidence = formatBrowserEvidence({
      url: src,
      capturedAt: Date.now(),
      loadState,
      cwd: thread?.cwd,
      inspection,
    })
    window.dispatchEvent(new CustomEvent(COMPOSER_INSERT_EVENT, { detail: { text: evidence, source: 'browser' } }))
    setAttached(true)
    window.setTimeout(() => setAttached(false), 1400)
  }

  const saveBrowserSource = async () => {
    const inspection = inspectPreviewFrame(frameRef.current, note)
    await api.addBrowserCurrentTabSource({
      url: src,
      title: inspection.title || new URL(src).hostname,
      excerpt: note || inspection.diagnostics.join(' '),
      capturedAt: Date.now(),
    })
    setSaved(true)
    window.setTimeout(() => setSaved(false), 1400)
  }

  return (
    <div className="flex h-full flex-col bg-card-2/40">
      <div className="flex items-center gap-2 border-b border-line px-3 py-1.5">
        <Icon name="globe" size={13} className="shrink-0 text-ink-3" />
        <select
          value={candidates.some((candidate) => candidate.url === src) ? src : ''}
          onChange={(e) => {
            if (!e.target.value) return
            setUrl(e.target.value)
            setSrc(e.target.value)
            setLoadState('loading')
            setNonce((n) => n + 1)
          }}
          className="h-6 max-w-[140px] rounded-md border border-line bg-card px-2 text-[13px] text-ink outline-none transition-colors focus:border-accent"
          title="Local preview candidates"
        >
          {candidates.map((candidate) => <option key={candidate.url} value={candidate.url}>{candidate.label}</option>)}
          {!candidates.some((candidate) => candidate.url === src) && <option value="">Custom URL</option>}
        </select>
        <input
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') go() }}
          placeholder="http://localhost:5173"
          className="h-6 min-w-0 flex-1 rounded-md border border-line bg-card px-2 text-[13px] text-ink outline-none transition-colors focus:border-accent"
          spellCheck={false}
        />
        <button onClick={go} className="flex h-6 w-6 items-center justify-center rounded text-ink-3 transition-colors hover:bg-ink-5 hover:text-ink" title="Go / reload"><Icon name="arrowRight" size={13} /></button>
        <button onClick={attachEvidence} className="flex h-6 items-center gap-1 rounded px-1.5 text-[12px] font-medium text-ink-3 transition-colors hover:bg-ink-5 hover:text-ink" title="Attach browser evidence to composer">
          <Icon name={attached ? 'check' : 'attach'} size={13} />{attached ? 'Attached' : 'Evidence'}
        </button>
        <button onClick={saveBrowserSource} className="flex h-6 items-center gap-1 rounded px-1.5 text-[12px] font-medium text-ink-3 transition-colors hover:bg-ink-5 hover:text-ink" title="Save explicit browser source">
          <Icon name={saved ? 'check' : 'plus'} size={13} />{saved ? 'Saved' : 'Source'}
        </button>
        <button onClick={onClose} className="flex h-6 w-6 items-center justify-center rounded text-ink-3 transition-colors hover:bg-ink-5 hover:text-ink" aria-label="Close preview"><Icon name="x" size={13} /></button>
      </div>
      <div className="flex items-center gap-2 border-b border-line px-3 py-1.5">
        <input
          value={note}
          onChange={(e) => setNote(e.target.value)}
          placeholder="Evidence note or failed click target"
          className="h-6 min-w-0 flex-1 rounded-md border border-line bg-card px-2 text-[13px] text-ink outline-none transition-colors placeholder:text-ink-4 focus:border-accent"
          spellCheck={false}
        />
        <span className="shrink-0 rounded bg-ink-5 px-1.5 text-[12px] text-ink-3">{loadState}</span>
      </div>
      {error && <div className="border-b border-line px-3 py-1.5 text-[13px] text-danger">{error}</div>}
      <iframe
        key={nonce}
        ref={frameRef}
        src={src}
        title="Preview"
        className="min-h-0 w-full flex-1 border-0 bg-white"
        sandbox="allow-scripts allow-forms"
        onLoad={() => setLoadState('loaded')}
        onError={() => { setLoadState('error'); setError('Preview failed to load this local URL.') }}
      />
    </div>
  )
}
