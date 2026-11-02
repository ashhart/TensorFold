import { useCallback, useEffect, useRef, useState } from 'react'
import type { QuickAskApi, QuickAskDraft, QuickAskSnapshot } from '../../../shared/quick-ask'

export function askError(error: unknown): string { return String(error instanceof Error ? error.message : error).replace(/^Error invoking remote method '[^']+':\s*(?:Error:\s*)?/, '').slice(0, 400) }
export function useQuickAsk(api: QuickAskApi) {
  const [snapshot, setSnapshot] = useState<QuickAskSnapshot | null>(null)
  const [draft, setDraft] = useState<QuickAskDraft | null>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [saving, setSaving] = useState(false)
  const draftRef = useRef<QuickAskDraft | null>(null), dirty = useRef(false), version = useRef(0)
  const revision = useRef(0), timer = useRef<ReturnType<typeof setTimeout>>()
  const writes = useRef(Promise.resolve()), mounted = useRef(true)
  const refreshVersion = useRef(0), sessionId = useRef<string | null>(null)
  const refresh = useCallback(async (refreshRoutes = false) => {
    const request = ++refreshVersion.current, editedAt = version.current
    const next = await api.snapshot({ refreshRoutes })
    if (!mounted.current || request !== refreshVersion.current) return
    setSnapshot(next)
    if (!dirty.current && editedAt === version.current && (sessionId.current !== next.session.id || next.session.draft.revision >= revision.current)) { sessionId.current = next.session.id; revision.current = next.session.draft.revision; draftRef.current = next.session.draft; setDraft(next.session.draft) }
  }, [api])
  const flush = useCallback((): Promise<void> => {
    clearTimeout(timer.current)
    const operation = writes.current.catch(() => {}).then(async () => {
      if (!dirty.current || !draftRef.current) return
      const captured = version.current, value = { ...draftRef.current, revision: revision.current }
      if (mounted.current) setSaving(true)
      try {
        const saved = await api.saveDraft(value)
        revision.current = saved.revision
        if (captured === version.current) { dirty.current = false; draftRef.current = saved; if (mounted.current) setDraft(saved) }
      } finally { if (mounted.current) setSaving(false) }
    })
    writes.current = operation
    return operation
  }, [api])
  const edit = useCallback((patch: Partial<Omit<QuickAskDraft, 'revision'>>) => {
    if (!draftRef.current) return
    const next = { ...draftRef.current, ...patch }
    draftRef.current = next; dirty.current = true; version.current++; setDraft(next); setError('')
    clearTimeout(timer.current)
    timer.current = setTimeout(() => { void flush().catch(error => { if (mounted.current) setError(askError(error)) }) }, 180)
  }, [flush])
  const action = useCallback(async (work: () => Promise<unknown>) => {
    setBusy(true); setError('')
    try { await flush(); await work(); await refresh(); return true }
    catch (error) { setError(askError(error)); return false }
    finally { setBusy(false) }
  }, [flush, refresh])
  useEffect(() => {
    mounted.current = true
    void refresh().catch(error => setError(askError(error)))
    const unsubscribe = api.onEvent(event => {
      if (event.kind === 'refresh') void refresh().catch(error => setError(askError(error)))
      if (event.kind === 'turn') setSnapshot(current => {
        if (!current || current.session.id !== event.sessionId) return current
        const turns = current.session.turns.some(turn => turn.id === event.turn.id) ? current.session.turns.map(turn => turn.id === event.turn.id ? event.turn : turn) : [...current.session.turns, event.turn]
        return { ...current, session: { ...current.session, turns } }
      })
      if (event.kind === 'flush') void flush().then(() => api.acknowledgeFlush({ requestId: event.requestId }), () => api.acknowledgeFlush({ requestId: event.requestId, error: 'save_failed' })).catch(() => {})
    })
    return () => { mounted.current = false; clearTimeout(timer.current); unsubscribe() }
  }, [api, flush, refresh])
  return { snapshot, draft, error, setError, busy, saving, edit, action, refresh, flush, revision }
}
