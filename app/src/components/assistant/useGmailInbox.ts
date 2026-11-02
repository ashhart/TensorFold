import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../../lib/ipc'
import { parseGmailInboxKey } from '../../../shared/gmail-inbox'
import type { GmailInboxSnapshot } from '../../../shared/gmail-inbox'
import type { GoogleWorkspaceThread } from '../../../shared/google-workspace'

export function useGmailInbox() {
  const [snapshot, setSnapshot] = useState<GmailInboxSnapshot>({ accounts: [], events: [] })
  const [opened, setOpened] = useState<{ key: string; scopeId?: string; thread: GoogleWorkspaceThread } | null>(null)
  const [error, setError] = useState(''), [busy, setBusy] = useState('')
  const mounted = useRef(true), generation = useRef(0), pending = useRef(false)
  const hostOnly = api.platform === 'web-remote' || api.platform === 'web'
  const accept = useCallback((next: GmailInboxSnapshot) => {
    setSnapshot(next)
    setOpened(current => {
      const target = current && parseGmailInboxKey(current.key)
      return target && next.accounts.some(account => account.id === target.accountId && account.revision === target.revision && account.enabled && account.connected && account.scopeId === current?.scopeId) ? current : null
    })
    setError('')
  }, [])
  const refresh = useCallback(async () => {
    if (hostOnly) return
    const own = ++generation.current
    try { const next = await api.gmailInboxSnapshot(); if (mounted.current && own === generation.current) accept(next) }
    catch { if (mounted.current && own === generation.current) { setSnapshot({ accounts: [], events: [] }); setOpened(null); setError('Gmail Inbox status could not be read. Refresh to try again.') } }
  }, [hostOnly, accept])
  useEffect(() => {
    mounted.current = true; void refresh()
    const reconcile = () => { if (!document.hidden && !pending.current) void refresh() }
    const timer = setInterval(reconcile, 15_000)
    window.addEventListener('focus', reconcile); document.addEventListener('visibilitychange', reconcile)
    return () => { mounted.current = false; generation.current++; clearInterval(timer); window.removeEventListener('focus', reconcile); document.removeEventListener('visibilitychange', reconcile) }
  }, [refresh])
  const action = async (name: string, run: () => Promise<void>) => {
    if (pending.current || hostOnly) return
    pending.current = true; generation.current++; setBusy(name); setError('')
    let failed = false
    try { await run() }
    catch { failed = true }
    finally {
      pending.current = false
      if (mounted.current) {
        setBusy(''); await refresh()
        if (failed && mounted.current) setError('Gmail could not complete this request. Refresh the connection status and try again.')
      }
    }
  }
  return {
    snapshot, opened, error, busy, refresh,
    sync: () => action('sync', async () => { await api.gmailInboxSync() }),
    read: (key: string) => action('read', async () => {
      const own = generation.current, thread = await api.gmailInboxThread(key)
      if (mounted.current && own === generation.current) setOpened({ key, scopeId: snapshot.accounts.find(account => account.id === parseGmailInboxKey(key)?.accountId)?.scopeId, thread })
    }),
  }
}
