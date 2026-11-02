import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../../lib/ipc'
import type { MailConnectionTarget, MailConnectionsSnapshot } from '../../../shared/mail-connections'

export function useMailConnections() {
  const [snapshot, setSnapshot] = useState<MailConnectionsSnapshot | null>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState<string[]>([])
  const mounted = useRef(true), generation = useRef(0), pending = useRef(new Set<string>())
  const hostOnly = api.platform === 'web-remote' || api.platform === 'web'
  const refresh = useCallback(async () => {
    if (hostOnly) return
    const own = ++generation.current
    try {
      const next = await api.mailConnections()
      if (mounted.current && generation.current === own) { setSnapshot(next); setError('') }
    } catch (reason) {
      if (mounted.current && generation.current === own) setError('Mail connection status could not be read. Refresh to try again.')
      throw reason
    }
  }, [hostOnly])
  useEffect(() => {
    mounted.current = true
    void refresh().catch(() => { if (mounted.current) setError('Mail connection status could not be read. Refresh to try again.') })
    return () => { mounted.current = false; generation.current++ }
  }, [refresh])
  const test = async (target: MailConnectionTarget) => {
    const key = `${target.provider}:${target.id}`
    if (hostOnly || pending.current.has(key)) return
    pending.current.add(key); generation.current++; setBusy([...pending.current]); setError('')
    let failed = false
    try { await api.testMailConnection(target) }
    catch { failed = true }
    finally {
      pending.current.delete(key)
      if (mounted.current) {
        setBusy([...pending.current])
        await refresh().catch(() => { })
        if (failed && mounted.current) setError('The account changed or its check could not finish. Refresh the connection status before trying again.')
      }
    }
  }
  const setInbox = async (id: string, revision: string, enabled: boolean) => {
    const key = `gmail:${id}`
    if (hostOnly || pending.current.has(key)) return
    pending.current.add(key); generation.current++; setBusy([...pending.current]); setError('')
    let failed = false
    try { await api.gmailInboxEnable({ accountId: id, revision: Number(revision), enabled }) }
    catch { failed = true }
    finally {
      pending.current.delete(key)
      if (mounted.current) {
        setBusy([...pending.current]); await refresh().catch(() => { })
        if (failed && mounted.current) setError('The account changed or its check could not finish. Refresh the connection status before trying again.')
      }
    }
  }
  return { snapshot, error, busy, hostOnly, refresh, test, setInbox }
}
