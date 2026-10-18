import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../../../../lib/ipc'
import type { GoogleWorkspaceAuthRequest, GoogleWorkspaceSnapshot } from '../../../../../shared/google-workspace'
import { googleClientImportError } from './google-setup-error'

export function useGoogleWorkspace() {
  const [snapshot, setSnapshot] = useState<GoogleWorkspaceSnapshot | null>(null)
  const [error, setError] = useState('')
  const [accountErrors, setAccountErrors] = useState<Record<string, string>>({})
  const [busy, setBusy] = useState('')
  const mounted = useRef(true)
  const version = useRef(0)
  const acting = useRef(false)
  const hostOnly = api.platform === 'web-remote'
  const refresh = useCallback(async () => {
    if (hostOnly) return
    const ownVersion = ++version.current
    const next = await api.googleWorkspaceStatus()
    if (mounted.current && ownVersion === version.current) { setSnapshot(next); setError('') }
  }, [hostOnly])
  useEffect(() => {
    mounted.current = true
    void refresh().catch(() => { if (mounted.current) setError('Google connections could not be read; check the local credential store, then retry.') })
    return () => { mounted.current = false; version.current++ }
  }, [refresh])
  const action = async (name: string, fn: () => Promise<unknown>, accountId?: string) => {
    if (hostOnly) { setError('Manage Google accounts on the host Mac.'); return false }
    if (acting.current) return false
    acting.current = true; setBusy(name); setError('')
    if (accountId) setAccountErrors((current) => ({ ...current, [accountId]: '' }))
    // An earlier background status read cannot overwrite the action's resulting snapshot.
    version.current++
    try { await fn(); await refresh(); return true }
    catch (reason) {
      if (mounted.current) {
        const message = name === 'import' ? googleClientImportError(reason) : 'This Google connection action failed; check account access, client configuration and the network, then retry.'
        if (accountId) setAccountErrors((current) => ({ ...current, [accountId]: message }))
        else setError(message)
      }
      return false
    } finally { acting.current = false; if (mounted.current) setBusy('') }
  }
  const waiting = snapshot?.authorization?.status === 'waiting'
  useEffect(() => {
    if (!waiting) return
    const timer = setInterval(() => { void refresh().catch(() => { if (mounted.current) setError('Sign-in status could not be refreshed.') }) }, 1500)
    return () => clearInterval(timer)
  }, [waiting, refresh])
  return {
    snapshot, error, accountErrors, busy, hostOnly, refresh, action,
    importClient: () => action('import', () => api.googleWorkspaceImportClient()),
    authorize: (request: GoogleWorkspaceAuthRequest) => action('authorize', () => api.googleWorkspaceAuthorize(request), request.accountId),
    cancel: () => action('cancel', () => api.googleWorkspaceCancelAuth(snapshot?.authorization?.id)),
    disconnect: (id: string) => action('disconnect-' + id, () => api.googleWorkspaceDisconnect(id), id),
  }
}
