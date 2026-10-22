import { useEffect, useRef, useState } from 'react'
import type { GoogleWorkspaceAccount } from '../../../../../shared/google-workspace'

// Fence every request by account+revision, including draft panes retained during reconnect.
export function useGoogleAccountRequest(account: GoogleWorkspaceAccount, refreshAccountStatus?: () => Promise<void>) {
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const epoch = useRef(0)
  const mounted = useRef(true)
  useEffect(() => {
    mounted.current = true
    setBusy(''); setError('')
    return () => { mounted.current = false; epoch.current++ }
  }, [account.id, account.revision])
  const run = async <T,>(name: string, work: () => Promise<T>, apply: (result: T) => void) => {
    const ownEpoch = ++epoch.current
    setBusy(name); setError('')
    try { const result = await work(); if (mounted.current && ownEpoch === epoch.current) { apply(result); return true } }
    catch {
      if (mounted.current && ownEpoch === epoch.current) {
        setError('Google could not complete this request; check the account permissions and connection, then retry.')
        // A failed token refresh or 401 can have removed the host's credentials.
        // Re-read that state so the parent replaces this pane with Reconnect;
        // neither exception text nor credentials need to cross this boundary.
        try { await refreshAccountStatus?.() }
        catch {
          if (mounted.current && ownEpoch === epoch.current) setError('Google could not complete this request or refresh the account status; use Refresh in Google Workspace, then retry.')
        }
      }
    }
    finally { if (mounted.current && ownEpoch === epoch.current) setBusy('') }
    return false
  }
  return { busy, error, run, clearError: () => setError('') }
}
