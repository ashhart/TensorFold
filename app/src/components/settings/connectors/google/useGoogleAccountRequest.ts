import { useEffect, useRef, useState } from 'react'
import { googleRequestError } from '../../../../../shared/google-workspace-errors'
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
    catch (reason) {
      if (mounted.current && ownEpoch === epoch.current) {
        setError(googleRequestError(reason))

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
