import { useEffect, useRef, useState } from 'react'
import type { GoogleWorkspaceAccount } from '../../../../../shared/google-workspace'

// Each pane is keyed by account+revision; old-account results cannot populate its replacement.
export function useGoogleAccountRequest(account: GoogleWorkspaceAccount) {
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const epoch = useRef(0)
  const mounted = useRef(true)
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; epoch.current++ } }, [account.id, account.revision])
  const run = async <T,>(name: string, work: () => Promise<T>, apply: (result: T) => void) => {
    const ownEpoch = ++epoch.current
    setBusy(name); setError('')
    try { const result = await work(); if (mounted.current && ownEpoch === epoch.current) { apply(result); return true } }
    catch { if (mounted.current && ownEpoch === epoch.current) setError('Google could not complete this request; check the account permissions and connection, then retry.') }
    finally { if (mounted.current && ownEpoch === epoch.current) setBusy('') }
    return false
  }
  return { busy, error, run, clearError: () => setError('') }
}
