import { useEffect, useRef, useState } from 'react'
import { api } from '../../../lib/ipc'
import { useAuthoredText } from '../../../lib/preferences/useAuthoredText'
import { Button, Field, Input, Toggle } from '../../ui'
import { OriginalDiagnostic } from '../../OriginalDiagnostic'
import type { MicrosoftMailSignIn } from '../../../../shared/microsoft-mail'
import type { SettingsSectionContext } from '../shared'

export function MicrosoftMailConnector({ ctx }: { ctx: SettingsSectionContext }) {
  const tx = useAuthoredText()
  const [clientId, setClientId] = useState('')
  const [allowSend, setAllowSend] = useState(false)
  const [session, setSession] = useState<MicrosoftMailSignIn | null>(null)
  const [pending, setPending] = useState(false)
  const [error, setError] = useState('')
  const [status, setStatus] = useState('')
  const active = useRef(true)
  const sessionRef = useRef<MicrosoftMailSignIn | null>(null)
  useEffect(() => { active.current = true; return () => { active.current = false; if (sessionRef.current) void api.microsoftMailCancel(sessionRef.current.id).catch(() => {}) } }, [])
  const start = async () => {
    setPending(true); setError(''); setStatus('')
    try {
      const next = await api.microsoftMailBegin(clientId.trim(), allowSend)
      if (!active.current) { void api.microsoftMailCancel(next.id).catch(() => {}); return }
      sessionRef.current = next; setSession(next)
    } catch (reason) { if (active.current) setError(reason instanceof Error ? reason.message : tx('Microsoft sign-in failed.')) }
    finally { if (active.current) setPending(false) }
  }
  const check = async () => {
    if (!session) return
    setPending(true); setError('')
    try {
      const result = await api.microsoftMailPoll(session.id)
      if (!active.current || sessionRef.current?.id !== session.id) return
      if (result.status === 'connected') {
        sessionRef.current = null; setSession(null)
        const settings = await api.getSettings()
        if (active.current) { ctx.setDraft(current => ({ ...current, email: settings.email })); setStatus(tx('Microsoft account connected. Sync email to read the inbox.')) }
      } else setStatus(tx('Waiting for Microsoft sign-in. Finish in your browser, then check again.'))
    } catch (reason) { if (active.current) setError(reason instanceof Error ? reason.message : tx('Microsoft sign-in failed.')) }
    finally { if (active.current) setPending(false) }
  }
  const cancel = async () => {
    const id = sessionRef.current?.id
    sessionRef.current = null; setSession(null); setStatus('')
    if (id) await api.microsoftMailCancel(id).catch(() => {})
  }
  const remote = api.platform === 'web-remote' || api.platform === 'web'
  return <div className="mb-4 space-y-3 rounded-xl border border-line bg-card p-3.5">
    <h3 className="text-[14px] font-medium text-ink">{tx('Outlook / Microsoft 365')}</h3>
    <p className="text-[13px] leading-relaxed text-ink-2">{tx('Use Microsoft sign-in on macOS, Windows or Linux. No Outlook installation or mailbox password is required.')}</p>
    {remote ? <p className="text-[13px] text-ink-2">{tx('Connect Microsoft mail in TensorFold on the host computer.')}</p> : <>
      <details className="text-[13px] text-ink-2"><summary className="cursor-pointer">{tx('Microsoft application setup')}</summary><p className="mt-2">{tx('Register a public-client application in Microsoft Entra, allow public client flows, and add delegated User.Read and Mail.Read permissions. Add Mail.Send only if you want approved sending. Your organization may require administrator consent.')}</p></details>
      {!session ? <>
        <Field label={tx('Application client ID')}><Input value={clientId} onChange={event => setClientId(event.target.value)} spellCheck={false} disabled={pending} placeholder="00000000-0000-0000-0000-000000000000" /></Field>
        <Toggle label={tx('Request sending permission')} checked={allowSend} onChange={setAllowSend} disabled={pending} />
        <p className="text-[12px] text-ink-3">{tx('Sending always requires a separate review of the account, recipient and exact message.')}</p>
        <Button size="sm" disabled={pending || !clientId.trim()} onClick={() => void start()}>{pending ? tx('Connecting…') : tx('Connect Microsoft account')}</Button>
      </> : <div className="space-y-2 text-[13px] text-ink-2">
        <p>{tx('Open Microsoft sign-in and enter this code:')} <strong className="select-all font-mono">{session.userCode}</strong></p>
        <p>{tx('Only enter a code you requested here. Check the application and account shown by Microsoft.')}</p>
        <p>{tx('Sign-in expires at {value0}', { value0: new Date(session.expiresAt).toLocaleTimeString() })}</p>
        <div className="flex flex-wrap gap-2"><Button size="sm" onClick={() => void api.openExternal(session.verificationUri)}>{tx('Open Microsoft sign-in')}</Button><Button size="sm" disabled={pending} onClick={() => void check()}>{tx('Check sign-in')}</Button><Button size="sm" variant="ghost" disabled={pending} onClick={() => void cancel()}>{tx('Cancel')}</Button></div>
      </div>}
    </>}
    {status && <p role="status" className="text-[13px] text-ink-2">{status}</p>}
    {error && <p role="alert" className="text-[13px] text-danger"><OriginalDiagnostic>{error}</OriginalDiagnostic></p>}
  </div>
}
