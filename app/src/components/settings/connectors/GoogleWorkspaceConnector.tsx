import { OriginalDiagnostic } from '../../OriginalDiagnostic'
import { useAuthoredText } from '../../../lib/preferences/useAuthoredText'
import { useEffect, useState } from 'react'
import { api } from '../../../lib/ipc'
import { Button } from '../../ui'
import { Section } from '../shared'
import { GoogleAccountSetup } from './google/GoogleAccountSetup'
import { GoogleActionReviews } from './google/GoogleActionReviews'
import { GoogleCalendarReader } from './google/GoogleCalendarReader'
import { GoogleMailDrafts } from './google/GoogleMailDrafts'
import { GoogleMailReader } from './google/GoogleMailReader'
import { useGoogleWorkspace } from './google/useGoogleWorkspace'

export function GoogleWorkspaceConnector() {
  const tx = useAuthoredText()
  const google = useGoogleWorkspace()
  const [selected, setSelected] = useState('')
  const [tab, setTab] = useState<'mail' | 'calendar'>('mail')
  const [draftOwner, setDraftOwner] = useState<string | null>(null)
  const snapshot = google.snapshot
  useEffect(() => { if (selected === 'new') return; if (snapshot && !snapshot.accounts.some((account) => account.id === selected)) setSelected(snapshot.accounts[0]?.id ?? 'new') }, [snapshot, selected])
  if (google.hostOnly) return <Section title={tx("Google Workspace")} desc={tx("Manage Google accounts locally in TensorFold on the host Mac.")}><p className="rounded-xl border border-line bg-card p-4 text-[13px] text-ink-2">{tx("Google Desktop OAuth client import, credentials and owner reviews are available only on the host Mac.")}</p></Section>
  const account = snapshot?.accounts.find((item) => item.id === selected)
  const connected = account?.status === 'connected'
  // A refreshed auth revision clears read panes, but dirty email edits stay
  // mounted for this account until the same verified identity reconnects.
  return <Section title={tx("Google Workspace")} desc={tx("Gmail and Calendar with your own Google sign-in, clear access choices and review before any external change.")}>
    <div className="space-y-4">
      {(api.platform === 'browser' || api.platform === 'web') && <p role="status" className="rounded-xl border border-warn/30 bg-warn/10 p-3 text-[13px] text-warn">{tx("Preview only; these are synthetic accounts and sample data, with no real Google sign-in or API requests.")}</p>}
      {google.error && <p role="alert" className="text-[13px] text-danger">{<OriginalDiagnostic>{google.error}</OriginalDiagnostic>}</p>}
      {!snapshot && <div className="rounded-xl border border-line bg-card p-4"><p role="status" className="text-[13px] text-ink-3">{google.error ? tx("Google account status is unavailable.") : tx("Loading Google connections…")}</p>{google.error && <Button className="mt-3" size="sm" onClick={() => google.action('refresh', google.refresh)}>{tx("Retry")}</Button>}</div>}
      {snapshot && <>
        {!snapshot.accounts.length && <p role="status" className="rounded-lg border border-line bg-card-2 p-3 text-[13px] text-ink-2">{snapshot.clientConfigured ? tx("No Google accounts are connected on this Mac. Sign in below and choose an account or use another address in Google's sign-in window.") : tx("No Google accounts are connected on this Mac. Set up Google sign-in below to add one.")}</p>}
        <div className="flex flex-wrap items-end gap-2 rounded-xl border border-line bg-card p-3.5"><label className="min-w-0 flex-1 text-[13px] text-ink-2">{tx("Google account")}<select aria-label={tx("Google account")} value={selected} disabled={!!google.busy || !!draftOwner} onChange={(event) => setSelected(event.target.value)} className="mt-1 w-full rounded-lg border border-line bg-card-2 px-3 py-2 text-[13px] text-ink"><option value="new">{tx("Connect another account")}</option>{snapshot.accounts.map((item) => <option key={item.id} value={item.id}>{item.email} · {item.status === 'connected' ? tx("Connected") : item.status === 'needs-reauthorization' ? tx("Reconnect needed") : tx("Disconnected on this Mac")}</option>)}</select></label><Button size="sm" disabled={!!google.busy} onClick={() => google.action('refresh', google.refresh)}>{tx("Refresh")}</Button>{account && <Button size="sm" disabled={!!google.busy || !!draftOwner || account.status === 'disconnected'} onClick={() => google.disconnect(account.id)}>{google.busy === 'disconnect-' + account.id ? tx("Disconnecting…") : tx("Disconnect account")}</Button>}</div>
        {draftOwner && <p role="status" className="text-[12px] text-warn">{connected ? tx("Save or discard your local email draft edits before changing accounts or Google permissions.") : tx("Your unsaved email edits remain on this page. Reconnect the same Google account to save them locally.")}</p>}
        {account && <div className="rounded-lg bg-card-2 p-3 text-[13px] text-ink-2"><p className="font-medium text-ink">{account.email}</p><p className="mt-1">{account.status === 'connected' ? tx("Google granted") : tx("Previously granted")}: {account.capabilities.map((capability) => ({ 'gmail.read': tx('Read Gmail'), 'gmail.send': tx('Send reviewed email'), 'calendar.read': tx('Read Calendar'), 'calendar.write': tx('Reviewed Calendar changes') }[capability])).join(', ') || tx("No access granted")}</p>{account.status !== 'connected' && <p className="mt-2 text-warn">{account.status === 'disconnected' ? tx("Disconnected locally on this Mac; reconnect with Google before reading or preparing new actions.") : tx("Reconnect with Google before reading or preparing new actions.")}</p>}{account.status !== 'connected' && account.lastError && <p role="alert" className="mt-2 break-words text-warn">{<OriginalDiagnostic>{account.lastError}</OriginalDiagnostic>}</p>}{google.accountErrors[account.id] && <p role="alert" className="mt-2 text-danger">{google.accountErrors[account.id]}</p>}</div>}
        <GoogleAccountSetup key={account ? `setup:${account.id}:${account.revision}` : 'setup:new'} snapshot={snapshot} account={account} busy={google.busy || (draftOwner && connected ? 'draft' : '')} permissionsLocked={!!draftOwner} importClient={() => { void google.importClient() }} authorize={(capabilities) => { void google.authorize({ capabilities, ...(account ? { accountId: account.id } : {}) }) }} cancel={() => { void google.cancel() }} />
        {account && (connected || draftOwner === account.id) && <div key={`tools:${account.id}`} className="space-y-4"><div role="tablist" aria-label={tx("Google Workspace tools")} className="flex flex-wrap gap-2"><Button role="tab" aria-selected={tab === 'mail'} size="sm" variant={tab === 'mail' ? 'primary' : 'default'} onClick={() => setTab('mail')}>{tx("Gmail")}</Button><Button role="tab" aria-selected={tab === 'calendar'} size="sm" variant={tab === 'calendar' ? 'primary' : 'default'} onClick={() => setTab('calendar')}>{tx("Calendar")}</Button></div><div hidden={tab !== 'mail'} className="space-y-4">{connected && account.capabilities.includes('gmail.read') ? <GoogleMailReader account={account} refresh={google.refresh} /> : <p className="text-[13px] text-ink-3">{tx("Reconnect with Read Gmail access to browse messages.")}</p>}<GoogleMailDrafts key={`draft:${account.id}`} account={account} refresh={google.refresh} onDirtyChange={(dirty) => setDraftOwner(dirty ? account.id : null)} /></div><div hidden={tab !== 'calendar'}>{connected && account.capabilities.includes('calendar.read') ? <GoogleCalendarReader account={account} refresh={google.refresh} /> : <p className="text-[13px] text-ink-3">{tx("Reconnect with Read Calendar access to view calendars and availability.")}</p>}</div></div>}
        <GoogleActionReviews key={`reviews:${account?.id ?? 'new'}`} reviews={snapshot.reviews.filter((review) => review.accountId === account?.id)} recentActions={(snapshot.recentActions ?? []).filter((item) => item.review.accountId === account?.id)} refresh={google.refresh} />
      </>}
    </div>
  </Section>
}
