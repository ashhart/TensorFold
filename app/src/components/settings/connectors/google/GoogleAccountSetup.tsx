import { useAuthoredText } from '../../../../lib/preferences/useAuthoredText'
import { authoredText } from '../../../../lib/preferences/language-packs'
import { useStore as useLocaleStore } from '../../../../lib/store'
import { useEffect, useState } from 'react'
import type { GoogleWorkspaceAccount, GoogleWorkspaceCapability, GoogleWorkspaceSnapshot } from '../../../../../shared/google-workspace'
import { api } from '../../../../lib/ipc'
import { Button, Toggle } from '../../../ui'

const CAPABILITIES: { id: GoogleWorkspaceCapability; label: string; detail: string }[] = [
  { id: 'gmail.read', get label() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Read Gmail") }, detail: 'Search and read messages without marking them read.' },
  { id: 'calendar.read', get label() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Read Calendar") }, detail: 'View calendars, events and availability.' },
  { id: 'gmail.send', get label() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Send reviewed emails") }, detail: 'Every send requires an exact-content owner review.' },
  { id: 'calendar.write', get label() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Apply reviewed calendar changes") }, detail: 'Create, edit or cancel an event after owner review.' },
]
export function GoogleAccountSetup({ snapshot, account, busy, permissionsLocked, importClient, authorize, cancel }: {
  snapshot: GoogleWorkspaceSnapshot; account?: GoogleWorkspaceAccount; busy: string;
  permissionsLocked?: boolean;
  importClient: () => void; authorize: (capabilities: GoogleWorkspaceCapability[]) => void; cancel: () => void;
}) {
  const tx = useAuthoredText()
  const [capabilities, setCapabilities] = useState<GoogleWorkspaceCapability[]>(['gmail.read', 'calendar.read'])
  const [linkError, setLinkError] = useState('')
  useEffect(() => { setCapabilities(account?.capabilities ?? ['gmail.read', 'calendar.read']) }, [account?.id, account?.revision])
  const waiting = snapshot.authorization?.status === 'waiting'
  const activeAccounts = snapshot.accounts.some((candidate) => candidate.status !== 'disconnected')
  const canImport = !busy && !waiting && !activeAccounts
  const chooseClient = () => { if (canImport) importClient() }
  return <div className="rounded-xl border border-line bg-card p-4">
    <h3 className="text-[14px] font-medium text-ink">{account ? tx("Account permissions") : tx("Connect your Google account")}</h3>
    <p className="mt-1 text-[13px] leading-relaxed text-ink-2">{tx("Your installation uses your own Google Desktop OAuth client, following Google's browser sign-in flow.")}</p>
    {!snapshot.clientConfigured && <div className="mt-4 space-y-3">
      <ol className="list-inside list-decimal space-y-2 text-[13px] leading-relaxed text-ink-2">
        <li>{tx("Create your Google Cloud project and configure its OAuth consent audience.")}</li>
        <li>{tx("Enable the Gmail API and Google Calendar API for that project.")}</li>
        <li>{tx("Create an OAuth client of type Desktop app and download its JSON file.")}</li>
        <li>{tx("Import that file below, choose access and sign in with Google.")}</li>
      </ol>
      <p className="text-[12px] leading-relaxed text-ink-2">{tx("For a project in Testing, add your account as a test user; Google's testing and account policies can require reconnection.")}</p>
      <div className="flex flex-wrap gap-2">
        <Button size="sm" onClick={() => { void api.openExternal('https://console.cloud.google.com/auth/overview').catch(() => setLinkError(tx("Could not open Google Cloud; check your browser and retry."))) }}>{tx("Google Cloud setup")}</Button>
        <Button size="sm" onClick={() => { void api.openExternal('https://console.cloud.google.com/apis/library').catch(() => setLinkError(tx("Could not open the API library; check your browser and retry."))) }}>{tx("Enable APIs")}</Button>
        <Button size="sm" variant="primary" disabled={!canImport} onClick={chooseClient}>{busy === 'import' ? tx("Importing…") : tx("Import Desktop client JSON")}</Button>
      </div>
    </div>}
    {snapshot.clientConfigured && <>
      <details className="mt-4 rounded-lg border border-line bg-card-2 p-3 text-[13px] text-ink-2">
        <summary className="cursor-pointer font-medium text-ink">{tx("Google Cloud project and Desktop client")}</summary>
        <p className="mt-2 text-[12px] leading-relaxed text-ink-2">{tx("Imported the wrong client or project? Disconnect every Google account first, then import a replacement Desktop app client JSON; reconnect each account through Google afterward.")}</p>
        <p className="mt-2 text-[12px] leading-relaxed text-ink-2">{tx("Replacing the client never signs in or disconnects an account automatically; its JSON and credentials remain on the host Mac.")}</p>
        {activeAccounts && <p className="mt-2 text-[12px] text-warn">{tx("Disconnect connected accounts and accounts needing reauthorization before replacing this client.")}</p>}
        <Button className="mt-3" size="sm" disabled={!canImport} onClick={chooseClient}>{busy === 'import' ? tx("Importing…") : tx("Replace Desktop client JSON")}</Button>
      </details>
      <div className="mt-4 grid gap-3 sm:grid-cols-2">{CAPABILITIES.map((capability) => <div key={capability.id} className="flex items-start justify-between gap-3 rounded-lg border border-line p-3">
        <div><p className="text-[13px] font-medium text-ink">{capability.label}</p><p className="mt-1 text-[12px] text-ink-2">{capability.detail}</p></div>
        <Toggle label={capability.label} disabled={!!busy || waiting || permissionsLocked} checked={capabilities.includes(capability.id)} onChange={(checked) => setCapabilities((old) => checked ? [...new Set([...old, capability.id, ...(capability.id === 'calendar.write' ? ['calendar.read' as const] : [])])] : old.filter((value) => value !== capability.id && !(capability.id === 'calendar.read' && value === 'calendar.write')))} />
      </div>)}</div>
      <p className="mt-3 text-[12px] leading-relaxed text-ink-2">{tx("Adding access opens Google consent again for the selected permission set; TensorFold checks the permissions Google actually grants.")}</p>
      <div className="mt-3 flex flex-wrap gap-2">
        <Button size="sm" variant="primary" disabled={!!busy || waiting || !capabilities.length} onClick={() => authorize(capabilities)}>{waiting ? tx("Waiting for Google…") : account ? tx("Reconnect with selected access") : tx("Sign in with Google")}</Button>
        {waiting && <Button size="sm" onClick={cancel}>{tx("Cancel sign-in")}</Button>}
      </div>
    </>}
    {snapshot.authorization?.status === 'failed' && <p role="alert" className="mt-3 text-[13px] text-danger">{tx("Google sign-in failed; retry or check your Desktop client configuration, enabled APIs and test-user access.")}</p>}
    {snapshot.authorization?.status === 'cancelled' && <p role="status" className="mt-3 text-[13px] text-ink-2">{tx("Sign-in cancelled; existing account permissions are unchanged.")}</p>}
    {linkError && <p role="alert" className="mt-3 text-[13px] text-danger">{linkError}</p>}
  </div>
}
