import { OriginalDiagnostic } from '../OriginalDiagnostic'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { GoogleWorkspaceConnector } from './connectors/GoogleWorkspaceConnector'
import { TelegramConnector } from './connectors/TelegramConnector'
import { useEffect, useState } from 'react'
import { CONNECTOR_CATALOG, CONNECTOR_SETUP_IDS, type ConnectorSetupId } from '../../../shared/connector-catalog'
import { consumeConnectorSetup, SETTINGS_SECTION_REQUEST_EVENT } from '../../lib/store'
import { api } from '../../lib/ipc'
import { BrandIcon, connectorBrand } from '../../lib/brand'
import type { AssistantControlSnapshot } from '../../lib/types'
import { Button } from '../ui'
import { Icon } from '../../lib/icons'
import { SettingsPane, type SettingsSectionContext } from './shared'
import { EmailConnector } from './connectors/EmailConnector'
import { SlackConnector } from './connectors/SlackConnector'
import { DiscordConnector } from './connectors/DiscordConnector'
import { MattermostConnector } from './connectors/MattermostConnector'
import { MatrixConnector } from './connectors/MatrixConnector'
import { GoogleChatConnector } from './connectors/GoogleChatConnector'
import { DocumentsConnector } from './connectors/DocumentsConnector'
import { BrowserConnector } from './connectors/BrowserConnector'
import { BotControlConnector } from './connectors/BotControlConnector'
import { WhatsAppBridgeConnector } from './connectors/WhatsAppBridgeConnector'

export function ConnectorsSettings({ ctx }: { ctx: SettingsSectionContext }) {
  const tx = useAuthoredText()
  const [selected, setSelected] = useState<ConnectorSetupId | null>(() => consumeConnectorSetup())
  const [snapshot, setSnapshot] = useState<AssistantControlSnapshot | null>(null)
  const [googleStatus, setGoogleStatus] = useState('Set up')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const refresh = async () => {
    setBusy(true)
    try {
      const [assistant, google] = await Promise.all([api.getAssistantSnapshot(), api.googleWorkspaceStatus()])
      setSnapshot(assistant)
      setGoogleStatus(google.accounts.some((account) => account.status === 'connected') ? 'Connected' : google.accounts.length ? 'Needs attention' : 'Set up')
      setError('')
    }
    catch (reason) { setError(reason instanceof Error ? reason.message : tx("Connection status could not be read.")) }
    finally { setBusy(false) }
  }
  useEffect(() => {
    let active = true
    void api.getAssistantSnapshot().then((next) => { if (active) setSnapshot(next) }).catch((reason) => { if (active) setError(reason instanceof Error ? reason.message : tx("Connection status could not be read.")) })
    void api.googleWorkspaceStatus().then((google) => { if (active) setGoogleStatus(google.accounts.some((account) => account.status === 'connected') ? 'Connected' : google.accounts.length ? 'Needs attention' : 'Set up') }).catch(() => { if (active) setGoogleStatus('Status unavailable') })
    return () => { active = false }
  }, [selected])
  useEffect(() => {
    const requested = () => { const id = consumeConnectorSetup(); if (id) setSelected(id) }
    window.addEventListener(SETTINGS_SECTION_REQUEST_EVENT, requested)
    return () => window.removeEventListener(SETTINGS_SECTION_REQUEST_EVENT, requested)
  }, [])
  return <SettingsPane>
    <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
      <div>
        {selected && <Button variant="ghost" size="sm" icon="back" onClick={() => setSelected(null)}>{tx("All connections")}</Button>}
        <h2 className="mt-2 text-[16px] font-semibold text-ink">{selected ? CONNECTOR_CATALOG[selected].name : tx("Connections")}</h2>
        <p className="mt-1 max-w-[600px] text-[13px] leading-relaxed text-ink-2">{selected ? tx("Connect your account, choose the sources it may read, then use it in Assistant or a bot routine.") : tx("Choose a service to set it up, or reconnect an existing account.")}</p>
      </div>
      {!selected && <Button size="sm" icon="refresh" disabled={busy} onClick={() => void refresh()}>{busy ? tx("Checking…") : tx("Refresh status")}</Button>}
    </div>
    {error && <p role="alert" className="mb-4 text-[13px] text-danger">{<OriginalDiagnostic>{error}</OriginalDiagnostic>}</p>}
    {!selected && <div className="grid gap-3 sm:grid-cols-2">{CONNECTOR_SETUP_IDS.map((id) => {
      const metadata = CONNECTOR_CATALOG[id]
      const state = snapshot?.connectors.find((connector) => connector.id === id)
      const status = id === 'google' ? googleStatus : state?.status === 'connected' ? tx("Connected") : state?.status === 'error' ? tx("Needs attention") : tx("Set up")
      return <button key={id} onClick={() => setSelected(id)} className="flex min-w-0 items-start gap-3 rounded-xl border border-line bg-card p-4 text-left transition-colors hover:border-line-strong focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent">
        {id === 'google' ? <span aria-hidden="true" className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-card-2 text-accent"><Icon name="mail" size={20} /></span> : <BrandIcon brand={connectorBrand(id)} size={32} />}
        <span className="min-w-0 flex-1"><span className="flex flex-wrap items-center justify-between gap-2"><span className="text-[14px] font-medium text-ink">{metadata.name}</span><span className="text-[12px] text-ink-2">{status}</span></span><span className="mt-1 block text-[13px] leading-relaxed text-ink-2">{metadata.description}</span>{state?.error && <span className="mt-2 block break-words text-[12px] text-danger">{<OriginalDiagnostic>{state.error}</OriginalDiagnostic>}</span>}</span>
      </button>
    })}</div>}
    {selected === 'google' && <GoogleWorkspaceConnector />}
    {selected === 'telegram' && <TelegramConnector ctx={ctx} />}
    {selected === 'email' && <EmailConnector ctx={ctx} />}
    {selected === 'slack' && <SlackConnector ctx={ctx} />}
    {selected === 'discord' && <DiscordConnector ctx={ctx} />}
    {selected === 'mattermost' && <MattermostConnector ctx={ctx} />}
    {selected === 'matrix' && <MatrixConnector ctx={ctx} />}
    {selected === 'googlechat' && <GoogleChatConnector ctx={ctx} />}
    {selected === 'docs' && <DocumentsConnector ctx={ctx} />}
    {selected === 'browser' && <BrowserConnector ctx={ctx} />}
    {selected === 'imessage' && <BotControlConnector ctx={ctx} focus={selected} />}
    {selected === 'whatsapp' && <><WhatsAppBridgeConnector ctx={ctx} /><details className="mt-5 rounded-xl border border-line p-4"><summary className="cursor-pointer text-[13px] font-medium text-ink">{tx("Advanced: WhatsApp Business Cloud")}</summary><p className="my-3 text-[13px] text-ink-2">{tx("This separate route needs Meta Business credentials and a verified public webhook.")}</p><BotControlConnector ctx={ctx} focus="whatsapp-cloud" /></details></>}
  </SettingsPane>
}
