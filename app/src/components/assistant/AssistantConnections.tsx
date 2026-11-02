import { useEffect, useState } from 'react'
import { MailConnectionHealthPanel } from './MailConnectionHealthPanel'
import { CONNECTOR_CATALOG, CONNECTOR_SETUP_IDS, type ConnectorSetupId } from '../../../shared/connector-catalog'
import { api } from '../../lib/ipc'
import { requestConnectorSetup, useStore } from '../../lib/store'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { OriginalDiagnostic } from '../OriginalDiagnostic'
import { Button } from '../ui'
import type { AssistantControlSnapshot, Settings } from '../../lib/types'

export function AssistantConnections({ onGoogle }: { onGoogle: () => void }) {
  const tx = useAuthoredText()
  const setView = useStore(state => state.setView)
  const [snapshot, setSnapshot] = useState<AssistantControlSnapshot | null>(null)
  const [settings, setSettings] = useState<Settings | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const refresh = async () => {
    setBusy(true)
    try { const [next, saved] = await Promise.all([api.getAssistantSnapshot(), api.getSettings()]); setSnapshot(next); setSettings(saved); setError('') }
    catch { setError(tx('Connection status could not be read.')) }
    finally { setBusy(false) }
  }
  useEffect(() => { void refresh() }, [])
  const open = (id: ConnectorSetupId) => {
    if (id === 'google') { onGoogle(); return }
    requestConnectorSetup(id); setView({ name: 'settings', section: 'connectors' })
  }
  return <div className="mx-auto max-w-[1000px] space-y-4 p-3 sm:p-5">
    <div className="flex flex-wrap items-center justify-between gap-2"><h2 className="text-[16px] font-semibold text-ink">{tx('Connections')}</h2><Button size="sm" disabled={busy} onClick={() => void refresh()}>{tx('Refresh status')}</Button></div>
    <p className="text-[13px] text-ink-2">{tx('Connect accounts here, then sync selected conversations in Inbox. Messages are context, never permission to run commands or contact someone.')}</p>
    {error && <p role="alert" className="text-[13px] text-danger"><OriginalDiagnostic>{error}</OriginalDiagnostic></p>}
    <MailConnectionHealthPanel />
    <div className="grid gap-3 sm:grid-cols-2">{CONNECTOR_SETUP_IDS.map(id => {
      const metadata = CONNECTOR_CATALOG[id]
      const state = snapshot?.connectors.find(connector => connector.id === (id === 'outlook' ? 'email' : id))
      const configured = id === 'outlook' ? settings?.email.sources.some(source => source.kind === 'microsoft_graph' && source.enabled) : id === 'google' ? false : state?.status === 'connected'
      return <button key={id} onClick={() => open(id)} className="min-w-0 rounded-xl border border-line bg-card p-4 text-left focus-visible:outline-2 focus-visible:outline-accent">
        <span className="flex flex-wrap justify-between gap-2 text-[14px] font-medium text-ink"><span>{metadata.name}</span><span className="text-[12px] text-ink-3">{id === 'google' ? tx('Open Google Workspace') : configured ? tx('Configured') : tx('Set up')}</span></span>
        <span className="mt-1 block text-[13px] text-ink-2">{tx(metadata.description)}</span>
        {id === 'imessage' && <span className="mt-2 block text-[12px] text-ink-3">{tx('Direct access on Mac, or BlueBubbles through a Mac from any desktop platform.')}</span>}
      </button>
    })}</div>
  </div>
}
