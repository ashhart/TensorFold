import { useEffect, useState } from 'react'
import { api } from '../../lib/ipc'
import { applyWhatsAppStatusPush } from '../../lib/whatsapp-status'
import type { Settings as TSettings } from '../../lib/types'
import type { ConnectorStatus } from '../../lib/connector-status'
import type { SettingsHookDeps } from './types'

// Telegram / iMessage / WhatsApp bot logic. Telegram + iMessage share the
// generic bot busy/status flags; WhatsApp has its own QR-bridge pairing state
// plus an IPC subscription effect that merges host status pushes into the local
// draft without persisting. Receives draft/commit/setDraft/draftRef from
// Settings.tsx and borrows flashConnectorStatus from the connector hook so
// iMessage sync results show the same transient confirmation.
export function useBotConnectors(
  { draft, setDraft, commit, draftRef }: SettingsHookDeps,
  flashConnectorStatus: (key: string, msg: string, state?: ConnectorStatus['state']) => void,
) {
  const [botBusy, setBotBusy] = useState('')
  const [botStatus, setBotStatus] = useState('')
  // iMessage local bridge + WhatsApp QR-bridge busy/feedback state.
  const [imessageBusy, setImessageBusy] = useState(false)
  const [imessageAccess, setImessageAccess] = useState<{ state: string; detail: string } | null>(null)
  const [whatsappBusy, setWhatsappBusy] = useState('')
  const [whatsappQrs, setWhatsappQrs] = useState<Record<string, string>>({})
  // Source ids whose bridge is connecting but hasn't yet emitted its first QR/status. The
  // startPairing IPC returns immediately (the bridge runs async), so without this the "Start
  // pairing" button would flick back to idle and the user would see nothing happen while the
  // QR is still being fetched. Cleared when a QR arrives, a status lands, or pairing errors.
  const [whatsappPairing, setWhatsappPairing] = useState<Record<string, boolean>>({})

  const commitBotControl = (botControl: TSettings['botControl']) => commit({ ...draft, botControl })
  const commitTelegram = (telegram: TSettings['botControl']['telegram']) => commitBotControl({ ...draft.botControl, telegram })
  const commitWhatsapp = (whatsapp: TSettings['botControl']['whatsapp']) => commitBotControl({ ...draft.botControl, whatsapp })
  const pollTelegram = async () => {
    setBotBusy('telegram')
    setBotStatus('')
    try {
      const result = await api.pollTelegram()
      setBotStatus(result.ok ? `Processed ${result.processed}, skipped ${result.skipped}` : (result.error || 'Telegram poll failed'))
      commitTelegram({ ...draft.botControl.telegram, lastPollAt: Date.now(), lastError: result.error, lastUpdateId: result.lastUpdateId })
    } catch (e: any) {
      setBotStatus(String(e?.message || e || 'Telegram poll failed').replace(/^Error invoking remote method '[^']+':\s*/, '').replace(/^Error:\s*/, ''))
    } finally {
      setBotBusy('')
    }
  }
  // iMessage local bridge: check Full Disk Access, toggle enable, and sync.
  const checkImessageAccess = async () => {
    setImessageBusy(true)
    try { setImessageAccess(await api.imessageAccess()) } finally { setImessageBusy(false) }
  }
  const commitImessage = (imessage: TSettings['imessage']) => commit({ ...draft, imessage })
  const toggleImessage = (enabled: boolean) => commitImessage({ ...draft.imessage, enabled })
  const syncImessage = async () => {
    setImessageBusy(true)
    try {
      await api.syncImessage()
      flashConnectorStatus('imessage', 'iMessage synced')
    } catch (e: any) {
      flashConnectorStatus('imessage', e?.message ?? 'Sync failed', 'error')
    } finally {
      setImessageBusy(false)
    }
  }
  // WhatsApp QR-bridge: multi-account pairing.
  const commitWhatsappBridge = (whatsapp: TSettings['whatsapp']) => commit({ ...draft, whatsapp })
  const addWhatsAppAccount = async () => {
    setWhatsappBusy('add')
    try { commitWhatsappBridge(await api.addWhatsAppSource('WhatsApp account')) } finally { setWhatsappBusy('') }
  }
  const removeWhatsAppAccount = async (id: string) => {
    setWhatsappBusy('remove')
    try { commitWhatsappBridge(await api.removeWhatsAppSource(id)) } finally { setWhatsappBusy('') }
  }
  const pairWhatsApp = async (id: string) => {
    setWhatsappBusy('pair-' + id)
    // Clear any stale error and mark this account as connecting until the first QR/status lands.
    setWhatsappPairing((p) => ({ ...p, [id]: true }))
    commitWhatsappBridge({ ...draftRef!.current.whatsapp, sources: draftRef!.current.whatsapp.sources.map((s) => s.id === id ? { ...s, lastError: null } : s) })
    try { await api.startWhatsAppPairing(id) } finally { setWhatsappBusy('') }
  }
  useEffect(() => {
    const offQr = api.onWhatsAppQr(({ sourceId, qr }) => {
      setWhatsappQrs((q) => ({ ...q, [sourceId]: qr }))
      // First QR resolves the "connecting" state — the scannable code is now showing.
      setWhatsappPairing((p) => { const next = { ...p }; delete next[sourceId]; return next })
    })
    const offStatus = api.onWhatsAppStatus(({ sourceId, paired, lastError }) => {
      // The host already persisted this runtime status. Merge it into the latest LOCAL
      // draft only: calling commit() here used to clear dirtyRef and save every in-progress
      // field merely because a QR socket emitted a background status update.
      setDraft((current) => applyWhatsAppStatusPush(current, { sourceId, paired, lastError }))
      // Any status (paired OR an error) ends the "connecting" state so the card stops spinning.
      setWhatsappPairing((p) => { const next = { ...p }; delete next[sourceId]; return next })
      if (paired) setWhatsappQrs((q) => { const next = { ...q }; delete next[sourceId]; return next })
    })
    return () => { offQr(); offStatus() }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  return {
    botBusy,
    botStatus,
    imessageBusy,
    imessageAccess,
    checkImessageAccess,
    toggleImessage,
    syncImessage,
    whatsappBusy,
    whatsappQrs,
    whatsappPairing,
    addWhatsAppAccount,
    removeWhatsAppAccount,
    pairWhatsApp,
    commitTelegram,
    commitWhatsapp,
    pollTelegram,
  }
}
