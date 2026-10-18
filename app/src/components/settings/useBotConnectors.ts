import { useEffect, useRef, useState } from 'react'
import { api } from '../../lib/ipc'
import { applyWhatsAppRuntimeSnapshot, applyWhatsAppStatusPush } from '../../lib/whatsapp-status'
import type { WhatsAppBridgeRuntimeSnapshot, WhatsAppBridgeSource, Settings as TSettings } from '../../lib/types'
import type { ConnectorStatus } from '../../lib/connector-status'
import type { SettingsHookDeps } from './types'

// Telegram / iMessage / WhatsApp bot logic. Telegram + iMessage share the
// generic bot busy/status flags; WhatsApp has its own QR-bridge pairing state
// plus an IPC subscription effect that merges host status pushes into the local
// draft without persisting. Receives draft/commit/setDraft from
// Settings.tsx and borrows flashConnectorStatus from the connector hook so
// iMessage sync results show the same transient confirmation.
export function useBotConnectors(
  { draft, setDraft, commit }: SettingsHookDeps,
  flashConnectorStatus: (key: string, msg: string, state?: ConnectorStatus['state']) => void,
) {
  const [botBusy, setBotBusy] = useState('')
  const [botStatus, setBotStatus] = useState('')
  // iMessage local bridge + WhatsApp QR-bridge busy/feedback state.
  const [imessageBusy, setImessageBusy] = useState(false)
  const [imessageAccess, setImessageAccess] = useState<{ state: string; detail: string } | null>(null)
  const [whatsappBusy, setWhatsappBusy] = useState('')
  const whatsappLocalOnly = api.platform === 'web-remote'
  const [whatsappRuntime, setWhatsappRuntime] = useState<Record<string, WhatsAppBridgeRuntimeSnapshot>>({})
  const whatsappRuntimeRef = useRef<Record<string, WhatsAppBridgeRuntimeSnapshot>>({})
  const [whatsappNow, setWhatsappNow] = useState(Date.now())
  const whatsappQrs = Object.fromEntries(Object.values(whatsappRuntime).filter((state) => state.qr && state.qrExpiresAt && state.qrExpiresAt > whatsappNow).map((state) => [state.sourceId, state.qr!]))
  // Source ids whose bridge is connecting but hasn't yet emitted its first QR/status. The
  // startPairing IPC returns immediately (the bridge runs async), so without this the "Start
  // pairing" button would flick back to idle and the user would see nothing happen while the
  // QR is still being fetched. Current host lifecycle resolves this local pending flag.
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
  // Account edits use owner-gated host patching so local drafts cannot overwrite runtime status.
  const mergeWhatsAppAccounts = (whatsapp: TSettings['whatsapp']) => setDraft((current) => ({ ...current, whatsapp }))
  const whatsappAction = async (busy: string, action: () => Promise<void>) => {
    if (whatsappLocalOnly) {
      flashConnectorStatus('whatsapp', 'Manage linked WhatsApp accounts and contact access in TensorFold on the host Mac.', 'error')
      return
    }
    setWhatsappBusy(busy)
    try { await action() } catch (error) {
      flashConnectorStatus('whatsapp', String(error instanceof Error ? error.message : error), 'error')
    } finally { setWhatsappBusy('') }
  }
  const addWhatsAppAccount = () => whatsappAction('add', async () => { mergeWhatsAppAccounts(await api.addWhatsAppSource('WhatsApp account')) })
  const removeWhatsAppAccount = (id: string) => whatsappAction('remove-' + id, async () => {
    mergeWhatsAppAccounts(await api.removeWhatsAppSource(id))
    const nextRuntime = { ...whatsappRuntimeRef.current }; delete nextRuntime[id]; whatsappRuntimeRef.current = nextRuntime
    setWhatsappRuntime((current) => { const next = { ...current }; delete next[id]; return next })
  })
  const updateWhatsAppAccount = (id: string, patch: Partial<Pick<WhatsAppBridgeSource, 'label' | 'enabled' | 'allowedChatJids' | 'allowSelfChat'>>) => whatsappAction('update-' + id, async () => {
    mergeWhatsAppAccounts(await api.updateWhatsAppSource(id, patch))
  })
  const disconnectWhatsAppAccount = (id: string) => whatsappAction('disconnect-' + id, async () => {
    await api.disconnectWhatsApp(id)
    setDraft((current) => applyWhatsAppStatusPush(current, { sourceId: id, enabled: false, lastError: null }))
  })
  const pairWhatsApp = (id: string) => whatsappAction('pair-' + id, async () => {
    setWhatsappPairing((current) => ({ ...current, [id]: true }))
    try { await api.startWhatsAppPairing(id) } catch (error) {
      setWhatsappPairing((current) => ({ ...current, [id]: false }))
      throw error
    }
  })
  useEffect(() => {
    // QR credentials and local runtime state never cross the paired web transport.
    // Settings also mounts this hook for unrelated sections: do not request denied IPC.
    if (whatsappLocalOnly) return
    let mounted = true
    const accept = (snapshot: WhatsAppBridgeRuntimeSnapshot) => {
      if (!mounted) return
      const next = applyWhatsAppRuntimeSnapshot(whatsappRuntimeRef.current, snapshot)
      if (next === whatsappRuntimeRef.current) return
      whatsappRuntimeRef.current = next
      setWhatsappRuntime(whatsappRuntimeRef.current)
      setWhatsappPairing((current) => ({ ...current, [snapshot.sourceId]: snapshot.phase === 'connecting' || snapshot.phase === 'reconnecting' }))
    }
    // Subscribe before the snapshot request, then retain newer push timestamps.
    const offRuntime = api.onWhatsAppRuntime(accept)
    const offStatus = api.onWhatsAppStatus((status) => {
      if (mounted) setDraft((current) => applyWhatsAppStatusPush(current, status))
    })
    void api.whatsappRuntime().then((snapshots) => { snapshots.forEach(accept) }).catch((error) => {
      if (mounted) flashConnectorStatus('whatsapp', String(error instanceof Error ? error.message : error), 'error')
    })
    const expiry = setInterval(() => { if (mounted) setWhatsappNow(Date.now()) }, 1000)
    return () => { mounted = false; offRuntime(); offStatus(); clearInterval(expiry) }
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
    whatsappLocalOnly,
    whatsappQrs,
    whatsappRuntime,
    whatsappNow,
    updateWhatsAppAccount,
    disconnectWhatsAppAccount,
    whatsappPairing,
    addWhatsAppAccount,
    removeWhatsAppAccount,
    pairWhatsApp,
    commitTelegram,
    commitWhatsapp,
    pollTelegram,
  }
}
