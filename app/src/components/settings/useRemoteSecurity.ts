import { useEffect, useState } from 'react'
import { api } from '../../lib/ipc'
import type { ControllerStatus, RemoteConnectionTestResult, RemoteControlPairingInfo, RemoteControlStatus, RemoteOrchestratorProfile, SecurityControlSnapshot, Settings as TSettings } from '../../lib/types'
import type { SettingsHookDeps } from './types'

// Remote-control + Security logic. Remote owns the Orchestrator pairing/token/
// client-test flow; security owns the local-memory purge + control snapshot.
// The Controller (Accessibility capture) also lives in the Security section, so
// its busy/status/enable handlers ride along here even though it's a distinct
// feature. Receives draft/commit from Settings.tsx and the purgeLocalMemory
// store action (security purge is the one destructive store op in this surface).
export function useRemoteSecurity(
  { draft, commit }: SettingsHookDeps,
  purgeLocalMemory: () => Promise<{ counts: Record<string, number>; preservedCredentialsAndSettings: string[]; preservedResiduals: Array<{ path: string; reason: string }> }>,
) {
  const [controllerBusy, setControllerBusy] = useState(false)
  const [controllerStatus, setControllerStatus] = useState<ControllerStatus | null>(null)
  const [remoteBusy, setRemoteBusy] = useState<'status' | 'pair' | 'regen' | 'test' | ''>('')
  const [remoteStatus, setRemoteStatus] = useState<RemoteControlStatus | null>(null)
  const [remotePairing, setRemotePairing] = useState<RemoteControlPairingInfo | null>(null)
  const [remoteTest, setRemoteTest] = useState<RemoteConnectionTestResult | null>(null)
  const [remoteNotice, setRemoteNotice] = useState('')
  const [securityControls, setSecurityControls] = useState<SecurityControlSnapshot | null>(null)
  const [securityBusy, setSecurityBusy] = useState('')
  const [securityNotice, setSecurityNotice] = useState('')

  const commitController = (controller: TSettings['controller']) => commit({ ...draft, controller })
  const commitRemoteControl = (remoteControl: TSettings['remoteControl']) => {
    commit({ ...draft, remoteControl })
    window.setTimeout(() => refreshRemoteStatus().catch(() => {}), 250)
  }
  const commitRemoteClient = (remoteClient: TSettings['remoteClient']) => commit({ ...draft, remoteClient })
  const commitRemoteOrchestrators = (remoteOrchestrators: RemoteOrchestratorProfile[]) => commit({ ...draft, remoteOrchestrators })
  const refreshControllerStatus = async () => {
    setControllerStatus(await api.getControllerStatus())
  }
  const refreshSecurityControls = async () => {
    setSecurityControls(await api.getSecurityControls())
  }
  const purgeLocalMemoryNow = async () => {
    const total = securityControls ? Object.values(securityControls.localMemoryCounts).reduce<number>((sum, value) => sum + Number(value || 0), 0) : 0
    if (!window.confirm(
      `Permanently purge ${total} local memory record${total === 1 ? '' : 's'} from this machine? App settings and configured provider/cloud credentials remain; linked WhatsApp device credentials are removed. Managed Git worktrees are preserved and reported so unmerged work is not silently deleted.`,
    )) return
    setSecurityBusy('purge')
    setSecurityNotice('')
    try {
      const result = await purgeLocalMemory()
      const purged = Object.values(result.counts).reduce<number>((sum, value) => sum + Number(value || 0), 0)
      const preserved = result.preservedCredentialsAndSettings.length
        ? ` Preserved credentials/settings: ${result.preservedCredentialsAndSettings.join('; ')}.`
        : ''
      const residuals = result.preservedResiduals.length
        ? ` Preserved residuals: ${result.preservedResiduals.map((item) => `${item.path} (${item.reason})`).join('; ')}.`
        : ' No scoped residuals were preserved.'
      setSecurityNotice(`Purged ${purged} local record${purged === 1 ? '' : 's'}.${preserved}${residuals}`)
      await refreshSecurityControls()
    } catch (e: any) {
      setSecurityNotice(String(e?.message || e || 'Purge failed').replace(/^Error invoking remote method '[^']+':\s*/, '').replace(/^Error:\s*/, ''))
    } finally {
      setSecurityBusy('')
    }
  }
  const refreshRemoteStatus = async () => {
    setRemoteBusy((busy) => busy || 'status')
    try { setRemoteStatus(await api.getRemoteStatus()) } finally { setRemoteBusy((busy) => busy === 'status' ? '' : busy) }
  }
  const revealRemotePairing = async () => {
    setRemoteBusy('pair')
    setRemoteNotice('')
    try {
      const info = await api.getRemotePairingInfo()
      setRemotePairing(info)
      setRemoteStatus(info.status)
      const payload = JSON.stringify({ serverUrl: info.url, token: info.token }, null, 2)
      try {
        await api.writeClipboard(payload)
        setRemoteNotice('Pairing details copied.')
      } catch {
        setRemoteNotice('Pairing details ready.')
      }
    } finally {
      setRemoteBusy('')
    }
  }
  const regenerateRemoteToken = async () => {
    setRemoteBusy('regen')
    setRemoteNotice('')
    try {
      const info = await api.regenerateRemoteToken()
      setRemotePairing(info)
      setRemoteStatus(info.status)
      setRemoteNotice('Token regenerated. Update every client that uses this Orchestrator.')
    } finally {
      setRemoteBusy('')
    }
  }
  const testRemoteClient = async () => {
    setRemoteBusy('test')
    setRemoteNotice('')
    try {
      const result = await api.testRemoteClient(draft.remoteClient)
      setRemoteTest(result)
      const now = Date.now()
      const serverUrl = draft.remoteClient.serverUrl.trim().replace(/\/+$/, '')
      const nextSettings = {
        ...draft,
        remoteClient: {
          ...draft.remoteClient,
          lastConnectedAt: result.ok ? now : draft.remoteClient.lastConnectedAt,
          lastError: result.ok ? null : (result.error ?? 'Connection failed.'),
        },
        remoteOrchestrators: draft.remoteOrchestrators,
      }
      if (result.ok && serverUrl) {
        const existing = draft.remoteOrchestrators.find((host) => host.serverUrl.replace(/\/+$/, '') === serverUrl)
        const host: RemoteOrchestratorProfile = {
          id: existing?.id ?? 'orch_' + Math.random().toString(36).slice(2, 9),
          name: result.snapshot?.serverName ?? existing?.name ?? 'TensorFold Orchestrator',
          serverUrl,
          authToken: draft.remoteClient.authToken,
          enabled: true,
          lastConnectedAt: now,
          lastError: null,
          createdAt: existing?.createdAt ?? now,
          updatedAt: now,
        }
        nextSettings.remoteOrchestrators = existing
          ? draft.remoteOrchestrators.map((item) => item.id === existing.id ? host : item)
          : [...draft.remoteOrchestrators, host]
        setRemoteNotice(existing ? 'Orchestrator updated.' : 'Orchestrator saved.')
      }
      commit(nextSettings)
    } finally {
      setRemoteBusy('')
    }
  }
  const enableController = async () => {
    setControllerBusy(true)
    try {
      const status = await api.enableController()
      setControllerStatus(status)
      commitController({ ...draft.controller, enabled: true, lastError: status.canCapture ? null : status.message })
    } finally {
      setControllerBusy(false)
    }
  }
  useEffect(() => { api.getControllerStatus().then(setControllerStatus).catch(() => {}) }, [])
  useEffect(() => { api.getSecurityControls().then(setSecurityControls).catch(() => {}) }, [])
  useEffect(() => { api.getRemoteStatus().then(setRemoteStatus).catch(() => {}) }, [])

  const studioStatus = remoteStatus ?? {
    enabled: draft.remoteControl.enabled,
    listening: false,
    serverName: draft.remoteControl.serverName,
    bindHost: draft.remoteControl.allowLan ? draft.remoteControl.bindHost : '127.0.0.1',
    port: draft.remoteControl.enabled ? draft.remoteControl.port : null,
    url: draft.remoteControl.enabled ? `http://${draft.remoteControl.allowLan ? draft.remoteControl.bindHost : '127.0.0.1'}:${draft.remoteControl.port}` : null,
    clients: 0,
    lastStartedAt: draft.remoteControl.lastStartedAt,
    lastClientAt: draft.remoteControl.lastClientAt,
    lastError: draft.remoteControl.lastError,
  } satisfies RemoteControlStatus
  const localMemoryTotal = securityControls ? Object.values(securityControls.localMemoryCounts).reduce<number>((sum, value) => sum + Number(value || 0), 0) : 0

  return {
    controllerBusy,
    controllerStatus,
    remoteBusy,
    remotePairing,
    remoteTest,
    remoteNotice,
    securityControls,
    securityBusy,
    securityNotice,
    commitRemoteControl,
    commitRemoteClient,
    commitRemoteOrchestrators,
    commitController,
    refreshControllerStatus,
    refreshSecurityControls,
    purgeLocalMemoryNow,
    revealRemotePairing,
    regenerateRemoteToken,
    refreshRemoteStatus,
    testRemoteClient,
    enableController,
    studioStatus,
    localMemoryTotal,
  }
}
