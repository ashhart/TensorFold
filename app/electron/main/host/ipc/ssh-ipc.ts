import type { SshHost } from '../../../agent/types'
import { auditPreview } from '../../../agent/security'
import { saveSettings } from '../../../store'
import { store } from '../../state'
import { auditEvent, requireSecurityRole } from '../audit'
import { handle } from '../ipc'

export interface SshManagerLike {
  listAllHosts(): unknown
  status(hostId: string): unknown
  disconnect(hostId: string): void
  test(hostId: string): Promise<unknown>
  ensureConnected(hostId: string): Promise<unknown>
  resolveSecret(promptId: string, value: string | null): void
  resolveTrust(promptId: string, ok: boolean): void
  listRemoteDir(hostId: string, dir: string): Promise<unknown>
}

export interface SshIpcDeps {
  sshManager: SshManagerLike
}

export function registerSshIpc(deps: SshIpcDeps) {
  const { sshManager } = deps

// --- SSH remote workspaces ---
handle('ssh:listHosts', async () => sshManager.listAllHosts())
handle('ssh:status', async (_e, hostId: string) => sshManager.status(String(hostId)))
handle('ssh:saveHost', async (_e, host: SshHost) => {
  requireSecurityRole('operator', 'manage SSH hosts')
  const now = Date.now()
  const existing = (store.settings.sshHosts ?? []).find((h) => h.id === host.id)
  const next: SshHost = { ...host, origin: 'manual', createdAt: existing?.createdAt ?? now, updatedAt: now }
  const hosts = existing
    ? (store.settings.sshHosts ?? []).map((h) => (h.id === host.id ? next : h))
    : [...(store.settings.sshHosts ?? []), next]
  store.settings = { ...store.settings, sshHosts: hosts }
  await auditEvent('ssh.host_saved', { host: auditPreview(next.hostName), origin: next.origin })
  await saveSettings(store.settings)
  return sshManager.listAllHosts()
})
handle('ssh:removeHost', async (_e, hostId: string) => {
  requireSecurityRole('operator', 'manage SSH hosts')
  const id = String(hostId)
  sshManager.disconnect(id)
  store.settings = { ...store.settings, sshHosts: (store.settings.sshHosts ?? []).filter((h) => h.id !== id) }
  await saveSettings(store.settings)
  return sshManager.listAllHosts()
})
handle('ssh:test', async (_e, hostId: string) => {
  requireSecurityRole('operator', 'connect to SSH hosts')
  return sshManager.test(String(hostId))
})
handle('ssh:connect', async (_e, hostId: string) => {
  requireSecurityRole('operator', 'connect to SSH hosts')
  await sshManager.ensureConnected(String(hostId))
  return sshManager.status(String(hostId))
})
handle('ssh:disconnect', async (_e, hostId: string) => {
  sshManager.disconnect(String(hostId))
  return { ok: true }
})
handle('ssh:resolveSecret', async (_e, promptId: string, value: string | null) => {
  sshManager.resolveSecret(String(promptId), value == null ? null : String(value))
  return { ok: true }
})
handle('ssh:resolveTrust', async (_e, promptId: string, ok: boolean) => {
  sshManager.resolveTrust(String(promptId), !!ok)
  return { ok: true }
})
handle('ssh:listRemoteDir', async (_e, hostId: string, dir: string) => {
  requireSecurityRole('operator', 'browse remote directories')
  return sshManager.listRemoteDir(String(hostId), String(dir || '.'))
})

}
