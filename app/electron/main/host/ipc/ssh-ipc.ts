import type { Settings, SshHost } from '../../../agent/types'
import { auditPreview } from '../../../agent/security'
import { saveSettings, vontraHomeDir } from '../../../store'
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
  reserveOperation?: () => { release(): void }
  profileRoot?: () => string
  /** Share core settingsChanges.run('settings', operation) when provided. */
  serializeSettings?: <T>(operation: () => Promise<T>) => Promise<T>
}
function text(value: unknown, limit: number, label: string, empty = false): string {
  if (typeof value !== 'string' || value.length > limit || !empty && !value.trim() || /[\u0000-\u001f\u007f]/.test(value)) throw new Error(`Invalid SSH ${label}.`)
  return value
}
function identity(value: unknown): string { const result = text(value, 320, 'identity'); if (!/^[A-Za-z0-9][A-Za-z0-9._:-]*$/.test(result)) throw new Error('Invalid SSH identity.'); return result }
export function validateSshHostInput(value: unknown): SshHost {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Invalid SSH host.')
  const raw = value as Record<string, unknown>
  if (Object.keys(raw).some(key => !['id','label','configAlias','hostName','user','port','identityFiles','proxyJump','forwardAgent','lastRemoteDir','platform','origin','createdAt','updatedAt'].includes(key))) throw new Error('Unsupported SSH host field.')
  const hostName = text(raw.hostName, 253, 'hostname')
  if (!/^[A-Za-z0-9_.:[\]%-]+$/.test(hostName) || hostName.startsWith('-') || !Number.isInteger(raw.port) || Number(raw.port) < 1 || Number(raw.port) > 65_535 || !Array.isArray(raw.identityFiles) || raw.identityFiles.length > 16) throw new Error('Choose a valid SSH hostname, port and key file list.')
  for (const key of ['createdAt','updatedAt']) if (!Number.isSafeInteger(raw[key]) || Number(raw[key]) < 0) throw new Error('Invalid SSH host revision.')
  if (raw.forwardAgent !== undefined && typeof raw.forwardAgent !== 'boolean' || !['manual','config'].includes(String(raw.origin))) throw new Error('Invalid SSH host options.')
  const optional = (key: string, limit: number, nullable = false) => raw[key] === undefined ? {} : raw[key] === null && nullable ? { [key]: null } : { [key]: text(raw[key], limit, key, true) }
  return { id: identity(raw.id), label: text(raw.label, 160, 'label'), hostName, port: Number(raw.port), identityFiles: raw.identityFiles.map(item => text(item, 16_384, 'key file')), origin: raw.origin as SshHost['origin'], createdAt: Number(raw.createdAt), updatedAt: Number(raw.updatedAt), ...optional('configAlias', 320, true), ...optional('user', 320), ...optional('proxyJump', 1024), ...(raw.forwardAgent === undefined ? {} : { forwardAgent: raw.forwardAgent as boolean }), ...optional('lastRemoteDir', 16_384, true), ...optional('platform', 160, true) }
}
export function registerSshIpc(deps: SshIpcDeps) {
  const { sshManager } = deps, profileRoot = deps.profileRoot ?? vontraHomeDir
  let tail: Promise<unknown> = Promise.resolve()
  const serialized = <T>(operation: () => Promise<T>): Promise<T> => {
    if (deps.serializeSettings) return deps.serializeSettings(operation)
    const next = tail.catch(() => {}).then(operation); tail = next.catch(() => {}); return next
  }
  const guarded = async <T>(action: () => Promise<T>, label: string): Promise<T> => {
    requireSecurityRole('operator', label)
    const root = profileRoot(), lease = deps.reserveOperation?.()
    try {
      if (profileRoot() !== root) throw new Error('The SSH profile changed; refresh before retrying.')
      const result = await action()
      requireSecurityRole('operator', label)
      if (profileRoot() !== root) throw new Error('The SSH profile changed; refresh before retrying.')
      return result
    } finally { lease?.release() }
  }
  const mutate = async (change: (current: Settings) => Settings, after?: () => void) => {
    const root = profileRoot()
    return guarded(() => serialized(async () => {
      const previous = store.settings, original = JSON.stringify(previous)
      const validateCommit = () => {
        requireSecurityRole('operator', 'manage SSH hosts')
        if (profileRoot() !== root || store.settings !== previous || JSON.stringify(store.settings) !== original) throw new Error('SSH settings or profile changed; refresh before retrying.')
      }
      validateCommit(); const next = change(previous)
      await saveSettings(next, { validateCommit }); validateCommit()
      store.settings = next; after?.()
      return sshManager.listAllHosts()
    }), 'manage SSH hosts')
  }
  handle('ssh:listHosts', async () => sshManager.listAllHosts())
  handle('ssh:status', async (_e, hostId: unknown) => sshManager.status(identity(hostId)))
  handle('ssh:saveHost', async (_e, input: unknown) => {
    requireSecurityRole('operator', 'manage SSH hosts')
    const host = validateSshHostInput(input)
    const result = await mutate(current => {
      const existing = (current.sshHosts ?? []).find(row => row.id === host.id)
      if (existing && existing.updatedAt !== host.updatedAt) throw new Error('This SSH host changed; refresh it before saving.')
      if (!existing && (current.sshHosts?.length ?? 0) >= 128) throw new Error('Keep at most 128 saved SSH hosts.')
      const now = Date.now(), next: SshHost = { ...host, origin: 'manual', createdAt: existing?.createdAt ?? now, updatedAt: now }
      return { ...current, sshHosts: existing ? current.sshHosts!.map(row => row.id === next.id ? next : row) : [...(current.sshHosts ?? []), next] }
    })
    await auditEvent('ssh.host_saved', { host: auditPreview(host.hostName), origin: 'manual' })
    return result
  })
  handle('ssh:removeHost', async (_e, hostId: unknown) => { const id = identity(hostId); return mutate(current => ({ ...current, sshHosts: (current.sshHosts ?? []).filter(row => row.id !== id) }), () => sshManager.disconnect(id)) })
  handle('ssh:test', async (_e, hostId: unknown) => { const id = identity(hostId); return guarded(() => sshManager.test(id), 'connect to SSH hosts') })
  handle('ssh:connect', async (_e, hostId: unknown) => { const id = identity(hostId); return guarded(async () => { await sshManager.ensureConnected(id); return sshManager.status(id) }, 'connect to SSH hosts') })
  handle('ssh:disconnect', async (_e, hostId: unknown) => guarded(async () => { sshManager.disconnect(identity(hostId)); return { ok: true } }, 'disconnect SSH hosts'))
  handle('ssh:resolveSecret', async (_e, promptId: unknown, value: unknown) => {
    requireSecurityRole('operator', 'answer an SSH credential prompt')
    if (value !== null && (typeof value !== 'string' || value.length > 16_384 || value.includes('\0'))) throw new Error('Invalid SSH credential response.')
    sshManager.resolveSecret(identity(promptId), value as string | null); return { ok: true }
  })
  handle('ssh:resolveTrust', async (_e, promptId: unknown, ok: unknown) => { requireSecurityRole('operator', 'answer an SSH host trust prompt'); if (typeof ok !== 'boolean') throw new Error('Choose an explicit SSH trust decision.'); sshManager.resolveTrust(identity(promptId), ok); return { ok: true } })
  handle('ssh:listRemoteDir', async (_e, hostId: unknown, dir: unknown) => { const id = identity(hostId), directory = text(dir, 16_384, 'directory', true) || '.'; return guarded(() => sshManager.listRemoteDir(id, directory), 'browse remote directories') })
}
