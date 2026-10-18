import { createHash, randomUUID } from 'node:crypto'
import path from 'node:path'
import type { ClusterBootstrapPreview, ClusterSshKey } from '../../../../shared/cluster-workspace'
import type { SshHost } from '../../../agent/types'
import type { SshExecOptions, SshExecResult } from '../../../agent/ssh-connection'
import type { ClusterForwardSession } from './ssh-forward'
import { createManagedSshKeyStore, type ClusterKeyDeps } from './ssh-key'
import { SSH_PREREQUISITE_PROGRAM, SSH_INSTALL_PROGRAM, SSH_CREDENTIAL_PROGRAM } from './ssh-install-script'

export interface ClusterAgentBundle { bytes: Buffer; sha256: string; version: '1'; entry: 'cluster-node.js' }
export interface ClusterSetupSession extends ClusterForwardSession { readonly host: SshHost; exec(command: string, options?: SshExecOptions): Promise<SshExecResult> }
export interface ClusterSshInspection {
  platform: 'darwin' | 'linux' | 'unknown'; nodeMissing?: boolean; arch: string; home: string; uid: number; nodePath: string; nodeVersion: string
  enginePath: string | null; engineVersion: string | null; supervisor: 'launchd-user' | 'systemd-user' | null
  serviceReady: boolean; lingering: boolean; homeOwned: boolean; serviceExists: boolean; unsafeExisting: boolean
  existing: { schema: 1; bundleSha256: string; port: number; enginePath: string; nodePath: string; state: string } | null
}
export interface ClusterBootstrapRecord {
  preview: ClusterBootstrapPreview; profileRoot: string; hostStamp: string; port: number; bundleSha256: string
  inspection: ClusterSshInspection
}
export interface ClusterBootstrapApplyResult {
  sshHostId: string; hostStamp: string; port: number; token: string; nodeId: string; verified: true
  service: 'launchd-user' | 'systemd-user'; reused: boolean
}
export interface ClusterSshSetupDeps {
  profileRoot(): string
  getSshHost(id: string): Promise<SshHost | null>
  sessionForHost(id: string): Promise<ClusterSetupSession>
  agentBundle(): Promise<ClusterAgentBundle>
  /** Store internally in protected profile state; never project inspection or profileRoot. */
  persistPreview(record: ClusterBootstrapRecord): Promise<void>
  loadPreview(id: string): Promise<ClusterBootstrapRecord | null>
  markApplied(previewId: string, commandId: string, result: { state: 'applying' | 'needs-review' | 'verified'; nodeId?: string; message?: string }): Promise<void>
  verifyAgent(input: { sshHostId: string; port: number; token: string; signal?: AbortSignal; session: ClusterSetupSession; assertCurrent: () => void }): Promise<{ protocol: 1; nodeId: string; authenticated: true }>
  key?: Omit<ClusterKeyDeps, 'profileRoot'>
  now?: () => number; newId?: () => string; delay?: (ms: number) => Promise<void>
}
const sha = (value: unknown) => createHash('sha256').update(JSON.stringify(value)).digest('hex')
const safePath = (value: unknown): value is string => typeof value === 'string' && value.startsWith('/') && value.length <= 16_384 && !/[\0\r\n]/.test(value)
const shell = (value: string) => `'${value.replace(/'/g, `'\\''`)}'`
const command = (program: string, node = 'node') => `${shell(node)} -e ${shell(program)}`
export function clusterSshHostStamp(host: SshHost): string { return sha([host.id, host.hostName, host.user, host.port, host.identityFiles, host.proxyJump, host.forwardAgent, host.configAlias, host.updatedAt]) }
function validBundle(bundle: ClusterAgentBundle): void {
  if (bundle.version !== '1' || bundle.entry !== 'cluster-node.js' || !Buffer.isBuffer(bundle.bytes) || !bundle.bytes.length || bundle.bytes.length > 4 * 1024 * 1024 || createHash('sha256').update(bundle.bytes).digest('hex') !== bundle.sha256) throw new Error('The app-owned headless agent bundle is missing, too large or failed integrity verification.')
}
function parseInspection(result: SshExecResult): ClusterSshInspection {
  if (result.code === 42 && /^TENSORFOLD_NODE_MISSING\r?\n(?:Darwin|Linux)?\r?\n?$/.test(result.stdout)) return { platform: result.stdout.includes('Darwin') ? 'darwin' : result.stdout.includes('Linux') ? 'linux' : 'unknown', nodeMissing: true, arch: 'unknown', home: '/unavailable/home', uid: -1, nodePath: '/unavailable/node', nodeVersion: '0.0.0', enginePath: null, engineVersion: null, supervisor: null, serviceReady: false, lingering: false, homeOwned: false, serviceExists: false, unsafeExisting: false, existing: null }
  if (result.code !== 0 || Buffer.byteLength(result.stdout) > 65_536) throw new Error('SSH prerequisite inspection could not be completed; no setup was applied.')
  let raw: any
  try { raw = JSON.parse(result.stdout) } catch { throw new Error('The remote prerequisite report is malformed; no setup was applied.') }
  if (!raw || !['darwin', 'linux'].includes(raw.platform) || typeof raw.arch !== 'string' || !safePath(raw.home) || !safePath(raw.nodePath) || typeof raw.nodeVersion !== 'string' || !/^\d+\.\d+\.\d+(?:[-+].*)?$/.test(raw.nodeVersion) || !Number.isSafeInteger(raw.uid) || raw.uid < 0 || raw.enginePath !== null && !safePath(raw.enginePath) || raw.engineVersion !== null && (typeof raw.engineVersion !== 'string' || raw.engineVersion.length > 4096) || ![null, 'launchd-user', 'systemd-user'].includes(raw.supervisor)) throw new Error('The remote prerequisite report has invalid or unsupported values.')
  for (const key of ['serviceReady', 'lingering', 'homeOwned', 'serviceExists', 'unsafeExisting']) if (typeof raw[key] !== 'boolean') throw new Error('The remote prerequisite report is incomplete.')
  const existing = raw.existing
  if (existing !== null && (!existing || existing.schema !== 1 || !/^[a-f0-9]{64}$/.test(existing.bundleSha256) || !Number.isInteger(existing.port) || !safePath(existing.enginePath) || !safePath(existing.nodePath) || typeof existing.state !== 'string')) throw new Error('The existing managed installation is not valid; preserve it for deliberate repair.')
  return { platform: raw.platform, arch: raw.arch.slice(0, 40), home: raw.home, uid: raw.uid, nodePath: raw.nodePath, nodeVersion: raw.nodeVersion, enginePath: raw.enginePath, engineVersion: raw.engineVersion, supervisor: raw.supervisor, serviceReady: raw.serviceReady, lingering: raw.lingering, homeOwned: raw.homeOwned, serviceExists: raw.serviceExists, unsafeExisting: raw.unsafeExisting, existing }
}
function recordDigest(record: Omit<ClusterBootstrapRecord, 'preview'>, preview: Omit<ClusterBootstrapPreview, 'digest'>): string { return sha({ ...record, preview }) }

export function createClusterSshSetup(deps: ClusterSshSetupDeps) {
  const now = deps.now ?? Date.now, newId = deps.newId ?? randomUUID, delay = deps.delay ?? (ms => new Promise(resolve => setTimeout(resolve, ms)))
  const keys = createManagedSshKeyStore({ ...deps.key, profileRoot: deps.profileRoot })
  const applying = new Set<string>()
  const capture = async (id: string, expected?: string) => {
    if (typeof id !== 'string' || !id || id.length > 320) throw new Error('Choose a saved SSH host.')
    const profileRoot = deps.profileRoot(), host = await deps.getSshHost(id)
    if (!host || host.id !== id || deps.profileRoot() !== profileRoot) throw new Error('The saved SSH host or active profile changed.')
    const stamp = clusterSshHostStamp(host)
    if (expected && expected !== stamp) throw new Error('The saved SSH host changed after preview; inspect it again.')
    const current = async () => {
      if (deps.profileRoot() !== profileRoot || clusterSshHostStamp((await deps.getSshHost(id)) ?? { id: '' } as SshHost) !== stamp) throw new Error('The saved SSH host or active profile changed during setup.')
    }
    return { profileRoot, stamp, current }
  }
  return {
    key(input: { action: 'inspect' | 'create' }): Promise<ClusterSshKey> {
      if (input?.action === 'inspect') return keys.inspect()
      if (input?.action === 'create') return keys.create()
      return Promise.reject(new Error('Choose inspect or create for the app-owned SSH key.'))
    },
    async preview(input: { sshHostId: string; port: number }): Promise<ClusterBootstrapPreview> {
      if (!input || !Number.isInteger(input.port) || input.port < 1 || input.port > 65_535) throw new Error('Choose a valid management loopback port.')
      const owner = await capture(input.sshHostId), bundle = await deps.agentBundle(); validBundle(bundle); await owner.current()
      const session = await deps.sessionForHost(input.sshHostId); await owner.current()
      if (clusterSshHostStamp(session.host) !== owner.stamp) throw new Error('The SSH session belongs to an older or different saved host; disconnect and inspect again.')
      const result = await session.exec(`command -v node >/dev/null 2>&1 || { printf '%s\\n' 'TENSORFOLD_NODE_MISSING'; uname -s; exit 42; }; ${command(SSH_PREREQUISITE_PROGRAM)}`, { timeoutMs: 20_000, maxBuffer: 65_536 })
      await owner.current()
      const inspection = parseInspection(result), blockers: string[] = []
      if (Number(inspection.nodeVersion.split('.')[0]) < 22) blockers.push('Install Node.js 22 or later on this host, then inspect again; TensorFold does not install a runtime automatically.')
      if (!inspection.nodeMissing && (!inspection.enginePath || !inspection.engineVersion)) blockers.push('Install tensorfold-native on this host, then inspect again; no engine or driver is installed automatically.')
      if (!inspection.nodeMissing && (!inspection.homeOwned || inspection.uid === 0)) blockers.push('Use a non-root SSH account with its own home directory for the reviewed per-user service.')
      if (!inspection.nodeMissing && (!inspection.serviceReady || !inspection.supervisor)) blockers.push('Start or configure the user service manager manually before applying this setup.')
      if (!inspection.nodeMissing && inspection.platform === 'linux' && !inspection.lingering) blockers.push('Enable user lingering through the host administrator before setup so its service survives SSH logout; TensorFold does not change this OS policy.')
      if (inspection.unsafeExisting || inspection.serviceExists && !inspection.existing) blockers.push('An unrecognized or partial managed installation exists; preserve and repair it manually before setup.')
      if (inspection.existing && (inspection.existing.bundleSha256 !== bundle.sha256 || inspection.existing.port !== input.port || inspection.existing.nodePath !== inspection.nodePath || inspection.existing.enginePath !== inspection.enginePath || inspection.existing.state !== 'started')) blockers.push('The existing installation differs from this preview; a deliberate upgrade or repair is required, and no files will be overwritten.')
      const publicPreview: Omit<ClusterBootstrapPreview, 'digest'> = {
        id: newId(), sshHostId: input.sshHostId, platform: inspection.platform, nodeAvailable: Number(inspection.nodeVersion.split('.')[0]) >= 22, engineAvailable: !!inspection.enginePath && !!inspection.engineVersion,
        steps: ['Use the already verified SSH host and its existing account authentication.', 'Create only a private TensorFold directory under this user’s home; preserve existing keys and unrelated files.', 'Transfer the integrity-checked app-owned agent bundle and loopback-only configuration over SSH stdin.', inspection.platform === 'darwin' ? 'Register a per-user launchd agent in the existing logged-in user domain.' : 'Register a systemd user service; leave administrator, lingering, driver and system-service changes to the user.', 'Read the new management credential only through the verified SSH channel into the desktop main process.', 'Require an authenticated nonce challenge before recording successful enrollment; no inference endpoint is exposed by setup.'],
        blockers, expiresAt: now() + 10 * 60_000,
      }
      const internal = { profileRoot: owner.profileRoot, hostStamp: owner.stamp, port: input.port, bundleSha256: bundle.sha256, inspection }
      const record = { ...internal, preview: { ...publicPreview, digest: recordDigest(internal, publicPreview) } }
      await owner.current(); await deps.persistPreview(record); await owner.current()
      return structuredClone(record.preview)
    },
    async apply(input: { previewId: string; digest: string; commandId: string; signal?: AbortSignal }): Promise<ClusterBootstrapApplyResult> {
      if (!input || typeof input.previewId !== 'string' || !/^[a-f0-9]{64}$/.test(input.digest) || !/^[A-Za-z0-9_-]{16,100}$/.test(input.commandId)) throw new Error('Choose a valid reviewed bootstrap preview and durable command identity.')
      const record = await deps.loadPreview(input.previewId)
      if (!record || record.preview.id !== input.previewId || record.preview.digest !== input.digest || record.preview.expiresAt <= now() || record.preview.blockers.length || record.profileRoot !== deps.profileRoot()) throw new Error('The setup preview is blocked, expired or no longer belongs to this profile; inspect again.')
      const { digest, ...plain } = record.preview, { preview: _preview, ...internal } = record
      if (recordDigest(internal, plain) !== digest) throw new Error('The saved setup preview failed integrity validation.')
      const owner = await capture(record.preview.sshHostId, record.hostStamp), bundle = await deps.agentBundle(); validBundle(bundle)
      if (bundle.sha256 !== record.bundleSha256) throw new Error('The app-owned agent changed after preview; inspect again.')
      const key = owner.profileRoot + ':' + record.preview.sshHostId
      if (applying.has(key)) throw new Error('This SSH host already has a setup operation in progress.')
      applying.add(key)
      let started = false
      const current = async () => { await owner.current(); if (input.signal?.aborted) throw new Error('Setup cancellation requires reconciliation before any retry.') }
      try {
        await current(); const session = await deps.sessionForHost(record.preview.sshHostId); await current()
        if (clusterSshHostStamp(session.host) !== owner.stamp) throw new Error('The SSH session identity changed after preview; reconnect and inspect again.')
        const inspected = parseInspection(await session.exec(command(SSH_PREREQUISITE_PROGRAM, record.inspection.nodePath), { timeoutMs: 20_000, maxBuffer: 65_536, signal: input.signal }))
        await current()
        if (sha(inspected) !== sha(record.inspection)) throw new Error('Host prerequisites changed after preview; inspect again before applying setup.')
        // This awaited write is the boundary before any remote filesystem/service mutation.
        await deps.markApplied(record.preview.id, input.commandId, { state: 'applying' }); await current(); started = true
        const payload = { schema: 1, home: inspected.home, uid: inspected.uid, platform: inspected.platform, nodePath: inspected.nodePath, enginePath: inspected.enginePath, port: record.port, commandId: input.commandId, bundleSha256: bundle.sha256, bundle: bundle.bytes.toString('base64') }
        const installed = await session.exec(command(SSH_INSTALL_PROGRAM, inspected.nodePath), { input: JSON.stringify(payload), timeoutMs: 60_000, maxBuffer: 16_384, signal: input.signal })
        await current()
        if (installed.code !== 0) throw new Error('Per-user setup did not settle; inspect the remote service and reconcile before retrying.')
        let install: any
        try { install = JSON.parse(installed.stdout) } catch { throw new Error('Setup returned an invalid receipt; reconcile the remote service before retrying.') }
        if (install.schema !== 1 || install.started !== true || typeof install.reused !== 'boolean' || install.dataDir !== path.posix.join(inspected.home, '.local/share/tensorfold-cluster/state')) throw new Error('Setup returned an invalid receipt; no successful installation was recorded.')
        let token = ''
        for (let attempt = 0; attempt < 8 && !token; attempt++) {
          await current()
          const credential = await session.exec(command(SSH_CREDENTIAL_PROGRAM, inspected.nodePath), { timeoutMs: 5000, maxBuffer: 512, signal: input.signal })
          await current()
          if (credential.code === 0 && /^[A-Za-z0-9_-]{43,160}$/.test(credential.stdout.trim())) token = credential.stdout.trim()
          else if (attempt < 7) await delay(500)
        }
        if (!token) throw new Error('The service has not provided its protected management credential; reconcile before retrying.')
        const proofRoot = owner.profileRoot
        const assertCurrent = () => { if (deps.profileRoot() !== proofRoot || input.signal?.aborted) throw new Error('The active setup scope changed.') }
        let proof: { protocol: 1; nodeId: string; authenticated: true }
        try { proof = await deps.verifyAgent({ sshHostId: record.preview.sshHostId, port: record.port, token, signal: input.signal, session, assertCurrent }) }
        catch { throw new Error('The agent could not complete the authenticated enrollment challenge; reconcile before retrying.') }
        await current()
        if (proof.protocol !== 1 || proof.authenticated !== true || typeof proof.nodeId !== 'string' || !proof.nodeId || proof.nodeId.length > 160 || /[\u0000-\u001f]/.test(proof.nodeId)) throw new Error('The agent did not prove its authenticated node identity; setup remains unverified.')
        await deps.markApplied(record.preview.id, input.commandId, { state: 'verified', nodeId: proof.nodeId }); await current()
        return { sshHostId: record.preview.sshHostId, hostStamp: clusterSshHostStamp(session.host), port: record.port, token, nodeId: proof.nodeId, verified: true, service: inspected.supervisor!, reused: install.reused }
      } catch (error) {
        if (started && deps.profileRoot() === record.profileRoot) await deps.markApplied(record.preview.id, input.commandId, { state: 'needs-review', message: 'Setup outcome is unverified; inspect its receipt and service before any retry.' }).catch(() => {})
        throw error
      } finally { applying.delete(key) }
    },
  }
}
