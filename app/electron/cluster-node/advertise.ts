import { spawn, type ChildProcess, type SpawnOptions } from 'node:child_process'
import { promises as fs, constants } from 'node:fs'
import { createHash } from 'node:crypto'
import { NodeHttpError } from './config'

export const NODE_DISCOVERY_SERVICE = '_tensorfold-cluster._tcp'
export interface NodeAdvertisement { stop(): Promise<void> }
export interface NodePublisher { start(nodeId: string, port: number, signal?: AbortSignal, transport?: 'ssh' | 'https'): Promise<NodeAdvertisement> }
export interface PublisherOptions {
  platform?: string
  access?: (file: string) => Promise<void>
  spawn?: (file: string, args: string[], options: SpawnOptions) => ChildProcess
  timeoutMs?: number
  diagnostic?: (message: string) => void
}
/** Opt-in OS publisher only; it advertises no model, account, token, path or hardware detail. */
export class OsNodePublisher implements NodePublisher {
  constructor(private options: PublisherOptions = {}) {}
  async start(nodeId: string, port: number, signal?: AbortSignal, transport: 'ssh' | 'https' = 'ssh'): Promise<NodeAdvertisement> {
    if (!/^[A-Za-z0-9_.:-]{1,160}$/.test(nodeId) || !Number.isInteger(port) || port < 1 || port > 65535) throw new NodeHttpError(400, 'Choose a valid node identity and advertised port.')
    if (transport !== 'ssh' && transport !== 'https') throw new NodeHttpError(400, 'Choose a supported enrollment transport.')
    const platform = this.options.platform ?? process.platform
    const file = platform === 'darwin' ? '/usr/bin/dns-sd' : platform === 'linux' ? '/usr/bin/avahi-publish-service' : null
    if (!file) throw new NodeHttpError(503, 'Node discovery publishing is supported only on macOS or Linux.')
    try { await (this.options.access ?? (file => fs.access(file, constants.X_OK)))(file) }
    catch { throw new NodeHttpError(503, platform === 'darwin' ? 'Node discovery requires the installed macOS dns-sd utility; nothing was published.' : 'Node discovery requires an installed avahi-publish-service utility and daemon; nothing was published.') }
    if (signal?.aborted) throw new NodeHttpError(503, 'Node discovery publishing was cancelled before startup.')
    const label = `TensorFold-${createHash('sha256').update(nodeId).digest('hex').slice(0, 24)}`
    const args = platform === 'darwin' ? ['-R', label, NODE_DISCOVERY_SERVICE, 'local.', String(port), 'protocol=1', `transport=${transport}`] : ['-s', label, NODE_DISCOVERY_SERVICE, String(port), 'protocol=1', `transport=${transport}`]
    const child = (this.options.spawn ?? spawn)(file, args, { shell: false, stdio: ['ignore', 'pipe', 'pipe'], env: { PATH: process.env.PATH, LANG: 'C', LC_ALL: 'C' } })
    let ended = false, stopping: Promise<void> | null = null, established = false, rejectStartup!: (error: Error) => void, resolveStartup!: () => void, output = '', total = 0
    const closed = new Promise<void>(resolve => child.once('close', () => { ended = true; if (!established) rejectStartup(new NodeHttpError(503, 'The OS publisher exited before confirming registration; nothing was published.')); else if (!stopping) this.options.diagnostic?.('Node discovery advertisement stopped; use an explicitly enrolled connection until publishing is restored.'); resolve() }))
    const stop = () => stopping ??= (async () => {
      if (ended) return
      child.kill('SIGTERM')
      const hard = setTimeout(() => { if (!ended) child.kill('SIGKILL') }, 2_000)
      let overdue: ReturnType<typeof setTimeout> | undefined
      try { await Promise.race([closed, new Promise<never>((_resolve, reject) => { overdue = setTimeout(() => reject(new NodeHttpError(503, 'The OS publisher did not confirm physical exit.')), 5_000) })]) }
      finally { clearTimeout(hard); if (overdue) clearTimeout(overdue) }
    })()
    const startup = new Promise<void>((resolve, reject) => { resolveStartup = resolve; rejectStartup = reject })
    const abort = () => { rejectStartup(new NodeHttpError(503, 'Node discovery publishing was cancelled.')); void stop().catch(() => {}) }
    signal?.addEventListener('abort', abort, { once: true })
    const receive = (part: Buffer) => {
      total += part.length
      if (total > 65_536) { rejectStartup(new NodeHttpError(503, 'The OS publisher exceeded its bounded diagnostic limit.')); void stop().catch(() => {}); return }
      if (established) return
      output = (output + part.toString('utf8')).slice(-8192)
      if (/Name now registered and active|Established under name/.test(output)) { established = true; resolveStartup() }
    }
    child.stdout?.on('data', receive); child.stderr?.on('data', receive)
    child.once('error', () => rejectStartup(new NodeHttpError(503, 'The installed OS publisher could not be started; nothing was published.')))
    const timer = setTimeout(() => rejectStartup(new NodeHttpError(503, 'The OS publisher did not confirm registration within its deadline; nothing was published.')), this.options.timeoutMs ?? 5_000)
    try { await startup; if (ended) throw new NodeHttpError(503, 'The OS publisher stopped during registration.'); return { stop: async () => { signal?.removeEventListener('abort', abort); await stop() } } }
    catch (error) { signal?.removeEventListener('abort', abort); await stop(); throw error }
    finally { clearTimeout(timer) }
  }
}
