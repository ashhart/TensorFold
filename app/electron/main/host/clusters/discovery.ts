import { spawn } from 'node:child_process'
import { createHash } from 'node:crypto'
import net from 'node:net'
import type { ClusterDiscoveryCandidate } from '../../../../shared/cluster-workspace'

export interface DiscoveryCommandResult { code: number | null; stdout: string; stderr: string }
export interface DiscoveryCommandOptions { signal: AbortSignal; timeoutMs: number; maxBytes: number }
export type DiscoveryCommandRunner = (command: string, args: string[], options: DiscoveryCommandOptions) => Promise<DiscoveryCommandResult>
export interface ClusterDiscoveryDeps { platform?: string; now?: () => number; run?: DiscoveryCommandRunner }
export const CLUSTER_BONJOUR_SERVICE = '_tensorfold-cluster._tcp'
const LIMIT = 32, TTL = 45_000
const clean = (text: string, limit = 160) => typeof text === 'string' && !!text.trim() && text.length <= limit && !/[\u0000-\u001f\u007f\u202a-\u202e\u2066-\u2069]/u.test(text)
function hostAllowed(host: string): boolean {
  if (!clean(host, 253)) return false
  const lower = host.toLowerCase().replace(/\.$/, '')
  if (/^[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?\.local$/i.test(lower) && !lower.includes('..')) return true
  if (net.isIP(lower) === 4) { const [a, b] = lower.split('.').map(Number); return a === 10 || a === 172 && b >= 16 && b <= 31 || a === 192 && b === 168 }
  return net.isIP(lower) === 6 && /^(?:fc|fd)/i.test(lower)
}
function candidate(label: string, host: string, port: number, now: number, via?: 'ssh' | 'https'): ClusterDiscoveryCandidate | null {
  host = host.replace(/\.$/, '')
  if (!clean(label) || !hostAllowed(host) || !Number.isInteger(port) || port < 1 || port > 65_535) return null
  return { id: createHash('sha256').update(JSON.stringify([label, host, port])).digest('hex'), label, host, port, protocol: 1, expiresAt: now + TTL, trusted: false, ...(via ? { via } : {}) }
}
export function parseAvahiCandidates(output: string, now: number): ClusterDiscoveryCandidate[] {
  const results: ClusterDiscoveryCandidate[] = []
  for (const line of output.slice(0, 65_536).split(/\r?\n/)) {
    const fields = line.split(';')
    if (fields[0] !== '=' || fields[4] !== CLUSTER_BONJOUR_SERVICE || fields[5] !== 'local' || !/(?:^|[\s";])protocol=1(?:$|[\s";])/.test(fields.slice(9).join(';'))) continue
    // Escaped discovery names are deliberately omitted rather than interpreted as commands.
    if (fields[3].includes('\\')) continue
    const value = candidate(fields[3], fields[6], Number(fields[8]), now, /(?:^|[\s";])transport=ssh(?:$|[\s";])/.test(fields.slice(9).join(';')) ? 'ssh' : undefined)
    if (value) results.push(value)
    if (results.length === LIMIT) break
  }
  return results
}
export function parseBonjourResolution(label: string, output: string, now: number): ClusterDiscoveryCandidate | null {
  const match = /can be reached at ([^\s:]+):(\d+)\s/.exec(output.slice(0, 65_536))
  if (!match || !/(?:^|[\s"])protocol=1(?:$|[\s"])/m.test(output)) return null
  return candidate(label, match[1], Number(match[2]), now, /(?:^|[\s"])transport=ssh(?:$|[\s"])/m.test(output) ? 'ssh' : undefined)
}
/** No subnet scan, endpoint probe, credential transmission or enrollment occurs here. */
export function createClusterDiscovery(deps: ClusterDiscoveryDeps = {}) {
  let pending: Promise<ClusterDiscoveryCandidate[]> | null = null, aborter: AbortController | null = null
  const run = deps.run ?? runDiscoveryCommand, now = deps.now ?? Date.now
  return {
    scan(options: { signal?: AbortSignal } = {}): Promise<ClusterDiscoveryCandidate[]> {
      if (pending) return pending
      const controller = new AbortController(); aborter = controller
      const cancel = () => controller.abort()
      options.signal?.addEventListener('abort', cancel, { once: true }); if (options.signal?.aborted) cancel()
      pending = (async () => {
        if (controller.signal.aborted) return []
        const platform = deps.platform ?? process.platform
        let found: ClusterDiscoveryCandidate[] = []
        if (platform === 'linux') found = parseAvahiCandidates((await run('/usr/bin/avahi-browse', ['--resolve', '--parsable', '--terminate', CLUSTER_BONJOUR_SERVICE], { signal: controller.signal, timeoutMs: 6000, maxBytes: 65_536 })).stdout, now())
        else if (platform === 'darwin') {
          const browse = await run('/usr/bin/dns-sd', ['-B', CLUSTER_BONJOUR_SERVICE, 'local.'], { signal: controller.signal, timeoutMs: 3000, maxBytes: 32_768 })
          const names = [...new Set(browse.stdout.split(/\r?\n/).flatMap(line => { const match = /\bAdd\s+\S+\s+\S+\s+local\.\s+_tensorfold-cluster\._tcp\.\s+(.+)$/.exec(line); return match && clean(match[1]) ? [match[1].trim()] : [] }))].slice(0, 16)
          // Two bounded resolver workers prevent an advertisement flood from spawning processes.
          let index = 0
          await Promise.all(Array.from({ length: Math.min(2, names.length) }, async () => {
            while (index < names.length && !controller.signal.aborted) {
              const name = names[index++]
              const resolved = await run('/usr/bin/dns-sd', ['-L', name, CLUSTER_BONJOUR_SERVICE, 'local.'], { signal: controller.signal, timeoutMs: 1500, maxBytes: 8192 })
              const value = parseBonjourResolution(name, resolved.stdout, now()); if (value) found.push(value)
            }
          }))
        } else throw new Error('Bonjour discovery is unavailable on this platform; add an explicit host instead.')
        if (controller.signal.aborted) return []
        return [...new Map(found.filter(item => item.expiresAt > now()).map(item => [item.id, item])).values()].slice(0, LIMIT)
      })().catch(error => { if (controller.signal.aborted) return []; throw error }).finally(() => { options.signal?.removeEventListener('abort', cancel); aborter = null; pending = null })
      return pending
    },
    cancel() { aborter?.abort() },
    async settle() { await pending?.catch(() => {}) },
  }
}
export const runDiscoveryCommand: DiscoveryCommandRunner = (command, args, options) => new Promise((resolve, reject) => {
  if (options.signal.aborted) { resolve({ code: null, stdout: '', stderr: 'Cancelled.' }); return }
  const child = spawn(command, args, { stdio: ['ignore', 'pipe', 'pipe'], shell: false })
  let stdout = '', stderr = '', size = 0, escalation: ReturnType<typeof setTimeout> | undefined
  const stop = () => { child.kill('SIGTERM'); escalation ??= setTimeout(() => child.kill('SIGKILL'), 1000) }
  const timer = setTimeout(stop, Math.max(1, Math.min(6000, options.timeoutMs)))
  options.signal.addEventListener('abort', stop, { once: true })
  const append = (target: 'stdout' | 'stderr', chunk: Buffer) => { const allowed = Math.max(0, options.maxBytes - size); size += chunk.length; if (target === 'stdout') stdout += chunk.subarray(0, allowed).toString(); else stderr += chunk.subarray(0, allowed).toString(); if (size > options.maxBytes) stop() }
  child.stdout?.on('data', chunk => append('stdout', chunk)); child.stderr?.on('data', chunk => append('stderr', chunk))
  const cleanUp = () => { clearTimeout(timer); if (escalation) clearTimeout(escalation); options.signal.removeEventListener('abort', stop) }
  child.once('error', error => { cleanUp(); reject(new Error((error as NodeJS.ErrnoException).code === 'ENOENT' ? 'Bonjour discovery command is not installed; add an explicit host or install the operating system discovery tools.' : 'Bonjour discovery could not start.')) })
  // Resolve on physical process close, not merely when cancellation is requested.
  child.once('close', code => { cleanUp(); resolve({ code, stdout, stderr }) })
})
