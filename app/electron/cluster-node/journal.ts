import { validateClusterCommandId, validateClusterNodeDeployment, validateClusterNodeJob } from '../../shared/cluster-validation'
import { promises as fs, constants } from 'node:fs'
import path from 'node:path'
import { randomUUID, createHash } from 'node:crypto'
import { privateDirectory, privateRead } from './config'
import type { NodeJournal, StoredNodeState } from './types'

const MAX_BYTES = 8 * 1024 * 1024
const MAX_JOBS = 2048
const empty = (): StoredNodeState => ({ schema: 1, nodeId: `node-${randomUUID()}`, revision: 0, jobs: [], receipts: [], cancelRequested: [], active: null, request: null })
function validate(state: StoredNodeState): void {
  if (!state || state.schema !== 1 || typeof state.nodeId !== 'string' || !/^[a-zA-Z0-9_-]{1,100}$/.test(state.nodeId) || !Number.isSafeInteger(state.revision) || state.revision < 0 || !Array.isArray(state.jobs) || state.jobs.length > MAX_JOBS || !Array.isArray(state.receipts) || state.receipts.length > MAX_JOBS * 3 || !Array.isArray(state.cancelRequested) || state.cancelRequested.length > MAX_JOBS) throw new Error('The node receipt journal has an invalid schema or limit.')
  const ids = new Set<string>(), commands = new Set<string>()
  for (const job of state.jobs) {
    if (!job || typeof job.id !== 'string' || ids.has(job.id) || !['ensure-model', 'start', 'stop', 'canary'].includes(job.kind) || !['queued', 'running', 'reconciling', 'needs-review', 'completed', 'failed', 'cancelled'].includes(job.phase) || typeof job.commandId !== 'string' || typeof job.step !== 'string' || typeof job.message !== 'string') throw new Error('The node receipt journal contains an invalid job.')
    validateClusterNodeJob(job)
    ids.add(job.id)
  }
  for (const receipt of state.receipts) {
    if (!receipt || typeof receipt.commandId !== 'string' || receipt.commandId.length > 160 || commands.has(receipt.commandId) || !/^[a-f0-9]{64}$/.test(receipt.digest) || !ids.has(receipt.jobId)) throw new Error('The node receipt journal contains an invalid command identity.')
    validateClusterCommandId(receipt.commandId)
    commands.add(receipt.commandId)
  }
  if (state.active) validateClusterNodeDeployment(state.active)
  if (state.active && (typeof state.active.id !== 'string' || !['preparing', 'starting', 'ready', 'stopping', 'stopped', 'needs-review', 'failed'].includes(state.active.state))) throw new Error('The node journal contains an invalid active deployment.')
}
export function commandDigest(value: unknown): string {
  const canonical = (value: any): any => Array.isArray(value) ? value.map(canonical) : value && typeof value === 'object' ? Object.fromEntries(Object.keys(value).sort().map(key => [key, canonical(value[key])])) : value
  return createHash('sha256').update(JSON.stringify(canonical(value))).digest('hex')
}
export class DiskNodeJournal implements NodeJournal {
  private state: StoredNodeState | null = null
  private loading: Promise<StoredNodeState> | null = null
  private tail: Promise<unknown> = Promise.resolve()
  constructor(private directory: string) {}
  async read(): Promise<StoredNodeState> { return structuredClone(await this.load()) }
  update<T>(edit: (state: StoredNodeState) => T): Promise<T> {
    const next = this.tail.then(async () => {
      const state = structuredClone(await this.load())
      const result = edit(state)
      state.revision++
      validate(state)
      await this.write(state)
      this.state = state
      return structuredClone(result)
    })
    this.tail = next.catch(() => {})
    return next
  }
  async drain(): Promise<void> { await this.tail }
  private async load(): Promise<StoredNodeState> {
    if (this.state) return this.state
    if (this.loading) return this.loading
    this.loading = (async () => {
      await privateDirectory(this.directory)
      let state: StoredNodeState
      try { state = JSON.parse((await privateRead(path.join(this.directory, 'state.json'), MAX_BYTES)).toString('utf8')); validate(state) }
      catch (error) { if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error; state = empty() }
      let changed = false
      for (const job of state.jobs) if (['queued', 'running', 'reconciling'].includes(job.phase)) { job.phase = 'needs-review'; job.step = 'restart-review'; job.message = 'The node restarted before the physical outcome was confirmed; nothing was replayed.'; job.updatedAt = Date.now(); changed = true }
      if (state.active && !['stopped', 'failed'].includes(state.active.state)) { state.active.state = 'needs-review'; state.active.endpointReady = false; state.active.canaryPassed = false; state.active.error = 'The node restarted without its original process ownership; verify and stop any earlier engine before continuing.'; changed = true }
      if (state.request?.phase === 'running') { state.request.phase = 'needs-review'; changed = true }
      if (changed) state.revision++
      await this.write(state)
      this.state = state
      return state
    })()
    try { return await this.loading } finally { this.loading = null }
  }
  private async write(state: StoredNodeState): Promise<void> {
    const body = JSON.stringify(state)
    if (Buffer.byteLength(body) > MAX_BYTES) throw new Error('The node receipt journal has reached its bounded storage limit.')
    await privateDirectory(this.directory)
    const file = path.join(this.directory, 'state.json'), temporary = path.join(this.directory, `.state-${randomUUID()}.tmp`)
    const existing = await fs.lstat(file).catch((error: NodeJS.ErrnoException) => { if (error.code === 'ENOENT') return null; throw error })
    if (existing?.isSymbolicLink()) throw new Error('The node receipt journal cannot be a symbolic link.')
    const handle = await fs.open(temporary, constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL | constants.O_NOFOLLOW, 0o600)
    try {
      await handle.writeFile(body); await handle.sync(); await handle.close()
      await fs.rename(temporary, file)
      const directory = await fs.open(this.directory, constants.O_RDONLY)
      try { await directory.sync() } finally { await directory.close() }
    } catch (error) { await handle.close().catch(() => {}); await fs.unlink(temporary).catch(() => {}); throw error }
  }
}
