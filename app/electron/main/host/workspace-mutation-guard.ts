import path from 'node:path'
import { promises as fs, realpathSync } from 'node:fs'

export interface WorkspaceMutationLeaseInput {
  threadId?: string | null
  workspaceKey?: string | null
}

export interface WorkspaceMutationLease {
  addThreadId?(threadId: string): void
  addWorkspaceKeys?(workspaceKeys: Iterable<string | null | undefined>): void
  release(): void
}

export function lexicalLocalWorkspaceKey(cwd: string | null | undefined): string | null {
  const raw = String(cwd ?? '').trim()
  if (!raw) return null
  const resolved = path.resolve(raw)
  try { return `local:${realpathSync(resolved)}` } catch { return `local:${resolved}` }
}

/** Canonical identity shared by restore leases and local run-start checks. */
export async function canonicalLocalWorkspaceKey(cwd: string | null | undefined): Promise<string | null> {
  const raw = String(cwd ?? '').trim()
  if (!raw) return null
  const resolved = path.resolve(raw)
  return `local:${await fs.realpath(resolved).catch(() => resolved)}`
}

export function sshWorkspaceKey(hostId: string | null | undefined, cwd: string | null | undefined): string | null {
  const host = String(hostId ?? '').trim()
  const raw = String(cwd ?? '').trim()
  if (!host || !raw) return null
  return `ssh:${host}:${path.posix.normalize(raw)}`
}

/**
 * Main-process lease registry. Acquisition/checks are synchronous once canonical keys are
 * prepared, so a run cannot interleave between the final idle check and lease acquisition.
 */
export class WorkspaceMutationGuard {
  private readonly threadLeases = new Set<string>()
  private readonly workspaceLeases = new Set<string>()
  private readonly startThreadReservations = new Set<string>()
  private readonly startReservations = new Map<string, number>()
  private globalBarrier = false
  private globalBarrierReason = 'Local-state security purge'
  private readonly admissionPauses = new Map<symbol, string>()
  private readonly idleWaiters = new Set<() => void>()

  private hasActiveLeases(): boolean {
    return this.threadLeases.size > 0 || this.workspaceLeases.size > 0 || this.startThreadReservations.size > 0 || this.startReservations.size > 0
  }

  private assertGlobalBarrierOpen(): void {
    const paused = this.admissionPauses.values().next().value
    if (paused) throw new Error(`${paused}; wait for it to finish before starting workspace work.`)
    if (this.globalBarrier) throw new Error(`${this.globalBarrierReason} is in progress; wait for it to finish before starting workspace work.`)
  }

  /** Close admission immediately without waiting on, or replacing, an updater/purge lease. */
  pauseAdmission(reason: string): Pick<WorkspaceMutationLease, 'release'> {
    const token = Symbol('admission-pause')
    this.admissionPauses.set(token, reason)
    return { release: () => { this.admissionPauses.delete(token) } }
  }

  private notifyIdle(): void {
    if (this.hasActiveLeases()) return
    for (const resolve of this.idleWaiters) resolve()
    this.idleWaiters.clear()
  }

  private assertWorkspaceStartAllowed(workspaceKeys: Iterable<string | null | undefined>): void {
    for (const key of workspaceKeys) {
      if (key && this.workspaceLeases.has(key)) {
        throw new Error('Workspace restoration is in progress; wait for it to finish before starting this room.')
      }
    }
  }

  /**
   * Close the run/restore admission gate and wait for every already-admitted workspace
   * lease to settle. The caller can abort long-lived work immediately after invoking this
   * method (the gate closes synchronously, before this promise yields), then await a truly
   * quiescent filesystem before shredding project/checkpoint state.
   */
  async acquireGlobal(reason = 'Local-state security purge'): Promise<WorkspaceMutationLease> {
    if (this.globalBarrier) throw new Error(`${this.globalBarrierReason} is already in progress.`)
    this.globalBarrier = true
    this.globalBarrierReason = reason
    if (this.hasActiveLeases()) {
      await new Promise<void>((resolve) => this.idleWaiters.add(resolve))
    }
    let released = false
    return {
      release: () => {
        if (released) return
        released = true
        this.globalBarrier = false
      },
    }
  }

  acquire(input: WorkspaceMutationLeaseInput): WorkspaceMutationLease {
    this.assertGlobalBarrierOpen()
    const threadId = String(input.threadId ?? '').trim() || null
    const workspaceKey = String(input.workspaceKey ?? '').trim() || null
    if (!threadId && !workspaceKey) throw new Error('Cannot acquire a workspace mutation lease without a host-owned target.')
    if ((threadId && (this.threadLeases.has(threadId) || this.startThreadReservations.has(threadId))) || (workspaceKey && (this.workspaceLeases.has(workspaceKey) || (this.startReservations.get(workspaceKey) ?? 0) > 0))) {
      throw new Error('Workspace restoration is already in progress.')
    }
    if (threadId) this.threadLeases.add(threadId)
    if (workspaceKey) this.workspaceLeases.add(workspaceKey)
    let released = false
    return {
      release: () => {
        if (released) return
        released = true
        if (threadId) this.threadLeases.delete(threadId)
        if (workspaceKey) this.workspaceLeases.delete(workspaceKey)
        this.notifyIdle()
      },
    }
  }

  reserveStart(input: { threadId?: string | null; workspaceKeys?: Iterable<string | null | undefined> }): WorkspaceMutationLease {
    this.assertGlobalBarrierOpen()
    const threadIds = new Set<string>()
    const keys = new Set<string>()
    const addThreadId = (raw: string) => {
      const threadId = String(raw ?? '').trim()
      if (!threadId || threadIds.has(threadId)) return
      if (this.threadLeases.has(threadId) || this.startThreadReservations.has(threadId)) throw new Error('Thread work is already starting or workspace restoration is in progress.')
      this.startThreadReservations.add(threadId)
      threadIds.add(threadId)
    }
    const addWorkspaceKeys = (values: Iterable<string | null | undefined>) => {
      const additions = [...new Set([...values].filter((key): key is string => !!key))].filter((key) => !keys.has(key))
      this.assertWorkspaceStartAllowed(additions)
      for (const key of additions) {
        this.startReservations.set(key, (this.startReservations.get(key) ?? 0) + 1)
        keys.add(key)
      }
    }
    const initialThreadId = String(input.threadId ?? '').trim() || null
    const initialKeys = [...new Set([...(input.workspaceKeys ?? [])].filter((key): key is string => !!key))]
    if (initialThreadId && (this.threadLeases.has(initialThreadId) || this.startThreadReservations.has(initialThreadId))) {
      throw new Error('Thread work is already starting or workspace restoration is in progress.')
    }
    this.assertWorkspaceStartAllowed(initialKeys)
    if (initialThreadId) {
      this.startThreadReservations.add(initialThreadId)
      threadIds.add(initialThreadId)
    }
    for (const key of initialKeys) {
      this.startReservations.set(key, (this.startReservations.get(key) ?? 0) + 1)
      keys.add(key)
    }
    let released = false
    return {
      addThreadId,
      addWorkspaceKeys,
      release: () => {
        if (released) return
        released = true
        for (const threadId of threadIds) this.startThreadReservations.delete(threadId)
        for (const key of keys) {
          const count = this.startReservations.get(key) ?? 0
          if (count <= 1) this.startReservations.delete(key)
          else this.startReservations.set(key, count - 1)
        }
        this.notifyIdle()
      },
    }
  }

  assertThreadStartAllowed(threadId: string, workspaceKey?: string | null): void {
    this.assertGlobalBarrierOpen()
    if (this.threadLeases.has(threadId) || this.startThreadReservations.has(threadId) || (!!workspaceKey && this.workspaceLeases.has(workspaceKey))) {
      throw new Error('Workspace restoration is in progress; wait for it to finish before starting this thread.')
    }
  }

  assertRoomStartAllowed(workspaceKeys: Iterable<string | null | undefined>): void {
    this.assertGlobalBarrierOpen()
    this.assertWorkspaceStartAllowed(workspaceKeys)
  }
}
