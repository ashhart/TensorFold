export interface SnapshotRefreshState<T> {
  snapshot: T | null
  loading: boolean
  error: string | null
  refreshedAt: number | null
}

export interface SnapshotRefreshOptions {
  timeoutMs?: number
  now?: () => number
}

export function createSnapshotRefresh<T>(
  read: () => Promise<T>,
  publish: (state: SnapshotRefreshState<T>) => void,
  options: SnapshotRefreshOptions = {},
) {
  const now = options.now ?? Date.now
  const timeoutMs = options.timeoutMs ?? 20_000
  let state: SnapshotRefreshState<T> = { snapshot: null, loading: false, error: null, refreshedAt: null }
  let active: Promise<void> | null = null
  let queued = false
  let disposed = false
  let cancelRead: (() => void) | null = null

  const update = (patch: Partial<SnapshotRefreshState<T>>) => {
    if (disposed) return
    state = { ...state, ...patch }
    publish(state)
  }

  const readWithDeadline = async () => {
    let timer: ReturnType<typeof setTimeout> | undefined
    const deadline = new Promise<never>((_resolve, reject) => {
      timer = setTimeout(() => reject(new Error('The assistant update timed out. Try Refresh again.')), timeoutMs)
      cancelRead = () => reject(new Error('Assistant view closed.'))
    })
    try {
      return await Promise.race([Promise.resolve().then(read), deadline])
    } finally {
      clearTimeout(timer)
      cancelRead = null
    }
  }

  const drain = async () => {
    update({ loading: true })
    try {
      do {
        queued = false
        try {
          const snapshot = await readWithDeadline()
          if (!queued) update({ snapshot, error: null, refreshedAt: now() })
        } catch (error) {
          if (!queued) update({ error: error instanceof Error ? error.message : String(error) })
        }
      } while (queued && !disposed)
    } finally {
      active = null
      update({ loading: false })
    }
  }

  return {
    snapshot: () => state,
    refresh(afterCurrent = false): Promise<void> {
      if (disposed) return Promise.resolve()
      if (active) {
        queued ||= afterCurrent
        return active
      }
      active = drain()
      return active
    },
    dispose() {
      disposed = true
      queued = false
      cancelRead?.()
    },
  }
}
