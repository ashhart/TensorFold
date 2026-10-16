export interface SettingsDraftState {
  dirty: boolean
  pending: boolean
  failed: boolean
}

export function createSettingsDraftPersistence<T>(options: {
  dirty: { current: boolean }
  setDraft: (next: T) => void
  save: (next: T) => Promise<T>
  changed?: (state: SettingsDraftState) => void
}) {
  let pending = 0
  let failed = false
  let revision = 0
  let queue: Promise<void> = Promise.resolve()
  const emit = () => options.changed?.({ dirty: options.dirty.current, pending: pending > 0, failed })
  return {
    edited() { revision++; options.dirty.current = true; failed = false; emit() },
    persist(next: T): Promise<void> {
      const ownRevision = ++revision
      options.dirty.current = true
      options.setDraft(next)
      pending++; failed = false; emit()
      // Preserve invocation order, including after a failed write. An older confirmation
      // must never erase newer typing or release that newer draft's restart protection.
      const work = queue.then(async () => {
        const saved = await options.save(next)
        if (revision === ownRevision) {
          options.dirty.current = false
          options.setDraft(saved)
        }
      }).catch((error) => {
        if (revision === ownRevision) { failed = true; options.dirty.current = true }
        throw error
      }).finally(() => { pending--; emit() })
      queue = work.catch(() => {})
      return work
    },
    discard(next: T) {
      if (pending) return false
      revision++; options.dirty.current = false; failed = false; options.setDraft(next); emit()
      return true
    },
  }
}
