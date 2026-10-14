import { useSyncExternalStore } from 'react'
import { api } from './ipc'
import type { PhoneSnapshot } from './types'
import { phoneError } from '../components/phone/phone-format'

export interface PhoneView { snapshot: PhoneSnapshot | null; error: string | null; pending: boolean }
let view: PhoneView = { snapshot: null, error: null, pending: false }
const listeners = new Set<() => void>()
let off: (() => void) | null = null
let revision = 0
let operations = 0
function publish(next: Partial<PhoneView>) {
  view = { ...view, ...next }
  for (const notify of listeners) notify()
}
export function applyPhoneSnapshot(snapshot: PhoneSnapshot) {
  revision++
  publish({ snapshot, error: null })
}
export async function refreshPhone(hard = false): Promise<PhoneSnapshot> {
  const next = hard ? await api.phoneRefresh() : await api.phoneSnapshot()
  applyPhoneSnapshot(next)
  return next
}
export async function runPhoneOperation(task: () => Promise<PhoneSnapshot | null>): Promise<PhoneSnapshot | null> {
  operations++
  publish({ pending: true, error: null })
  try {
    const next = await task()
    if (next) applyPhoneSnapshot(next)
    return next
  } catch (error) {
    publish({ error: phoneError(error) })
    throw error
  } finally {
    operations--
    publish({ pending: operations > 0 })
  }
}
function subscribe(notify: () => void) {
  listeners.add(notify)
  if (!off) {
    off = api.onPhoneEvent(applyPhoneSnapshot)
    const startedAtRevision = revision
    void api.phoneSnapshot().then((snapshot) => {
      if (startedAtRevision === revision) applyPhoneSnapshot(snapshot)
    }).catch((error) => publish({ error: phoneError(error) }))
  }
  return () => {
    listeners.delete(notify)
    if (listeners.size === 0) { off?.(); off = null }
  }
}
export function usePhone(): PhoneView {
  const current = useSyncExternalStore(subscribe, () => view)
  // Subscribing mirrors service state only; capturing a screen always requires a
  // button click or the tools of a chat to which the user explicitly grants access.
  return current
}
