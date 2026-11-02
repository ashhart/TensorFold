import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../../lib/ipc'
import type { AssistantControlSnapshot } from '../../lib/types'
import { createSnapshotRefresh, type SnapshotRefreshState } from './snapshot-refresh'

export const ASSISTANT_REFRESH_MS = 15_000

export function useAssistantSnapshot() {
  const [state, setState] = useState<SnapshotRefreshState<AssistantControlSnapshot>>({
    snapshot: null, loading: false, error: null, refreshedAt: null,
  })
  const control = useRef<{ refresh: () => Promise<void> } | null>(null)
  const [paused, setPaused] = useState(false)

  useEffect(() => {
    const controller = createSnapshotRefresh(() => api.getAssistantSnapshot(), setState)
    let timer: ReturnType<typeof setTimeout> | undefined
    let stopped = false
    const visible = () => document.visibilityState !== 'hidden'
    const schedule = () => {
      clearTimeout(timer)
      if (!stopped && visible()) timer = setTimeout(() => { void reconcile() }, ASSISTANT_REFRESH_MS)
    }
    const reconcile = async () => {
      if (stopped || !visible()) return
      await controller.refresh()
      schedule()
    }
    const refresh = async () => {
      await controller.refresh(true)
      schedule()
    }
    const visibility = () => {
      setPaused(!visible())
      clearTimeout(timer)
      if (visible()) void reconcile()
    }
    const onFocus = () => { void reconcile() }
    control.current = { refresh }
    document.addEventListener('visibilitychange', visibility)
    window.addEventListener('focus', onFocus)
    window.addEventListener('online', onFocus)
    visibility()
    return () => {
      stopped = true
      clearTimeout(timer)
      controller.dispose()
      control.current = null
      document.removeEventListener('visibilitychange', visibility)
      window.removeEventListener('focus', onFocus)
      window.removeEventListener('online', onFocus)
    }
  }, [])

  const refresh = useCallback(() => control.current?.refresh() ?? Promise.resolve(), [])
  return { ...state, paused, refresh }
}
