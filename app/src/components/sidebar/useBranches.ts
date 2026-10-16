import { useEffect, useState } from 'react'
import { api } from '../../lib/ipc'

// Git branch per local folder for the thread rows: one quiet git status read per folder, at most
// every 30 s, rechecked when the folder list changes or the window regains focus.

const TTL_MS = 30_000
const MAX_FOLDERS = 16
const cache = new Map<string, { branch: string | null; at: number }>()
const inflight = new Map<string, Promise<void>>()

function snapshot(cwds: string[]): Record<string, string | null> {
  const out: Record<string, string | null> = {}
  for (const cwd of cwds) out[cwd] = cache.get(cwd)?.branch ?? null
  return out
}

export function useBranches(cwds: string[]): Record<string, string | null> {
  const key = [...new Set(cwds)].slice(0, MAX_FOLDERS).sort().join('\n')
  const [branches, setBranches] = useState<Record<string, string | null>>(() => snapshot(key ? key.split('\n') : []))

  useEffect(() => {
    const list = key ? key.split('\n') : []
    let alive = true
    const refresh = () => {
      const now = Date.now()
      for (const cwd of list) {
        const hit = cache.get(cwd)
        if (hit && now - hit.at < TTL_MS) continue
        let pending = inflight.get(cwd)
        if (!pending) {
          pending = api.gitStatus(cwd)
            .then((g) => { cache.set(cwd, { branch: g.branch, at: Date.now() }) })
            .catch(() => { cache.set(cwd, { branch: null, at: Date.now() }) })
            .finally(() => { inflight.delete(cwd) })
          inflight.set(cwd, pending)
        }
        void pending.then(() => { if (alive) setBranches(snapshot(list)) })
      }
      setBranches(snapshot(list))
    }
    refresh()
    window.addEventListener('focus', refresh)
    return () => { alive = false; window.removeEventListener('focus', refresh) }
  }, [key])

  return branches
}
