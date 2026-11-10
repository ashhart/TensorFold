import { useEffect, useState } from 'react'



export function useWorking() {
  const [working, setWorking] = useState<string | null>(null)
  const run = async (key: string, fn: () => Promise<unknown>): Promise<void> => {
    if (working) return
    setWorking(key)
    try { await fn() } finally { setWorking(null) }
  }
  return { working, run, is: (key: string) => working === key, any: working != null }
}


export function useElapsed(since: number | null | undefined, active = true): number {
  const [now, setNow] = useState(() => Date.now() / 1000)
  useEffect(() => {
    if (!active || since == null) return
    setNow(Date.now() / 1000)
    const t = setInterval(() => setNow(Date.now() / 1000), 1000)
    return () => clearInterval(t)
  }, [since, active])
  return since == null ? 0 : Math.max(0, now - since)
}


export const elapsed = (s: number) => (s < 60 ? `${Math.floor(s)} s` : `${Math.floor(s / 60)} min ${String(Math.floor(s % 60)).padStart(2, '0')} s`)
