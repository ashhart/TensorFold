import { useEffect, useState } from 'react'

/** One action at a time on a screen, and which one: the button that was pressed shows it is working (`aria-busy`),
 *  the others wait. A click that does nothing visible is the worst feedback there is. */
export function useWorking() {
  const [working, setWorking] = useState<string | null>(null)
  const run = async (key: string, fn: () => Promise<unknown>): Promise<void> => {
    if (working) return
    setWorking(key)
    try { await fn() } finally { setWorking(null) }
  }
  return { working, run, is: (key: string) => working === key, any: working != null }
}

/** Seconds since `since` (epoch seconds), ticking every second while `active`: a long operation shows it is alive. */
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

/** "1 min 05 s", "45 s" */
export const elapsed = (s: number) => (s < 60 ? `${Math.floor(s)} s` : `${Math.floor(s / 60)} min ${String(Math.floor(s % 60)).padStart(2, '0')} s`)
