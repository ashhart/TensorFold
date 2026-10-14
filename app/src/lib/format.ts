export function relativeTime(ts: number): string {
  const diff = Date.now() - ts
  const s = Math.floor(diff / 1000)
  if (s < 60) return 'just now'
  const m = Math.floor(s / 60)
  if (m < 60) return `${m}m ago`
  const h = Math.floor(m / 60)
  if (h < 24) return `${h}h ago`
  const d = Math.floor(h / 24)
  if (d < 7) return `${d}d ago`
  return new Date(ts).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}

export function dateBucket(ts: number): string {
  const now = new Date()
  const d = new Date(ts)
  const startToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime()
  const startYesterday = startToday - 86_400_000
  const start7 = startToday - 6 * 86_400_000
  if (ts >= startToday) return 'Today'
  if (ts >= startYesterday) return 'Yesterday'
  if (ts >= start7) return 'Previous 7 days'
  if (d.getFullYear() === now.getFullYear()) return d.toLocaleDateString(undefined, { month: 'long' })
  return d.getFullYear().toString()
}

// Compact duration like "2m", "1h 4m", "45s" for goal/run timing.
export function formatDuration(ms: number): string {
  const s = Math.max(0, Math.round(ms / 1000))
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  if (m < 60) return `${m}m`
  const h = Math.floor(m / 60)
  const rm = m % 60
  return rm ? `${h}h ${rm}m` : `${h}h`
}

export function basename(p: string): string {
  if (!p) return ''
  const parts = p.replace(/\/+$/, '').split('/')
  return parts[parts.length - 1] || p
}

export function shortPath(p: string, max = 2): string {
  if (!p) return ''
  const home = (p.match(/^\/Users\/[^/]+/) || [])[0]
  let s = home ? p.replace(home, '~') : p
  const parts = s.split('/').filter(Boolean)
  if (parts.length <= max + 1) return s.startsWith('~') ? s : '/' + parts.join('/')
  return (s.startsWith('~') ? '~/…/' : '…/') + parts.slice(-max).join('/')
}
