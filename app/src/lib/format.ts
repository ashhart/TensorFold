import { currentUiLanguage, localDateText } from './preferences/current-locale'
import { authoredText } from './preferences/language-packs'
export function relativeTime(ts: number): string {
  const locale = currentUiLanguage()
  if (locale !== 'en') {
    if (locale === 'la') return localDateText(ts)
    const seconds = Math.max(0, Math.floor((Date.now() - ts) / 1000))
    const format = new Intl.RelativeTimeFormat(locale, { numeric: 'auto', style: 'short' })
    if (seconds < 60) return format.format(0, 'second')
    if (seconds < 3600) return format.format(-Math.floor(seconds / 60), 'minute')
    if (seconds < 86400) return format.format(-Math.floor(seconds / 3600), 'hour')
    if (seconds < 604800) return format.format(-Math.floor(seconds / 86400), 'day')
    return localDateText(ts, { month: 'short', day: 'numeric' })
  }
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
  if (ts >= startToday) return authoredText(currentUiLanguage(), 'Today')
  if (ts >= startYesterday) return authoredText(currentUiLanguage(), 'Yesterday')
  if (ts >= start7) return authoredText(currentUiLanguage(), 'Previous 7 days')
  if (d.getFullYear() === now.getFullYear()) return currentUiLanguage() === 'en' ? d.toLocaleDateString(undefined, { month: 'long' }) : localDateText(d, { month: 'long' })
  return d.getFullYear().toString()
}

// Compact duration like "2m", "1h 4m", "45s" for goal/run timing.
export function formatDuration(ms: number): string {
  const s = Math.max(0, Math.round(ms / 1000)), locale = currentUiLanguage()
  if (locale !== 'en' && locale !== 'la') {
    const unit = (value: number, name: 'second' | 'minute' | 'hour') => new Intl.NumberFormat(locale, { style: 'unit', unit: name, unitDisplay: 'narrow' }).format(value)
    if (s < 60) return unit(s, 'second')
    const minutes = Math.floor(s / 60)
    if (minutes < 60) return unit(minutes, 'minute')
    return [unit(Math.floor(minutes / 60), 'hour'), ...(minutes % 60 ? [unit(minutes % 60, 'minute')] : [])].join(' ')
  }
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
