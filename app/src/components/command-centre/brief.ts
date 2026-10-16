import { relativeTime } from '../../lib/format'
import type { Automation } from '../../lib/types'

export function briefOutputLines(automation: Automation | null) {
  const output = automation?.runs.find((run) => run.status === 'ok' && run.output)?.output ?? ''
  return output.split('\n').map((line) => line.replace(/^\s*\d+[.)]\s*/, '').trim()).filter(Boolean)
}

export function firstLine(text: string) {
  return text.split('\n').map((line) => line.trim()).find(Boolean) ?? ''
}

export function briefSchedule(automation: Automation) {
  const last = automation.lastRunAt ? `Last ${relativeTime(automation.lastRunAt)}` : 'Not run yet'
  const next = automation.nextRunAt ? `next ${scheduleTime(automation.nextRunAt)}` : automation.scheduleKind
  return `${last} / ${next} / ${automation.model ?? 'model'}`
}

export function scheduleTime(ts: number) {
  const d = new Date(ts)
  const today = new Date()
  const tomorrow = new Date(today.getFullYear(), today.getMonth(), today.getDate() + 1).getTime()
  const startToday = new Date(today.getFullYear(), today.getMonth(), today.getDate()).getTime()
  const prefix = ts >= startToday && ts < tomorrow ? 'Today' : d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
  return `${prefix} ${d.toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' })}`
}
