import type { ConnectorStatus } from '../../lib/types'
import type { Tone } from './model'

export function connectorHealthBadgeClass(tone: Tone) {
  if (tone === 'critical') return 'bg-danger/10 text-danger'
  if (tone === 'high') return 'bg-warn/12 text-warn'
  if (tone === 'good') return 'bg-positive/12 text-positive'
  return 'bg-ink-5 text-ink-3'
}

export function connectorTone(status: ConnectorStatus): Tone {
  if (status === 'error') return 'critical'
  if (status === 'pending') return 'low'
  return 'good'
}

export function statusBadgeClass(status: ConnectorStatus) {
  if (status === 'error') return 'bg-danger/10 text-danger'
  if (status === 'pending') return 'bg-ink-5 text-ink-3'
  return 'bg-positive/12 text-positive'
}

export function toneText(tone: Tone) {
  if (tone === 'critical') return 'text-danger'
  if (tone === 'high') return 'text-warn'
  if (tone === 'good') return 'text-positive'
  return 'text-ink-3'
}

export function toneBg(tone: Tone) {
  if (tone === 'critical') return 'bg-danger/10'
  if (tone === 'high') return 'bg-warn/12'
  if (tone === 'good') return 'bg-positive/12'
  return 'bg-ink-5'
}

export function toneDot(tone: Tone) {
  if (tone === 'critical') return 'bg-danger'
  if (tone === 'high') return 'bg-warn'
  if (tone === 'good') return 'bg-positive'
  return 'bg-accent'
}

export function toneBadge(tone: Tone) {
  if (tone === 'critical') return 'bg-danger/10 text-danger'
  if (tone === 'high') return 'bg-warn/12 text-warn'
  if (tone === 'good') return 'bg-positive/12 text-positive'
  return 'bg-ink-5 text-ink-3'
}
