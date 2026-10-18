import { currentAuthoredText } from './preferences/display'
import { currentUiLanguage } from './preferences/current-locale'
import type { EngineStatus, ModelPreflight, PreflightVerdict, PullProgress } from './types'

// Pure display helpers for the built-in engine (no IPC), shared by the Models page and the picker.

const GIB = 1024 ** 3

export function formatBytes(bytes: number | null | undefined): string {
  if (bytes == null || !Number.isFinite(bytes)) return '?'
  if (bytes >= GIB) return `${(bytes / GIB).toFixed(1)} GB`
  if (bytes >= 1024 ** 2) return `${Math.round(bytes / 1024 ** 2)} MB`
  return `${Math.max(0, Math.round(bytes / 1024))} KB`
}

export function pullPercent(pull: PullProgress): number | null {
  if (!pull.totalBytes) return null
  return Math.max(0, Math.min(100, Math.round((pull.receivedBytes / pull.totalBytes) * 100)))
}

export interface VerdictBadge {
  label: string
  tone: 'ok' | 'warn' | 'bad' | 'muted'
}

export const VERDICT_BADGES: Record<PreflightVerdict, VerdictBadge> = {
  ok: { label: 'Supported', tone: 'ok' },
  installed: { label: 'Installed', tone: 'ok' },
  unsupported: { label: 'Unsupported family', tone: 'muted' },
  format: { label: 'Unsupported format', tone: 'muted' },
  gated: { label: 'Gated', tone: 'warn' },
  'too-big': { label: 'Too big for this Mac', tone: 'bad' },
  disk: { label: 'Not enough disk', tone: 'bad' },
  error: { label: 'Check failed', tone: 'warn' },
}

export function canDownload(preflight: ModelPreflight | null | undefined): boolean {
  return preflight?.verdict === 'ok' || preflight?.verdict === 'installed'
}

export function engineStateLabel(status: EngineStatus | null | undefined): string {
  if (currentUiLanguage() !== 'en') {
    if (!status) return currentAuthoredText('Checking engine')
    if (!status.available) return currentAuthoredText('Engine missing from this build')
    if (status.state === 'ready') return currentAuthoredText('Running {model}', { model: status.model ?? '' })
    if (status.state === 'starting') return status.message ? `${currentAuthoredText('Diagnostic, original text')}: ${status.message}` : currentAuthoredText('Loading {model}', { model: status.model ?? '' })
    if (status.state === 'failed') return status.message ? `${currentAuthoredText('Diagnostic, original text')}: ${status.message}` : currentAuthoredText('The engine stopped')
    return status.version ? currentAuthoredText('Engine {version} idle', { version: status.version }) : currentAuthoredText('Engine idle')
  }
  if (!status) return 'Checking engine'
  if (!status.available) return 'Engine missing from this build'
  if (status.state === 'ready') return `Running ${status.model}`
  if (status.state === 'starting') return status.message ?? `Loading ${status.model}`
  if (status.state === 'failed') return status.message ?? 'The engine stopped'
  return status.version ? `Engine ${status.version} idle` : 'Engine idle'
}
