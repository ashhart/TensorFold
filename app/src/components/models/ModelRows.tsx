import { useState, type ReactNode } from 'react'
import clsx from 'clsx'
import { Icon } from '../../lib/icons'
import { Button } from '../ui'
import { VERDICT_BADGES, canDownload, formatBytes, pullPercent, type VerdictBadge } from '../../lib/engine'
import type { EngineInstalledModel, EngineStatus, HubModelSummary, ModelPreflight, PullProgress } from '../../lib/types'
import { repoDisplayName } from '../../lib/turn-stats-line'

// Rows for the Models page: installed checkpoints, downloads in flight, and Hub repos with preflight badges.

const TONES: Record<VerdictBadge['tone'], string> = {
  ok: 'bg-positive/10 text-ink',
  warn: 'bg-warn/10 text-ink',
  bad: 'bg-danger/10 text-danger',
  muted: 'bg-ink-5 text-ink-2',
}

export function Badge({ badge }: { badge: VerdictBadge }) {
  return <span className={clsx('shrink-0 rounded-md px-1.5 py-0.5 text-[12px] font-medium', TONES[badge.tone])}>{badge.label}</span>
}

function Row({ children, testId }: { children: ReactNode; testId?: string }) {
  return <div data-testid={testId} className="min-w-0 rounded-xl border border-line bg-card px-3.5 py-3 shadow-card">{children}</div>
}

function Facts({ items }: { items: Array<string | null | false | undefined> }) {
  const shown = items.filter(Boolean) as string[]
  return <div className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-[13px] text-ink-2">{shown.map((item) => <span key={item}>{item}</span>)}</div>
}

export function InstalledRow({ model, status, onLoad, onUnload, onDelete, onTask, onChat, hubCache, locked = false, minimal = false }: {
  minimal?: boolean
  model: EngineInstalledModel
  status: EngineStatus | null
  onLoad: () => void
  onUnload: () => void
  onDelete: () => Promise<void>
  onTask: () => void
  onChat: () => void
  hubCache: string
  locked?: boolean
}) {
  const [confirming, setConfirming] = useState(false)
  const [busy, setBusy] = useState(false)
  const active = status?.model === model.repo
  const running = active && status?.state === 'ready'
  const loading = active && status?.state === 'starting'
  const failed = active && status?.state === 'failed'
  const badge: VerdictBadge | null = running ? { label: 'Running', tone: 'ok' } : loading ? { label: 'Loading', tone: 'warn' } : failed ? { label: 'Stopped', tone: 'bad' } : model.fitsMemory === false ? VERDICT_BADGES['too-big'] : null
  const remove = async () => {
    setBusy(true)
    try { await onDelete() } finally { setBusy(false); setConfirming(false) }
  }
  return (
    <Row testId="models-installed-row">
      <div className="flex flex-wrap items-start gap-3">
        <span className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-surface text-ink-2"><Icon name="cpu" size={16} /></span>
        <div className="min-w-0 flex-1 basis-40">
          <div className="flex min-w-0 items-center gap-2">
            <span className="break-words text-[14px] font-medium text-ink" title={model.localPath ?? model.repo}>{model.title ?? repoDisplayName(model.repo)}</span>
            {badge && <Badge badge={badge} />}
          </div>
          <Facts items={[
            model.family,
            model.format,
            model.source === 'library' ? 'Linked local model' : model.source === 'cache' ? 'Shared cache' : null,
            model.contextTokens ? `${Math.round(model.contextTokens / 1024)}k context` : null,
            model.weightsBytes != null ? `${formatBytes(model.weightsBytes)} on disk` : null,
            model.memoryFloorBytes != null ? `needs ${formatBytes(model.memoryFloorBytes)} memory` : null,
            model.drafter ? (model.drafterInstalled ? `drafter ${model.drafter}` : `drafter ${model.drafter} missing`) : null,
          ]} />
          {failed && status?.message && <div className="mt-1.5 text-[13px] text-danger">{status.message}</div>}
        </div>
        <div className="flex shrink-0 items-center gap-1.5">
          {/* A task gets the project folder and file tools; a chat only talks. */}
          {running && !minimal && <Button size="xs" variant="primary" icon="pencil" onClick={onTask}>New worker</Button>}
          {running && !minimal && <Button size="xs" variant="default" icon="message" onClick={onChat}>New chat</Button>}
          {active && !failed
            ? <Button size="xs" variant="default" icon="stop" onClick={onUnload} disabled={locked}>Unload</Button>
            : <Button size="xs" variant={running ? 'default' : 'primary'} icon="play" onClick={onLoad} disabled={model.fitsMemory === false || loading || locked || status?.state === 'starting'}>Load</Button>}
          {model.source === 'download' && <Button size="xs" variant="ghost" icon="trash" aria-label={`Delete ${model.repo}`} onClick={() => setConfirming(true)} disabled={busy || running || loading} />}
        </div>
      </div>
      {confirming && (
        <div className="mt-3 flex flex-wrap items-center gap-2 rounded-lg border border-danger/25 bg-danger/10 px-3 py-2 text-[13px] text-ink-2">
          <span className="min-w-0 flex-1">Delete {model.repo}{model.weightsBytes != null ? ` (${formatBytes(model.weightsBytes)})` : ''} from the shared Hugging Face cache at {hubCache}? Other apps that read this cache lose it too.</span>
          <Button size="xs" variant="danger" icon="trash" onClick={remove} disabled={busy}>{busy ? 'Deleting' : 'Delete'}</Button>
          <Button size="xs" variant="ghost" onClick={() => setConfirming(false)} disabled={busy}>Cancel</Button>
        </div>
      )}
    </Row>
  )
}

export function PullRow({ pull, onCancel, onRetry }: { pull: PullProgress; onCancel: () => void; onRetry: () => void }) {
  const percent = pullPercent(pull)
  const running = pull.state === 'running'
  return (
    <Row testId="models-pull-row">
      <div className="flex items-center gap-3">
        <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-surface text-ink-2"><Icon name={running ? 'arrowDown' : pull.state === 'done' ? 'check' : 'warning'} size={16} /></span>
        <div className="min-w-0 flex-1">
          <div className="flex items-center justify-between gap-2">
            <span className="truncate text-[14px] font-medium text-ink">{pull.repo}</span>
            <span className="shrink-0 text-[13px] text-ink-3">
              {formatBytes(pull.receivedBytes)}{pull.totalBytes ? ` of ${formatBytes(pull.totalBytes)}` : ''}{percent != null ? ` · ${percent}%` : ''}
            </span>
          </div>
          <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-ink-5">
            <div className={clsx('h-full rounded-full transition-[width] duration-300', pull.state === 'failed' ? 'bg-danger' : 'bg-brand', running && percent == null && 'animate-pulse')} style={{ width: `${percent ?? (running ? 30 : 100)}%` }} />
          </div>
          <div className="mt-1 truncate font-mono text-[12px] text-ink-3">{pull.message ?? pull.lastLine ?? pull.phase}</div>
        </div>
        {running
          ? <Button size="xs" variant="default" icon="x" onClick={onCancel}>Cancel</Button>
          : pull.state !== 'done' && <Button size="xs" variant="default" icon="refresh" onClick={onRetry}>Resume</Button>}
      </div>
    </Row>
  )
}

export interface PreflightState {
  loading: boolean
  result: ModelPreflight | null
  error: string | null
}

export function HubRow({ title, repo, note, summary, preflight, pull, onCheck, onDownload, testId }: {
  title?: string
  repo: string
  note?: string
  summary?: HubModelSummary
  preflight: PreflightState | undefined
  pull: PullProgress | undefined
  onCheck: () => void
  onDownload: () => void
  testId?: string
}) {
  const result = preflight?.result ?? null
  const badges: VerdictBadge[] = []
  if (result) badges.push(VERDICT_BADGES[result.verdict])
  else if (summary?.gated) badges.push(VERDICT_BADGES.gated)
  if (result && result.verdict !== 'too-big' && result.fitsMemory === false) badges.push(VERDICT_BADGES['too-big'])
  if (result && result.verdict !== 'disk' && result.fitsDisk === false) badges.push(VERDICT_BADGES.disk)
  const downloading = pull?.state === 'running'
  return (
    <Row testId={testId}>
      <div className="flex flex-wrap items-start gap-3">
        <div className="min-w-0 flex-1 basis-40">
          <div className="flex min-w-0 flex-wrap items-center gap-2">
            <span className="truncate text-[14px] font-medium text-ink">{title ?? repo}</span>
            {badges.map((badge) => <Badge key={badge.label} badge={badge} />)}
            {preflight?.loading && <span className="text-[12px] text-ink-2">Checking…</span>}
          </div>
          {title && <div className="mt-0.5 truncate font-mono text-[12px] text-ink-2" title={repo}>{repo}</div>}
          <Facts items={[
            result?.family ?? (summary?.modelType ? `model_type ${summary.modelType}` : null),
            result?.totalBytes != null ? `${formatBytes(result.totalBytes)} download` : null,
            result?.memoryFloorBytes != null ? `needs ${formatBytes(result.memoryFloorBytes)} of ${formatBytes(result.memoryLimitBytes)} allowed` : null,
            result?.drafter ? `with drafter ${result.drafter.repo}` : null,
            summary?.downloads != null ? `${summary.downloads.toLocaleString()} downloads` : null,
            note,
          ]} />
          {result && result.reasons.length > 0 && <div className="mt-1.5 text-[13px] leading-snug text-ink-2">{result.reasons[0]}</div>}
          {preflight?.error && <div className="mt-1.5 text-[13px] text-danger">{preflight.error}</div>}
        </div>
        <div className="flex shrink-0 items-center gap-1.5">
          {!result && !preflight?.loading && <Button size="xs" variant="ghost" icon="check" onClick={onCheck}>Check</Button>}
          {result?.verdict === 'installed' && !downloading
            ? <span className="inline-flex items-center gap-1 text-[13px] text-ink-2"><Icon name="check" size={13} />Installed</span>
            : (
              <Button size="xs" variant="default" icon="arrowDown" onClick={onDownload} disabled={!canDownload(result) || downloading}>
                {downloading ? 'Downloading' : 'Download'}
              </Button>
            )}
        </div>
      </div>
    </Row>
  )
}
