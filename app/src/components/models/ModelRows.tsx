import { currentIntlLocale } from '../../lib/preferences/current-locale'
import { OriginalDiagnostic } from '../OriginalDiagnostic'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { useEffect, useRef, useState, type ReactNode } from 'react'
import clsx from 'clsx'
import { Icon } from '../../lib/icons'
import { Button } from '../ui'
import { VERDICT_BADGES, canDownload, formatBytes, pullPercent, type VerdictBadge } from '../../lib/engine'
import type { EngineInstalledModel, EngineStatus, HubModelSummary, ModelPreflight, PullProgress } from '../../lib/types'
import { repoDisplayName } from '../../lib/turn-stats-line'
import { downloadEstimate, downloadStateLabel, isDownloadPending } from './download-display'

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

export function InstalledRow({ model, status, onLoad, onUnload, onDelete, onTask, onChat, onSettings, hubCache, locked = false, minimal = false, favorite = false, onFavorite, favoriteDisabled = false, using = false, chatBlocked = false }: {
  minimal?: boolean
  model: EngineInstalledModel
  status: EngineStatus | null
  onLoad: () => void
  onUnload: () => void
  onDelete: () => Promise<void>
  onTask: () => void
  onChat: () => void
  onSettings?: () => void
  hubCache: string
  locked?: boolean
  favorite?: boolean
  onFavorite?: () => void
  favoriteDisabled?: boolean
  using?: boolean
  chatBlocked?: boolean
}) {
  const tx = useAuthoredText()
  const [confirming, setConfirming] = useState(false)
  const [busy, setBusy] = useState(false)
  const [deleteError, setDeleteError] = useState<string | null>(null)
  const deleteLock = useRef(false)
  const [menu, setMenu] = useState(false)
  const menuRef = useRef<HTMLDivElement>(null)
  const moreRef = useRef<HTMLButtonElement>(null)
  const active = status?.model === model.repo
  const running = active && status?.state === 'ready'
  const loading = active && status?.state === 'starting'
  const failed = active && status?.state === 'failed'
  const name = model.title ?? repoDisplayName(model.repo)
  const badge: VerdictBadge = running ? { label: tx("Running"), tone: 'ok' } : loading ? { label: tx("Loading"), tone: 'warn' } : failed ? { label: tx("Needs attention"), tone: 'bad' } : model.fitsMemory === false ? { label: tx("Needs more memory"), tone: 'warn' } : model.fitsMemory === true ? { label: tx("Fits this Mac"), tone: 'ok' } : { label: tx("Compatibility checked on use"), tone: 'muted' }
  useEffect(() => {
    if (!menu) return
    const outside = (event: MouseEvent) => { if (!menuRef.current?.contains(event.target as Node)) setMenu(false) }
    const escape = (event: KeyboardEvent) => { if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); setMenu(false); moreRef.current?.focus() } }
    document.addEventListener('mousedown', outside); document.addEventListener('keydown', escape, true)
    menuRef.current?.querySelector<HTMLButtonElement>('[role="menuitem"]:not(:disabled)')?.focus()
    return () => { document.removeEventListener('mousedown', outside); document.removeEventListener('keydown', escape, true) }
  }, [menu])
  const action = (callback: () => void) => { setMenu(false); callback() }
  const remove = async () => {
    if (deleteLock.current) return
    deleteLock.current = true; setBusy(true); setDeleteError(null)
    try { await onDelete(); setConfirming(false) } catch (error) { setDeleteError(error instanceof Error ? error.message : tx("This download could not be deleted. Try again.")) } finally { deleteLock.current = false; setBusy(false) }
  }
  return (
    <div data-testid="models-installed-row" className="min-w-0 border-b border-line py-4 last:border-b-0">
      <div className="flex min-w-0 flex-wrap items-center gap-3">
        {onFavorite && <button type="button" onClick={onFavorite} disabled={favoriteDisabled} aria-label={favorite ? tx("Remove {model} from favourites", { model: name }) : tx("Add {model} to favourites", { model: name })} aria-pressed={favorite} className={clsx('shrink-0 rounded-lg p-1.5 focus-visible:outline-2 focus-visible:outline-accent disabled:opacity-40', favorite ? 'text-accent' : 'text-ink-3 hover:text-ink')}><Icon name={favorite ? 'starFilled' : 'star'} size={17} /></button>}
        <div className="min-w-0 flex-1 basis-48">
          <button type="button" onClick={onSettings} disabled={!onSettings} aria-label={tx("Details for {value0}", { value0: name })} className="block max-w-full break-words rounded text-left text-[15px] font-medium text-ink hover:text-accent focus-visible:outline-2 focus-visible:outline-accent disabled:text-ink">{name}</button>
          <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-[13px] text-ink-2">
            <Badge badge={badge} />
            {model.weightsBytes != null && <span>{formatBytes(model.weightsBytes)} {tx("on disk")}</span>}
            {model.memoryFloorBytes != null && <span>{formatBytes(model.memoryFloorBytes)} {tx("memory")}</span>}
          </div>
          {failed && status?.message && <p className="mt-2 break-words text-[13px] text-danger">{status.message}</p>}
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <Button size="sm" variant={running ? 'default' : 'primary'} icon={running && !minimal ? 'message' : 'play'} onClick={running ? onChat : onLoad} disabled={busy || using || loading || (running ? minimal || chatBlocked : model.fitsMemory === false || locked || status?.state === 'starting')} title={!running && locked ? tx("Finish active local work before changing models") : model.fitsMemory === false ? tx("This model needs more available memory") : undefined}>{using || loading ? tx("Loading…") : running ? minimal ? tx("Running") : tx("Chat") : minimal ? tx("Start server") : tx("Use model")}</Button>
          <div ref={menuRef} className="relative">
            <button ref={moreRef} type="button" aria-label={tx("More actions for {value0}", { value0: name })} aria-haspopup="menu" aria-expanded={menu} onClick={() => setMenu((open) => !open)} className="rounded-lg px-2 py-2 text-ink-2 hover:bg-card-2 hover:text-ink focus-visible:outline-2 focus-visible:outline-accent"><Icon name="dots" size={17} /></button>
            {menu && <div role="menu" aria-label={tx("Actions for {value0}", { value0: name })} className="absolute right-0 top-full z-30 mt-1 min-w-48 rounded-xl border border-line-strong bg-card p-1 shadow-pop" onKeyDown={(event) => {
              if (!['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) return
              event.preventDefault(); const buttons = [...event.currentTarget.querySelectorAll<HTMLButtonElement>('[role="menuitem"]:not(:disabled)')]; const index = buttons.indexOf(document.activeElement as HTMLButtonElement); const next = event.key === 'Home' ? 0 : event.key === 'End' ? buttons.length - 1 : (index + (event.key === 'ArrowUp' ? -1 : 1) + buttons.length) % buttons.length; buttons[next]?.focus()
            }}>
              {onSettings && <button role="menuitem" className="block w-full rounded-lg px-3 py-2 text-left text-[13px] text-ink-2 hover:bg-card-2 focus:bg-card-2" onClick={() => action(onSettings)}>{tx("Model details")}</button>}
              {running && !minimal && <button role="menuitem" className="block w-full rounded-lg px-3 py-2 text-left text-[13px] text-ink-2 hover:bg-card-2 focus:bg-card-2" onClick={() => action(onTask)}>{tx("New worker")}</button>}
              {active && !failed && <button role="menuitem" disabled={locked} className="block w-full rounded-lg px-3 py-2 text-left text-[13px] text-ink-2 hover:bg-card-2 focus:bg-card-2 disabled:opacity-40" onClick={() => action(onUnload)}>{tx("Unload model")}</button>}
              {model.source === 'download' && <button role="menuitem" disabled={busy || running || loading} className="block w-full rounded-lg px-3 py-2 text-left text-[13px] text-danger hover:bg-card-2 focus:bg-card-2 disabled:opacity-40" onClick={() => action(() => setConfirming(true))}>{tx("Delete download…")}</button>}
            </div>}
          </div>
        </div>
      </div>
      {confirming && <div className="mt-3 flex flex-wrap items-center gap-2 rounded-lg border border-danger/25 bg-danger/10 px-3 py-2 text-[13px] text-ink-2"><span className="min-w-0 flex-1 break-words">{tx("Delete ")}{model.repo}{model.weightsBytes != null ? ` (${formatBytes(model.weightsBytes)})` : ''} {tx("from the shared Hugging Face cache at ")}{hubCache}{tx("? Other apps that read this cache lose it too.")}</span>{deleteError && <p role="alert" className="w-full text-danger">{deleteError}</p>}<Button size="xs" variant="danger" icon="trash" onClick={() => void remove()} disabled={busy}>{busy ? tx("Deleting") : tx("Delete")}</Button><Button size="xs" variant="ghost" onClick={() => setConfirming(false)} disabled={busy}>{tx("Cancel")}</Button></div>}
    </div>
  )
}

export function PullRow({ pull, onCancel, onRetry }: { pull: PullProgress; onCancel: () => void | Promise<unknown>; onRetry: () => void | Promise<unknown> }) {
  const tx = useAuthoredText()
  const [now, setNow] = useState(Date.now)
  const [pending, setPending] = useState(false)
  const [actionError, setActionError] = useState<string | null>(null)
  const acting = useRef(false)
  useEffect(() => {
    if (pull.state !== 'running') return
    const timer = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(timer)
  }, [pull.state])
  const percent = pullPercent(pull)
  const running = pull.state === 'running'
  const queued = pull.state === 'queued'
  const verifying = pull.state === 'verifying'
  const estimate = downloadEstimate(pull, now)
  const act = async (callback: () => void | Promise<unknown>) => {
    if (acting.current) return
    acting.current = true
    setPending(true)
    setActionError(null)
    try { await callback() } catch (error) { setActionError(error instanceof Error ? error.message : tx("The download action could not complete. Try again.")) }
    finally { acting.current = false; setPending(false) }
  }
  return (
    <Row testId="models-pull-row">
      <div className="flex flex-wrap items-start gap-3">
        <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-surface text-ink-2"><Icon name={isDownloadPending(pull) ? 'arrowDown' : pull.state === 'done' ? 'check' : 'warning'} size={16} /></span>
        <div className="min-w-0 flex-1">
          <div className="flex min-w-0 flex-wrap items-center justify-between gap-x-3 gap-y-1">
            <span className="min-w-0 break-all text-[14px] font-medium text-ink">{pull.repo}</span>
            <span className="text-[12px] text-ink-2">{downloadStateLabel(pull)}</span>
          </div>
          {!queued && <div role="progressbar" aria-label={tx("Download progress for {value0}", { value0: pull.repo })} aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent ?? undefined} aria-valuetext={verifying ? 'Download complete, verifying checkpoint' : percent != null ? `${percent}%` : 'Size unknown'} className="mt-2 h-1.5 overflow-hidden rounded-full bg-ink-5">
            <div className={clsx('h-full rounded-full transition-[width] duration-300', pull.state === 'failed' ? 'bg-danger' : 'bg-brand', (running || verifying) && percent == null && 'animate-pulse')} style={{ width: `${percent ?? (pull.state === 'done' || running || verifying ? 100 : 0)}%` }} />
          </div>}
          {!queued && <div className="mt-1.5 flex flex-wrap gap-x-3 gap-y-1 text-[12px] text-ink-2">
            <span>{formatBytes(pull.receivedBytes)}{pull.totalBytes ? tx(" of {value0}", { value0: formatBytes(pull.totalBytes) }) : tx(" saved")}{percent != null ? ` · ${percent}%` : ''}</span>
            {running && <span title={tx("Estimated from bytes written to the model cache; the remaining time changes with transfer speed.")}>{estimate ?? tx("Measuring transfer speed…")}</span>}
          </div>}
          {pull.companionRepos?.length ? <div className="mt-1 break-all text-[12px] text-ink-2">{tx("Includes companion: ")}{pull.companionRepos.join(', ')}</div> : null}
          {pull.message && <div className={clsx('mt-1.5 break-words text-[12px] leading-relaxed', pull.state === 'failed' ? 'text-danger' : 'text-ink-2')}>{pull.message}</div>}
          {actionError && <p role="alert" className="mt-1.5 break-words text-[12px] text-danger">{actionError}</p>}
        </div>
        {(running || queued)
          ? <Button size="xs" variant="default" icon="x" disabled={pending} onClick={() => void act(onCancel)} aria-label={`${queued ? 'Remove queued download' : 'Cancel download'} ${pull.repo}`}>{queued ? tx("Remove") : tx("Cancel")}</Button>
          : !verifying && pull.state !== 'done' && <Button size="xs" variant="default" icon="refresh" disabled={pending} onClick={() => void act(onRetry)} aria-label={tx("Resume download {value0}", { value0: pull.repo })}>{tx("Resume")}</Button>}
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
  const tx = useAuthoredText()
  const result = preflight?.result ?? null
  const badges: VerdictBadge[] = []
  if (result) badges.push(VERDICT_BADGES[result.verdict])
  else if (summary?.gated) badges.push(VERDICT_BADGES.gated)
  if (result && result.verdict !== 'too-big' && result.fitsMemory === false) badges.push(VERDICT_BADGES['too-big'])
  if (result && result.verdict !== 'disk' && result.fitsDisk === false) badges.push(VERDICT_BADGES.disk)
  const downloading = isDownloadPending(pull)
  return (
    <Row testId={testId}>
      <div className="flex flex-wrap items-start gap-3">
        <div className="min-w-0 flex-1 basis-40">
          <div className="flex min-w-0 flex-wrap items-center gap-2">
            <span className="truncate text-[14px] font-medium text-ink">{title ?? repo}</span>
            {badges.map((badge) => <Badge key={badge.label} badge={badge} />)}
            {preflight?.loading && <span className="text-[12px] text-ink-2">{tx("Checking…")}</span>}
          </div>
          {title && <div className="mt-0.5 truncate font-mono text-[12px] text-ink-2" title={repo}>{repo}</div>}
          <Facts items={[
            result?.family ?? (summary?.modelType ? `model_type ${summary.modelType}` : null),
            result?.totalBytes != null ? `${formatBytes(result.totalBytes)} download` : null,
            result?.memoryFloorBytes != null ? `needs ${formatBytes(result.memoryFloorBytes)} of ${formatBytes(result.memoryLimitBytes)} allowed` : null,
            result?.drafter ? `with drafter ${result.drafter.repo}` : null,
            summary?.downloads != null ? `${summary.downloads.toLocaleString(currentIntlLocale())} downloads` : null,
            note,
          ]} />
          {result && result.reasons.length > 0 && <div className="mt-1.5 text-[13px] leading-snug text-ink-2">{result.reasons[0]}</div>}
          {preflight?.error && <div className="mt-1.5 text-[13px] text-danger">{<OriginalDiagnostic>{preflight.error}</OriginalDiagnostic>}</div>}
        </div>
        <div className="flex shrink-0 items-center gap-1.5">
          {!result && !preflight?.loading && <Button size="xs" variant="ghost" icon="check" onClick={onCheck}>{tx("Check")}</Button>}
          {result?.verdict === 'installed' && !downloading
            ? <span className="inline-flex items-center gap-1 text-[13px] text-ink-2"><Icon name="check" size={13} />{tx("Installed")}</span>
            : (
              <Button size="xs" variant="default" icon="arrowDown" onClick={onDownload} disabled={!canDownload(result) || downloading}>
                {pull?.state === 'queued' ? tx("Queued") : pull?.state === 'verifying' ? tx("Verifying") : downloading ? tx("Downloading") : tx("Download")}
              </Button>
            )}
        </div>
      </div>
    </Row>
  )
}
