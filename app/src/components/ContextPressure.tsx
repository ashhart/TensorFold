import { useEffect, useMemo, useState } from 'react'
import clsx from 'clsx'
import { api } from '../lib/ipc'
import type { ContextPressureAction, ContextPressureSnapshot, ContextPressureStatus, ContextPressureThread } from '../lib/types'
import { Icon } from '../lib/icons'
import { formatUsageTokens } from '../lib/provider-usage'
import { Button } from './ui'

const ACTION_LABEL: Record<ContextPressureAction, string> = {
  continue: 'Continue',
  summarize_next: 'Summarize next',
  compact_now: 'Compact now',
  reduce_pins: 'Reduce pins',
  split_thread: 'Split thread',
}

const STATUS_CLASS: Record<ContextPressureStatus, string> = {
  idle: 'text-ink-3 bg-ink-5',
  healthy: 'text-positive bg-positive/12',
  warm: 'text-ink-2 bg-ink-5',
  pressured: 'text-warn bg-warn/12',
  critical: 'text-danger bg-danger/10',
}

export function ContextPressure() {
  const [snapshot, setSnapshot] = useState<ContextPressureSnapshot | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const refresh = async () => {
    setLoading(true)
    setError('')
    try {
      setSnapshot(await api.getContextPressure())
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void refresh()
  }, [])

  const localThreads = useMemo(() => snapshot?.threads.filter((thread) => thread.kind === 'local') ?? [], [snapshot])
  const primary = localThreads[0] ?? snapshot?.threads[0] ?? null
  const registry = snapshot?.registry ?? []

  return (
    <section className="rounded-lg border border-line bg-card p-3.5 shadow-card">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2 text-[13px] font-semibold text-ink">
            <Icon name="cpu" size={15} className="text-ink-3" />
            <span>Context pressure</span>
          </div>
          <div className="mt-1 text-[13px] text-ink-3">
            {snapshot ? `${snapshot.totals.localThreads} local threads / ${formatTokens(snapshot.totals.estimatedTokens)} estimated` : 'Local model telemetry'}
          </div>
        </div>
        <Button size="xs" variant="ghost" icon="clock" onClick={refresh} disabled={loading}>
          {loading ? 'Refreshing' : 'Refresh'}
        </Button>
      </div>

      {error && (
        <div className="mt-3 rounded-lg border border-danger/25 bg-danger/10 p-2.5 text-[13px] leading-snug text-danger">
          {error}
        </div>
      )}

      {!snapshot && !error && (
        <div className="mt-3 rounded-lg bg-ink-5 px-3 py-2.5 text-[13px] text-ink-3">
          {loading ? 'Estimating context pressure...' : 'No pressure snapshot loaded.'}
        </div>
      )}

      {snapshot && (
        <>
          <div className="mt-3 grid grid-cols-3 gap-2">
            <PressureStat label="Max pressure" value={formatPercent(snapshot.totals.maxPressureRatio)} detail={primary?.model ?? 'No local model'} />
            <PressureStat label="Pinned" value={formatTokens(snapshot.totals.pinnedTokens)} detail="Pinned + recent" />
            <PressureStat label="Summarizable" value={formatTokens(snapshot.totals.summarizableTokens)} detail="Foldable middle" />
          </div>

          {primary ? (
            <PrimaryThread thread={primary} />
          ) : (
            <div className="mt-3 rounded-lg bg-ink-5 px-3 py-2.5 text-[13px] text-ink-3">
              No local-model thread history yet.
            </div>
          )}

          <div className="mt-4">
            <div className="mb-2 flex items-center justify-between gap-2">
              <div className="text-[13px] font-medium text-ink-2">Model registry</div>
              <div className="text-[12px] text-ink-4">{registry.length} local profiles</div>
            </div>
            <div className="space-y-1.5">
              {registry.slice(0, 4).map((profile) => (
                <div key={profile.id} className="grid grid-cols-[minmax(0,1fr)_70px_70px] items-center gap-2 rounded-lg bg-card-2 px-2.5 py-2 text-[13px]">
                  <div className="min-w-0">
                    <div className="truncate font-medium text-ink">{profile.model}</div>
                    <div className="truncate text-[12px] text-ink-4">{profile.providerName} / {profile.source}</div>
                  </div>
                  <div className="text-right font-mono text-[12px] text-ink-3">{formatTokens(profile.nominalContextTokens)}</div>
                  <div className="text-right font-mono text-[12px] text-ink-3">{formatTokens(profile.recommendedWorkingTokens)}</div>
                </div>
              ))}
              {registry.length === 0 && (
                <div className="rounded-lg bg-ink-5 px-3 py-2.5 text-[13px] text-ink-3">
                  Add a local provider to populate the registry.
                </div>
              )}
            </div>
          </div>
        </>
      )}
    </section>
  )
}

function PrimaryThread({ thread }: { thread: ContextPressureThread }) {
  const pct = Math.min(100, Math.round(thread.pressureRatio * 100))
  return (
    <div className="mt-3 rounded-lg bg-card-2 p-3">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <div className="truncate text-[13px] font-medium text-ink">{thread.title}</div>
            <span className={clsx('shrink-0 rounded-md px-1.5 py-0.5 text-[12px] font-medium capitalize', STATUS_CLASS[thread.status])}>
              {thread.status}
            </span>
          </div>
          <div className="mt-0.5 truncate text-[13px] text-ink-4">
            {thread.providerName} / {thread.model ?? 'default'} / {timeAgo(thread.updatedAt)}
          </div>
        </div>
        <div className="shrink-0 text-right">
          <div className="text-[13px] font-semibold text-ink">{formatPercent(thread.pressureRatio)}</div>
          <div className="text-[12px] text-ink-4">{formatTokens(thread.estimatedTokens)} / {formatTokens(thread.workingLimitTokens)}</div>
        </div>
      </div>

      <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-ink-5">
        <div className={clsx('h-full rounded-full', thread.status === 'critical' ? 'bg-danger' : thread.status === 'pressured' ? 'bg-warn' : 'bg-ink-2')} style={{ width: `${pct}%` }} />
      </div>

      <div className="mt-3 grid grid-cols-2 gap-2 text-[13px]">
        <GuidanceBlock title="Pinned" value={formatTokens(thread.guidance.pinnedTokens + thread.guidance.recentTokens)} items={thread.guidance.pinned} />
        <GuidanceBlock title="Summarizable" value={formatTokens(thread.guidance.summarizableTokens)} items={thread.guidance.summarizable} />
      </div>

      <div className="mt-3 rounded-lg bg-ink-5 px-3 py-2.5">
        <div className="flex items-center justify-between gap-2">
          <span className="text-[13px] font-medium text-ink">{ACTION_LABEL[thread.guidance.recommendedAction]}</span>
          <span className="text-[12px] text-ink-4">Tools {formatTokens(thread.guidance.toolTokens)}</span>
        </div>
        <div className="mt-1 text-[13px] leading-snug text-ink-3">{thread.guidance.reason}</div>
        <div className="mt-1 text-[13px] leading-snug text-ink-4">{thread.guidance.cacheHint}</div>
      </div>
    </div>
  )
}

function GuidanceBlock({ title, value, items }: { title: string; value: string; items: string[] }) {
  return (
    <div className="min-w-0 rounded-lg bg-ink-5 px-2.5 py-2">
      <div className="flex items-center justify-between gap-2">
        <span className="text-[13px] font-medium text-ink-2">{title}</span>
        <span className="font-mono text-[12px] text-ink-3">{value}</span>
      </div>
      <div className="mt-1 space-y-0.5">
        {(items.length ? items : ['None']).slice(0, 3).map((item) => (
          <div key={item} className="truncate text-[12px] text-ink-4">{item}</div>
        ))}
      </div>
    </div>
  )
}

function PressureStat({ label, value, detail }: { label: string; value: string; detail: string }) {
  return (
    <div className="min-w-0 rounded-lg bg-ink-5 px-2.5 py-2">
      <div className="truncate text-[12px] text-ink-4">{label}</div>
      <div className="mt-0.5 truncate text-[15px] font-semibold text-ink">{value}</div>
      <div className="mt-0.5 truncate text-[12px] text-ink-4">{detail}</div>
    </div>
  )
}

function formatPercent(value: number) {
  return `${Math.round(value * 100)}%`
}

const formatTokens = formatUsageTokens

function timeAgo(ts: number) {
  const delta = Math.max(0, Date.now() - ts)
  const minutes = Math.floor(delta / 60_000)
  if (minutes < 1) return 'now'
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  const days = Math.floor(hours / 24)
  return `${days}d ago`
}
