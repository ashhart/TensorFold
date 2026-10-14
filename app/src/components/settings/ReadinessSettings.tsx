import { useEffect, useMemo, useState } from 'react'
import clsx from 'clsx'
import { api } from '../../lib/ipc'
import { Icon } from '../../lib/icons'
import { buildReadinessDoctor, listReadinessProviderProbes, summarizeDoctor, type DoctorArea, type DoctorCheck, type DoctorStatus, type ProbeState, type RemoteSnapshotProbeValue } from '../../lib/readiness-doctor'
import type { ConnectorEvent, CliDefinition, CliToolStatus, ConnectorDefinition, PersistedProviderStatus, PersistenceFailure, Readiness, SecurityControlSnapshot, WebhookEndpoint } from '../../lib/types'
import { Button } from '../ui'
import { Section, SettingsPane, Spinner, type SettingsSectionContext } from './shared'

const AREA_LABELS: Record<DoctorArea, string> = {
  models: 'Models',
  connectors: 'Connectors',
  remote: 'Remote',
  terminals: 'Terminals',
  security: 'Security',
}

const AREA_ORDER: DoctorArea[] = ['models', 'remote', 'connectors', 'terminals', 'security']

const STATUS_META: Record<DoctorStatus, { label: string; icon: 'check' | 'warning' | 'refresh' | 'x'; className: string }> = {
  ok: { label: 'Ready', icon: 'check', className: 'border-positive/30 bg-positive/10 text-positive' },
  warning: { label: 'Needs review', icon: 'warning', className: 'border-warn/40 bg-warn/10 text-warn' },
  error: { label: 'Blocked', icon: 'x', className: 'border-danger/35 bg-danger/10 text-danger' },
  checking: { label: 'Checking', icon: 'refresh', className: 'border-line bg-ink-5 text-ink-3' },
}

function areaGroups(checks: DoctorCheck[]) {
  return AREA_ORDER.map((area) => ({ area, checks: checks.filter((check) => check.area === area) })).filter((group) => group.checks.length)
}

function statusPill(status: DoctorStatus) {
  const meta = STATUS_META[status]
  return (
    <span className={clsx('inline-flex h-6 shrink-0 items-center gap-1.5 rounded-md border px-2 text-[12px] font-medium', meta.className)}>
      {status === 'checking' ? <Spinner size={12} /> : <Icon name={meta.icon} size={12} />}
      {meta.label}
    </span>
  )
}

export function ReadinessSettings({ ctx }: { ctx: SettingsSectionContext }) {
  const [providerReadiness, setProviderReadiness] = useState<Record<string, ProbeState<Readiness>>>({})
  const [localClis, setLocalClis] = useState<ProbeState<Array<CliDefinition | CliToolStatus>>>({ status: 'checking' })
  const [remoteProbes, setRemoteProbes] = useState<Record<string, ProbeState<RemoteSnapshotProbeValue>>>({})
  const [connectors, setConnectors] = useState<ProbeState<ConnectorDefinition[]>>({ status: 'checking' })
  const [connectorEvents, setConnectorEvents] = useState<ConnectorEvent[]>([])
  const [providerStatus, setProviderStatus] = useState<PersistedProviderStatus | null>(null)
  const [webhooks, setWebhooks] = useState<WebhookEndpoint[]>([])
  const [securityControls, setSecurityControls] = useState<SecurityControlSnapshot | null>(ctx.securityControls ?? null)
  const [persistenceFailure, setPersistenceFailure] = useState<PersistenceFailure | null>(ctx.securityControls?.lastPersistenceFailure ?? null)
  const [refreshNonce, setRefreshNonce] = useState(0)

  const providerEntries = useMemo(() => listReadinessProviderProbes(ctx.draft), [ctx.draft.providers])
  const remoteHosts = ctx.draft.remoteOrchestrators ?? []

  useEffect(() => {
    api.getProviderStatus().then(setProviderStatus).catch(() => setProviderStatus(null))
    api.getWebhooks().then((snapshot) => setWebhooks(snapshot.endpoints)).catch(() => setWebhooks([]))
    api.getSecurityControls().then((snapshot) => {
      setSecurityControls(snapshot)
      setPersistenceFailure(snapshot.lastPersistenceFailure)
    }).catch(() => {})
  }, [refreshNonce])

  useEffect(() => {
    setSecurityControls(ctx.securityControls ?? null)
    setPersistenceFailure(ctx.securityControls?.lastPersistenceFailure ?? null)
  }, [ctx.securityControls])

  useEffect(() => api.onPersistenceFailure(setPersistenceFailure), [])

  useEffect(() => {
    let cancelled = false
    const initial = Object.fromEntries(providerEntries.map((entry) => [entry.key, { status: 'checking' as const }]))
    setProviderReadiness(initial)
    for (const entry of providerEntries) {
      api.getModelReadiness({ providerId: entry.providerId, accountId: entry.accountId })
        .then((value) => {
          if (cancelled) return
          setProviderReadiness((current) => ({ ...current, [entry.key]: { status: 'ok', value } }))
        })
        .catch((error) => {
          if (cancelled) return
          setProviderReadiness((current) => ({ ...current, [entry.key]: { status: 'error', message: String(error?.message ?? error) } }))
        })
    }
    return () => { cancelled = true }
  }, [providerEntries, refreshNonce])

  useEffect(() => {
    let cancelled = false
    setLocalClis({ status: 'checking' })
    // Probed statuses (installed/version) let the doctor prove each launcher distinctly; the
    // plain registry fallback still renders entries as 'not probed yet' rather than hiding them.
    api.listCliTools()
      .then((value) => { if (!cancelled) setLocalClis({ status: 'ok', value }) })
      .catch(() => api.listClis()
        .then((value) => { if (!cancelled) setLocalClis({ status: 'ok', value }) })
        .catch((error) => { if (!cancelled) setLocalClis({ status: 'error', message: String(error?.message ?? error) }) }))
    return () => { cancelled = true }
  }, [refreshNonce])

  useEffect(() => {
    let cancelled = false
    setConnectors({ status: 'checking' })
    api.getAssistantSnapshot()
      .then((snapshot) => { if (!cancelled) { setConnectors({ status: 'ok', value: snapshot.connectors }); setConnectorEvents(snapshot.events ?? []) } })
      .catch((error) => { if (!cancelled) setConnectors({ status: 'error', message: String(error?.message ?? error) }) })
    return () => { cancelled = true }
  }, [refreshNonce])

  useEffect(() => {
    let cancelled = false
    const enabled = remoteHosts.filter((host) => host.enabled && host.serverUrl?.trim() && host.authToken?.trim())
    setRemoteProbes(Object.fromEntries(enabled.map((host) => [host.id, { status: 'checking' as const }])))
    for (const host of enabled) {
      api.getRemoteSnapshot(host.id)
        .then((value) => { if (!cancelled) setRemoteProbes((current) => ({ ...current, [host.id]: { status: value.ok === false ? 'error' : 'ok', value } })) })
        .catch((error) => { if (!cancelled) setRemoteProbes((current) => ({ ...current, [host.id]: { status: 'error', message: String(error?.message ?? error) } })) })
    }
    return () => { cancelled = true }
  }, [remoteHosts, refreshNonce])

  const checks = useMemo(() => buildReadinessDoctor({
    settings: ctx.draft,
    providerReadiness,
    providerStatus,
    localClis,
    remoteProbes,
    studioRemote: ctx.studioStatus,
    connectors,
    connectorEvents,
    webhooks,
    persistenceFailure,
    securityControls,
  }), [ctx.draft, providerReadiness, providerStatus, localClis, remoteProbes, ctx.studioStatus, connectors, connectorEvents, webhooks, persistenceFailure, securityControls])
  const summary = summarizeDoctor(checks)
  const grouped = areaGroups(checks)

  const jump = (section?: string) => {
    if (section && typeof ctx.goToSection === 'function') ctx.goToSection(section)
  }

  return (
    <SettingsPane>
      <Section title="Readiness" desc="Live checks for the runtime pieces TensorFold depends on.">
        <div className="mb-4 grid gap-2 sm:grid-cols-4">
          <SummaryCell label="Ready" value={summary.ok} status="ok" />
          <SummaryCell label="Review" value={summary.warning} status="warning" />
          <SummaryCell label="Blocked" value={summary.error} status="error" />
          <SummaryCell label="Checking" value={summary.checking} status="checking" />
        </div>
        <div className="mb-5 flex items-center justify-between gap-3 rounded-lg border border-line bg-card px-3 py-2.5">
          <div className="min-w-0 text-[13px] text-ink-2">
            {summary.total} checks across models, connectors, remote orchestration, terminals, and security.
          </div>
          <Button variant="default" size="sm" icon="refresh" onClick={() => setRefreshNonce((value) => value + 1)}>Refresh</Button>
        </div>
        <div className="space-y-5">
          {grouped.map((group) => (
            <div key={group.area}>
              <div className="mb-2 flex items-center gap-2 text-[13px] font-semibold uppercase tracking-[0.06em] text-ink-3">
                <Icon name={group.area === 'models' ? 'cpu' : group.area === 'connectors' ? 'plug' : group.area === 'remote' ? 'monitor' : group.area === 'terminals' ? 'terminal' : 'shield'} size={13} />
                {AREA_LABELS[group.area]}
              </div>
              <div className="overflow-hidden rounded-lg border border-line bg-card">
                {group.checks.map((item, index) => (
                  <div key={item.id} className={clsx('flex items-start gap-3 px-3 py-3', index > 0 && 'border-t border-line')}>
                    {statusPill(item.status)}
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-[13px] font-medium text-ink">{item.title}</div>
                      <div className="mt-0.5 text-[13px] leading-relaxed text-ink-3">{item.detail}</div>
                    </div>
                    {item.section && item.action && (
                      <Button variant="ghost" size="sm" icon="arrowRight" onClick={() => jump(item.section)}>{item.action}</Button>
                    )}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </Section>
    </SettingsPane>
  )
}

function SummaryCell({ label, value, status }: { label: string; value: number; status: DoctorStatus }) {
  const meta = STATUS_META[status]
  return (
    <div className={clsx('rounded-lg border px-3 py-2.5', meta.className)}>
      <div className="text-[20px] font-semibold leading-none">{value}</div>
      <div className="mt-1 text-[12px] font-medium">{label}</div>
    </div>
  )
}
