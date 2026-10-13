import type { ConnectorEvent, CliDefinition, CliToolStatus, ConnectorDefinition, ConnectorId, PersistedProviderStatus, PersistenceFailure, Readiness, RemoteControlWorkSnapshot, SecurityControlSnapshot, Settings, WebhookEndpoint, SecretRotationMetadata } from './types'
import { providerStatusView } from './provider-status'

export type DoctorStatus = 'ok' | 'warning' | 'error' | 'checking'
export type DoctorArea = 'models' | 'connectors' | 'remote' | 'terminals' | 'security'

export interface ProbeState<T = unknown> {
  status: 'checking' | 'ok' | 'error'
  value?: T
  message?: string
}

export interface DoctorCheck {
  id: string
  area: DoctorArea
  title: string
  status: DoctorStatus
  detail: string
  section?: string
  action?: string
}

export interface DoctorSummary {
  ok: number
  warning: number
  error: number
  checking: number
  total: number
}

// The slice of a remote host snapshot the doctor derives subchecks from. Callers pass the
// full RemoteOrchestratorSnapshotResult; only these fields are read.
export interface RemoteSnapshotProbeValue {
  ok: boolean
  error?: string
  generatedAt?: number
  work?: RemoteControlWorkSnapshot
}

export interface ReadinessDoctorInput {
  settings: Settings
  providerReadiness?: Record<string, ProbeState<Readiness>>
  // Persisted provider rate-limit status (VST-037): folds real session/weekly pace into the
  // per-provider readiness detail, and emits a warn subcheck when a window is exhausted.
  providerStatus?: PersistedProviderStatus | null
  localClis?: ProbeState<Array<CliDefinition | CliToolStatus>>
  remoteProbes?: Record<string, ProbeState<RemoteSnapshotProbeValue>>
  studioRemote?: {
    enabled: boolean
    listening: boolean
    clients?: number
    url?: string | null
    lastError?: string | null
  }
  // Live connector proof (assistant snapshot connectors): sync status, sample items,
  // capabilities, permission posture. Without it a configured connector caps at 'warning' —
  // ready-without-proof is exactly what VST-001 forbids.
  connectors?: ProbeState<ConnectorDefinition[]>
  // Recent connector events (VST-013): successful OUTBOUND ones prove write capability.
  connectorEvents?: Array<Pick<ConnectorEvent, 'connectorId' | 'direction' | 'createdAt' | 'approvalStatus'>>
  webhooks?: WebhookEndpoint[]
  persistenceFailure?: PersistenceFailure | null
  securityControls?: Pick<SecurityControlSnapshot, 'localStateKeyMode' | 'externalSecretKeyRequired' | 'copiedProfileDecryptable' | 'legacySafeStorageSecrets' | 'legacySafeStorageBackupPath' | 'quarantinedProtectedFiles' | 'quarantinedProtectedPaths' | 'quarantineBackupPaths'> | null
  now?: number
}

export interface ProviderReadinessProbeEntry {
  providerId: string
  accountId: string | null
  key: string
}

const MASK = '••••••••'
const HEARTBEAT_STALE_MS = 90_000
const SECRET_STALE_AFTER_MS = 90 * 24 * 60 * 60_000

function hasValue(value: unknown): boolean {
  if (typeof value !== 'string') return value != null
  const trimmed = value.trim()
  return Boolean(trimmed && trimmed !== 'undefined' && trimmed !== 'null')
}

function hasSecret(value: unknown): boolean {
  if (typeof value !== 'string') return Boolean(value)
  const trimmed = value.trim()
  return Boolean(trimmed && (trimmed === MASK || trimmed.length > 0))
}

function hasAllowlist(value: unknown): boolean {
  return typeof value === 'string' && value.split(/[\s,;]+/).some((item) => item.trim().length > 0)
}

function ago(from: number | null | undefined, now: number): string {
  if (!from) return 'never'
  const delta = Math.max(0, now - from)
  if (delta < 45_000) return 'just now'
  if (delta < 90_000) return '1m ago'
  if (delta < 60 * 60_000) return `${Math.round(delta / 60_000)}m ago`
  if (delta < 48 * 60 * 60_000) return `${Math.round(delta / (60 * 60_000))}h ago`
  return `${Math.round(delta / (24 * 60 * 60_000))}d ago`
}

function plural(count: number, word: string): string {
  return `${count} ${word}${count === 1 ? '' : 's'}`
}

function check(id: string, area: DoctorArea, title: string, status: DoctorStatus, detail: string, section?: string, action?: string): DoctorCheck {
  return { id, area, title, status, detail, section, action }
}

function credentialAgeCheck(id: string, area: DoctorArea, title: string, section: string, values: Record<string, unknown>, rotations: SecretRotationMetadata | undefined, now: number): DoctorCheck | null {
  const configured = Object.entries(values).filter(([, value]) => hasSecret(value))
  if (!configured.length) return null
  const states = configured.map(([slot]) => {
    const rotatedAt = rotations?.[slot]
    return { slot, state: !Number.isFinite(rotatedAt) ? 'unknown' : now - Number(rotatedAt) > SECRET_STALE_AFTER_MS ? 'stale' : 'fresh' }
  })
  const unknown = states.filter((item) => item.state === 'unknown')
  const stale = states.filter((item) => item.state === 'stale')
  if (unknown.length) return check(id, area, title, 'warning', `Credential rotation age is unknown for: ${unknown.map((item) => item.slot).join(', ')}. Rotate or re-save a new value to establish provenance.`, section, 'Rotate')
  if (stale.length) return check(id, area, title, 'warning', `Credentials are older than 90 days: ${stale.map((item) => item.slot).join(', ')}.`, section, 'Rotate')
  return check(id, area, title, 'ok', 'Credential rotation age is recorded and under 90 days.', section)
}

function isCliStatus(cli: CliDefinition | CliToolStatus): cli is CliToolStatus {
  return typeof (cli as CliToolStatus).installed === 'boolean'
}

interface ConnectorProofInput {
  state: 'loaded' | 'checking' | 'unavailable'
  byId: Map<ConnectorId, ConnectorDefinition>
  now: number
  // Write-capability proof (VST-013): newest successful OUTBOUND action per connector.
  outboundOkAt?: Map<ConnectorId, number>
}

// Connector families whose mode claims WRITE (outbound actions) — they must prove it with
// a recorded successful outbound before rendering fully ok. Kept in sync with the runtime
// source of truth (electron/agent/connector-action-safety.ts) by a test.
export const WRITE_CAPABLE_CONNECTORS = new Set<ConnectorId>([
  'email', 'slack', 'discord', 'mattermost', 'matrix', 'googlechat', 'whatsapp', 'telegram', 'calendar', 'controller', 'webhook',
])

// Evidence gate: a connector whose configuration looks complete still needs live proof
// (a successful sync or an indexed sample item) before it may render 'ok'.
function proofVerdict(id: ConnectorId, title: string, baseDetail: string, proof: ConnectorProofInput, section: string, checkId: string): DoctorCheck {
  if (proof.state === 'checking') return check(checkId, 'connectors', title, 'checking', 'Verifying sync status and sample items.', section)
  if (proof.state === 'unavailable') return check(checkId, 'connectors', title, 'warning', `${baseDetail} Proof unavailable — the connector snapshot did not load, so readiness is unproven.`, section, 'Retry')
  const live = proof.byId.get(id)
  if (!live) return check(checkId, 'connectors', title, 'warning', `${baseDetail} No live connector state yet — run a sync or setup test to prove it.`, section, 'Test')
  if (live.error || live.status === 'error') return check(checkId, 'connectors', title, 'error', `${title}: ${live.error || 'connector reported an error state.'}`, section, 'Open')
  if (live.permissionStatus === 'denied') return check(checkId, 'connectors', title, 'error', `${title} permissions are denied: ${live.permissionDetail || 'access was refused.'}`, section, 'Fix')
  const lastSync = live.lastSyncAt ?? null
  const lastEvent = live.lastEventAt ?? null
  if (!lastSync && !lastEvent) {
    return check(checkId, 'connectors', title, 'warning', `${baseDetail} Configured but unproven — no successful sync and no indexed sample item yet.`, section, 'Test')
  }
  const parts: string[] = [baseDetail.replace(/\.$/, '')]
  parts.push(`last sync ${ago(lastSync, proof.now)}`)
  if (lastEvent) parts.push(`sample item ${ago(lastEvent, proof.now)}`)
  if (live.tools.length) parts.push(live.tools.length === 1 ? '1 capability' : `${live.tools.length} capabilities`)
  if (live.pendingApprovals > 0) parts.push(`${plural(live.pendingApprovals, 'approval')} pending`)
  if (live.status === 'mock') {
    return check(checkId, 'connectors', title, 'warning', `${parts.join(' · ')} · simulated (demo mode), not proven against a real backend.`, section)
  }
  if (live.permissionStatus === 'degraded') {
    return check(checkId, 'connectors', title, 'warning', `${parts.join(' · ')} · degraded: ${live.permissionDetail || 'partial access.'}`, section, 'Review')
  }
  // Write proof (VST-013): read evidence alone doesn't prove a read+write connector.
  if (WRITE_CAPABLE_CONNECTORS.has(id)) {
    const outboundAt = proof.outboundOkAt?.get(id) ?? null
    if (!outboundAt) {
      return check(checkId, 'connectors', title, 'warning', `${parts.join(' · ')} · write unproven — no successful outbound action recorded yet.`, section, 'Test')
    }
    parts.push(`outbound ok ${ago(outboundAt, proof.now)}`)
  }
  return check(checkId, 'connectors', title, 'ok', `${parts.join(' · ')}.`, section)
}

function sourceCheck(
  id: ConnectorId,
  title: string,
  sources: Array<{ enabled?: boolean; label?: string; lastError?: string | null }>,
  ready: (source: any) => string | null,
  emptyDetail: string,
  proof: ConnectorProofInput,
  section = 'connectors',
): DoctorCheck {
  const checkId = `connector:${id === 'docs' ? 'docs' : id}`
  if (!sources.length) return check(checkId, 'connectors', title, 'warning', emptyDetail, section, 'Configure')
  const enabled = sources.filter((source) => source.enabled !== false)
  if (!enabled.length) return check(checkId, 'connectors', title, 'warning', `${title} has sources, but none are enabled.`, section, 'Enable')
  const errored = enabled.find((source) => source.lastError)
  if (errored) return check(checkId, 'connectors', title, 'error', `${errored.label || title}: ${errored.lastError}`, section, 'Open')
  const missing = enabled.map((source) => ready(source)).find(Boolean)
  if (missing) return check(checkId, 'connectors', title, 'error', missing, section, 'Fix')
  return proofVerdict(id, title, `${plural(enabled.length, 'enabled source')} configured.`, proof, section, checkId)
}

function providerProbeKey(providerId: string, accountId: string | null | undefined): string {
  return `${providerId}:${accountId ?? 'default'}`
}

export function listReadinessProviderProbes(settings: Settings): ProviderReadinessProbeEntry[] {
  return (settings.providers ?? []).flatMap((provider) => {
    const enabledAccounts = (provider.accounts ?? []).filter((account) => account.enabled !== false)
    const accountIds = enabledAccounts.length ? enabledAccounts.map((account) => account.id) : [provider.defaultAccountId ?? null]
    return accountIds.map((accountId) => ({ providerId: provider.id, accountId, key: providerProbeKey(provider.id, accountId) }))
  })
}

export function buildReadinessDoctor(input: ReadinessDoctorInput): DoctorCheck[] {
  const s = input.settings
  const now = input.now ?? Date.now()
  const out: DoctorCheck[] = []

  if (!s.providers.length) {
    out.push(check('models:none', 'models', 'Model providers', 'error', 'No model provider is configured.', 'models', 'Connect'))
  } else {
    out.push(check('models:count', 'models', 'Model providers', 'ok', `${plural(s.providers.length, 'provider')} configured.`, 'models'))
  }

  const activeProvider = s.providers.find((provider) => provider.id === s.activeProviderId)
  if (!activeProvider) {
    out.push(check('models:active', 'models', 'Active model', 'error', 'No active provider is selected.', 'models', 'Choose'))
  } else {
    const key = providerProbeKey(activeProvider.id, s.activeProviderAccountId ?? activeProvider.defaultAccountId ?? null)
    const probe = input.providerReadiness?.[key]
    if (!probe || probe.status === 'checking') {
      out.push(check('models:active', 'models', 'Active model', 'checking', `Checking ${activeProvider.name || 'active provider'} readiness.`, 'models'))
    } else if (probe.status === 'error') {
      out.push(check('models:active', 'models', 'Active model', 'error', probe.message || 'Readiness check failed.', 'models', 'Open'))
    } else if (probe.value?.ready) {
      out.push(check('models:active', 'models', 'Active model', 'ok', `${activeProvider.name || 'Active provider'} is ready.`, 'models'))
    } else {
      out.push(check('models:active', 'models', 'Active model', 'error', probe.value?.message || 'The active provider is not ready.', 'models', 'Fix'))
    }
  }

  for (const provider of s.providers) {
    const accounts = (provider.accounts ?? []).filter((account) => account.enabled !== false)
    const accountIds = accounts.length ? accounts.map((account) => account.id) : [provider.defaultAccountId ?? null]
    for (const accountId of accountIds) {
      const label = accounts.find((account) => account.id === accountId)?.name
      const title = label ? `${provider.name} · ${label}` : provider.name || 'Provider'
      const key = providerProbeKey(provider.id, accountId)
      const probe = input.providerReadiness?.[key]
      if (!probe || probe.status === 'checking') {
        out.push(check(`provider:${key}`, 'models', title, 'checking', 'Checking authentication and endpoint readiness.', 'models'))
      } else if (probe.status === 'error') {
        out.push(check(`provider:${key}`, 'models', title, 'error', probe.message || 'Provider readiness check failed.', 'models', 'Open'))
      } else if (probe.value?.ready) {
        // Real rate-limit windows (VST-037): fold pace into the ready detail, and if a
        // window is exhausted, downgrade to a warn with its ETA so a "ready" provider that
        // is actually rate-limited can't read as fully green.
        const status = providerStatusView(input.providerStatus?.statuses?.[key], now)
        const rate = status ? status.windows.map((w) => `${w.label} ${Math.round(w.usedPercent ?? 0)}%${w.etaText ? ` (${w.etaText})` : ''}`).join(' · ') : ''
        const exhausted = status?.windows.find((w) => w.pace === 'exhausted')
        const readyDetail = [probe.value.detail, rate].filter(Boolean).join(' · ')
        if (exhausted) {
          out.push(check(`provider:${key}`, 'models', title, 'warning', `Rate limited — ${exhausted.label} at 100%${exhausted.etaText ? ` (${exhausted.etaText})` : ''}.`, 'models'))
        } else {
          out.push(check(`provider:${key}`, 'models', title, 'ok', readyDetail ? `Ready: ${readyDetail}` : 'Ready.', 'models'))
        }
      } else {
        out.push(check(`provider:${key}`, 'models', title, 'error', probe.value?.message || 'Provider is not ready.', 'models', 'Fix'))
      }
    }
  }

  const proofProbe = input.connectors
  // Newest successful outbound per connector (VST-013): pending/denied approvals don't
  // count — only actions that actually went out.
  const outboundOkAt = new Map<ConnectorId, number>()
  for (const event of input.connectorEvents ?? []) {
    if (event.direction !== 'outbound') continue
    if (event.approvalStatus && event.approvalStatus !== 'approved') continue
    const prev = outboundOkAt.get(event.connectorId) ?? 0
    if (event.createdAt > prev) outboundOkAt.set(event.connectorId, event.createdAt)
  }
  const proof: ConnectorProofInput = {
    state: !proofProbe ? 'unavailable' : proofProbe.status === 'checking' ? 'checking' : proofProbe.status === 'error' ? 'unavailable' : 'loaded',
    byId: new Map((proofProbe?.value ?? []).map((connector) => [connector.id, connector])),
    now,
    outboundOkAt,
  }

  out.push(sourceCheck('email', 'Email', s.email.sources, (source) => {
    if (source.kind === 'imap' && (!hasValue(source.host) || !hasValue(source.user) || !hasSecret(source.password))) return `${source.label || 'IMAP'} needs host, user, and password.`
    if (source.kind !== 'imap' && !hasValue(source.path)) return `${source.label || 'Email source'} needs a path.`
    return null
  }, 'Email has no inbox sources.', proof))
  out.push(sourceCheck('slack', 'Slack', s.slack.sources, (source) => source.kind === 'api' && !hasSecret(source.token) ? `${source.label || 'Slack'} needs a bot token.` : null, 'Slack has no workspace sources.', proof))
  out.push(sourceCheck('discord', 'Discord', s.discord.sources, (source) => !hasSecret(source.token) ? `${source.label || 'Discord'} needs a bot token.` : null, 'Discord has no bot sources.', proof))
  out.push(sourceCheck('mattermost', 'Mattermost', s.mattermost.sources, (source) => !hasValue(source.baseUrl) || !hasSecret(source.token) ? `${source.label || 'Mattermost'} needs a server URL and token.` : null, 'Mattermost has no server sources.', proof))
  out.push(sourceCheck('matrix', 'Matrix', s.matrix.sources, (source) => !hasValue(source.baseUrl) || !hasSecret(source.token) ? `${source.label || 'Matrix'} needs a homeserver URL and access token.` : null, 'Matrix has no homeserver sources.', proof))
  out.push(sourceCheck('googlechat', 'Google Chat', s.googlechat.sources, (source) => !hasSecret(source.webhookUrl) && !hasSecret(source.accessToken) ? `${source.label || 'Google Chat'} needs a webhook URL or access token.` : null, 'Google Chat has no space sources.', proof))
  out.push(sourceCheck('docs', 'Documents', s.documents.sources, (source) => !hasValue(source.path) ? `${source.label || 'Document source'} needs a path.` : null, 'Documents has no file or folder sources.', proof))
  out.push(sourceCheck('browser', 'Browser', s.browser.sources, (source) => source.kind === 'history_export' && !hasValue(source.path) ? `${source.label || 'Browser history'} needs an export path.` : null, 'Browser has no current-tab or history sources.', proof))
  out.push(sourceCheck('calendar', 'Calendar', s.calendar.sources, (source) => !hasValue(source.path) ? `${source.label || 'Calendar source'} needs an ICS path.` : null, 'Calendar has no ICS sources.', proof))

  if (s.botControl.telegram.enabled) {
    if (!hasSecret(s.botControl.telegram.botToken)) out.push(check('connector:telegram', 'connectors', 'Telegram bot', 'error', 'Telegram is enabled but has no bot token.', 'connectors', 'Fix'))
    else if (!hasAllowlist(s.botControl.telegram.allowedChatIds)) out.push(check('connector:telegram', 'connectors', 'Telegram bot', 'warning', 'Telegram is enabled without a chat allowlist.', 'connectors', 'Restrict'))
    else if (!hasAllowlist(s.botControl.telegram.allowedUserIds)) out.push(check('connector:telegram', 'connectors', 'Telegram bot', 'warning', 'Telegram is enabled without a sender user-ID allowlist.', 'connectors', 'Restrict'))
    else if (s.botControl.telegram.lastError) out.push(check('connector:telegram', 'connectors', 'Telegram bot', 'error', s.botControl.telegram.lastError, 'connectors', 'Open'))
    else out.push(proofVerdict('telegram', 'Telegram bot', 'Enabled with chat and sender allowlists.', proof, 'connectors', 'connector:telegram'))
  } else {
    out.push(check('connector:telegram', 'connectors', 'Telegram bot', 'warning', 'Telegram bot control is disabled.', 'connectors', 'Configure'))
  }

  if (s.botControl.whatsapp.enabled) {
    const cloud = s.botControl.whatsapp
    const missing = [
      !hasSecret(cloud.accessToken) && 'access token',
      !hasValue(cloud.phoneNumberId) && 'phone number ID',
      !hasSecret(cloud.verifyToken) && 'verify token',
      !hasSecret(cloud.appSecret) && 'Meta app secret',
      !hasAllowlist(cloud.allowedChatIds) && 'sender allowlist',
    ].filter(Boolean) as string[]
    if (missing.length) out.push(check('connector:whatsapp-cloud', 'connectors', 'WhatsApp Cloud bot', 'error', `WhatsApp Cloud bot needs ${missing.join(', ')}.`, 'connectors', 'Fix'))
    else if (cloud.lastError) out.push(check('connector:whatsapp-cloud', 'connectors', 'WhatsApp Cloud bot', 'error', cloud.lastError, 'connectors', 'Open'))
    else out.push(proofVerdict('whatsapp', 'WhatsApp Cloud bot', 'Enabled with signed Meta webhook verification and a sender allowlist.', proof, 'connectors', 'connector:whatsapp-cloud'))
  }

  if (s.whatsapp.sources?.some((source) => source.enabled !== false)) {
    const bad = s.whatsapp.sources.find((source) => source.enabled !== false && source.lastError)
    out.push(bad
      ? check('connector:whatsapp', 'connectors', 'WhatsApp bridge', 'error', bad.lastError || 'WhatsApp bridge error.', 'connectors', 'Open')
      : proofVerdict('whatsapp', 'WhatsApp bridge', 'At least one bridge account is enabled.', proof, 'connectors', 'connector:whatsapp'))
  } else {
    out.push(check('connector:whatsapp', 'connectors', 'WhatsApp bridge', 'warning', 'WhatsApp bridge has no enabled paired accounts.', 'connectors', 'Pair'))
  }

  if (s.controller.enabled) {
    out.push(s.controller.lastError
      ? check('connector:controller', 'connectors', 'Controller', 'error', s.controller.lastError, 'security', 'Open')
      : check('connector:controller', 'connectors', 'Controller', 'ok', 'Current-window capture is enabled.', 'security'))
  } else {
    out.push(check('connector:controller', 'connectors', 'Controller', 'warning', 'Controller is disabled.', 'security', 'Configure'))
  }

  const studio = input.studioRemote
  if (!studio || !studio.enabled) out.push(check('remote:studio', 'remote', 'This Mac remote control', 'warning', 'Desktop remote control is disabled.', 'orchestrator', 'Enable'))
  else if (studio.lastError) out.push(check('remote:studio', 'remote', 'This Mac remote control', 'error', studio.lastError, 'orchestrator', 'Open'))
  else if (!studio.listening) out.push(check('remote:studio', 'remote', 'This Mac remote control', 'error', 'Remote control is enabled but not listening.', 'orchestrator', 'Open'))
  else out.push(check('remote:studio', 'remote', 'This Mac remote control', 'ok', `Listening${studio.url ? ` at ${studio.url}` : ''}${studio.clients ? ` · ${plural(studio.clients, 'client')}` : ''}.`, 'orchestrator'))

  for (const host of s.remoteOrchestrators ?? []) {
    const name = host.name || 'Orchestrator'
    if (!host.enabled) {
      out.push(check(`remote:${host.id}`, 'remote', name, 'warning', 'Host is saved but disabled.', 'orchestrator', 'Enable'))
      continue
    }
    if (!hasValue(host.serverUrl) || !hasSecret(host.authToken)) {
      out.push(check(`remote:${host.id}`, 'remote', name, 'error', 'Enabled host needs a server URL and auth token.', 'orchestrator', 'Fix'))
      continue
    }
    const probe = input.remoteProbes?.[host.id]
    if (!probe || probe.status === 'checking') {
      out.push(check(`remote:${host.id}`, 'remote', name, 'checking', 'Checking host snapshot endpoint.', 'orchestrator'))
      continue
    }
    if (probe.status === 'error' || probe.value?.ok === false) {
      out.push(check(`remote:${host.id}`, 'remote', name, 'error', probe.message || probe.value?.error || host.lastError || 'Host check failed.', 'orchestrator', 'Open'))
      continue
    }
    out.push(check(`remote:${host.id}`, 'remote', name, 'ok', 'Snapshot endpoint responded.', 'orchestrator'))

    // Explicit health subchecks (VST-001): heartbeat freshness, active work, pending
    // approvals, and terminal health each stand alone instead of hiding in summary fields.
    const generatedAt = probe.value?.generatedAt
    if (!generatedAt) {
      out.push(check(`remote:${host.id}:heartbeat`, 'remote', `${name} · heartbeat`, 'warning', 'Host responded without a heartbeat timestamp.', 'orchestrator'))
    } else if (now - generatedAt > HEARTBEAT_STALE_MS) {
      out.push(check(`remote:${host.id}:heartbeat`, 'remote', `${name} · heartbeat`, 'warning', `Last heartbeat ${ago(generatedAt, now)} — snapshot may be stale.`, 'orchestrator', 'Open'))
    } else {
      out.push(check(`remote:${host.id}:heartbeat`, 'remote', `${name} · heartbeat`, 'ok', `Heartbeat ${ago(generatedAt, now)}.`, 'orchestrator'))
    }

    const work = probe.value?.work
    if (!work) {
      out.push(check(`remote:${host.id}:work`, 'remote', `${name} · active work`, 'warning', 'Snapshot has no work summary; job health is unknown.', 'orchestrator'))
      continue
    }
    const unhealthyJobs = work.jobs.failed + work.jobs.deadLetter + work.jobs.stalled
    const leaseTrouble = work.leases.expired + work.leases.missingRunning
    if (unhealthyJobs || leaseTrouble) {
      const bits: string[] = []
      if (work.jobs.failed) bits.push(`${work.jobs.failed} failed`)
      if (work.jobs.deadLetter) bits.push(`${work.jobs.deadLetter} dead-letter`)
      if (work.jobs.stalled) bits.push(`${work.jobs.stalled} stalled`)
      if (work.leases.expired) bits.push(`${work.leases.expired} expired lease${work.leases.expired === 1 ? '' : 's'}`)
      if (work.leases.missingRunning) bits.push(`${work.leases.missingRunning} running without a lease`)
      out.push(check(`remote:${host.id}:work`, 'remote', `${name} · active work`, 'warning', `${bits.join(', ')} · ${work.jobs.running} running, ${work.jobs.queued} queued.`, 'orchestrator', 'Open'))
    } else {
      out.push(check(`remote:${host.id}:work`, 'remote', `${name} · active work`, 'ok', `${work.jobs.running} running · ${work.jobs.queued} queued · leases healthy.`, 'orchestrator'))
    }

    if (work.approvals.pending > 0) {
      out.push(check(`remote:${host.id}:approvals`, 'remote', `${name} · approvals`, 'warning', `${plural(work.approvals.pending, 'approval')} waiting (${work.approvals.tool} tool · ${work.approvals.connector} connector · ${work.approvals.mission} mission).`, 'orchestrator', 'Review'))
    } else {
      out.push(check(`remote:${host.id}:approvals`, 'remote', `${name} · approvals`, 'ok', 'No approvals waiting.', 'orchestrator'))
    }

    const terminals = work.terminals
    if (!terminals) {
      out.push(check(`remote:${host.id}:terminals`, 'remote', `${name} · terminals`, 'warning', 'Snapshot has no terminal summary; terminal health is unknown.', 'orchestrator'))
    } else if (terminals.attention > 0) {
      out.push(check(`remote:${host.id}:terminals`, 'remote', `${name} · terminals`, 'warning', `${terminals.attention} of ${plural(terminals.total, 'terminal')} need attention.`, 'orchestrator', 'Open'))
    } else {
      out.push(check(`remote:${host.id}:terminals`, 'remote', `${name} · terminals`, 'ok', terminals.total ? `${terminals.running} running of ${plural(terminals.total, 'terminal')}.` : 'No terminals hosted.', 'orchestrator'))
    }
  }

  const cliProbe = input.localClis
  if (!cliProbe || cliProbe.status === 'checking') {
    out.push(check('terminal:clis', 'terminals', 'Local CLI launchers', 'checking', 'Checking installed terminal launchers.', 'agent'))
  } else if (cliProbe.status === 'error') {
    out.push(check('terminal:clis', 'terminals', 'Local CLI launchers', 'error', cliProbe.message || 'Could not list installed CLIs.', 'agent', 'Open'))
  } else {
    const clis = cliProbe.value ?? []
    if (!clis.length) {
      out.push(check('terminal:clis', 'terminals', 'Local CLI launchers', 'error', 'No terminal launchers are available.', 'agent', 'Open'))
    } else {
      const launchable = clis.filter((cli) => !cli.command || (isCliStatus(cli) && cli.installed && cli.state !== 'broken')).length
      out.push(check('terminal:clis', 'terminals', 'Local CLI launchers', 'ok', `${launchable} of ${plural(clis.length, 'launcher')} ready to launch.`, 'agent'))
      for (const cli of clis) {
        const id = `terminal:cli:${cli.id}`
        const title = cli.builtin ? cli.name : `${cli.name} · custom`
        if (!cli.command) {
          out.push(check(id, 'terminals', title, 'ok', 'Plain login shell — always available.', 'agent'))
          continue
        }
        if (!isCliStatus(cli)) {
          out.push(check(id, 'terminals', title, 'checking', `Registered as \`${cli.command}\`; binary not probed yet.`, 'agent'))
          continue
        }
        if (cli.installed && cli.state === 'broken') {
          out.push(check(id, 'terminals', title, 'warning', `\`${cli.effectiveCommand ?? cli.command} --version\` failed: ${cli.problem || 'no version printed'}`, 'agent'))
          continue
        }
        if (cli.installed) {
          const version = cli.version ? `v${cli.version.replace(/^v/, '')}` : 'version unknown'
          const via = cli.configuredPath ? 'configured path' : 'PATH'
          const provider = cli.providerIds.length ? ` · linked to ${plural(cli.providerIds.length, 'provider account')}` : ''
          const base = `Installed (${version}) via ${via}: ${cli.effectiveCommand ?? cli.command}${provider}.`
          // A newer published version is informational, not a readiness failure — surface it as
          // a warning sub-note with the update command so the launcher still reads as usable.
          if (cli.advisory?.status === 'update_available') {
            out.push(check(id, 'terminals', title, 'warning', `${base} Update available: ${cli.advisory.latestVersion}${cli.advisory.updateCommand ? ` — run \`${cli.advisory.updateCommand}\`` : ''}.`, 'agent', cli.advisory.canUpdate ? 'Update' : undefined))
          } else {
            out.push(check(id, 'terminals', title, 'ok', cli.advisory?.status === 'ahead' ? `${base} (dev build, ahead of ${cli.advisory.latestVersion})` : base, 'agent'))
          }
        } else {
          out.push(check(id, 'terminals', title, 'warning', `Binary \`${cli.effectiveCommand ?? cli.command}\` not found on PATH or the configured path.`, 'agent', 'Install'))
        }
      }
    }
  }

  if (s.approvalMode === 'auto') out.push(check('security:approval', 'security', 'Approval mode', 'warning', 'Full Auto auto-runs only read-only, Git-read, test, and build commands; mutations, network/package work, destructive actions, and unknown commands still ask.', 'agent', 'Review'))
  else if (s.approvalMode === 'readonly') out.push(check('security:approval', 'security', 'Approval mode', 'ok', 'Read-only mode blocks file edits and commands.', 'agent'))
  else out.push(check('security:approval', 'security', 'Approval mode', 'ok', 'Approve Edits is active.', 'agent'))
  if (s.allowTaskNetwork) out.push(check('security:network', 'security', 'Task network access', 'warning', 'Sandboxed task commands may access the network.', 'agent', 'Review'))
  else out.push(check('security:network', 'security', 'Task network access', 'ok', 'Sandboxed task command network access is off.', 'agent'))
  if (input.persistenceFailure) {
    const failure = input.persistenceFailure
    out.push(check(
      'security:persistence',
      'security',
      'Local persistence',
      'error',
      `Could not save ${failure.label} ${ago(failure.at, now)}. Recent changes may not survive restart. ${failure.message}`,
    ))
  } else {
    out.push(check('security:persistence', 'security', 'Local persistence', 'ok', 'No background write failure has been recorded this session.'))
  }

  const localSecrets = input.securityControls
  if (localSecrets) {
    if (localSecrets.quarantinedProtectedFiles > 0) {
      const examples = localSecrets.quarantinedProtectedPaths.slice(0, 2).join(', ')
      const backups = localSecrets.quarantineBackupPaths.slice(0, 2).join(', ')
      out.push(check(
        'security:local-secrets',
        'security',
        'Protected state quarantine',
        'error',
        `${plural(localSecrets.quarantinedProtectedFiles, 'protected file')} could not be decoded and is write-blocked. Affected: ${examples || 'see logs'}. Recovery ${localSecrets.quarantineBackupPaths.length === 1 ? 'copy' : 'copies'}: ${backups || 'backup failed; startup should be stopped'}. Restore the matching key/file or explicitly purge the quarantined state.`,
        'security',
        'Review',
      ))
    } else if (localSecrets.legacySafeStorageSecrets > 0) {
      const backup = localSecrets.legacySafeStorageBackupPath
        ? ` The original settings are preserved at ${localSecrets.legacySafeStorageBackupPath}.`
        : ' The original settings backup could not be created; avoid clearing the affected entries.'
      out.push(check(
        'security:local-secrets',
        'security',
        'Local secret recovery',
        'error',
        `${plural(localSecrets.legacySafeStorageSecrets, 'credential')} could not be decrypted by this build.${backup} Re-enter each credential to replace the preserved legacy ciphertext.`,
        'security',
        'Review',
      ))
    } else if (localSecrets.copiedProfileDecryptable || localSecrets.localStateKeyMode === 'profile-key' || !localSecrets.externalSecretKeyRequired) {
      out.push(check(
        'security:local-secrets',
        'security',
        'Local secret storage',
        'warning',
        localSecrets.localStateKeyMode === 'external-env-key' && !localSecrets.externalSecretKeyRequired
          ? 'An external key is active, but managed fail-closed policy is off. Set VONTRA_REQUIRE_EXTERNAL_SECRET_KEY=1 so a future launch cannot silently fall back to a profile key.'
          : 'Profile-key or transition mode encrypts secret fields, but a complete TensorFold profile copy may include a decryption key. Managed deployments should hold VONTRA_SECRET_KEY_BASE64 separately and set VONTRA_REQUIRE_EXTERNAL_SECRET_KEY=1.',
        'security',
        'Review',
      ))
    } else {
      out.push(check('security:local-secrets', 'security', 'Local secret storage', 'ok', 'An independently-held external key protects copied TensorFold profiles.', 'security'))
    }
  }

  const ageChecks: Array<DoctorCheck | null> = []
  const addSources = (prefix: string, title: string, sources: Array<any>, slots: (source: any) => Record<string, unknown>) => {
    for (const source of sources.filter((item) => item.enabled !== false)) ageChecks.push(credentialAgeCheck(`secret-age:${prefix}:${source.id}`, 'connectors', `${title} · ${source.label || source.id}`, 'connectors', slots(source), source.secretRotations, now))
  }
  addSources('email', 'Email', s.email.sources, (source) => ({ password: source.password }))
  addSources('slack', 'Slack', s.slack.sources, (source) => ({ token: source.token }))
  addSources('discord', 'Discord', s.discord.sources, (source) => ({ token: source.token }))
  addSources('mattermost', 'Mattermost', s.mattermost.sources, (source) => ({ token: source.token }))
  addSources('matrix', 'Matrix', s.matrix.sources, (source) => ({ token: source.token }))
  addSources('googlechat', 'Google Chat', s.googlechat.sources, (source) => ({ webhookUrl: source.webhookUrl, accessToken: source.accessToken }))
  for (const source of s.whatsapp.sources.filter((item) => item.enabled !== false && item.paired)) {
    // The actual Baileys keys never enter Settings/the renderer. A synthetic configured marker
    // lets the common freshness policy evaluate only the non-secret, host-authored timestamp.
    ageChecks.push(credentialAgeCheck(
      `secret-age:whatsapp-bridge:${source.id}`,
      'connectors',
      `WhatsApp bridge · ${source.label || source.id} credential age`,
      'connectors',
      { linkedDeviceCredential: 'configured' },
      { linkedDeviceCredential: source.credentialRotatedAt ?? null },
      now,
    ))
  }
  for (const provider of s.providers) {
    if (provider.kind === 'local') continue
    const providerSecrets: Record<string, unknown> = { apiKey: provider.apiKey, serverPassword: provider.serverPassword }
    for (const [key, value] of Object.entries(provider.cliEnv ?? {})) providerSecrets[`cliEnv:${key}`] = value
    ageChecks.push(credentialAgeCheck(`secret-age:provider:${provider.id}`, 'models', `${provider.name} credential age`, 'models', providerSecrets, provider.secretRotations, now))
    for (const account of provider.accounts?.filter((item) => item.enabled !== false) ?? []) {
      const accountSecrets: Record<string, unknown> = { apiKey: account.apiKey }
      for (const [key, value] of Object.entries(account.env ?? {})) accountSecrets[`env:${key}`] = value
      ageChecks.push(credentialAgeCheck(`secret-age:provider:${provider.id}:${account.id}`, 'models', `${provider.name} · ${account.name} credential age`, 'models', accountSecrets, account.secretRotations, now))
    }
  }
  if (s.botControl.telegram.enabled) ageChecks.push(credentialAgeCheck('secret-age:telegram', 'connectors', 'Telegram credential age', 'connectors', { botToken: s.botControl.telegram.botToken }, s.botControl.telegram.secretRotations, now))
  if (s.botControl.whatsapp.enabled) ageChecks.push(credentialAgeCheck('secret-age:whatsapp-cloud', 'connectors', 'WhatsApp Cloud credential age', 'connectors', { accessToken: s.botControl.whatsapp.accessToken, verifyToken: s.botControl.whatsapp.verifyToken, appSecret: s.botControl.whatsapp.appSecret }, s.botControl.whatsapp.secretRotations, now))
  if (s.remoteControl.enabled) ageChecks.push(credentialAgeCheck('secret-age:remote-control', 'remote', 'This Mac remote-control credential age', 'orchestrator', { authToken: s.remoteControl.authToken }, s.remoteControl.secretRotations, now))
  if (s.remoteClient.enabled) ageChecks.push(credentialAgeCheck('secret-age:remote-client', 'remote', 'Remote client credential age', 'orchestrator', { authToken: s.remoteClient.authToken }, s.remoteClient.secretRotations, now))
  for (const host of s.remoteOrchestrators.filter((item) => item.enabled)) ageChecks.push(credentialAgeCheck(`secret-age:remote:${host.id}`, 'remote', `${host.name || 'Orchestrator'} credential age`, 'orchestrator', { authToken: host.authToken }, host.secretRotations, now))
  for (const webhook of input.webhooks?.filter((item) => item.enabled) ?? []) ageChecks.push(credentialAgeCheck(`secret-age:webhook:${webhook.id}`, 'connectors', `${webhook.name} webhook credential age`, 'webhooks', { secretRef: webhook.secretRef }, webhook.secretRotations, now))
  if (s.voice.ttsProvider !== 'browser' || s.voice.sttProvider !== 'browser') ageChecks.push(credentialAgeCheck('secret-age:voice', 'connectors', 'Voice provider credential age', 'voice', { apiKey: s.voice.apiKey, elevenLabsApiKey: s.voice.elevenLabsApiKey }, s.voice.secretRotations, now))
  out.push(...ageChecks.filter((item): item is DoctorCheck => !!item))

  return out
}

export function summarizeDoctor(checks: DoctorCheck[]): DoctorSummary {
  const summary: DoctorSummary = { ok: 0, warning: 0, error: 0, checking: 0, total: checks.length }
  for (const item of checks) summary[item.status] += 1
  return summary
}

export function worstDoctorStatus(checks: DoctorCheck[]): DoctorStatus {
  if (checks.some((item) => item.status === 'error')) return 'error'
  if (checks.some((item) => item.status === 'checking')) return 'checking'
  if (checks.some((item) => item.status === 'warning')) return 'warning'
  return 'ok'
}

export function readinessProbeKey(providerId: string, accountId?: string | null): string {
  return providerProbeKey(providerId, accountId)
}
