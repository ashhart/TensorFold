// Node & inference dashboard (VST-014): ONE derived view over every place work can run —
// this machine, remote Orchestrator hosts, cloud providers, and CLI subscriptions — each
// with an explicit health verdict (online / degraded / offline / unknown), auth state,
// live-work counts, and its last error. Pure derivation: probes and snapshots come in,
// cards and a usage rollup come out; nothing here can render "healthy" without evidence.

import type { CliToolStatus, PersistedProviderStatus, Provider, RemoteControlWorkSnapshot, TokenUsageSnapshot } from './types'
import type { TerminalActivitySummary } from './terminal-sessions'

export type NodeKind = 'local' | 'orchestrator' | 'cloud-provider' | 'cli-subscription'
export type NodeHealth = 'online' | 'degraded' | 'offline' | 'unknown'
export type NodeAuthState = 'ok' | 'unauthenticated' | 'unknown'

export interface NodeCard {
  id: string
  kind: NodeKind
  name: string
  health: NodeHealth
  detail: string
  latencyMs: number | null
  // Age of the newest evidence (heartbeat / snapshot / model list) — staleness is visible.
  evidenceAgeMs: number | null
  authState: NodeAuthState
  activeJobs: number | null
  pendingApprovals: number | null
  terminals: TerminalActivitySummary | null
  lastError: string | null
}

export interface RemoteHostProbe {
  hostId: string
  name: string
  ok: boolean
  error?: string | null
  generatedAt?: number | null
  latencyMs?: number | null
  work?: RemoteControlWorkSnapshot | null
}

export interface NodeDashboardInput {
  now: number
  // This machine's own live-work counts (always present — we ARE the local node).
  local: { activeJobs: number; pendingApprovals: number; terminals: TerminalActivitySummary }
  remoteHosts?: RemoteHostProbe[]
  // A remote heartbeat older than this is DEGRADED; a failed probe is OFFLINE.
  remoteStaleAfterMs?: number
  providers?: Provider[]
  providerStatus?: PersistedProviderStatus | null
  cliTools?: CliToolStatus[]
  tokenUsage?: TokenUsageSnapshot | null
}

export interface NodeUsageRollup {
  totalTokens: number
  // Per-account totals so multi-account spend is visible at a glance.
  byAccount: Record<string, number>
  // Newest activity timestamp across the usage windows (null = never).
  latestAt: number | null
}

export interface NodeDashboard {
  nodes: NodeCard[]
  usage: NodeUsageRollup
  // Every failing/degraded node's message, for a one-glance failure strip.
  failures: string[]
}

const DEFAULT_REMOTE_STALE_MS = 3 * 60_000
// A CLI transport implies a subscription-backed engine worth its own node card.
const CLI_SUBSCRIPTION_TRANSPORTS = new Set(['codex-cli', 'claude-cli', 'cursor-cli', 'gemini-cli', 'pi-cli'])

export function buildNodeDashboard(input: NodeDashboardInput): NodeDashboard {
  const staleAfter = input.remoteStaleAfterMs ?? DEFAULT_REMOTE_STALE_MS
  const nodes: NodeCard[] = []

  nodes.push({
    id: 'node:local',
    kind: 'local',
    name: 'This machine',
    health: 'online',
    detail: `${input.local.activeJobs} active job${input.local.activeJobs === 1 ? '' : 's'} · ${input.local.pendingApprovals} pending approval${input.local.pendingApprovals === 1 ? '' : 's'} · ${terminalDetail(input.local.terminals)}`,
    latencyMs: null,
    evidenceAgeMs: 0,
    authState: 'ok',
    activeJobs: input.local.activeJobs,
    pendingApprovals: input.local.pendingApprovals,
    terminals: input.local.terminals,
    lastError: null,
  })

  for (const host of input.remoteHosts ?? []) {
    const age = host.generatedAt ? Math.max(0, input.now - host.generatedAt) : null
    let health: NodeHealth
    let detail: string
    if (!host.ok) {
      health = 'offline'
      detail = host.error?.trim() || 'Host is unreachable.'
    } else if (age == null) {
      health = 'degraded'
      detail = 'Connected, but the snapshot carries no heartbeat timestamp.'
    } else if (age > staleAfter) {
      health = 'degraded'
      detail = `Heartbeat is stale (${Math.round(age / 1000)}s old).`
    } else {
      health = 'online'
      const work = host.work
      detail = work
        ? `${work.jobs.running} running · ${work.approvals.pending} approvals · ${work.terminals ? terminalDetail(work.terminals) : 'terminal state unknown'}`
        : 'Connected.'
    }
    nodes.push({
      id: `node:remote:${host.hostId}`,
      kind: 'orchestrator',
      name: host.name || host.hostId,
      health,
      detail,
      latencyMs: host.latencyMs ?? null,
      evidenceAgeMs: age,
      authState: host.ok ? 'ok' : /unauthori|401|token/i.test(host.error ?? '') ? 'unauthenticated' : 'unknown',
      activeJobs: host.work ? host.work.jobs.running + host.work.jobs.queued : null,
      pendingApprovals: host.work?.approvals.pending ?? null,
      terminals: host.work?.terminals ? { total: host.work.terminals.total, running: host.work.terminals.running, attention: host.work.terminals.attention } : null,
      lastError: host.ok ? null : host.error?.trim() || 'unreachable',
    })
  }

  const statuses = input.providerStatus?.statuses ?? {}
  for (const provider of input.providers ?? []) {
    const isCliSubscription = CLI_SUBSCRIPTION_TRANSPORTS.has(provider.transport ?? '')
    if (isCliSubscription) continue // CLI engines get their card from the installed-CLI probe below.
    const status = Object.values(statuses).find((entry) => entry.providerId === provider.id) ?? null
    const evidenceAge = status ? Math.max(0, input.now - status.updatedAt) : null
    const exhausted = status ? rateWindowsExhausted(status.raw) : false
    nodes.push({
      id: `node:provider:${provider.id}`,
      kind: 'cloud-provider',
      name: provider.name || provider.id,
      health: exhausted ? 'degraded' : status ? 'online' : 'unknown',
      detail: exhausted
        ? 'A rate-limit window is exhausted — runs will throttle or fail until it resets.'
        : status
          ? 'Live rate-limit telemetry present.'
          : 'No live telemetry yet — health is unknown until a run reports.',
      latencyMs: null,
      evidenceAgeMs: evidenceAge,
      authState: provider.apiKey || provider.transport ? 'ok' : 'unknown',
      activeJobs: null,
      pendingApprovals: null,
      terminals: null,
      lastError: null,
    })
  }

  for (const cli of input.cliTools ?? []) {
    if (!cli.providerIds.length && !cli.builtin) continue
    const authState: NodeAuthState = cli.installed ? 'ok' : 'unknown'
    nodes.push({
      id: `node:cli:${cli.id}`,
      kind: 'cli-subscription',
      name: cli.name,
      health: cli.installed ? 'online' : 'offline',
      detail: cli.installed
        ? `${cli.version ?? 'installed'}${cli.advisory?.status === 'update_available' ? ` · update available (${cli.latestVersion})` : ''}`
        : 'Binary not found on PATH — runs through this engine will fail.',
      latencyMs: null,
      evidenceAgeMs: null,
      authState,
      activeJobs: null,
      pendingApprovals: null,
      terminals: null,
      lastError: cli.installed ? null : `${cli.name} is not installed`,
    })
  }

  return {
    nodes,
    usage: rollupUsage(input.tokenUsage ?? null),
    failures: nodes
      .filter((node) => node.health === 'offline' || node.health === 'degraded')
      .map((node) => `${node.name}: ${node.lastError ?? node.detail}`),
  }
}

function terminalDetail(terminals: TerminalActivitySummary): string {
  if (!terminals.total) return '0 terminals'
  return `${terminals.running} active of ${terminals.total} terminal${terminals.total === 1 ? '' : 's'}`
}

// Sum the snapshot's daily points into one total + per-account rollup.
export function rollupUsage(snapshot: TokenUsageSnapshot | null): NodeUsageRollup {
  if (!snapshot) return { totalTokens: 0, byAccount: {}, latestAt: null }
  let totalTokens = 0
  const byAccount: Record<string, number> = {}
  for (const point of snapshot.daily ?? []) {
    totalTokens += point.total ?? 0
    for (const [account, tokens] of Object.entries(point.accounts ?? {})) {
      byAccount[account] = (byAccount[account] ?? 0) + (tokens ?? 0)
    }
  }
  return { totalTokens, byAccount, latestAt: snapshot.latestAt ?? null }
}

export function formatEvidenceAge(ageMs: number | null): string {
  if (ageMs == null) return 'age unknown'
  const seconds = Math.max(0, Math.floor(ageMs / 1000))
  if (seconds < 60) return seconds < 5 ? 'live now' : `${seconds}s ago`
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes}m ago`
  return `${Math.floor(minutes / 60)}h ago`
}

function rateWindowsExhausted(raw: unknown): boolean {
  if (!raw || typeof raw !== 'object') return false
  // A codex rate-limit snapshot carries windows with used_percent — 100 means exhausted.
  const candidates = Object.values(raw as Record<string, unknown>)
  return candidates.some((window) => {
    const percent = (window as { used_percent?: unknown })?.used_percent
    return typeof percent === 'number' && percent >= 100
  })
}
