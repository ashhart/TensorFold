import { useEffect, useState } from 'react'
import clsx from 'clsx'
import { Icon, type IconName } from '../lib/icons'
import { BrandIcon, connectorBrand } from '../lib/brand'
import { Button, Input, Toggle } from './ui'
import { api } from '../lib/ipc'
import { useStore } from '../lib/store'
import { WebhooksPanel } from './WebhooksPanel'
import type { AssistantControlSnapshot, CapabilityHealthState, ConnectorDefinition, ConnectorStatus, LocalScriptCapability, McpServerConfig, McpServerStatus, McpSnapshot, PackagedPluginCapability, WebhookSnapshot } from '../lib/types'
import { BUILTIN_CAPABILITIES, type BuiltInCapability } from '../lib/capabilities'

interface Connector {
  id: string
  name: string
  icon: IconName
  tag: string
  desc: string
  fields: string[]
  difficulty: 'Easy' | 'Medium' | 'Hard'
}

const CONNECTORS: Connector[] = [
  { id: 'telegram', name: 'Telegram', icon: 'message', tag: 'Bot API', difficulty: 'Easy', desc: 'Talk to your agent from any Telegram chat.', fields: ['Bot token', 'Allowed user IDs'] },
  { id: 'email', name: 'Email', icon: 'mail', tag: 'IMAP / SMTP', difficulty: 'Easy', desc: 'Read and reply to mail; trigger tasks from your inbox.', fields: ['Email address', 'App password', 'IMAP host', 'SMTP host'] },
  { id: 'outlook', name: 'Outlook / 365', icon: 'mail', tag: 'Graph · OAuth', difficulty: 'Medium', desc: 'Microsoft 365 mailbox via Microsoft Graph.', fields: ['Azure app (client) ID', 'Tenant', 'Redirect URI'] },
  { id: 'whatsapp', name: 'WhatsApp', icon: 'message', tag: 'Cloud API / bridge', difficulty: 'Hard', desc: 'Official Cloud API, or a local web bridge (opt-in).', fields: ['Mode', 'Phone number ID', 'Access token'] },
  { id: 'imessage', name: 'iMessage', icon: 'message', tag: 'macOS · BlueBubbles', difficulty: 'Hard', desc: 'Send/receive iMessage through a BlueBubbles server.', fields: ['Server URL', 'Password'] },
]

const DIFF_COLOR: Record<Connector['difficulty'], string> = {
  Easy: 'text-positive', Medium: 'text-warn', Hard: 'text-ink-3',
}

const STATUS_COLOR: Record<ConnectorStatus, string> = {
  connected: 'bg-positive',
  mock: 'bg-positive',
  pending: 'bg-ink-4',
  error: 'bg-danger',
}

const MCP_STATUS_COLOR: Record<McpServerStatus, string> = {
  connected: 'bg-positive',
  connecting: 'bg-warn',
  stopped: 'bg-ink-4',
  error: 'bg-danger',
}

const HEALTH_TONE: Record<CapabilityHealthState['status'], string> = {
  unknown: 'bg-ink-5 text-ink-3',
  ok: 'bg-positive/10 text-positive',
  warning: 'bg-warn/10 text-warn',
  error: 'bg-danger/10 text-danger',
}

function capabilityHealthCheck(capability: BuiltInCapability, enabled: boolean): CapabilityHealthState {
  if (!enabled) {
    return { status: 'warning', checkedAt: Date.now(), lastError: 'Capability is disabled in Settings.' }
  }
  if (capability.status === 'available') {
    return { status: 'ok', checkedAt: Date.now(), lastError: null }
  }
  return {
    status: 'warning',
    checkedAt: Date.now(),
    lastError: 'Capability contract is registered, but its worker is not installed yet.',
  }
}

function shortTime(value: number | null | undefined): string {
  if (!value) return 'never checked'
  const minutes = Math.max(0, Math.round((Date.now() - value) / 60_000))
  if (minutes < 1) return 'just now'
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.round(minutes / 60)
  return hours < 24 ? `${hours}h ago` : `${Math.round(hours / 24)}d ago`
}

type ExecutableTrustItem = {
  key: string
  icon: IconName
  name: string
  kind: string
  enabled: boolean
  detail: string
  trustLabel: string
  trustTone: string
  sandbox: string
  checksum: string | null
  verifiedAt: number | null
}

function trustBadge(checksum: string | null | undefined, verifiedAt: number | null | undefined) {
  if (verifiedAt) return { label: 'signed', tone: 'bg-positive/10 text-positive' }
  if (checksum) return { label: 'checksum', tone: 'bg-warn/10 text-warn' }
  return { label: 'unverified', tone: 'bg-danger/10 text-danger' }
}

function scriptTrustItem(script: LocalScriptCapability): ExecutableTrustItem {
  const trust = trustBadge(script.checksumSha256, script.signatureVerifiedAt)
  return {
    key: `script:${script.id}`,
    icon: 'terminal',
    name: script.name || script.id,
    kind: 'Local script',
    enabled: script.enabled !== false,
    detail: script.path,
    trustLabel: trust.label,
    trustTone: trust.tone,
    sandbox: script.sandboxProfile ?? 'none',
    checksum: script.checksumSha256 ?? null,
    verifiedAt: script.signatureVerifiedAt ?? null,
  }
}

function pluginTrustItem(plugin: PackagedPluginCapability): ExecutableTrustItem {
  const trust = trustBadge(plugin.checksumSha256, plugin.signatureVerifiedAt)
  return {
    key: `plugin:${plugin.id}`,
    icon: 'plug',
    name: plugin.name || plugin.id,
    kind: plugin.trusted ? 'Trusted plugin' : 'Packaged plugin',
    enabled: plugin.enabled !== false,
    detail: plugin.manifestPath,
    trustLabel: trust.label,
    trustTone: trust.tone,
    sandbox: plugin.sandboxProfile ?? 'none',
    checksum: plugin.checksumSha256 ?? null,
    verifiedAt: plugin.signatureVerifiedAt ?? null,
  }
}

function CapabilityTile({
  capability,
  enabled,
  health,
  onToggle,
  onCheck,
}: {
  capability: BuiltInCapability
  enabled: boolean
  health?: CapabilityHealthState
  onToggle: (enabled: boolean) => void
  onCheck: () => void
}) {
  const healthStatus = health?.status ?? 'unknown'
  return (
    <div className="rounded-xl border border-line bg-card p-3.5 shadow-card">
      <div className="flex items-start gap-3">
        <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg text-white shadow-[inset_0_1px_rgba(255,255,255,.18)]" style={{ background: capability.color }}>
          <Icon name={capability.icon} size={19} strokeWidth={1.9} />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex items-start gap-2">
            <span className="text-[14px] font-medium text-ink">{capability.name}</span>
            <span className={clsx('rounded-md px-1.5 py-0.5 text-[12px] font-medium', capability.status === 'available' ? 'bg-positive/10 text-positive' : 'bg-ink-5 text-ink-3')}>
              {capability.status === 'available' ? 'Ready' : 'Contract'}
            </span>
            <span className={clsx('rounded-md px-1.5 py-0.5 text-[12px] font-medium', HEALTH_TONE[healthStatus])}>{healthStatus}</span>
            <Toggle checked={enabled} onChange={onToggle} label={`${enabled ? 'Disable' : 'Enable'} ${capability.name}`} />
          </div>
          <p className="mt-1 text-[13px] leading-snug text-ink-2">{capability.description}</p>
          <div className="mt-2 flex items-center justify-between gap-2">
            <div className="min-w-0">
              <div className="font-mono text-[12px] text-ink-3">@{capability.mention}</div>
              <div className="mt-0.5 truncate text-[12px] text-ink-3">{health?.lastError ?? shortTime(health?.checkedAt)}</div>
            </div>
            <Button variant="ghost" size="xs" icon="check" onClick={onCheck}>Check</Button>
          </div>
        </div>
      </div>
    </div>
  )
}

function SourceCapabilityCard({
  icon,
  title,
  status,
  statusTone,
  detail,
  provenance,
  tools,
}: {
  icon: IconName
  title: string
  status: string
  statusTone: string
  detail: string
  provenance: string
  tools: string
}) {
  return (
    <div className="rounded-xl border border-line bg-card p-3.5 shadow-card">
      <div className="flex items-start gap-3">
        <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-ink-5 text-ink-2">
          <Icon name={icon} size={18} />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-[14px] font-medium text-ink">{title}</span>
            <span className={clsx('rounded-md px-1.5 py-0.5 text-[12px] font-medium', statusTone)}>{status}</span>
          </div>
          <p className="mt-1 text-[13px] leading-snug text-ink-2">{detail}</p>
          <div className="mt-2 grid grid-cols-1 gap-1 text-[12px] text-ink-3">
            <span className="truncate">{tools}</span>
            <span className="truncate font-mono">{provenance}</span>
          </div>
        </div>
      </div>
    </div>
  )
}

function ExecutableTrustCard({ item }: { item: ExecutableTrustItem }) {
  return (
    <div className="rounded-xl border border-line bg-card p-3.5 shadow-card">
      <div className="flex items-start gap-3">
        <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-ink-5 text-ink-2">
          <Icon name={item.icon} size={18} />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="truncate text-[14px] font-medium text-ink">{item.name}</span>
            <span className={clsx('rounded-md px-1.5 py-0.5 text-[12px] font-medium', item.enabled ? 'bg-positive/10 text-positive' : 'bg-ink-5 text-ink-3')}>{item.enabled ? 'enabled' : 'disabled'}</span>
            <span className={clsx('rounded-md px-1.5 py-0.5 text-[12px] font-medium', item.trustTone)}>{item.trustLabel}</span>
          </div>
          <p className="mt-1 text-[13px] leading-snug text-ink-2">{item.kind}</p>
          <div className="mt-2 grid grid-cols-1 gap-1 text-[12px] text-ink-3">
            <span className="truncate font-mono">{item.detail}</span>
            <span className="truncate">sandbox:{item.sandbox}{item.checksum ? ` · sha256:${item.checksum.slice(0, 12)}` : ''}{item.verifiedAt ? ` · verified ${shortTime(item.verifiedAt)}` : ''}</span>
          </div>
        </div>
      </div>
    </div>
  )
}

function connectorStatusTone(status: ConnectorStatus | 'inactive'): string {
  if (status === 'connected' || status === 'mock') return 'bg-positive/10 text-positive'
  if (status === 'error') return 'bg-danger/10 text-danger'
  return 'bg-ink-5 text-ink-3'
}

function connectorCapabilityDetail(connector: ConnectorDefinition | null | undefined, fallback: string): string {
  if (!connector) return fallback
  if (connector.error) return connector.error
  return connector.description
}

function connectorToolLabel(connector: ConnectorDefinition | null | undefined): string {
  if (!connector) return '0 tools registered'
  const approvals = connector.pendingApprovals ? ` · ${connector.pendingApprovals} approvals` : ''
  return `${connector.tools.length} tool${connector.tools.length === 1 ? '' : 's'}${approvals}`
}

function sourceCards(snapshot: AssistantControlSnapshot | null, mcp: McpSnapshot | null, webhooks: WebhookSnapshot | null) {
  const email = snapshot?.connectors.find((connector) => connector.id === 'email')
  const calendar = snapshot?.connectors.find((connector) => connector.kind === 'calendar')
  const mcpErrors = mcp?.servers.filter((server) => server.runtime.status === 'error').length ?? 0
  const mcpConnected = mcp?.servers.filter((server) => server.runtime.status === 'connected').length ?? 0
  const mcpTools = mcp?.tools.length ?? 0
  const mcpResources = mcp?.resources.length ?? 0
  const mcpPrompts = mcp?.prompts.length ?? 0
  const webhookEnabled = webhooks?.endpoints.filter((endpoint) => endpoint.enabled).length ?? 0
  const webhookListening = webhooks?.listener.listening ?? false
  const webhookStatus = webhookListening ? 'connected' : webhookEnabled ? 'pending' : 'inactive'
  const mcpStatus = mcpErrors ? 'error' : mcpConnected ? 'connected' : (mcp?.servers.length ? 'pending' : 'inactive')
  return [
    {
      icon: 'mail' as IconName,
      title: 'Email',
      status: email?.status === 'mock' ? 'contract' : (email?.status ?? 'pending'),
      statusTone: connectorStatusTone(email?.status ?? 'pending'),
      detail: connectorCapabilityDetail(email, 'Local email connector is not configured yet.'),
      tools: connectorToolLabel(email),
      provenance: email?.transport ? `transport:${email.transport}` : 'transport:local connector',
    },
    {
      icon: 'clock' as IconName,
      title: 'Calendar',
      status: calendar?.status ?? 'pending',
      statusTone: connectorStatusTone(calendar?.status ?? 'pending'),
      detail: connectorCapabilityDetail(calendar, 'Calendar connector contract is reserved for the connector phase.'),
      tools: connectorToolLabel(calendar),
      provenance: calendar?.transport ? `transport:${calendar.transport}` : 'provenance:contract only',
    },
    {
      icon: 'globe' as IconName,
      title: 'Webhooks',
      status: webhookStatus,
      statusTone: webhookStatus === 'connected' ? 'bg-positive/10 text-positive' : webhookStatus === 'pending' ? 'bg-warn/10 text-warn' : 'bg-ink-5 text-ink-3',
      detail: webhooks ? `${webhooks.endpoints.length} endpoint${webhooks.endpoints.length === 1 ? '' : 's'} configured; listener ${webhookListening ? 'online' : 'offline'}.` : 'Loading webhook listener and endpoint state.',
      tools: `${webhookEnabled} enabled endpoint${webhookEnabled === 1 ? '' : 's'}`,
      provenance: webhooks ? `loopback:${webhooks.listener.port ?? 'off'} routes:${webhooks.listener.routes}` : 'provenance:loading',
    },
    {
      icon: 'plug' as IconName,
      title: 'MCP',
      status: mcpStatus,
      statusTone: mcpStatus === 'connected' ? 'bg-positive/10 text-positive' : mcpStatus === 'error' ? 'bg-danger/10 text-danger' : 'bg-ink-5 text-ink-3',
      detail: mcp ? `${mcp.servers.length} server${mcp.servers.length === 1 ? '' : 's'} registered; ${mcpConnected} connected.` : 'Loading MCP server runtime state.',
      tools: `${mcpTools} tool${mcpTools === 1 ? '' : 's'} · ${mcpResources} resource${mcpResources === 1 ? '' : 's'} · ${mcpPrompts} prompt${mcpPrompts === 1 ? '' : 's'}`,
      provenance: mcp?.servers.find((server) => server.runtime.provenance)?.runtime.provenance?.commandSha256 ? `sha256:${mcp.servers.find((server) => server.runtime.provenance)?.runtime.provenance?.commandSha256.slice(0, 12)}` : 'provenance:not checked',
    },
  ]
}

type McpDraft = Partial<McpServerConfig> & { argsText?: string; envAllowlistText?: string }

export function Plugins() {
  const settings = useStore((s) => s.settings)
  const saveSettings = useStore((s) => s.saveSettings)
  const [open, setOpen] = useState<string | null>(null)
  const [snapshot, setSnapshot] = useState<AssistantControlSnapshot | null>(null)
  const [mcp, setMcp] = useState<McpSnapshot | null>(null)
  const [webhooks, setWebhooks] = useState<WebhookSnapshot | null>(null)
  const [mcpBusy, setMcpBusy] = useState<string | null>(null)
  const [mcpDraft, setMcpDraft] = useState<McpDraft | null>(null)
  const executableTrustItems = [
    ...(settings?.capabilities?.localScripts ?? []).map(scriptTrustItem),
    ...(settings?.capabilities?.packagedPlugins ?? []).map(pluginTrustItem),
  ]

  useEffect(() => {
    let alive = true
    api.getAssistantSnapshot().then((next) => { if (alive) setSnapshot(next) }).catch(() => {})
    api.getMcpSnapshot().then((next) => { if (alive) setMcp(next) }).catch(() => {})
    api.getWebhooks().then((next) => { if (alive) setWebhooks(next) }).catch(() => {})
    return () => { alive = false }
  }, [])

  const reloadMcp = async () => setMcp(await api.getMcpSnapshot())
  const updateCapabilityEnabled = async (capabilityId: string, enabled: boolean) => {
    if (!settings) return
    const disabled = new Set(settings.capabilities?.disabledCapabilityIds ?? [])
    if (enabled) disabled.delete(capabilityId)
    else disabled.add(capabilityId)
    await saveSettings({
      ...settings,
      capabilities: {
        ...settings.capabilities,
        disabledCapabilityIds: [...disabled],
        health: settings.capabilities?.health ?? {},
      },
    })
  }
  const checkCapability = async (capability: BuiltInCapability, enabled: boolean) => {
    if (!settings) return
    await saveSettings({
      ...settings,
      capabilities: {
        ...settings.capabilities,
        disabledCapabilityIds: settings.capabilities?.disabledCapabilityIds ?? [],
        health: {
          ...(settings.capabilities?.health ?? {}),
          [capability.id]: capabilityHealthCheck(capability, enabled),
        },
      },
    })
  }
  const startMcpDraft = (server?: McpServerConfig) => {
    setMcpDraft(server ? { ...server, env: undefined, argsText: server.args.join(' '), envAllowlistText: server.dockerGateway?.envAllowlist?.join(', ') ?? '' } : {
      name: 'New MCP server',
      transport: 'stdio',
      launchMode: 'direct',
      command: '',
      args: [],
      argsText: '',
      cwd: '',
      dockerGateway: null,
      envAllowlistText: '',
      enabled: true,
      autoConnect: false,
      approvalMode: 'ask',
    })
  }
  const saveMcpDraft = async () => {
    if (!mcpDraft) return
    const payload: Partial<McpServerConfig> = {
      id: mcpDraft.id,
      name: mcpDraft.name?.trim() || 'MCP server',
      transport: 'stdio',
      command: mcpDraft.command?.trim() || '',
      args: splitArgs(mcpDraft.argsText || ''),
      cwd: mcpDraft.cwd?.trim() || null,
      launchMode: mcpDraft.launchMode === 'docker_gateway' ? 'docker_gateway' : 'direct',
      dockerGateway: mcpDraft.launchMode === 'docker_gateway'
        ? {
            image: mcpDraft.dockerGateway?.image?.trim() || '',
            network: mcpDraft.dockerGateway?.network ?? 'none',
            readOnlyRootfs: mcpDraft.dockerGateway?.readOnlyRootfs !== false,
            envAllowlist: splitCsv(mcpDraft.envAllowlistText || ''),
            mounts: mcpDraft.dockerGateway?.mounts ?? [],
            workdir: mcpDraft.dockerGateway?.workdir ?? null,
          }
        : null,
      enabled: mcpDraft.enabled ?? true,
      autoConnect: mcpDraft.autoConnect ?? false,
      approvalMode: mcpDraft.approvalMode ?? 'ask',
    }
    setMcpBusy('save')
    try {
      setMcp(await api.saveMcpServer(payload))
      setMcpDraft(null)
    } finally {
      setMcpBusy(null)
    }
  }
  const withMcpBusy = async (label: string, action: () => Promise<McpSnapshot>) => {
    setMcpBusy(label)
    try { setMcp(await action()) } finally { setMcpBusy(null) }
  }

  return (
    <div className="scroll-thin h-full overflow-y-auto xl:pr-[304px]">
      <div className="mx-auto w-full max-w-[760px] px-7 py-9">
        <h1 className="text-[20px] font-semibold tracking-[-0.02em] text-ink">Sources & Connectors</h1>
        <p className="mt-1 text-[13px] text-ink-3">Email, messages, documents, services, and MCP tool servers.</p>

        <div className="mt-3 flex items-start gap-2 rounded-xl border border-line bg-card-2 px-3.5 py-2.5 text-[13px] text-ink-2">
          <Icon name="plug" size={15} className="mt-0.5 shrink-0 text-ink-3" />
          <span>{snapshot ? `${snapshot.connectors.filter((c) => c.status !== 'pending').length}/${snapshot.connectors.length} connector contracts active · ${snapshot.gaps.length} gaps · ${snapshot.pendingApprovals} pending approvals.` : 'Telegram and Email are first in the queue; Outlook, Dynamics, WhatsApp and iMessage follow behind explicit opt-in connection flows.'}</span>
        </div>

        <h2 className="mt-8 text-[13px] font-medium uppercase tracking-[0.05em] text-ink-3">Built-in capabilities</h2>
        <div className="mt-2.5 grid grid-cols-1 gap-2.5 sm:grid-cols-2">
          {BUILTIN_CAPABILITIES.map((capability) => {
            const enabled = !(settings?.capabilities?.disabledCapabilityIds ?? []).includes(capability.id)
            return (
              <CapabilityTile
                key={capability.id}
                capability={capability}
                enabled={enabled}
                health={settings?.capabilities?.health?.[capability.id]}
                onToggle={(next) => updateCapabilityEnabled(capability.id, next)}
                onCheck={() => checkCapability(capability, enabled)}
              />
            )
          })}
        </div>
        <p className="mt-2 text-[13px] text-ink-3">Type <span className="font-mono">@</span> in the composer to attach one of these capabilities to a chat or task.</p>

        <h2 className="mt-8 text-[13px] font-medium uppercase tracking-[0.05em] text-ink-3">Runtime capability health</h2>
        <div className="mt-2.5 grid grid-cols-1 gap-2.5 sm:grid-cols-2">
          {sourceCards(snapshot, mcp, webhooks).map((card) => (
            <SourceCapabilityCard key={card.title} {...card} />
          ))}
        </div>

        {executableTrustItems.length > 0 && (
          <>
            <h2 className="mt-8 text-[13px] font-medium uppercase tracking-[0.05em] text-ink-3">Executable trust</h2>
            <p className="mt-1 text-[13px] text-ink-3">Local scripts and packaged plugins stay approval-gated; review signatures, checksums, and sandbox intent before enabling.</p>
            <div className="mt-2.5 grid grid-cols-1 gap-2.5 sm:grid-cols-2">
              {executableTrustItems.map((item) => <ExecutableTrustCard key={item.key} item={item} />)}
            </div>
          </>
        )}

        <h2 className="mt-8 text-[13px] font-medium uppercase tracking-[0.05em] text-ink-3">Connectors</h2>
        <div className="mt-2.5 grid grid-cols-1 gap-2.5 sm:grid-cols-2">
          {CONNECTORS.map((c) => {
            const expanded = open === c.id
            const contract = snapshot?.connectors.find((x) => x.id === c.id)
            const status = contract?.status ?? 'pending'
            return (
              <div key={c.id} className="overflow-hidden rounded-xl border border-line bg-card shadow-card">
                <div className="flex items-start gap-3 p-3.5">
                  <BrandIcon brand={connectorBrand(c.id)} size={36} />
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <span className="text-[14px] font-medium text-ink">{c.name}</span>
                      <span className={clsx('text-[12px] font-medium', DIFF_COLOR[c.difficulty])}>{c.difficulty}</span>
                    </div>
                    <div className="text-[12px] text-ink-3">{c.tag}</div>
                    <p className="mt-1 text-[13px] leading-snug text-ink-2">{c.desc}</p>
                  </div>
                </div>
                <div className="flex items-center justify-between border-t border-line px-3.5 py-2">
                  <span className="flex items-center gap-1.5 text-[13px] text-ink-3"><span className={clsx('h-1.5 w-1.5 rounded-full', STATUS_COLOR[status])} />{status === 'mock' ? 'Contract ready' : status.replace(/_/g, ' ')}</span>
                  <button onClick={() => setOpen(expanded ? null : c.id)} className="rounded-lg px-2.5 py-1 text-[13px] font-medium text-ink-2 transition-colors hover:bg-ink-5 hover:text-ink">{expanded ? 'Close' : 'Configure'}</button>
                </div>
                {expanded && (
                  <div className="border-t border-line bg-card-2 px-3.5 py-3">
                    {contract && <div className="mb-2 rounded-lg border border-line bg-card px-2.5 py-2 text-[13px] text-ink-3">{contract.tools.length} narrow tools · transport {contract.transport ?? 'local only'}</div>}
                    <div className="space-y-2">
                      {c.fields.map((f) => (
                        <div key={f}>
                          <div className="mb-1 text-[12px] font-medium text-ink-3">{f}</div>
                          <div className="h-8 rounded-lg border border-line bg-card px-3 text-[13px] leading-8 text-ink-4">—</div>
                        </div>
                      ))}
                    </div>
                    <div className="mt-2.5 flex items-center gap-1.5 text-[12px] text-ink-3"><Icon name="clock" size={12} />Saving + live connection arrive in the Connectors phase.</div>
                  </div>
                )}
              </div>
            )
          })}
        </div>

        <div className="mt-9 flex items-end justify-between">
          <div>
            <h2 className="text-[13px] font-medium uppercase tracking-[0.05em] text-ink-3">MCP servers</h2>
            <p className="mt-1 text-[13px] text-ink-3">Local stdio servers expose tools to the model as approval-gated TensorFold functions.</p>
          </div>
          <div className="flex gap-2">
            <Button variant="default" icon="play" disabled={!!mcpBusy || !mcp?.servers.length} onClick={() => withMcpBusy('refresh', () => api.refreshMcpServer())}>{mcpBusy === 'refresh' ? 'Refreshing' : 'Refresh'}</Button>
            <Button variant="primary" icon="plus" onClick={() => startMcpDraft()}>Add server</Button>
          </div>
        </div>

        <div className="mt-3 rounded-xl border border-line bg-card-2 px-3.5 py-2.5 text-[13px] text-ink-2">
          {mcp ? `${mcp.servers.filter((server) => server.runtime.status === 'connected').length}/${mcp.servers.length} servers connected · ${mcp.tools.length} tools · ${mcp.resources.length} resources · ${mcp.prompts.length} prompts.` : 'Loading MCP servers…'}
        </div>

        {mcpDraft && (
          <div className="mt-3 rounded-xl border border-accent/40 bg-card p-3.5 shadow-card">
            <div className="mb-3 flex items-center justify-between">
              <div>
                <div className="text-[13px] font-medium text-ink">{mcpDraft.id ? 'Edit MCP server' : 'New MCP server'}</div>
                <div className="text-[13px] text-ink-3">Stdio transport; environment secrets stay in the main process.</div>
              </div>
              <Button variant="ghost" size="sm" onClick={() => setMcpDraft(null)}>Cancel</Button>
            </div>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              <Labeled label="Name"><Input value={mcpDraft.name || ''} onChange={(e) => setMcpDraft({ ...mcpDraft, name: e.target.value })} placeholder="Filesystem" /></Labeled>
              <Labeled label="Command"><Input value={mcpDraft.command || ''} onChange={(e) => setMcpDraft({ ...mcpDraft, command: e.target.value })} placeholder="npx" spellCheck={false} /></Labeled>
              <Labeled label="Arguments"><Input value={mcpDraft.argsText || ''} onChange={(e) => setMcpDraft({ ...mcpDraft, argsText: e.target.value })} placeholder="-y @modelcontextprotocol/server-filesystem ~/Projects" spellCheck={false} /></Labeled>
              <Labeled label="Working directory"><Input value={mcpDraft.cwd || ''} onChange={(e) => setMcpDraft({ ...mcpDraft, cwd: e.target.value })} placeholder="optional" spellCheck={false} /></Labeled>
            </div>
            <div className="mt-3 rounded-xl border border-line bg-card-2 p-3">
              <div className="mb-2 flex flex-wrap items-center gap-2">
                <span className="text-[13px] font-medium text-ink-2">Launch mode</span>
                <button onClick={() => setMcpDraft({ ...mcpDraft, launchMode: 'direct', dockerGateway: null })} className={clsx('rounded-lg px-2.5 py-1.5 text-[13px] font-medium transition-colors', (mcpDraft.launchMode ?? 'direct') !== 'docker_gateway' ? 'bg-accent text-accent-ink' : 'bg-ink-5 text-ink-2 hover:text-ink')}>Direct</button>
                <button onClick={() => setMcpDraft({ ...mcpDraft, launchMode: 'docker_gateway', dockerGateway: { image: mcpDraft.dockerGateway?.image ?? '', network: mcpDraft.dockerGateway?.network ?? 'none', readOnlyRootfs: mcpDraft.dockerGateway?.readOnlyRootfs !== false, envAllowlist: splitCsv(mcpDraft.envAllowlistText || ''), mounts: mcpDraft.dockerGateway?.mounts ?? [], workdir: mcpDraft.dockerGateway?.workdir ?? null } })} className={clsx('rounded-lg px-2.5 py-1.5 text-[13px] font-medium transition-colors', mcpDraft.launchMode === 'docker_gateway' ? 'bg-accent text-accent-ink' : 'bg-ink-5 text-ink-2 hover:text-ink')}>Docker gateway</button>
              </div>
              {mcpDraft.launchMode === 'docker_gateway' && (
                <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                  <Labeled label="Docker image digest"><Input value={mcpDraft.dockerGateway?.image || ''} onChange={(e) => setMcpDraft({ ...mcpDraft, dockerGateway: { ...(mcpDraft.dockerGateway ?? { network: 'none', readOnlyRootfs: true, mounts: [], envAllowlist: [] }), image: e.target.value } })} placeholder="mcp/server@sha256:…" spellCheck={false} /></Labeled>
                  <Labeled label="Env allowlist"><Input value={mcpDraft.envAllowlistText || ''} onChange={(e) => setMcpDraft({ ...mcpDraft, envAllowlistText: e.target.value })} placeholder="API_TOKEN, TENANT_ID" spellCheck={false} /></Labeled>
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-[13px] text-ink-3">Network</span>
                    {(['none', 'bridge'] as const).map((mode) => (
                      <button key={mode} onClick={() => setMcpDraft({ ...mcpDraft, dockerGateway: { ...(mcpDraft.dockerGateway ?? { image: '', readOnlyRootfs: true, mounts: [], envAllowlist: [] }), network: mode } })} className={clsx('rounded-lg px-2.5 py-1.5 text-[13px] font-medium transition-colors', (mcpDraft.dockerGateway?.network ?? 'none') === mode ? 'bg-accent text-accent-ink' : 'bg-ink-5 text-ink-2 hover:text-ink')}>{mode}</button>
                    ))}
                    {mcpDraft.dockerGateway?.network === 'host' && <span className="text-[12px] text-danger">Host networking is blocked; choose none or bridge.</span>}
                  </div>
                  <SwitchRow label="Read-only rootfs" checked={mcpDraft.dockerGateway?.readOnlyRootfs !== false} onChange={(readOnlyRootfs) => setMcpDraft({ ...mcpDraft, dockerGateway: { ...(mcpDraft.dockerGateway ?? { image: '', network: 'none', mounts: [], envAllowlist: [] }), readOnlyRootfs } })} />
                </div>
              )}
            </div>
            <div className="mt-3 flex flex-wrap items-center gap-4">
              <SwitchRow label="Enabled" checked={mcpDraft.enabled ?? true} onChange={(enabled) => setMcpDraft({ ...mcpDraft, enabled })} />
              <SwitchRow label="Auto-connect" checked={mcpDraft.autoConnect ?? false} onChange={(autoConnect) => setMcpDraft({ ...mcpDraft, autoConnect })} />
              <div className="flex items-center gap-2">
                <span className="text-[13px] text-ink-3">Policy</span>
                <button onClick={() => setMcpDraft({ ...mcpDraft, approvalMode: 'ask' })} className={clsx('rounded-lg px-2.5 py-1.5 text-[13px] font-medium transition-colors', (mcpDraft.approvalMode ?? 'ask') === 'ask' ? 'bg-accent text-accent-ink' : 'bg-ink-5 text-ink-2 hover:text-ink')}>Ask</button>
                <button onClick={() => setMcpDraft({ ...mcpDraft, approvalMode: 'trusted' })} className={clsx('rounded-lg px-2.5 py-1.5 text-[13px] font-medium transition-colors', mcpDraft.approvalMode === 'trusted' ? 'bg-accent text-accent-ink' : 'bg-ink-5 text-ink-2 hover:text-ink')}>Trusted</button>
              </div>
              <Button variant="primary" size="sm" icon="check" onClick={saveMcpDraft} disabled={!!mcpBusy || !mcpDraft.command?.trim() || (mcpDraft.launchMode === 'docker_gateway' && !mcpDraft.dockerGateway?.image?.trim())}>{mcpBusy === 'save' ? 'Saving' : 'Save server'}</Button>
            </div>
          </div>
        )}

        {mcp && mcp.servers.length > 0 ? (
          <div className="mt-3 space-y-2.5">
            {mcp.servers.map((server) => {
              const connected = server.runtime.status === 'connected'
              return (
                <div key={server.id} className="rounded-xl border border-line bg-card p-3.5 shadow-card">
                  <div className="flex items-start gap-3">
                    <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-ink-5 text-ink-2"><Icon name="plug" size={18} /></span>
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="text-[14px] font-medium text-ink">{server.name}</span>
                        <span className="flex items-center gap-1.5 text-[12px] text-ink-3"><span className={clsx('h-1.5 w-1.5 rounded-full', MCP_STATUS_COLOR[server.runtime.status])} />{server.runtime.status}</span>
                        {server.launchMode === 'docker_gateway' && <span className="rounded-md bg-ink-5 px-1.5 py-0.5 text-[12px] font-medium text-ink-2">docker gateway</span>}
                        {server.approvalMode === 'trusted' && <span className="rounded-md bg-accent-soft px-1.5 py-0.5 text-[12px] font-medium text-accent">trusted</span>}
                      </div>
                      <div className="mt-1 truncate font-mono text-[13px] text-ink-3">{server.command} {server.args.join(' ')}</div>
                      {server.runtime.lastError && <div className="mt-1 text-[13px] text-danger">{server.runtime.lastError}</div>}
                    </div>
                    <div className="flex shrink-0 flex-wrap justify-end gap-1.5">
                      <Button variant="ghost" size="xs" icon="edit" onClick={() => startMcpDraft(server)}>Edit</Button>
                      {connected
                        ? <Button variant="default" size="xs" icon="stop" disabled={!!mcpBusy} onClick={() => withMcpBusy(`disconnect:${server.id}`, () => api.disconnectMcpServer(server.id))}>Stop</Button>
                        : <Button variant="default" size="xs" icon="play" disabled={!!mcpBusy || !server.enabled} onClick={() => withMcpBusy(`connect:${server.id}`, () => api.connectMcpServer(server.id))}>Connect</Button>}
                      <Button variant="ghost" size="xs" icon="trash" disabled={!!mcpBusy} onClick={() => withMcpBusy(`delete:${server.id}`, () => api.deleteMcpServer(server.id))}>Delete</Button>
                    </div>
                  </div>
                  <div className="mt-3 flex flex-wrap items-center gap-2 text-[13px] text-ink-3">
                    <span>{server.runtime.tools.length} tool{server.runtime.tools.length === 1 ? '' : 's'}</span>
                    <span>{server.runtime.resources.length} resource{server.runtime.resources.length === 1 ? '' : 's'}</span>
                    <span>{server.runtime.prompts.length} prompt{server.runtime.prompts.length === 1 ? '' : 's'}</span>
                    {server.runtime.pid && <span>pid {server.runtime.pid}</span>}
                    {server.runtime.lastConnectedAt && <span>connected {new Date(server.runtime.lastConnectedAt).toLocaleTimeString()}</span>}
                    <button className="font-medium text-ink-2 hover:text-ink" onClick={() => withMcpBusy(`refresh:${server.id}`, () => api.refreshMcpServer(server.id))}>refresh</button>
                  </div>
                  {server.runtime.tools.length > 0 && (
                    <div className="mt-2 grid grid-cols-1 gap-1.5 sm:grid-cols-2">
                      {server.runtime.tools.slice(0, 6).map((tool) => (
                        <div key={tool.callName} className="flex items-center gap-2 rounded-lg border border-line bg-card-2 px-2.5 py-2">
                          <Icon name={tool.readOnlyHint ? 'search' : tool.destructiveHint ? 'warning' : 'terminal'} size={13} className={tool.destructiveHint ? 'text-warn' : 'text-ink-3'} />
                          <div className="min-w-0 flex-1">
                            <div className="truncate text-[13px] font-medium text-ink-2">{tool.title || tool.name}</div>
                            <div className="truncate font-mono text-[12px] text-ink-4">{tool.callName}</div>
                          </div>
                          {tool.readOnlyHint && <Button variant="ghost" size="xs" onClick={() => api.testMcpTool(tool.callName).then(() => reloadMcp()).catch(() => reloadMcp())}>Test</Button>}
                        </div>
                      ))}
                    </div>
                  )}
                  {server.runtime.logs.length > 0 && (
                    <div className="mt-2 rounded-lg bg-card-2 px-2.5 py-2 font-mono text-[12px] leading-relaxed text-ink-3">
                      {server.runtime.logs.slice(-2).map((line, i) => <div key={i} className="truncate">{line}</div>)}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        ) : (
          <div className="mt-3 flex flex-col items-center justify-center rounded-xl border border-dashed border-line py-10 text-center">
            <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-ink-5 text-ink-3"><Icon name="grid" size={20} /></span>
            <div className="mt-2.5 text-[13px] font-medium text-ink-2">No MCP servers yet</div>
            <div className="mt-0.5 max-w-[340px] text-[13px] text-ink-3">Add a stdio server to make external tools available to Chat and Code runs.</div>
          </div>
        )}
      </div>
      <WebhooksPanel />
    </div>
  )
}

function Labeled({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <div className="mb-1.5 text-[13px] font-medium text-ink-2">{label}</div>
      {children}
    </label>
  )
}

function SwitchRow({ label, checked, onChange }: { label: string; checked: boolean; onChange: (checked: boolean) => void }) {
  return (
    <div className="flex items-center gap-2">
      <Toggle label={label} checked={checked} onChange={onChange} />
      <span className="text-[13px] text-ink-3">{label}</span>
    </div>
  )
}

function splitArgs(value: string): string[] {
  const args: string[] = []
  let current = ''
  let quote: '"' | "'" | null = null
  let escaping = false
  for (const char of value) {
    if (escaping) { current += char; escaping = false; continue }
    if (char === '\\') { escaping = true; continue }
    if (quote) {
      if (char === quote) quote = null
      else current += char
      continue
    }
    if (char === '"' || char === "'") { quote = char; continue }
    if (/\s/.test(char)) {
      if (current) { args.push(current); current = '' }
      continue
    }
    current += char
  }
  if (current) args.push(current)
  return args
}

function splitCsv(value: string): string[] {
  return [...new Set(value.split(',').map((item) => item.trim()).filter(Boolean))]
}
