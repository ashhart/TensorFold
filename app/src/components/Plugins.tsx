import { currentIntlLocale } from '../lib/preferences/current-locale'
import { statusText } from '../lib/preferences/display'
import { OriginalDiagnostic } from './OriginalDiagnostic'
import { authoredText } from '../lib/preferences/language-packs'
import { useStore as useLocaleStore } from '../lib/store'
import { useAuthoredText } from '../lib/preferences/useAuthoredText'
import { useEffect, useRef, useState } from 'react'
import clsx from 'clsx'
import { Icon, type IconName } from '../lib/icons'
import { BrandIcon, connectorBrand } from '../lib/brand'
import { Button, Input, Toggle } from './ui'
import { api } from '../lib/ipc'
import { useStore, requestConnectorSetup } from '../lib/store'
import { CONNECTOR_CATALOG, CONNECTOR_SETUP_IDS } from '../../shared/connector-catalog'
import { WebhooksPanel } from './WebhooksPanel'
import type { AssistantControlSnapshot, CapabilityHealthState, ConnectorDefinition, ConnectorStatus, LocalScriptCapability, McpServerConfig, McpServerStatus, McpSnapshot, PackagedPluginCapability, WebhookSnapshot } from '../lib/types'
import { BUILTIN_CAPABILITIES, type BuiltInCapability } from '../lib/capabilities'
import { executableEvidence, formatMcpArguments, matchesPluginSearch, parseMcpArguments, pluginRegistrationEnabled } from '../lib/plugin-library'

const CONNECTORS = CONNECTOR_SETUP_IDS.map((id) => {  return (({ id, name: CONNECTOR_CATALOG[id].name, tag: CONNECTOR_CATALOG[id].tag, desc: CONNECTOR_CATALOG[id].description })) })

const LIBRARY_TABS = ['Built-in', 'Installed', 'Connections', 'Advanced'] as const
type LibraryTab = typeof LIBRARY_TABS[number]

const MCP_STATUS_COLOR: Record<McpServerStatus, string> = {
  connected: 'bg-positive',
  connecting: 'bg-warn',
  stopped: 'bg-ink-4',
  error: 'bg-danger',
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
  if (!value) return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "never checked")
  const minutes = Math.max(0, Math.round((Date.now() - value) / 60_000))
  if (minutes < 1) return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "just now")
  if (minutes < 60) return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "{value0}m ago", { value0: String(minutes) })
  const hours = Math.round(minutes / 60)
  return hours < 24 ? authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "{value0}h ago", { value0: String(hours) }) : authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "{value0}d ago", { value0: String(Math.round(hours / 24)) })
}

type ExecutableTrustItem = {
  key: string
  id: string
  source: 'script' | 'plugin'
  icon: IconName
  name: string
  kind: string
  enabled: boolean
  canEnable: boolean
  description: string | null
  detail: string
  evidence: string
  access: string
  checksum: string | null
  recordedAt: number | null
  tools: string[]
}

function scriptTrustItem(script: LocalScriptCapability): ExecutableTrustItem {
  return {
    key: `script:${script.id}`, id: script.id, source: 'script', icon: 'terminal',
    name: script.name || script.id, kind: 'Local script', enabled: script.enabled !== false,
    canEnable: true, description: script.description ?? null, detail: script.path,
    evidence: executableEvidence(script), access: script.sandboxProfile ?? 'No execution profile configured',
    checksum: script.checksumSha256 ?? null, recordedAt: script.signatureVerifiedAt ?? null, tools: [],
  }
}

function pluginTrustItem(plugin: PackagedPluginCapability): ExecutableTrustItem {
  return {
    key: `plugin:${plugin.id}`, id: plugin.id, source: 'plugin', icon: 'plug',
    name: plugin.name || plugin.id, kind: plugin.trusted ? 'Trust approved' : 'Trust required',
    enabled: pluginRegistrationEnabled(plugin), canEnable: plugin.trusted === true,
    description: plugin.description ?? null, detail: plugin.manifestPath,
    evidence: executableEvidence(plugin), access: 'Runs inside the app with host access',
    checksum: plugin.checksumSha256 ?? null, recordedAt: plugin.signatureVerifiedAt ?? null,
    tools: plugin.toolNames ?? [],
  }
}

function CapabilityTile({
  capability,
  enabled,
  health,
  onToggle,
  onCheck,
  disabled,
}: {
  capability: BuiltInCapability
  enabled: boolean
  health?: CapabilityHealthState
  onToggle: (enabled: boolean) => void
  onCheck: () => void
  disabled: boolean
}) {
  const tx = useAuthoredText()
  const healthStatus = health?.status ?? 'unknown'
  return (
    <div className="rounded-xl border border-line bg-card p-3.5 shadow-card">
      <div className="flex items-start gap-3">
        <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg text-white shadow-[inset_0_1px_rgba(255,255,255,.18)]" style={{ background: capability.color }}>
          <Icon name={capability.icon} size={19} strokeWidth={1.9} />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex items-start justify-between gap-2">
            <span className="text-[14px] font-medium text-ink">{capability.name}</span>
            <Toggle disabled={disabled} checked={enabled} onChange={onToggle} label={`${enabled ? 'Disable' : 'Enable'} ${capability.name}`} />
          </div>
          <span className={clsx('mt-1 inline-block rounded-md px-1.5 py-0.5 text-[12px] font-medium', enabled && capability.status === 'available' ? 'bg-positive/10 text-positive' : 'bg-ink-5 text-ink-2')}>
            {!enabled ? tx("Disabled") : capability.status === 'available' ? tx("Available") : tx("Worker required")}
          </span>
          <p className="mt-1 text-[13px] leading-snug text-ink-2">{capability.description}</p>
          <div className="mt-2 flex items-center justify-between gap-2">
            <div className="min-w-0">
              <div className="font-mono text-[12px] text-ink-2">@{capability.mention}</div>
              <div className={clsx("mt-0.5 text-[12px] leading-snug", healthStatus === 'error' ? 'text-danger' : 'text-ink-2')}>{health?.lastError ?? (capability.status === 'contract' ? tx("The document worker is not installed.") : tx("Uses your existing tool approval settings."))}</div>
            </div>
            <Button variant="ghost" size="xs" icon="check" className="shrink-0 whitespace-nowrap" disabled={disabled} onClick={onCheck}>{tx("Check setup")}</Button>
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
  const tx = useAuthoredText()
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
          <div className="mt-2 grid grid-cols-1 gap-1 text-[12px] text-ink-2">
            <span className="truncate">{tools}</span>
            <span className="truncate font-mono">{provenance}</span>
          </div>
        </div>
      </div>
    </div>
  )
}

function ExecutableTrustCard({ item, disabled, onToggle }: { item: ExecutableTrustItem; disabled: boolean; onToggle: (enabled: boolean) => void }) {
  const tx = useAuthoredText()
  return (
    <div className="rounded-xl border border-line bg-card p-4 shadow-card">
      <div className="flex items-start gap-3">
        <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-ink-5 text-ink-2"><Icon name={item.icon} size={18} /></span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="break-words text-[14px] font-medium text-ink">{item.name}</span>
            <span className={clsx('rounded-md px-1.5 py-0.5 text-[12px] font-medium', item.enabled ? 'bg-positive/10 text-positive' : 'bg-ink-5 text-ink-2')}>{item.enabled ? tx("Enabled in settings") : item.canEnable ? tx("Disabled") : tx("Blocked")}</span>
          </div>
          <p className="mt-1 text-[13px] leading-snug text-ink-2">{item.description || (item.source === 'plugin' ? tx("A local packaged plugin registered on this Mac.") : tx("A local script registered on this Mac."))}</p>
        </div>
        <Toggle checked={item.enabled} disabled={disabled || (!item.enabled && !item.canEnable)} label={`${item.enabled ? 'Disable' : 'Enable'} ${item.name}`} onChange={onToggle} />
      </div>
      <div className="mt-3 flex flex-wrap gap-2 text-[12px] text-ink-2">
        <span className="rounded-md bg-ink-5 px-2 py-1">{item.kind}</span>
        <span className="rounded-md bg-ink-5 px-2 py-1">{item.evidence}</span>
      </div>
      <p className="mt-2 text-[13px] leading-snug text-ink-2">{item.access}</p>
      {!item.canEnable && <p className="mt-2 text-[13px] leading-snug text-ink-2">{tx("This registration has no persisted trust approval, so it cannot load.")}</p>}
      <details className="mt-3 text-[13px] text-ink-2">
        <summary className="cursor-pointer font-medium text-ink-2">{tx("Registration details")}</summary>
        <div className="mt-2 space-y-2 rounded-lg bg-card-2 p-3">
          <div className="break-all font-mono text-[12px]">{item.detail}</div>
          {item.checksum && <div className="break-all font-mono text-[12px]">{tx("SHA-256 record: ")}{item.checksum}</div>}
          {item.recordedAt && <div>{tx("Signature record saved ")}{shortTime(item.recordedAt)}.</div>}
          {item.source === 'plugin' && <>
            <div>{item.tools.length ? tx("Configured tools: {value0}", { value0: item.tools.join(', ') }) : tx("No tool names recorded in settings.")}</div>
            <p>{tx("Configured metadata does not confirm successful loading or restrict host access.")}</p>
          </>}
        </div>
      </details>
    </div>
  )
}

function LibraryEmpty({ title, detail }: { title: string; detail: string }) {
  const tx = useAuthoredText()
  return <div className="mt-4 rounded-xl border border-dashed border-line px-5 py-10 text-center">
    <Icon name="plug" size={24} className="mx-auto text-ink-2" />
    <p className="mt-3 text-[14px] font-medium text-ink-2">{title}</p>
    <p className="mx-auto mt-1 max-w-[380px] text-[13px] leading-relaxed text-ink-2">{detail}</p>
  </div>
}

function connectorStatusTone(status: ConnectorStatus | 'inactive'): string {
  if (status === 'connected') return 'bg-positive/10 text-positive'
  if (status === 'error') return 'bg-danger/10 text-danger'
  return 'bg-ink-5 text-ink-2'
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
      status: calendar?.status === 'mock' ? 'contract' : (calendar?.status ?? 'pending'),
      statusTone: connectorStatusTone(calendar?.status ?? 'pending'),
      detail: connectorCapabilityDetail(calendar, 'Calendar integration is not available yet.'),
      tools: connectorToolLabel(calendar),
      provenance: calendar?.transport ? `transport:${calendar.transport}` : 'provenance:contract only',
    },
    {
      icon: 'globe' as IconName,
      title: 'Webhooks',
      status: webhookStatus,
      statusTone: webhookStatus === 'connected' ? 'bg-positive/10 text-positive' : webhookStatus === 'pending' ? 'bg-warn/10 text-ink-2' : 'bg-ink-5 text-ink-2',
      detail: webhooks ? `${webhooks.endpoints.length} endpoint${webhooks.endpoints.length === 1 ? '' : 's'} configured; listener ${webhookListening ? 'online' : 'offline'}.` : 'Loading webhook listener and endpoint state.',
      tools: `${webhookEnabled} enabled endpoint${webhookEnabled === 1 ? '' : 's'}`,
      provenance: webhooks ? `loopback:${webhooks.listener.port ?? 'off'} routes:${webhooks.listener.routes}` : 'provenance:loading',
    },
    {
      icon: 'plug' as IconName,
      title: 'MCP',
      status: mcpStatus,
      statusTone: mcpStatus === 'connected' ? 'bg-positive/10 text-positive' : mcpStatus === 'error' ? 'bg-danger/10 text-danger' : 'bg-ink-5 text-ink-2',
      detail: mcp ? `${mcp.servers.length} server${mcp.servers.length === 1 ? '' : 's'} registered; ${mcpConnected} connected.` : 'Loading MCP server runtime state.',
      tools: `${mcpTools} tool${mcpTools === 1 ? '' : 's'} · ${mcpResources} resource${mcpResources === 1 ? '' : 's'} · ${mcpPrompts} prompt${mcpPrompts === 1 ? '' : 's'}`,
      provenance: mcp?.servers.find((server) => server.runtime.provenance)?.runtime.provenance?.commandSha256 ? `sha256:${mcp.servers.find((server) => server.runtime.provenance)?.runtime.provenance?.commandSha256.slice(0, 12)}` : 'provenance:not checked',
    },
  ]
}

type McpDraft = Partial<McpServerConfig> & { argsText?: string; envAllowlistText?: string }

export function Plugins() {
  const tx = useAuthoredText()
  const settings = useStore((s) => s.settings)
  const status = useStore((s) => s.status)
  const threads = useStore((s) => s.threads)
  const [tab, setTab] = useState<LibraryTab>('Built-in')
  const [query, setQuery] = useState('')
  const [snapshot, setSnapshot] = useState<AssistantControlSnapshot | null>(null)
  const [mcp, setMcp] = useState<McpSnapshot | null>(null)
  const [webhooks, setWebhooks] = useState<WebhookSnapshot | null>(null)
  const [mcpBusy, setMcpBusy] = useState<string | null>(null)
  const [mcpDraft, setMcpDraft] = useState<McpDraft | null>(null)
  const [settingsBusy, setSettingsBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [loadErrors, setLoadErrors] = useState<string[]>([])
  const [loading, setLoading] = useState(true)
  const settingsLock = useRef(false)
  const mcpLock = useRef(false)
  const refreshId = useRef(0)
  const activeRuns = threads.some((thread) => ['thinking', 'streaming', 'tool', 'awaiting'].includes(status[thread.id] ?? (thread.status === 'running' ? 'thinking' : 'idle')))
  const executableTrustItems = [
    ...(settings?.capabilities?.packagedPlugins ?? []).map(pluginTrustItem),
    ...(settings?.capabilities?.localScripts ?? []).map(scriptTrustItem),
  ]
  const builtins = BUILTIN_CAPABILITIES.filter((item) => matchesPluginSearch(query, item.name, item.description, item.mention))
  const installed = executableTrustItems.filter((item) => matchesPluginSearch(query, item.name, item.description, item.kind, ...item.tools))
  const connectors = CONNECTORS.filter((item) => matchesPluginSearch(query, item.name, item.desc, item.tag))
  const servers = mcp?.servers.filter((item) => matchesPluginSearch(query, item.name, item.command, ...item.args)) ?? []

  const refreshLibrary = async () => {
    const id = ++refreshId.current
    setLoading(true)
    const next = await Promise.allSettled([api.getAssistantSnapshot(), api.getMcpSnapshot(), api.getWebhooks()])
    if (refreshId.current !== id) return
    const labels = ['Connections', 'MCP servers', 'Webhooks']
    const failures: string[] = []
    next.forEach((result, i) => {
      if (result.status === 'rejected') failures.push(`${labels[i]}: ${message(result.reason)}`)
    })
    if (next[0].status === 'fulfilled') setSnapshot(next[0].value)
    if (next[1].status === 'fulfilled') setMcp(next[1].value)
    if (next[2].status === 'fulfilled') setWebhooks(next[2].value)
    setLoadErrors(failures)
    setLoading(false)
  }
  useEffect(() => {
    void refreshLibrary()
    return () => { refreshId.current++ }
  }, [])

  const changeSettings = async (update: (current: NonNullable<typeof settings>) => NonNullable<typeof settings>) => {
    if (settingsLock.current || activeRuns) return
    const current = useStore.getState().settings
    if (!current) return
    settingsLock.current = true
    setSettingsBusy(true)
    setError(null)
    try { await useStore.getState().saveSettings(update(current)) }
    catch (reason) { setError(message(reason)) }
    finally { settingsLock.current = false; setSettingsBusy(false) }
  }
  const updateCapabilityEnabled = (capabilityId: string, enabled: boolean) => changeSettings((current) => {
    const disabled = new Set(current.capabilities.disabledCapabilityIds)
    if (enabled) disabled.delete(capabilityId)
    else disabled.add(capabilityId)
    return { ...current, capabilities: { ...current.capabilities, disabledCapabilityIds: [...disabled] } }
  })
  const checkCapability = (capability: BuiltInCapability, enabled: boolean) => changeSettings((current) => ({
    ...current, capabilities: { ...current.capabilities, health: { ...current.capabilities.health, [capability.id]: capabilityHealthCheck(capability, enabled) } },
  }))
  const updateExecutableEnabled = (item: ExecutableTrustItem, enabled: boolean) => changeSettings((current) => ({
    ...current,
    capabilities: item.source === 'plugin'
      ? { ...current.capabilities, packagedPlugins: current.capabilities.packagedPlugins.map((plugin) => plugin.id === item.id ? { ...plugin, enabled: enabled && plugin.trusted === true } : plugin) }
      : { ...current.capabilities, localScripts: current.capabilities.localScripts.map((script) => script.id === item.id ? { ...script, enabled } : script) },
  }))
  const startMcpDraft = (server?: McpServerConfig) => {
    setMcpDraft(server ? { ...server, env: undefined, argsText: formatMcpArguments(server.args), envAllowlistText: server.dockerGateway?.envAllowlist?.join(', ') ?? '' } : {
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
    if (!mcpDraft || mcpLock.current) return
    mcpLock.current = true
    refreshId.current++
    setLoading(false)
    setMcpBusy('save')
    setError(null)
    try {
      const payload: Partial<McpServerConfig> = {
        id: mcpDraft.id,
        name: mcpDraft.name?.trim() || 'MCP server',
        transport: 'stdio',
        command: mcpDraft.command?.trim() || '',
        args: parseMcpArguments(mcpDraft.argsText || ''),
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
      setMcp(await api.saveMcpServer(payload))
      setMcpDraft(null)
    } catch (reason) {
      setError(message(reason))
    } finally {
      mcpLock.current = false
      setMcpBusy(null)
    }
  }
  const withMcpBusy = async (label: string, action: () => Promise<McpSnapshot>) => {
    if (mcpLock.current) return
    mcpLock.current = true
    refreshId.current++
    setLoading(false)
    setMcpBusy(label)
    setError(null)
    try { setMcp(await action()) }
    catch (reason) { setError(message(reason)) }
    finally { mcpLock.current = false; setMcpBusy(null) }
  }
  const testMcpTool = (callName: string) => withMcpBusy(`test:${callName}`, async () => {
    const result = await api.testMcpTool(callName)
    const next = await api.getMcpSnapshot()
    if (!result.ok) { setMcp(next); throw new Error(result.output || tx("The tool check failed.")) }
    return next
  })

  return (
    <div className="scroll-thin h-full overflow-y-auto xl:pr-[304px]">
      <div className="mx-auto w-full max-w-[800px] px-4 py-7 sm:px-7 sm:py-9">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h1 className="text-[20px] font-semibold tracking-[-0.02em] text-ink">{tx("Plugins")}</h1>
            <p className="mt-1 text-[13px] leading-relaxed text-ink-2">{tx("Choose the tools and connections your assistant can use.")}</p>
          </div>
          <Button variant="ghost" size="sm" icon="refresh" disabled={loading || !!mcpBusy} onClick={() => void refreshLibrary()}>{loading ? tx("Checking") : tx("Refresh status")}</Button>
        </div>
        <nav aria-label={tx("Plugin categories")} className="mt-6 grid grid-cols-2 gap-1 rounded-xl border border-line bg-card-2 p-1 sm:grid-cols-4">
          {LIBRARY_TABS.map((name) => <button key={name} type="button" aria-pressed={tab === name} onClick={() => setTab(name)} className={clsx('rounded-lg px-3 py-2 text-[13px] font-medium transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent', tab === name ? 'bg-card text-ink shadow-card' : 'text-ink-2 hover:bg-ink-5 hover:text-ink')}>
            {name}{name === 'Installed' && executableTrustItems.length > 0 ? ` · ${executableTrustItems.length}` : ''}
          </button>)}
        </nav>
        <div className="relative mt-4">
          <Icon name="search" size={16} className="pointer-events-none absolute left-3 top-2.5 text-ink-2" />
          <Input aria-label={tx("Search plugins and connections")} value={query} onChange={(event) => setQuery(event.target.value)} placeholder={tab === 'Built-in' ? tx("Search built-in tools") : tab === 'Installed' ? tx("Search installed plugins and scripts") : tab === 'Connections' ? tx("Search connections") : tx("Search MCP servers and services")} className="pl-9" />
        </div>
        {(error || loadErrors.length > 0) && <div role="alert" className="mt-3 rounded-xl border border-danger/30 bg-danger/5 p-3 text-[13px] leading-relaxed text-danger">
          {error && <p>{<OriginalDiagnostic>{error}</OriginalDiagnostic>}</p>}{loadErrors.map((detail) => <p key={detail}>{detail}</p>)}
          {loadErrors.length > 0 && <button type="button" className="mt-2 font-medium underline" disabled={loading} onClick={() => void refreshLibrary()}>{tx("Retry status check")}</button>}
        </div>}
        {activeRuns && (tab === 'Built-in' || tab === 'Installed') && <p role="status" className="mt-3 text-[13px] leading-relaxed text-ink-2">{tx("Finish or stop active tasks before changing tools, because settings changes reload registered plugins.")}</p>}

        {tab === 'Built-in' && <section aria-label={tx("Built-in tools")} className="mt-5">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <h2 className="text-[14px] font-medium text-ink">{tx("Included with TensorFold")}</h2>
            <span className="text-[12px] text-ink-2">{BUILTIN_CAPABILITIES.filter((item) => item.status === 'available').length} {tx("available · ")}{BUILTIN_CAPABILITIES.filter((item) => item.status === 'contract').length} {tx("need a worker")}</span>
          </div>
          <p className="mt-1 text-[13px] leading-relaxed text-ink-2">{tx("Type ")}<span className="font-mono">@</span> {tx("in a chat to select a tool; access still follows your approval settings.")}</p>
          <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
            {builtins.map((capability) => {
              const enabled = !(settings?.capabilities?.disabledCapabilityIds ?? []).includes(capability.id)
              return <CapabilityTile key={capability.id} capability={capability} enabled={enabled} health={settings?.capabilities?.health?.[capability.id]} disabled={!settings || settingsBusy || activeRuns} onToggle={(next) => void updateCapabilityEnabled(capability.id, next)} onCheck={() => void checkCapability(capability, enabled)} />
            })}
          </div>
          {!builtins.length && <LibraryEmpty title={tx("No built-in tools match")} detail="Try another name, such as browser or documents." />}
        </section>}

        {tab === 'Installed' && <section aria-label={tx("Installed plugins")} className="mt-5">
          <h2 className="text-[14px] font-medium text-ink">{tx("On this Mac")}</h2>
          <p className="mt-1 text-[13px] leading-relaxed text-ink-2">{tx("Manage existing local plugin and script registrations; enabling a trusted plugin loads its code into the app.")}</p>
          <div className="mt-3 space-y-3">{installed.map((item) => <ExecutableTrustCard key={item.key} item={item} disabled={!settings || settingsBusy || activeRuns} onToggle={(enabled) => void updateExecutableEnabled(item, enabled)} />)}</div>
          {!installed.length && <LibraryEmpty title={query.trim() ? tx("No installed plugins match") : tx("No local plugins registered")} detail={query.trim() ? 'Try another name or clear your search.' : 'Built-in tools are available in the first tab; external tool servers can be configured in Advanced.'} />}
          <p className="mt-3 text-[12px] leading-relaxed text-ink-2">{tx("Plugin registration does not isolate code or confirm successful loading; review the source before approving trust.")}</p>
        </section>}

        {tab === 'Connections' && <section aria-label={tx("Connections")} className="mt-5">
          <h2 className="text-[14px] font-medium text-ink">{tx("Connect your services")}</h2>
          <p className="mt-1 text-[13px] leading-relaxed text-ink-2">{tx("Setup opens the connection settings with credential fields, scope controls and connection checks.")}</p>
          <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
            {connectors.map((connector) => {
              const contract = snapshot?.connectors.find((item) => item.id === connector.id)
              const state = contract?.status
              const label = !snapshot ? tx("Status unavailable") : state === 'mock' ? tx("Demo connection") : state === 'connected' ? tx("Connected") : state === 'error' ? tx("Needs attention") : tx("Not connected")
              return <div key={connector.id} className="flex flex-col rounded-xl border border-line bg-card p-4 shadow-card">
                <div className="flex items-start gap-3">
                  <BrandIcon brand={connectorBrand(connector.id)} size={36} />
                  <div className="min-w-0"><h3 className="text-[14px] font-medium text-ink">{connector.name}</h3><p className="mt-0.5 text-[12px] text-ink-2">{connector.tag}</p></div>
                </div>
                <p className="mt-3 flex-1 text-[13px] leading-relaxed text-ink-2">{connector.desc}</p>
                {contract?.error && <p className="mt-2 break-words text-[13px] text-danger">{<OriginalDiagnostic>{contract.error}</OriginalDiagnostic>}</p>}
                <div className="mt-3 flex flex-wrap items-center justify-between gap-2 border-t border-line pt-3">
                  <span className={clsx('text-[12px]', state === 'connected' ? 'text-positive' : state === 'error' ? 'text-danger' : 'text-ink-2')}>{label}</span>
                  <Button variant="default" size="xs" icon="chevronRight" onClick={() => { requestConnectorSetup(connector.id); useStore.getState().setView({ name: 'settings', section: 'connectors' }) }}>{state === 'connected' ? tx("Manage") : tx("Set up")}</Button>
                </div>
              </div>
            })}
          </div>
          {!connectors.length && <LibraryEmpty title={tx("No connections match")} detail="Try another service name or clear your search." />}
        </section>}

        {tab === 'Advanced' && <section aria-label={tx("Advanced plugin connections")}>
        <div className="mt-5 flex flex-wrap items-end justify-between gap-3">
          <div>
            <h2 className="text-[14px] font-medium text-ink">{tx("MCP servers")}</h2>
            <p className="mt-1 text-[13px] text-ink-2">{tx("Connect a tool server; use Ask to review each action, or explicitly choose Trusted.")}</p>
          </div>
          <div className="flex gap-2">
            <Button variant="default" icon="play" disabled={!!mcpBusy || !mcp?.servers.length} onClick={() => withMcpBusy('refresh', () => api.refreshMcpServer())}>{mcpBusy === 'refresh' ? tx("Refreshing") : tx("Refresh")}</Button>
            <Button variant="primary" icon="plus" disabled={!!mcpBusy} onClick={() => startMcpDraft()}>{tx("Add server")}</Button>
          </div>
        </div>

        <div className="mt-3 rounded-xl border border-line bg-card-2 px-3.5 py-2.5 text-[13px] text-ink-2">
          {mcp ? tx("{value0}/{value1} servers connected · {value2} tools · {value3} resources · {value4} prompts.", { value0: mcp.servers.filter((server) => server.runtime.status === 'connected').length, value1: mcp.servers.length, value2: mcp.tools.length, value3: mcp.resources.length, value4: mcp.prompts.length }) : loading ? tx("Checking MCP servers…") : tx("MCP status unavailable; retry the status check.")}
        </div>

        {mcpDraft && (
          <fieldset disabled={!!mcpBusy} className="mt-3 min-w-0 rounded-xl border border-accent/40 bg-card p-3.5 shadow-card">
            <div className="mb-3 flex items-center justify-between">
              <div>
                <div className="text-[13px] font-medium text-ink">{mcpDraft.id ? tx("Edit MCP server") : tx("New MCP server")}</div>
                <div className="text-[13px] text-ink-2">{tx("Stdio transport; environment secrets stay in the main process.")}</div>
              </div>
              <Button variant="ghost" size="sm" disabled={!!mcpBusy} onClick={() => setMcpDraft(null)}>{tx("Cancel")}</Button>
            </div>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              <Labeled label={tx("Name")}><Input value={mcpDraft.name || ''} onChange={(e) => setMcpDraft({ ...mcpDraft, name: e.target.value })} placeholder={tx("Filesystem")} /></Labeled>
              <Labeled label={tx("Command")}><Input value={mcpDraft.command || ''} onChange={(e) => setMcpDraft({ ...mcpDraft, command: e.target.value })} placeholder={tx("npx")} spellCheck={false} /></Labeled>
              <Labeled label={tx("Arguments")}><Input value={mcpDraft.argsText || ''} onChange={(e) => setMcpDraft({ ...mcpDraft, argsText: e.target.value })} placeholder="-y @modelcontextprotocol/server-filesystem ~/Projects" spellCheck={false} /></Labeled>
              <Labeled label={tx("Working directory")}><Input value={mcpDraft.cwd || ''} onChange={(e) => setMcpDraft({ ...mcpDraft, cwd: e.target.value })} placeholder={tx("optional")} spellCheck={false} /></Labeled>
            </div>
            <div className="mt-3 rounded-xl border border-line bg-card-2 p-3">
              <div className="mb-2 flex flex-wrap items-center gap-2">
                <span className="text-[13px] font-medium text-ink-2">{tx("Launch mode")}</span>
                <button onClick={() => setMcpDraft({ ...mcpDraft, launchMode: 'direct', dockerGateway: null })} className={clsx('rounded-lg px-2.5 py-1.5 text-[13px] font-medium transition-colors', (mcpDraft.launchMode ?? 'direct') !== 'docker_gateway' ? 'bg-accent text-accent-ink' : 'bg-ink-5 text-ink-2 hover:text-ink')}>{tx("Direct")}</button>
                <button onClick={() => setMcpDraft({ ...mcpDraft, launchMode: 'docker_gateway', dockerGateway: { image: mcpDraft.dockerGateway?.image ?? '', network: mcpDraft.dockerGateway?.network ?? 'none', readOnlyRootfs: mcpDraft.dockerGateway?.readOnlyRootfs !== false, envAllowlist: splitCsv(mcpDraft.envAllowlistText || ''), mounts: mcpDraft.dockerGateway?.mounts ?? [], workdir: mcpDraft.dockerGateway?.workdir ?? null } })} className={clsx('rounded-lg px-2.5 py-1.5 text-[13px] font-medium transition-colors', mcpDraft.launchMode === 'docker_gateway' ? 'bg-accent text-accent-ink' : 'bg-ink-5 text-ink-2 hover:text-ink')}>{tx("Docker gateway")}</button>
              </div>
              {mcpDraft.launchMode === 'docker_gateway' && (
                <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                  <Labeled label={tx("Docker image digest")}><Input value={mcpDraft.dockerGateway?.image || ''} onChange={(e) => setMcpDraft({ ...mcpDraft, dockerGateway: { ...(mcpDraft.dockerGateway ?? { network: 'none', readOnlyRootfs: true, mounts: [], envAllowlist: [] }), image: e.target.value } })} placeholder={tx("mcp/server@sha256:…")} spellCheck={false} /></Labeled>
                  <Labeled label={tx("Env allowlist")}><Input value={mcpDraft.envAllowlistText || ''} onChange={(e) => setMcpDraft({ ...mcpDraft, envAllowlistText: e.target.value })} placeholder={"API_TOKEN, TENANT_ID"} spellCheck={false} /></Labeled>
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-[13px] text-ink-2">{tx("Network")}</span>
                    {(['none', 'bridge'] as const).map((mode) => (
                      <button key={mode} onClick={() => setMcpDraft({ ...mcpDraft, dockerGateway: { ...(mcpDraft.dockerGateway ?? { image: '', readOnlyRootfs: true, mounts: [], envAllowlist: [] }), network: mode } })} className={clsx('rounded-lg px-2.5 py-1.5 text-[13px] font-medium transition-colors', (mcpDraft.dockerGateway?.network ?? 'none') === mode ? 'bg-accent text-accent-ink' : 'bg-ink-5 text-ink-2 hover:text-ink')}>{mode}</button>
                    ))}
                    {mcpDraft.dockerGateway?.network === 'host' && <span className="text-[12px] text-danger">{tx("Host networking is blocked; choose none or bridge.")}</span>}
                  </div>
                  <SwitchRow label={tx("Read-only rootfs")} checked={mcpDraft.dockerGateway?.readOnlyRootfs !== false} onChange={(readOnlyRootfs) => setMcpDraft({ ...mcpDraft, dockerGateway: { ...(mcpDraft.dockerGateway ?? { image: '', network: 'none', mounts: [], envAllowlist: [] }), readOnlyRootfs } })} />
                </div>
              )}
            </div>
            <div className="mt-3 flex flex-wrap items-center gap-4">
              <SwitchRow label={tx("Enabled")} checked={mcpDraft.enabled ?? true} onChange={(enabled) => setMcpDraft({ ...mcpDraft, enabled })} />
              <SwitchRow label={tx("Auto-connect")} checked={mcpDraft.autoConnect ?? false} onChange={(autoConnect) => setMcpDraft({ ...mcpDraft, autoConnect })} />
              <div className="flex items-center gap-2">
                <span className="text-[13px] text-ink-2">{tx("Policy")}</span>
                <button onClick={() => setMcpDraft({ ...mcpDraft, approvalMode: 'ask' })} className={clsx('rounded-lg px-2.5 py-1.5 text-[13px] font-medium transition-colors', (mcpDraft.approvalMode ?? 'ask') === 'ask' ? 'bg-accent text-accent-ink' : 'bg-ink-5 text-ink-2 hover:text-ink')}>{tx("Ask")}</button>
                <button onClick={() => setMcpDraft({ ...mcpDraft, approvalMode: 'trusted' })} className={clsx('rounded-lg px-2.5 py-1.5 text-[13px] font-medium transition-colors', mcpDraft.approvalMode === 'trusted' ? 'bg-accent text-accent-ink' : 'bg-ink-5 text-ink-2 hover:text-ink')}>{tx("Trusted")}</button>
              </div>
              <Button variant="primary" size="sm" icon="check" onClick={saveMcpDraft} disabled={!!mcpBusy || !mcpDraft.command?.trim() || (mcpDraft.launchMode === 'docker_gateway' && !mcpDraft.dockerGateway?.image?.trim())}>{mcpBusy === 'save' ? tx("Saving") : tx("Save server")}</Button>
            </div>
          </fieldset>
        )}

        {servers.length > 0 ? (
          <div className="mt-3 space-y-2.5">
            {servers.map((server) => {
              const connected = server.runtime.status === 'connected'
              return (
                <div key={server.id} className="rounded-xl border border-line bg-card p-3.5 shadow-card">
                  <div className="flex flex-col items-start gap-3 sm:flex-row">
                    <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-ink-5 text-ink-2"><Icon name="plug" size={18} /></span>
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="text-[14px] font-medium text-ink">{server.name}</span>
                        <span className="flex items-center gap-1.5 text-[12px] text-ink-2"><span className={clsx('h-1.5 w-1.5 rounded-full', MCP_STATUS_COLOR[server.runtime.status])} />{statusText(server.runtime.status, tx)}</span>
                        {server.launchMode === 'docker_gateway' && <span className="rounded-md bg-ink-5 px-1.5 py-0.5 text-[12px] font-medium text-ink-2">{tx("docker gateway")}</span>}
                        {server.approvalMode === 'trusted' && <span className="rounded-md bg-accent-soft px-1.5 py-0.5 text-[12px] font-medium text-accent">{tx("trusted")}</span>}
                      </div>
                      <div className="mt-1 truncate font-mono text-[13px] text-ink-2">{server.command} {server.args.join(' ')}</div>
                      {server.runtime.lastError && <div className="mt-1 text-[13px] text-danger">{<OriginalDiagnostic>{server.runtime.lastError}</OriginalDiagnostic>}</div>}
                    </div>
                    <div className="flex shrink-0 flex-wrap justify-end gap-1.5">
                      <Button variant="ghost" size="xs" icon="edit" disabled={!!mcpBusy} onClick={() => startMcpDraft(server)}>{tx("Edit")}</Button>
                      {connected
                        ? <Button variant="default" size="xs" icon="stop" disabled={!!mcpBusy} onClick={() => withMcpBusy(`disconnect:${server.id}`, () => api.disconnectMcpServer(server.id))}>{tx("Stop")}</Button>
                        : <Button variant="default" size="xs" icon="play" disabled={!!mcpBusy || !server.enabled} onClick={() => withMcpBusy(`connect:${server.id}`, () => api.connectMcpServer(server.id))}>{tx("Connect")}</Button>}
                      <Button variant="ghost" size="xs" icon="trash" disabled={!!mcpBusy} onClick={() => withMcpBusy(`delete:${server.id}`, () => api.deleteMcpServer(server.id))}>{tx("Delete")}</Button>
                    </div>
                  </div>
                  <div className="mt-3 flex flex-wrap items-center gap-2 text-[13px] text-ink-2">
                    <span>{tx("Tools: {count}", { count: server.runtime.tools.length })}</span>
                    <span>{tx("Resources: {count}", { count: server.runtime.resources.length })}</span>
                    <span>{tx("Prompts: {count}", { count: server.runtime.prompts.length })}</span>
                    {server.runtime.pid && <span>{tx("pid ")}{server.runtime.pid}</span>}
                    {server.runtime.lastConnectedAt && <span>{tx("connected ")}{new Date(server.runtime.lastConnectedAt).toLocaleTimeString(currentIntlLocale())}</span>}
                    <button disabled={!!mcpBusy} className="font-medium text-ink-2 hover:text-ink disabled:opacity-40" onClick={() => withMcpBusy(`refresh:${server.id}`, () => api.refreshMcpServer(server.id))}>{tx("refresh")}</button>
                  </div>
                  {server.runtime.tools.length > 0 && (
                    <div className="mt-2 grid grid-cols-1 gap-1.5 sm:grid-cols-2">
                      {server.runtime.tools.slice(0, 6).map((tool) => (
                        <div key={tool.callName} className="flex items-center gap-2 rounded-lg border border-line bg-card-2 px-2.5 py-2">
                          <Icon name={tool.readOnlyHint ? 'search' : tool.destructiveHint ? 'warning' : 'terminal'} size={13} className={tool.destructiveHint ? 'text-ink-2' : 'text-ink-2'} />
                          <div className="min-w-0 flex-1">
                            <div className="truncate text-[13px] font-medium text-ink-2">{tool.title || tool.name}</div>
                            <div className="truncate font-mono text-[12px] text-ink-4">{tool.callName}</div>
                          </div>
                          {tool.readOnlyHint && <Button variant="ghost" size="xs" disabled={!!mcpBusy} onClick={() => void testMcpTool(tool.callName)}>{tx("Test")}</Button>}
                        </div>
                      ))}
                    </div>
                  )}
                  {server.runtime.logs.length > 0 && (
                    <div className="mt-2 rounded-lg bg-card-2 px-2.5 py-2 font-mono text-[12px] leading-relaxed text-ink-2">
                      {server.runtime.logs.slice(-2).map((line, i) => <div key={i} className="truncate">{line}</div>)}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        ) : (
          <div className="mt-3 flex flex-col items-center justify-center rounded-xl border border-dashed border-line py-10 text-center">
            <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-ink-5 text-ink-2"><Icon name="grid" size={20} /></span>
            <div className="mt-2.5 text-[13px] font-medium text-ink-2">{query.trim() ? tx("No MCP servers match") : tx("No MCP servers yet")}</div>
            <div className="mt-0.5 max-w-[340px] text-[13px] text-ink-2">{query.trim() ? tx("Try another server name or clear your search.") : tx("Add a stdio server to make external tools available to Chat and Code runs.")}</div>
          </div>
        )}
        <h2 className="mt-7 text-[14px] font-medium text-ink">{tx("Runtime status")}</h2>
        <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
          {sourceCards(snapshot, mcp, webhooks).filter((card) => matchesPluginSearch(query, card.title, card.detail, card.tools)).map((card) => <SourceCapabilityCard key={card.title} {...card} />)}
        </div>
        {matchesPluginSearch(query, 'webhooks incoming events endpoints') && <WebhooksPanel />}
        </section>}
      </div>
    </div>
  )
}

function Labeled({ label, children }: { label: string; children: React.ReactNode }) {
  const tx = useAuthoredText()
  return (
    <label className="block">
      <div className="mb-1.5 text-[13px] font-medium text-ink-2">{label}</div>
      {children}
    </label>
  )
}

function SwitchRow({ label, checked, onChange }: { label: string; checked: boolean; onChange: (checked: boolean) => void }) {
  const tx = useAuthoredText()
  return (
    <div className="flex items-center gap-2">
      <Toggle label={label} checked={checked} onChange={onChange} />
      <span className="text-[13px] text-ink-2">{label}</span>
    </div>
  )
}

function message(reason: unknown): string {
  return reason instanceof Error ? reason.message : String(reason)
}

function splitCsv(value: string): string[] {
  return [...new Set(value.split(',').map((item) => item.trim()).filter(Boolean))]
}
