import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { ModelHelperDefaultsPanel } from '../models/ModelHelperDefaultsPanel'
import clsx from 'clsx'
import { BrandIcon, providerBrand } from '../../lib/brand'
import { Icon } from '../../lib/icons'
import { Button, Field, Input, Segmented, Toggle } from '../ui'
import { CliProviderSetup } from './CliProviderSetup'
import { ProviderToolsPanel } from './ProviderToolsPanel'
import { isLoopbackEndpoint } from '../../lib/provider-discovery'
import { modelDisplayName, providerAccountChoices } from '../../lib/provider-accounts'
import type { ModelFailoverResetPolicy, Provider, ProviderAccountKind, ProviderDiscoveryKind, Settings } from '../../lib/types'
import { useState } from 'react'
import {
  HTTP_CLOUD_PROVIDER_PRESETS,
  LOCAL_RUNTIME_PRESETS,
  SELECT_CLASS,
  AdvancedSection,
  Section,
  SettingsPane,
  SpendGuardControls,
  Spinner,
  type CloudTransportType,
  type LocalRuntimeKey,
  type LocalSetupLocation,
  type ProviderSetupMode,
  type SettingsSectionContext,
} from './shared'

export function ModelsSettings({ ctx }: { ctx: SettingsSectionContext }) {
  const tx = useAuthoredText()
  const {
    draft,
    patch,
    updateProvider,
    commitProviders,
    updateProviderAccount,
    commitProviderAccount,
    commitProviderSpendBudget,
    commitProviderAccountSpendBudget,
    addProviderAccount,
    addCliSubscriptionPair,
    removeProviderAccount,
    accountProfileLabel,
    accountProfileHint,
    accountHomeHint,
    loginCommand,
    patchProviderAccountEnv,
    copyLoginCommand,
    copiedLoginAccount,
    selectedCloudPreset,
    selectedLocalRuntime,
    providerSetup,
    patchProviderSetup,
    selectLocalRuntime,
    localBaseUrl,
    localNodeName,
    localRuntimeDescription,
    providerExists,
    addConfiguredProvider,
    addCliProvider,
    testConfiguredProvider,
    setupBusy,
    setupStatus,
    addProvider,
    removeProvider,
    localDiscoveryKind,
    isOpenRouterProvider,
    isAiGatewayProvider,
    discoveryLabel,
    applyLocalDiscovery,
    discoverBusy,
    conn,
    connect,
  } = ctx

  // Provider whose model checklist modal is open (#46).
  const [editingModelsFor, setEditingModelsFor] = useState<string | null>(null)
  const editingProvider = editingModelsFor ? draft.providers.find((p) => p.id === editingModelsFor) ?? null : null

  return (
<SettingsPane>
        <ModelHelperDefaultsPanel draft={draft} patch={patch} />
        <ProviderToolsPanel ctx={ctx} />
        <SurfaceModelSection
          draft={draft}
          patch={patch}
          title={tx("Assistant model")}
          desc={tx("The model the Assistant surface uses for summaries and reply drafting. Any cloud, CLI, or local provider. Defaults to your active model when unset.")}
          providerKey="assistantProviderId"
          accountKey="assistantProviderAccountId"
          modelKey="assistantModel"
        />
        <SurfaceModelSection
          draft={draft}
          patch={patch}
          title={tx("Coding tasks model")}
          desc={tx("The model new coding tasks launch with (the home composer seeds from it; you can still change it per task). Defaults to your active model when unset.")}
          providerKey="codeProviderId"
          accountKey="codeProviderAccountId"
          modelKey="codeModel"
        />
        <Section title={tx("Models & accounts")} desc={tx("Add model sources here. Pick which agent uses which source inside rooms, automations, or solo runs.")}>
          <div className="space-y-3">
            <div className="rounded-xl border border-line bg-card p-3.5 shadow-card">
              <div className="mb-3 flex flex-wrap items-start justify-between gap-3">
                <div>
                  <div className="text-[14px] font-semibold text-ink">{tx("Add provider or node")}</div>
                  <div className="mt-0.5 text-[13px] leading-snug text-ink-3">{tx("Pick the provider type first, then TensorFold fills in the sensible endpoint shape.")}</div>
                </div>
                <Segmented<ProviderSetupMode>
                  value={providerSetup.mode}
                  onChange={(mode) => patchProviderSetup({ mode })}
                  options={[{ value: 'cloud', label: tx("Cloud"), icon: 'globe' }, { value: 'local', label: tx("Local node"), icon: 'cpu' }]}
                />
              </div>

              {providerSetup.mode === 'cloud' ? (
                <div className="space-y-3">
                  {/* Provider type is chosen once, here, at add-time (#44). The card
                      itself no longer lets you flip transport after creation. */}
                  <Field label={tx("Provider type")}>
                    <Segmented<CloudTransportType>
                      value={providerSetup.transportType}
                      onChange={(transportType) => patchProviderSetup({ transportType, providerName: '', apiKey: '', cliPath: '', cliProfile: '' })}
                      options={[
                        { value: 'http', label: tx("HTTP endpoint"), icon: 'globe' },
                        { value: 'codex-cli', label: tx("Codex CLI"), icon: 'terminal' },
                        { value: 'claude-cli', label: tx("Claude CLI"), icon: 'terminal' },
                      ]}
                    />
                  </Field>

                  {providerSetup.transportType === 'http' && (
                    <div className="grid grid-cols-1 gap-3 sm:grid-cols-[1fr_1.2fr]">
                      <Field label={tx("Cloud provider")}>
                        <select
                          value={providerSetup.cloudKey}
                          onChange={(e) => patchProviderSetup({ cloudKey: e.target.value, providerName: '', apiKey: '', cliPath: '' })}
                          className={SELECT_CLASS}
                        >
                          {HTTP_CLOUD_PROVIDER_PRESETS.map((preset) => <option key={preset.key} value={preset.key}>{preset.label}</option>)}
                        </select>
                      </Field>
                      <div className="flex min-w-0 items-center gap-3 rounded-lg border border-line bg-card-2 px-3 py-2.5">
                        <BrandIcon brand={providerBrand(selectedCloudPreset.provider)} size={30} />
                        <div className="min-w-0 flex-1">
                          <div className="flex items-center gap-2">
                            <span className="truncate text-[13px] font-medium text-ink">{selectedCloudPreset.label}</span>
                            <span className="shrink-0 rounded-full border border-line bg-card px-2 py-0.5 text-[12px] text-ink-3">{selectedCloudPreset.auth === 'cli-local' ? tx("CLI auth") : tx("API key")}</span>
                          </div>
                          <div className="mt-0.5 line-clamp-2 text-[13px] leading-snug text-ink-3">{selectedCloudPreset.desc}</div>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* CLI (Codex/Claude) gets the simplified single-detect / multi-wizard
                      flow; HTTP keeps the name + key fields. */}
                  {providerSetup.transportType !== 'http' ? (
                    <CliProviderSetup
                      key={providerSetup.transportType}
                      transport={providerSetup.transportType}
                      cliSubscriptionMode={providerSetup.cliSubscriptionMode}
                      cliPath={providerSetup.cliPath}
                      cliProfile={providerSetup.cliProfile}
                      onPatch={patchProviderSetup}
                      addCliProvider={addCliProvider}
                    />
                  ) : (
                    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                      <Field label={tx("Display name")} hint={tx("optional")}>
                        <Input value={providerSetup.providerName} onChange={(e) => patchProviderSetup({ providerName: e.target.value })} placeholder={selectedCloudPreset.provider.name} spellCheck={false} />
                      </Field>
                      {selectedCloudPreset.auth === 'cli-local' ? (
                        <Field label={tx("CLI binary")} hint={tx("uses Pi's own auth")}>
                          <Input value={providerSetup.cliPath || selectedCloudPreset.provider.cliPath || 'pi'} onChange={(e) => patchProviderSetup({ cliPath: e.target.value })} placeholder={tx("pi")} spellCheck={false} />
                        </Field>
                      ) : (
                        <Field label={selectedCloudPreset.provider.transport === 'opencode' || selectedCloudPreset.provider.transport === 'kilo' ? tx("Server password") : tx("API key")} hint={selectedCloudPreset.provider.transport === 'opencode' || selectedCloudPreset.provider.transport === 'kilo' ? tx("optional") : undefined}>
                          <Input type="password" value={providerSetup.apiKey} onChange={(e) => patchProviderSetup({ apiKey: e.target.value })} placeholder={selectedCloudPreset.provider.transport === 'opencode' ? tx("OPENCODE_SERVER_PASSWORD (blank if none)") : tx("Paste provider key")} spellCheck={false} />
                        </Field>
                      )}
                      <div className="flex items-end sm:col-span-2">
                        <div className="w-full rounded-lg border border-line bg-card-2 px-3 py-2 text-[13px] leading-snug text-ink-3">
                          {selectedCloudPreset.auth === 'cli-local'
                            ? <>{tx("Authenticate by launching ")}<span className="font-mono text-ink-2">{tx("pi")}</span> {tx("once and configuring its provider, or provide a documented provider environment key to the TensorFold process. A generic key field is intentionally unavailable because Pi would require exposing it in process arguments.")}</>
                            : providerExists(selectedCloudPreset.provider)
                            ? tx("A matching provider already exists. You can still add another one for a separate account or key.")
                            : tx("This will create a provider that can be selected in chats, rooms, missions, and automations.")}
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="space-y-3">
                  <div className="grid grid-cols-1 gap-3 sm:grid-cols-[1fr_1fr]">
                    <Field label={tx("Where is it running?")}>
                      <Segmented<LocalSetupLocation>
                        value={providerSetup.localLocation}
                        onChange={(localLocation) => patchProviderSetup({ localLocation, host: localLocation === 'this-mac' ? '127.0.0.1' : providerSetup.host })}
                        options={[{ value: 'this-mac', label: tx("This Mac"), icon: 'monitor' }, { value: 'network', label: tx("Network"), icon: 'globe' }]}
                      />
                    </Field>
                    <Field label={tx("Runtime")}>
                      <select value={providerSetup.localRuntime} onChange={(e) => selectLocalRuntime(e.target.value as LocalRuntimeKey)} className={SELECT_CLASS}>
                        {LOCAL_RUNTIME_PRESETS.map((runtime) => <option key={runtime.key} value={runtime.key}>{runtime.label}</option>)}
                      </select>
                    </Field>
                  </div>

                  <div className="flex min-w-0 items-center gap-3 rounded-lg border border-line bg-card-2 px-3 py-2.5">
                    <BrandIcon brand={providerBrand({ name: selectedLocalRuntime.label, kind: 'local', source: selectedLocalRuntime.source })} size={30} />
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-[13px] font-medium text-ink">{selectedLocalRuntime.label}</div>
                      <div className="mt-0.5 line-clamp-2 text-[13px] leading-snug text-ink-3">{localRuntimeDescription()}</div>
                    </div>
                    <span className="shrink-0 rounded-full border border-line bg-card px-2 py-0.5 text-[12px] text-ink-3">:{providerSetup.port || selectedLocalRuntime.defaultPort}</span>
                  </div>

                  <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                    <Field label={tx("Node name")} hint={tx("optional")}>
                      <Input value={providerSetup.nodeName} onChange={(e) => patchProviderSetup({ nodeName: e.target.value })} placeholder={providerSetup.localLocation === 'this-mac' ? tx("This Mac") : tx("DGX Spark")} spellCheck={false} />
                    </Field>
                    <Field label={tx("Provider name")} hint={tx("optional")}>
                      <Input value={providerSetup.providerName} onChange={(e) => patchProviderSetup({ providerName: e.target.value })} placeholder={localNodeName()} spellCheck={false} />
                    </Field>
                    <Field label={tx("Host / IP")}>
                      <Input value={providerSetup.localLocation === 'this-mac' ? '127.0.0.1' : providerSetup.host} onChange={(e) => patchProviderSetup({ host: e.target.value })} disabled={providerSetup.localLocation === 'this-mac'} placeholder="192.168.1.45" spellCheck={false} />
                    </Field>
                    <Field label={tx("Port")}>
                      <Input value={providerSetup.port} onChange={(e) => patchProviderSetup({ port: e.target.value.replace(/[^\d]/g, '') })} placeholder={selectedLocalRuntime.defaultPort} inputMode="numeric" />
                    </Field>
                    <Field label={tx("API token")} hint={providerSetup.localRuntime === 'omlx' && providerSetup.localLocation === 'this-mac' ? tx("auto from ~/.omlx") : tx("optional")}>
                      <Input type="password" value={providerSetup.apiKey} onChange={(e) => patchProviderSetup({ apiKey: e.target.value })} placeholder={selectedLocalRuntime.defaultApiKey || tx("Bearer token")} spellCheck={false} />
                    </Field>
                    <Field label={tx("API path")}>
                      <Input value={providerSetup.basePath} onChange={(e) => patchProviderSetup({ basePath: e.target.value })} placeholder="/v1" spellCheck={false} />
                    </Field>
                  </div>

                  <div className="rounded-lg border border-line bg-card-2 px-3 py-2 text-[13px] leading-snug text-ink-3">
                    <span className="font-medium text-ink-2">{tx("Endpoint:")}</span> <span className="font-mono text-ink-3">{localBaseUrl()}</span>
                  </div>
                </div>
              )}

              {/* CLI providers add/connect from inside CliProviderSetup; the generic
                  Test / Add bar is only for HTTP endpoints and local nodes. */}
              {!(providerSetup.mode === 'cloud' && providerSetup.transportType !== 'http') && (
              <div className="mt-3 flex flex-wrap items-center justify-between gap-2 border-t border-line pt-3">
                <div className="min-w-0 flex-1">
                  {setupStatus && (
                    <div className={clsx('inline-flex max-w-full items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-[13px]', setupStatus.state === 'ok' ? 'border-positive/30 bg-positive/10 text-positive' : 'border-danger/30 bg-danger/5 text-danger')}>
                      <Icon name={setupStatus.state === 'ok' ? 'check' : 'warning'} size={13} className="shrink-0" />
                      <span className="break-words">{setupStatus.msg}</span>
                    </div>
                  )}
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  {providerSetup.mode === 'local' && providerSetup.localLocation === 'this-mac' && providerSetup.localRuntime !== 'custom' && (
                    <Button variant="default" size="sm" icon="cpu" onClick={() => applyLocalDiscovery(providerSetup.localRuntime as ProviderDiscoveryKind)} disabled={discoverBusy === 'new'}>
                      {discoverBusy === 'new' ? tx("Discovering") : tx("Discover this Mac")}
                    </Button>
                  )}
                  <Button variant="default" size="sm" icon="signal" onClick={() => testConfiguredProvider()} disabled={setupBusy} title={tx("Sends a fixed test prompt with no files or conversation history; provider usage may apply.")}>
                    {setupBusy ? tx("Testing") : tx("Test response")}
                  </Button>
                  <Button variant="default" size="sm" icon="plus" onClick={() => addConfiguredProvider(false)} disabled={setupBusy}>{tx("Add only")}</Button>
                  <Button variant="primary" size="sm" icon="plug" onClick={() => addConfiguredProvider(true)} disabled={setupBusy}>
                    {setupBusy ? tx("Connecting") : tx("Add and connect")}
                  </Button>
                </div>
              </div>
              )}
            </div>

            <div className="flex items-center justify-between gap-3 border-t border-line pt-4">
              <div>
                <div className="text-[13px] font-semibold text-ink">{tx("Configured providers")}</div>
                <div className="mt-0.5 text-[13px] text-ink-3">{tx("Advanced details for endpoints, accounts, subscriptions, and model overrides.")}</div>
              </div>
              <span className="rounded-full border border-line bg-card px-2.5 py-1 text-[12px] text-ink-3">{draft.providers.length}</span>
            </div>

            {draft.providers.map((p) => (
              <div key={p.id} className="rounded-xl border border-line bg-card p-3.5 shadow-card">
                <div className="mb-3 flex items-center gap-2">
                  <BrandIcon brand={providerBrand(p)} size={28} />
                  <input
                    value={p.name}
                    onChange={(e) => updateProvider(p.id, { name: e.target.value })}
                    className="flex-1 bg-transparent text-[14px] font-medium text-ink outline-none"
                  />
                  {/* Transport is locked to the type chosen when the provider was
                      added (#44) — shown read-only here so it can't be flipped by
                      accident. Cloud/local stays adjustable for HTTP providers. */}
                  <span className="inline-flex shrink-0 items-center gap-1.5 rounded-lg border border-line bg-card-2 px-2.5 py-1 text-[12px] font-medium text-ink-3">
                    <Icon name={p.transport === 'codex-cli' || p.transport === 'claude-cli' || p.transport === 'pi-cli' ? 'terminal' : 'globe'} size={13} />
                    {p.transport === 'codex-cli' ? tx("Codex CLI") : p.transport === 'claude-cli' ? tx("Claude CLI") : p.transport === 'pi-cli' ? tx("Pi CLI") : 'HTTP'}
                  </span>
                  {p.transport !== 'codex-cli' && p.transport !== 'claude-cli' && p.transport !== 'pi-cli' && (
                    <Segmented value={p.kind || 'cloud'} onChange={(v) => commitProviders(draft.providers.map((x) => x.id === p.id ? { ...x, kind: v } : x))} options={[{ value: 'cloud', label: tx("Cloud") }, { value: 'local', label: tx("Local") }]} />
                  )}
                  <button onClick={() => removeProvider(p.id)} className="flex h-8 w-8 items-center justify-center rounded-lg text-ink-3 transition-colors hover:bg-danger/10 hover:text-danger" aria-label={tx("Remove provider")}>
                    <Icon name="trash" size={15} />
                  </button>
                </div>
                {p.transport === 'codex-cli' ? (
                  <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
                    <Field label={tx("CLI binary")} hint={tx("uses codex login")}>
                      <Input value={p.cliPath || 'codex'} onChange={(e) => updateProvider(p.id, { cliPath: e.target.value })} placeholder={tx("codex")} spellCheck={false} />
                    </Field>
                    <Field label={tx("Profile")} hint={tx("optional")}>
                      <Input value={p.cliProfile || ''} onChange={(e) => updateProvider(p.id, { cliProfile: e.target.value })} placeholder={tx("default")} spellCheck={false} />
                    </Field>
                    <Field label={tx("Sandbox")}>
                      <Segmented value={p.cliSandbox || 'workspace-write'} onChange={(v) => commitProviders(draft.providers.map((x) => x.id === p.id ? { ...x, cliSandbox: v } : x))} options={[{ value: 'read-only', label: tx("Read"), icon: 'file' }, { value: 'workspace-write', label: tx("Write"), icon: 'edit' }, { value: 'danger-full-access', label: tx("Full"), icon: 'warning' }]} />
                    </Field>
                    <Field label={tx("CLI approval")} hint={tx("non-interactive")}>
                      <Segmented value={p.cliApprovalPolicy || 'on-request'} onChange={(v) => commitProviders(draft.providers.map((x) => x.id === p.id ? { ...x, cliApprovalPolicy: v } : x))} options={[{ value: 'on-request', label: tx("Ask") }, { value: 'untrusted', label: tx("Guard") }, { value: 'never', label: tx("Never") }]} />
                    </Field>
                  </div>
                ) : p.transport === 'claude-cli' ? (
                  <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
                    <Field label={tx("CLI binary")} hint={tx("uses claude auth")}>
                      <Input value={p.cliPath || 'claude'} onChange={(e) => updateProvider(p.id, { cliPath: e.target.value })} placeholder={tx("claude")} spellCheck={false} />
                    </Field>
                    <Field label={tx("Settings file")} hint={tx("optional")}>
                      <Input value={p.cliProfile || ''} onChange={(e) => updateProvider(p.id, { cliProfile: e.target.value })} placeholder="/path/to/settings.json" spellCheck={false} />
                    </Field>
                    <Field label={tx("Permission mode")}>
                      <Segmented value={p.cliPermissionMode || 'default'} onChange={(v) => commitProviders(draft.providers.map((x) => x.id === p.id ? { ...x, cliPermissionMode: v } : x))} options={[{ value: 'default', label: tx("Default") }, { value: 'auto', label: tx("Auto") }, { value: 'acceptEdits', label: tx("Edits") }, { value: 'plan', label: tx("Plan") }]} />
                    </Field>
                    <div className="flex items-end pb-1 text-[13px] leading-snug text-ink-3">
                      {tx("Claude CLI runs through your local Claude Code login and its own permission/MCP surface.")}</div>
                  </div>
                ) : p.transport === 'pi-cli' ? (
                  <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
                    <Field label={tx("CLI binary")} hint={tx("uses Pi's own auth")}>
                      <Input value={p.cliPath || 'pi'} onChange={(e) => updateProvider(p.id, { cliPath: e.target.value })} placeholder={tx("pi")} spellCheck={false} />
                    </Field>
                    <div className="flex items-end pb-1 text-[13px] leading-snug text-ink-3">
                      {tx("Configure a provider inside Pi, or launch TensorFold with a provider-specific key such as ")}<span className="ml-1 font-mono text-ink-2">OPENAI_API_KEY</span>{tx(". TensorFold never sends a generic key through ")}<span className="mx-1 font-mono text-ink-2">--api-key</span>.
                    </div>
                    {!!p.apiKey?.trim() && (
                      <div className="sm:col-span-2 flex flex-wrap items-center justify-between gap-2 rounded-lg border border-warn/30 bg-warn/10 px-3 py-2.5 text-[13px] text-warn">
                        <span>{tx("This legacy raw key cannot be handed to Pi safely. Clear it, then use Pi auth or a provider environment key.")}</span>
                        <Button variant="default" size="xs" onClick={() => updateProvider(p.id, { apiKey: '' })}>{tx("Clear incompatible key")}</Button>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
                    <Field label={tx("Base URL")}>
                      <Input value={p.baseUrl} onChange={(e) => updateProvider(p.id, { baseUrl: e.target.value })} placeholder="https://api.openai.com/v1" spellCheck={false} />
                    </Field>
                    <Field label={tx("API key")} hint={p.kind === 'local' ? tx("often any value") : undefined}>
                      <Input type="password" value={p.apiKey} onChange={(e) => updateProvider(p.id, { apiKey: e.target.value })} placeholder={"sk-…"} spellCheck={false} />
                    </Field>
                    {localDiscoveryKind(p) && (
                      <div className="sm:col-span-2 flex flex-wrap items-center justify-between gap-2 rounded-lg border border-line bg-card-2 px-3 py-2.5">
                        <div className="min-w-0 text-[13px] leading-snug text-ink-3">
                          {!isLoopbackEndpoint(p.baseUrl)
                            ? <>{tx("Probe the node at ")}<span className="font-mono text-ink-2">{p.baseUrl || tx("this address")}</span> {tx("for its model list — works over the network (e.g. a Mac TensorFold on your LAN).")}</>
                            : localDiscoveryKind(p) === 'omlx'
                              ? <>{tx("Read ")}<span className="font-mono text-ink-2">~/.omlx/settings.json</span> {tx("for the local port, API key, and model list.")}</>
                              : <>{tx("Probe the local server and refresh the model list from its native API.")}</>}
                        </div>
                        <Button variant="default" size="sm" icon="cpu" onClick={() => { const kind = localDiscoveryKind(p); if (kind) applyLocalDiscovery(kind, p.id) }} disabled={discoverBusy === p.id}>
                          {discoverBusy === p.id ? tx("Discovering") : tx("Discover {value0}", { value0: discoveryLabel(localDiscoveryKind(p)!) })}
                        </Button>
                      </div>
                    )}
                    {isOpenRouterProvider(p) && (
                      <div className="sm:col-span-2 rounded-lg border border-line bg-card-2 px-3 py-2.5 text-[13px] leading-snug text-ink-3">
                        {tx("Connect refreshes OpenRouter's current tool-capable catalogue, including free variants and routers such as ")}<span className="font-mono text-ink-2">openrouter/free</span>.
                      </div>
                    )}
                    {isAiGatewayProvider(p) && (
                      <div className="sm:col-span-2 rounded-lg border border-line bg-card-2 px-3 py-2.5 text-[13px] leading-snug text-ink-3">
                        {tx("Use gateway-qualified model ids such as ")}<span className="font-mono text-ink-2">openai/gpt-5.4</span>{tx(". If the gateway does not expose a models endpoint, TensorFold keeps the curated gateway list.")}</div>
                    )}
                  </div>
                )}
                <div className="mt-3">
                  <SpendGuardControls
                    budget={p.spendBudget}
                    title={tx("Provider spend guard")}
                    scope={p.name}
                    desc={tx("Warn or block routes when this provider reaches its budget.")}
                    onChange={(patch) => commitProviderSpendBudget(p.id, patch)}
                  />
                </div>
                {/* Local runtimes (oMLX, Ollama, LM Studio) have no subscriptions
                    or per-account logins, so the accounts panel only applies to
                    cloud/CLI providers (#46). The whole panel is advanced —
                    per-account failover policy, config home, OAuth tokens, model
                    overrides — so it collapses by default. */}
                {p.kind !== 'local' && (
                <div className="mt-3">
                <AdvancedSection title={tx("Accounts & subscriptions")} icon="layers">
                  <div className="mb-2 flex items-center justify-between gap-2">
                    <div>
                      <div className="text-[13px] font-medium text-ink-2">{tx("Accounts / subscriptions")}</div>
                      <div className="text-[12px] text-ink-3">{tx("Pick one per chat, room seat, automation, or ticket run.")}</div>
                    </div>
                    <div className="flex shrink-0 items-center gap-2">
                      {(p.transport === 'codex-cli' || p.transport === 'claude-cli') && (
                        <Button variant="subtle" size="sm" icon="layers" onClick={() => addCliSubscriptionPair(p)}>{tx("Personal + work")}</Button>
                      )}
                      <Button variant="default" size="sm" icon="plus" onClick={() => addProviderAccount(p)}>{tx("Add account")}</Button>
                    </div>
                  </div>
                  {(p.transport === 'codex-cli' || p.transport === 'claude-cli') && (
                    <div className="mb-2 rounded-lg border border-line bg-card px-2.5 py-2 text-[13px] leading-snug text-ink-3">
                      {tx("Create one account per subscription. Run the login command once for each account, then select that account in chats, room agents, projects, missions, and automations.")}</div>
                  )}
                  {!(p.accounts ?? []).length && (
                    <div className="rounded-lg border border-dashed border-line px-2.5 py-2 text-[13px] text-ink-3">
                      {tx("No named accounts yet. TensorFold will use the provider's default login/key.")}</div>
                  )}
                  <div className="space-y-2">
                    {(p.accounts ?? []).map((account) => (
                      <div key={account.id} className="rounded-lg border border-line bg-card p-2.5">
                        <div className="mb-2 flex items-center gap-2">
                          <Toggle label={tx("Enable {value0} on {value1}", { value0: account.name || tx("account"), value1: p.name })} checked={account.enabled !== false} onChange={(enabled) => commitProviderAccount(p.id, account.id, { enabled })} />
                          <Input value={account.name} onChange={(e) => updateProviderAccount(p.id, account.id, { name: e.target.value })} placeholder={tx("Account name")} />
                          <button
                            onClick={() => commitProviders(draft.providers.map((provider) => provider.id === p.id ? { ...provider, defaultAccountId: provider.defaultAccountId === account.id ? null : account.id } : provider))}
                            className={clsx('shrink-0 rounded-md border px-2 py-1 text-[12px] font-medium transition-colors', p.defaultAccountId === account.id ? 'border-accent bg-accent-soft text-accent' : 'border-line text-ink-3 hover:border-line-strong hover:text-ink')}
                          >
                            {p.defaultAccountId === account.id ? tx("Default") : tx("Make default")}
                          </button>
                          <button onClick={() => removeProviderAccount(p.id, account.id)} className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-ink-3 transition-colors hover:bg-danger/10 hover:text-danger" aria-label={tx("Remove account")}>
                            <Icon name="trash" size={14} />
                          </button>
                        </div>
                        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                          <Field label={tx("Kind")}>
                            <Segmented value={account.kind || 'subscription'} onChange={(v) => commitProviderAccount(p.id, account.id, { kind: v as ProviderAccountKind })} options={[{ value: 'subscription', label: tx("Subscription") }, { value: 'api-key', label: tx("API key") }, { value: 'profile', label: tx("Profile") }]} />
                          </Field>
                          <Field label={tx("Failover")}>
                            <div className="flex min-h-9 items-center justify-between gap-3 rounded-lg border border-line bg-card-2 px-2.5 py-2">
                              <div className="min-w-0">
                                <div className="text-[13px] font-medium text-ink-2">{tx("Automatic failover")}</div>
                                <div className="truncate text-[12px] text-ink-3">{tx("Use another equivalent account when this one hits a limit.")}</div>
                              </div>
                              <Toggle label={tx("Automatic failover for {value0} on {value1}", { value0: account.name || tx("account"), value1: p.name })} checked={account.autoFailover === true} onChange={(autoFailover) => commitProviderAccount(p.id, account.id, { autoFailover })} />
                            </div>
                          </Field>
                          <Field label={tx("After reset")} hint={account.autoFailover ? tx("failover policy") : tx("enable failover first")}>
                            <select
                              value={account.failoverResetPolicy ?? 'return_after_reset'}
                              onChange={(e) => commitProviderAccount(p.id, account.id, { failoverResetPolicy: e.target.value as ModelFailoverResetPolicy })}
                              disabled={account.autoFailover !== true}
                              className={SELECT_CLASS}
                            >
                              <option value="return_after_reset">{tx("Return to this account")}</option>
                              <option value="stay_on_failover">{tx("Stay on failover account")}</option>
                              <option value="rotate_on_limit">{tx("Rotate only on next limit")}</option>
                            </select>
                          </Field>
                          {(p.transport === 'codex-cli' || p.transport === 'claude-cli') ? (
                            <Field label={accountProfileLabel(p)} hint={accountProfileHint(p)}>
                              <Input value={account.cliProfile || ''} onChange={(e) => updateProviderAccount(p.id, account.id, { cliProfile: e.target.value })} placeholder={p.transport === 'codex-cli' ? tx("work") : '/path/to/settings.json'} spellCheck={false} />
                            </Field>
                          ) : (
                            <Field label={tx("API key")}>
                              <Input type="password" value={account.apiKey || ''} onChange={(e) => updateProviderAccount(p.id, account.id, { apiKey: e.target.value })} placeholder={"sk-..."} spellCheck={false} />
                            </Field>
                          )}
                          {(p.transport === 'codex-cli' || p.transport === 'claude-cli') && (
                            <Field label={tx("Config home")} hint={accountHomeHint(p)}>
                              <Input value={account.cliHome || ''} onChange={(e) => updateProviderAccount(p.id, account.id, { cliHome: e.target.value })} placeholder={p.transport === 'codex-cli' ? '~/.codex-work' : '~/.claude-work'} spellCheck={false} />
                            </Field>
                          )}
                          {p.transport === 'claude-cli' && (
                            <Field label={tx("OAuth token")} hint={tx("from claude setup-token")}>
                              <Input type="password" value={account.env?.CLAUDE_CODE_OAUTH_TOKEN || ''} onChange={(e) => patchProviderAccountEnv(p.id, account, 'CLAUDE_CODE_OAUTH_TOKEN', e.target.value)} placeholder="CLAUDE_CODE_OAUTH_TOKEN" spellCheck={false} />
                            </Field>
                          )}
                          <Field label={tx("Models override")} hint={tx("optional")}>
                            <Input value={(account.models ?? []).join(', ')} onChange={(e) => updateProviderAccount(p.id, account.id, { models: e.target.value.split(',').map((s) => s.trim()).filter(Boolean) })} placeholder={p.models.join(', ') || tx("inherit provider models")} spellCheck={false} />
                          </Field>
                        </div>
                        <div className="mt-2">
                          <SpendGuardControls
                            budget={account.spendBudget}
                            title={tx("Account spend guard")}
                            scope={`${account.name || "account"} on ${p.name}`}
                            desc={tx("Warn, stop, or fail over this subscription when it reaches a limit.")}
                            onChange={(patch) => commitProviderAccountSpendBudget(p.id, account.id, patch)}
                          />
                        </div>
                        {loginCommand(p, account) && (
                          <div className="mt-2 rounded-lg border border-line bg-card-2 px-2.5 py-2">
                            <div className="mb-1.5 flex items-center justify-between gap-2">
                              <span className="text-[12px] font-medium text-ink-3">{tx("Login command")}</span>
                              <Button variant="ghost" size="xs" icon="copy" onClick={() => copyLoginCommand(p, account)}>
                                {copiedLoginAccount === account.id ? tx("Copied") : tx("Copy")}
                              </Button>
                            </div>
                            <code className="block select-text overflow-x-auto whitespace-nowrap rounded-md bg-ink-5 px-2 py-1.5 font-mono text-[12px] text-ink-2">
                              {loginCommand(p, account)}
                            </code>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </AdvancedSection>
                </div>
                )}

                <div className="mt-2.5">
                  <div className="mb-1.5 flex items-center justify-between gap-2">
                    <span className="text-[13px] font-medium text-ink-2">{tx("Models")}</span>
                    <div className="flex items-center gap-2.5 text-[12px]">
                      {conn[p.id]?.state === 'ok' && <span className="flex items-center gap-1 text-positive"><Icon name="check" size={12} />{conn[p.id]?.msg}</span>}
                      {conn[p.id]?.state === 'error' && <span className="flex max-w-[230px] items-center gap-1 truncate text-danger" title={conn[p.id]?.msg}><Icon name="warning" size={12} className="shrink-0" /><span className="truncate">{conn[p.id]?.msg}</span></span>}
                      {(!conn[p.id] || conn[p.id]?.state === 'loading') && <span className="text-ink-3">{p.models.length} {tx("selected")}</span>}
                      <button
                        onClick={() => connect(p)}
                        disabled={conn[p.id]?.state === 'loading'}
                        className="inline-flex items-center gap-1.5 rounded-md border border-line px-2 py-1 font-medium text-ink-2 transition-colors hover:border-line-strong hover:text-ink disabled:opacity-60"
                      >
                        {conn[p.id]?.state === 'loading' ? <Spinner size={12} /> : <BrandIcon brand={providerBrand(p)} size={16} className="rounded-[5px]" />}
                        {conn[p.id]?.state === 'loading' ? tx("Checking...") : conn[p.id]?.state === 'ok' ? tx("Refresh") : tx("Connect")}
                      </button>
                    </div>
                  </div>
                  {/* Models are an interactive checklist (#46): the comma-separated
                      text field is replaced by a modal over the discovered catalogue. */}
                  <div className="flex flex-wrap items-center gap-2 rounded-lg border border-line bg-card-2 px-3 py-2.5">
                    <span className="min-w-0 flex-1 truncate text-[13px] text-ink-3" title={p.models.join(', ')}>
                      {p.models.length ? p.models.join(', ') : tx("No models selected yet")}
                    </span>
                    <Button variant="default" size="sm" icon="edit" onClick={() => setEditingModelsFor(p.id)}>{tx("Edit models")}</Button>
                  </div>
                </div>

                {/* Explicit save (#45). Provider fields now edit the draft only —
                    nothing persists on blur, so rapid edits + discovery races can't
                    spawn duplicate providers. One click commits the whole card. */}
                <div className="mt-3 flex items-center justify-end gap-2 border-t border-line pt-3">
                  <Button variant="primary" size="sm" icon="check" onClick={() => commitProviders(draft.providers)}>{tx("Save changes")}</Button>
                </div>
              </div>
            ))}

            {editingProvider && (
              <ModelChecklistModal
                provider={editingProvider}
                onClose={() => setEditingModelsFor(null)}
                onSave={(models) => { commitProviders(draft.providers.map((x) => (x.id === editingProvider.id ? { ...x, models } : x))); setEditingModelsFor(null) }}
              />
            )}

            <Button variant="ghost" icon="plus" onClick={addProvider} className="w-full justify-center text-ink-3">{tx("Add blank advanced provider")}</Button>
          </div>
        </Section>

              </SettingsPane>

  )
}

// Models checklist modal (#46): pick which discovered models the provider
// exposes, instead of editing a fragile comma-separated string. Falls back to
// the provider's current model list when no discovery has run yet, and lets you
// add an id by hand for endpoints whose catalogue TensorFold can't enumerate.
function ModelChecklistModal({ provider, onClose, onSave }: { provider: Provider; onClose: () => void; onSave: (models: string[]) => void }) {
  const tx = useAuthoredText()
  const discovered = provider.discoveredModels?.length ? provider.discoveredModels : provider.models
  // Union of the discovered catalogue and any already-selected models, so a
  // hand-added model that isn't in the catalogue still appears (and stays
  // checked) rather than silently vanishing.
  const initialCatalogue = Array.from(new Set([...(discovered ?? []), ...provider.models]))
  const [catalogue, setCatalogue] = useState<string[]>(initialCatalogue)
  const [selected, setSelected] = useState<Set<string>>(new Set(provider.models))
  const [customModel, setCustomModel] = useState('')

  const toggle = (model: string) => {
    setSelected((current) => {
      const next = new Set(current)
      if (next.has(model)) next.delete(model)
      else next.add(model)
      return next
    })
  }
  const addCustom = () => {
    const model = customModel.trim()
    if (!model) return
    setCatalogue((current) => (current.includes(model) ? current : [...current, model]))
    setSelected((current) => new Set(current).add(model))
    setCustomModel('')
  }
  // Preserve provider model ordering for already-selected ids; append the rest.
  const save = () => {
    const order = [...provider.models, ...catalogue]
    const ordered = order.filter((model, index) => selected.has(model) && order.indexOf(model) === index)
    onSave(ordered)
  }

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/40 p-5 backdrop-blur-sm" onClick={onClose}>
      <div className="flex max-h-[min(560px,calc(100vh-80px))] w-[min(480px,calc(100vw-40px))] flex-col overflow-hidden rounded-2xl border border-line bg-bg shadow-[0_24px_90px_rgb(0_0_0/0.45)]" onClick={(e) => e.stopPropagation()}>
        <div className="flex shrink-0 items-center justify-between gap-3 border-b border-line px-4 py-3">
          <div className="min-w-0">
            <div className="truncate text-[14px] font-semibold text-ink">{tx("Edit models")}</div>
            <div className="mt-0.5 truncate text-[13px] text-ink-3">{provider.name} · {selected.size} {tx("selected")}</div>
          </div>
          <button onClick={onClose} aria-label={tx("Close")} className="flex h-8 w-8 items-center justify-center rounded-lg text-ink-3 transition-colors hover:bg-ink-5 hover:text-ink">
            <Icon name="x" size={15} />
          </button>
        </div>
        <div className="scroll-thin min-h-0 flex-1 overflow-y-auto px-4 py-3">
          {catalogue.length === 0 ? (
            <div className="rounded-lg border border-dashed border-line px-3 py-6 text-center text-[13px] text-ink-3">
              {tx("No models discovered yet. Use Connect to refresh, or add one below.")}</div>
          ) : (
            <div className="space-y-1.5">
              {catalogue.map((model) => (
                <label key={model} className="flex cursor-pointer items-center gap-2.5 rounded-lg border border-line bg-card px-3 py-2 text-[13px] text-ink-2 transition-colors hover:border-line-strong">
                  <input type="checkbox" checked={selected.has(model)} onChange={() => toggle(model)} className="h-4 w-4 accent-[var(--accent)]" />
                  <span className="min-w-0 flex-1 truncate font-mono text-[13px]">{model}</span>
                </label>
              ))}
            </div>
          )}
        </div>
        <div className="shrink-0 border-t border-line px-4 py-3">
          <div className="mb-2.5 flex items-center gap-2">
            <Input value={customModel} onChange={(e) => setCustomModel(e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); addCustom() } }} placeholder={tx("Add a model id")} spellCheck={false} />
            <Button variant="default" size="sm" icon="plus" onClick={addCustom}>{tx("Add")}</Button>
          </div>
          <div className="flex items-center justify-end gap-2">
            <Button variant="ghost" size="sm" onClick={onClose}>{tx("Cancel")}</Button>
            <Button variant="primary" size="sm" icon="check" onClick={save}>{tx("Save")}</Button>
          </div>
        </div>
      </div>
    </div>
  )
}

// One per-surface default model picker (VST-015): provider -> account -> model, with "same as
// active model" as the unset state. Shared by the Assistant (chat) and Coding-tasks (code)
// surfaces so both defaults get the exact same selection behaviour.
function SurfaceModelSection({ draft, patch, title, desc, providerKey, accountKey, modelKey }: {
  draft: Settings
  patch: (p: Partial<Settings>) => void
  title: string
  desc: string
  providerKey: 'assistantProviderId' | 'codeProviderId'
  accountKey: 'assistantProviderAccountId' | 'codeProviderAccountId'
  modelKey: 'assistantModel' | 'codeModel'
}) {
  const tx = useAuthoredText()
  const providerId = draft[providerKey] ?? null
  const provider = providerId ? draft.providers.find((p) => p.id === providerId) ?? null : null
  const accounts = provider ? providerAccountChoices(provider) : []
  const account = accounts.find((choice) => choice.id === (draft[accountKey] ?? null))
    ?? accounts.find((choice) => choice.id === (provider?.defaultAccountId ?? null))
    ?? accounts[0]
  const models = account?.models?.length ? account.models : (provider?.models ?? [])
  return (
    <Section title={title} desc={desc}>
      <div className="flex flex-wrap items-center gap-2">
        <select
          className={SELECT_CLASS}
          value={providerId ?? ''}
          onChange={(e) => {
            const pid = e.target.value
            if (!pid) { patch({ [providerKey]: null, [accountKey]: null, [modelKey]: null }); return }
            const prov = draft.providers.find((p) => p.id === pid)
            const nextAccounts = prov ? providerAccountChoices(prov) : []
            const nextAccount = nextAccounts.find((choice) => choice.id === (prov?.defaultAccountId ?? null)) ?? nextAccounts[0]
            const nextModels = nextAccount?.models?.length ? nextAccount.models : (prov?.models ?? [])
            patch({ [providerKey]: pid, [accountKey]: nextAccount?.id ?? null, [modelKey]: nextModels[0] ?? null })
          }}
        >
          <option value="">{tx("Same as active model")}</option>
          {draft.providers.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
        </select>
        {provider && (
          <>
            <select
              className={SELECT_CLASS}
              value={account?.id ?? ''}
              onChange={(e) => {
                const nextAccount = accounts.find((choice) => (choice.id ?? '') === e.target.value) ?? accounts[0]
                const nextModels = nextAccount?.models?.length ? nextAccount.models : provider.models
                patch({ [accountKey]: nextAccount?.id ?? null, [modelKey]: nextModels[0] ?? null })
              }}
            >
              {accounts.map((choice) => <option key={choice.id ?? 'default'} value={choice.id ?? ''}>{choice.name}</option>)}
            </select>
            <select className={SELECT_CLASS} value={draft[modelKey] ?? ''} onChange={(e) => patch({ [modelKey]: e.target.value || null })}>
              {models.length === 0 && <option value="">{tx("(no models — connect this provider first)")}</option>}
              {models.map((m) => <option key={m} value={m}>{modelDisplayName(provider, m)}</option>)}
            </select>
          </>
        )}
      </div>
    </Section>
  )
}
