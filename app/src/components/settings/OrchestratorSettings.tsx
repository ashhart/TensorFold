import { useEffect, useState } from 'react'
import clsx from 'clsx'
import { Icon } from '../../lib/icons'
import { api } from '../../lib/ipc'
import { Button, Field, Input, Toggle } from '../ui'
import { AdvancedSection, Section, SettingsPane, type SettingsSectionContext } from './shared'
import { QrCode } from './QrCode'
import { MobileConnectionPanel } from './MobileConnectionPanel'

export function OrchestratorSettings({ ctx }: { ctx: SettingsSectionContext }) {
  const [copiedPairing, setCopiedPairing] = useState<'url' | 'token' | null>(null)
  const [checkingHostId, setCheckingHostId] = useState<string | null>(null)
  const {
    draft,
    setDraft,
    patch,
    commit,
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
    browseCwd,
    updateStatus,
    checkForUpdates,
    openUpdateDownload,
    emailBusy,
    slackBusy,
    documentBusy,
    browserBusy,
    browserTabUrl,
    setBrowserTabUrl,
    browserTabTitle,
    setBrowserTabTitle,
    browserTabExcerpt,
    setBrowserTabExcerpt,
    browserHistoryPath,
    setBrowserHistoryPath,
    botBusy,
    botStatus,
    controllerBusy,
    controllerStatus,
    remoteBusy,
    remotePairing,
    remoteTest,
    remoteNotice,
    securityControls,
    securityBusy,
    securityNotice,
    voiceNames,
    commitRemoteControl,
    commitRemoteClient,
    commitRemoteOrchestrators,
    commitController,
    commitVoice,
    commitTelegram,
    commitWhatsapp,
    addEmailSource,
    updateEmailSource,
    removeEmailSource,
    syncEmail,
    addSlackSource,
    updateSlackSource,
    removeSlackSource,
    syncSlack,
    addDocumentSource,
    updateDocumentSource,
    removeDocumentSource,
    syncDocuments,
    addBrowserCurrentTabSource,
    addBrowserHistorySource,
    updateBrowserSource,
    removeBrowserSource,
    syncBrowser,
    pollTelegram,
    refreshControllerStatus,
    refreshSecurityControls,
    purgeLocalMemoryNow,
    revealRemotePairing,
    regenerateRemoteToken,
    refreshRemoteStatus,
    testRemoteClient,
    enableController,
    conn,
    connect,
    studioStatus,
    localMemoryTotal,
  } = ctx
  const copyPairingValue = async (kind: 'url' | 'token', value: string | null | undefined) => {
    const text = String(value || '').trim()
    if (!text) return
    try {
      const result = await api.writeClipboard(text)
      if (!result.ok) return
      setCopiedPairing(kind)
      window.setTimeout(() => setCopiedPairing((current) => current === kind ? null : current), 1400)
    } catch {
      // Clipboard access can be denied by the shell; the value remains visible for manual copy.
    }
  }
  const savedHosts = draft.remoteOrchestrators
  const defaultHost = savedHosts.find((host) => host.enabled) ?? null
  const makeDefaultHost = (hostId: string) => {
    const host = savedHosts.find((item) => item.id === hostId)
    if (!host) return
    commitRemoteOrchestrators([{ ...host, enabled: true, updatedAt: Date.now() }, ...savedHosts.filter((item) => item.id !== hostId)])
  }
  const testSavedHost = async (hostId: string) => {
    const host = savedHosts.find((item) => item.id === hostId)
    if (!host) return
    setCheckingHostId(hostId)
    try {
      const result = await api.getRemoteSnapshot(hostId)
      const now = Date.now()
      commitRemoteOrchestrators(savedHosts.map((item) => item.id === hostId
        ? {
            ...item,
            name: result.ok ? (result.name || item.name) : item.name,
            lastConnectedAt: result.ok ? now : item.lastConnectedAt,
            lastError: result.ok ? null : (result.error ?? 'Connection failed.'),
            updatedAt: now,
          }
        : item))
    } finally {
      setCheckingHostId(null)
    }
  }

  return (
<SettingsPane>
        <Section title="Orchestrator" desc="Let this TensorFold coordinate long-running work while another TensorFold app watches and guides it.">
          <div className="space-y-3">
            <div className="rounded-xl border border-line bg-card p-3.5 shadow-card">
              <div className="mb-3 flex items-start justify-between gap-3">
                <div className="flex min-w-0 items-start gap-2.5">
                  <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-ink-5 text-ink-2"><Icon name="monitor" size={17} /></span>
                  <div className="min-w-0">
                    <div className="text-[13px] font-medium text-ink">Host this TensorFold as an Orchestrator</div>
                    <div className="mt-0.5 text-[13px] leading-snug text-ink-3">
                      Keep rooms, missions, and overnight jobs running on this machine while clients watch and guide them.
                    </div>
                  </div>
                </div>
                <Toggle
                  label="Orchestrator server" checked={draft.remoteControl.enabled}
                  onChange={(enabled) => commitRemoteControl({ ...draft.remoteControl, enabled, lastError: null })}
                />
              </div>

              <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
                <Field label="Orchestrator name">
                  <Input
                    value={draft.remoteControl.serverName}
                    onChange={(e) => setDraft((d) => ({ ...d, remoteControl: { ...d.remoteControl, serverName: e.target.value } }))}
                    onBlur={() => commitRemoteControl(draft.remoteControl)}
                    placeholder="TensorFold Orchestrator"
                    spellCheck={false}
                  />
                </Field>
                <Field label="Orchestrator URL">
                  <div className="flex min-h-9 items-center rounded-lg border border-line bg-card-2 px-3 font-mono text-[13px] text-ink-2">
                    {studioStatus.url ?? 'Disabled'}
                  </div>
                </Field>
              </div>

              <div className="mt-3">
                <AdvancedSection title="Network" icon="globe">
                  <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
                    <Field label="Port" hint={studioStatus.listening ? 'listening' : 'idle'}>
                      <Input
                        type="number"
                        min={1}
                        max={65535}
                        value={draft.remoteControl.port}
                        onChange={(e) => setDraft((d) => ({ ...d, remoteControl: { ...d.remoteControl, port: parseInt(e.target.value, 10) || 8765 } }))}
                        onBlur={() => commitRemoteControl(draft.remoteControl)}
                        spellCheck={false}
                      />
                    </Field>
                    <Field label="Bind interface" hint="Local only">
                      <div className="flex min-h-9 items-center rounded-lg border border-line bg-card-2 px-3 text-[13px] text-ink-2">
                        127.0.0.1 · use an SSH tunnel or TLS front-end for another machine
                      </div>
                    </Field>
                  </div>
                </AdvancedSection>
              </div>

              <div className="mt-3 flex flex-wrap items-center gap-2 text-[13px] text-ink-3">
                <span className={clsx('rounded-full border px-2 py-1', studioStatus.listening ? 'border-positive/30 bg-positive/10 text-positive' : 'border-line bg-card-2')}>
                  {studioStatus.listening ? 'Listening' : draft.remoteControl.enabled ? 'Starting' : 'Off'}
                </span>
                <span className="rounded-full border border-line bg-card-2 px-2 py-1">{studioStatus.clients} client{studioStatus.clients === 1 ? '' : 's'}</span>
                {studioStatus.lastStartedAt && <span>Started {new Date(studioStatus.lastStartedAt).toLocaleString()}</span>}
                {studioStatus.lastClientAt && <span>Client seen {new Date(studioStatus.lastClientAt).toLocaleString()}</span>}
              </div>

              <div className="mt-3 flex flex-wrap gap-2">
                <Button variant="default" size="sm" icon="copy" onClick={revealRemotePairing} disabled={remoteBusy === 'pair'}>
                  {remoteBusy === 'pair' ? 'Preparing' : 'Show pairing'}
                </Button>
                <Button variant="ghost" size="sm" icon="spark" onClick={regenerateRemoteToken} disabled={remoteBusy === 'regen'}>
                  {remoteBusy === 'regen' ? 'Regenerating' : 'Regenerate token'}
                </Button>
                <Button variant="ghost" size="sm" icon="search" onClick={() => refreshRemoteStatus()} disabled={remoteBusy === 'status'}>
                  Refresh
                </Button>
              </div>

              {remotePairing && (
                <div className="mt-3 rounded-lg border border-line bg-card-2 p-3">
                  <div className="text-[12px] font-medium uppercase tracking-[0.08em] text-ink-3">Pairing</div>
                  <div className="mt-2 grid grid-cols-1 gap-2 sm:grid-cols-2">
                    <div className="min-w-0 rounded-md border border-line bg-card px-2.5 py-2">
                      <div className="mb-1 flex items-center justify-between gap-2">
                        <div className="text-[12px] text-ink-3">Server URL</div>
                        <Button
                          variant="ghost"
                          size="xs"
                          icon={copiedPairing === 'url' ? 'check' : 'copy'}
                          onClick={() => copyPairingValue('url', remotePairing.url)}
                          disabled={!remotePairing.url}
                        >
                          {copiedPairing === 'url' ? 'Copied' : 'Copy'}
                        </Button>
                      </div>
                      <div className="mt-0.5 truncate font-mono text-[13px] text-ink-2" title={remotePairing.url ?? ''}>{remotePairing.url ?? 'Enable Orchestrator first'}</div>
                    </div>
                    <div className="min-w-0 rounded-md border border-line bg-card px-2.5 py-2">
                      <div className="mb-1 flex items-center justify-between gap-2">
                        <div className="text-[12px] text-ink-3">Token</div>
                        <Button
                          variant="ghost"
                          size="xs"
                          icon={copiedPairing === 'token' ? 'check' : 'copy'}
                          onClick={() => copyPairingValue('token', remotePairing.token)}
                        >
                          {copiedPairing === 'token' ? 'Copied' : 'Copy'}
                        </Button>
                      </div>
                      <div className="mt-0.5 truncate font-mono text-[13px] text-ink-2" title={remotePairing.token}>{remotePairing.token}</div>
                    </div>
                  </div>
                </div>
              )}

              {(studioStatus.lastError || remoteNotice) && (
                <div className={clsx('mt-2.5 rounded-lg border px-3 py-2 text-[13px]', studioStatus.lastError ? 'border-danger/30 bg-danger/5 text-danger' : 'border-line bg-card-2 text-ink-3')}>
                  {studioStatus.lastError || remoteNotice}
                </div>
              )}

              <PairedDevicesPanel enabled={draft.remoteControl.enabled} serverUrl={studioStatus.url} />
            </div>

            <MobileConnectionPanel onEnabled={remoteControl => { setDraft(d => ({ ...d, remoteControl })); void refreshRemoteStatus() }} />

            <div className="rounded-xl border border-line bg-card p-3.5 shadow-card">
              <div className="mb-3 flex items-start justify-between gap-3">
                <div className="flex min-w-0 items-start gap-2.5">
                  <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-ink-5 text-ink-2"><Icon name="sidebar" size={17} /></span>
                  <div className="min-w-0">
                    <div className="text-[13px] font-medium text-ink">Connect to an Orchestrator</div>
                    <div className="mt-0.5 text-[13px] leading-snug text-ink-3">
                      Point this app at an Orchestrator URL and token, then inspect and guide work from this machine.
                    </div>
                  </div>
                </div>
                <Toggle label="Orchestrator client" checked={draft.remoteClient.enabled} onChange={(enabled) => commitRemoteClient({ ...draft.remoteClient, enabled, lastError: null })} />
              </div>

              <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
                <Field label="Orchestrator URL">
                  <Input
                    value={draft.remoteClient.serverUrl}
                    onChange={(e) => setDraft((d) => ({ ...d, remoteClient: { ...d.remoteClient, serverUrl: e.target.value } }))}
                    onBlur={() => commitRemoteClient(draft.remoteClient)}
                    placeholder="http://192.168.1.50:8765"
                    spellCheck={false}
                  />
                </Field>
                <Field label="Token">
                  <Input
                    type="password"
                    value={draft.remoteClient.authToken}
                    onChange={(e) => setDraft((d) => ({ ...d, remoteClient: { ...d.remoteClient, authToken: e.target.value } }))}
                    onBlur={() => commitRemoteClient(draft.remoteClient)}
                    placeholder="paste pairing token"
                    spellCheck={false}
                  />
                </Field>
              </div>

              <div className="mt-3 flex flex-wrap items-center gap-2">
                <Button variant="primary" size="sm" icon="plug" onClick={testRemoteClient} disabled={remoteBusy === 'test' || !draft.remoteClient.enabled}>
                  {remoteBusy === 'test' ? 'Testing' : 'Test connection'}
                </Button>
                {draft.remoteClient.lastConnectedAt && <span className="text-[13px] text-ink-3">Last connected {new Date(draft.remoteClient.lastConnectedAt).toLocaleString()}</span>}
              </div>

              {remoteTest && (
                <div className={clsx('mt-2.5 rounded-lg border px-3 py-2 text-[13px]', remoteTest.ok ? 'border-positive/30 bg-positive/10 text-positive' : 'border-danger/30 bg-danger/5 text-danger')}>
                  {remoteTest.ok
                    ? `Connected to ${remoteTest.snapshot?.serverName ?? 'Orchestrator'} with ${remoteTest.snapshot?.activeThreadIds.length ?? 0} active task${remoteTest.snapshot?.activeThreadIds.length === 1 ? '' : 's'}.`
                    : remoteTest.error}
                </div>
              )}
              {draft.remoteClient.lastError && !remoteTest && (
                <div className="mt-2.5 rounded-lg border border-danger/30 bg-danger/5 px-3 py-2 text-[13px] text-danger">
                  {draft.remoteClient.lastError}
                </div>
              )}

              <div className="mt-4 border-t border-line pt-3">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <div className="text-[13px] font-medium text-ink">Run everything on an Orchestrator</div>
                    <div className="mt-0.5 max-w-[420px] text-[13px] leading-relaxed text-ink-3">
                      New tasks, chats, and terminals from this machine start on the selected host — close this laptop and the work keeps running there. Projects pinned to their own Orchestrator keep their host; local-only projects stay local. The composer chip always shows where a run will land.
                    </div>
                  </div>
                  <Toggle
                    label="Run everything on an Orchestrator" checked={draft.defaultExecutionTarget === 'orchestrator'}
                    onChange={(on) => commit({ ...draft, defaultExecutionTarget: on ? 'orchestrator' : 'local' })}
                  />
                </div>
                {draft.defaultExecutionTarget === 'orchestrator' && (
                  <div className="mt-2.5">
                    {savedHosts.filter((host) => host.enabled).length === 0 ? (
                      <div className="rounded-lg border border-warn/40 bg-warn/10 px-3 py-2 text-[13px] text-warn">
                        No enabled Orchestrator host — new work will NOT start until one is connected below.
                      </div>
                    ) : (
                      <Field label="Default host">
                        <select
                          value={draft.defaultOrchestratorId ?? ''}
                          onChange={(e) => commit({ ...draft, defaultOrchestratorId: e.target.value || null })}
                          className="h-8 w-full max-w-[280px] rounded-md border border-line bg-card px-2 text-[13px] text-ink outline-none"
                        >
                          <option value="">First enabled host</option>
                          {savedHosts.filter((host) => host.enabled).map((host) => (
                            <option key={host.id} value={host.id}>{host.name}</option>
                          ))}
                        </select>
                      </Field>
                    )}
                  </div>
                )}
              </div>

              <div className="mt-4 border-t border-line pt-3">
                <div className="mb-2 flex items-center justify-between gap-3">
                  <div>
                    <div className="text-[13px] font-medium uppercase tracking-[0.08em] text-ink-3">Saved Orchestrators</div>
                    <div className="mt-0.5 text-[13px] text-ink-3">Projects can target these hosts when their run location is Orchestrator.</div>
                  </div>
                  <div className="flex items-center gap-2">
                    {defaultHost && <span className="max-w-[180px] truncate rounded-full border border-line bg-card-2 px-2 py-1 text-[12px] text-ink-3">Default: {defaultHost.name}</span>}
                    <span className="rounded-full border border-line bg-card-2 px-2 py-1 text-[12px] text-ink-3">{savedHosts.length}</span>
                  </div>
                </div>
                {savedHosts.length === 0 ? (
                  <div className="rounded-lg border border-line bg-card-2 px-3 py-2 text-[13px] text-ink-3">No saved Orchestrators yet. Test a connection to save one.</div>
                ) : (
                  <div className="space-y-2">
                    {savedHosts.map((host) => {
                      const isDefault = defaultHost?.id === host.id
                      const checking = checkingHostId === host.id
                      return (
                      <div key={host.id} className={clsx('flex items-center gap-2 rounded-lg border bg-card-2 px-3 py-2', isDefault ? 'border-accent/40' : 'border-line')}>
                        <span className={clsx('h-2 w-2 shrink-0 rounded-full', checking ? 'animate-pulse bg-accent' : host.lastError ? 'bg-danger' : host.enabled ? 'bg-positive' : 'bg-ink-4')} />
                        <div className="min-w-0 flex-1">
                          <div className="flex min-w-0 items-center gap-2">
                            <div className="truncate text-[13px] font-medium text-ink">{host.name}</div>
                            {isDefault && <span className="shrink-0 rounded-full border border-accent/30 bg-accent/10 px-1.5 py-0.5 text-[11px] text-accent">Default</span>}
                            {!host.enabled && <span className="shrink-0 rounded-full border border-line bg-card px-1.5 py-0.5 text-[11px] text-ink-4">Disabled</span>}
                          </div>
                          <div className="truncate font-mono text-[12px] text-ink-3">{host.serverUrl}</div>
                          <div className={clsx('mt-0.5 truncate text-[12px]', host.lastError ? 'text-danger' : 'text-ink-4')}>
                            {checking
                              ? 'Checking connection...'
                              : host.lastError
                                ? host.lastError
                                : host.lastConnectedAt
                                  ? `Last checked ${new Date(host.lastConnectedAt).toLocaleString()}`
                                  : 'Not checked yet'}
                          </div>
                        </div>
                        {!isDefault && (
                          <Button variant="ghost" size="xs" onClick={() => makeDefaultHost(host.id)}>
                            Make default
                          </Button>
                        )}
                        <Button
                          variant="ghost"
                          size="xs"
                          icon={checking ? 'clock' : 'plug'}
                          onClick={() => void testSavedHost(host.id)}
                          disabled={checkingHostId !== null}
                        >
                          {checking ? 'Checking' : 'Test'}
                        </Button>
                        <Button
                          variant="ghost"
                          size="xs"
                          onClick={() => commitRemoteOrchestrators(draft.remoteOrchestrators.map((item) => item.id === host.id ? { ...item, enabled: !item.enabled, updatedAt: Date.now() } : item))}
                        >
                          {host.enabled ? 'Disable' : 'Enable'}
                        </Button>
                        <Button
                          variant="danger"
                          size="xs"
                          icon="trash"
                          aria-label={`Remove ${host.name}`}
                          onClick={() => commitRemoteOrchestrators(draft.remoteOrchestrators.filter((item) => item.id !== host.id))}
                        />
                      </div>
                    )})}
                  </div>
                )}
              </div>
            </div>
          </div>
        </Section>
              </SettingsPane>

  )
}

// Paired devices (VST-035): each phone/laptop/browser pairs ONCE via a one-time link and
// gets its own revocable session token — the shared owner token never leaves this machine.
// The link opens the host-served web client with #pair=<credential>, which redeems itself.
// A fragment never reaches HTTP request logs or intermediaries.
function PairedDevicesPanel({ enabled, serverUrl }: { enabled: boolean; serverUrl: string | null }) {
  const [devices, setDevices] = useState<Array<{ id: string; label: string; createdAt: number; lastSeenAt: number; revoked: boolean; clientInfo?: string | null }>>([])
  const [label, setLabel] = useState('')
  const [busy, setBusy] = useState(false)
  const [issued, setIssued] = useState<{ link: string; expiresAt: number } | null>(null)
  const [copied, setCopied] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const refresh = () => {
    api.listDeviceSessions().then(setDevices).catch(() => setDevices([]))
  }
  useEffect(() => {
    refresh()
    if (!enabled) return
    const interval = setInterval(refresh, 5000)
    return () => clearInterval(interval)
  }, [enabled])

  const issue = async () => {
    if (busy) return
    setBusy(true)
    setError(null)
    setIssued(null)
    try {
      const link = await api.issueDevicePairing(label.trim() || 'Paired device')
      const base = (link.url || serverUrl || '').replace(/\/$/, '')
      if (!base) {
        setError('Enable the Orchestrator first — the pairing link needs its URL.')
        return
      }
      setIssued({ link: `${base}/#pair=${encodeURIComponent(link.credential)}`, expiresAt: link.expiresAt })
      setLabel('')
      refresh()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not issue a pairing link.')
    } finally {
      setBusy(false)
    }
  }

  const revoke = async (id: string) => {
    try {
      await api.revokeDeviceSession(id)
      refresh()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not revoke the device.')
    }
  }

  const live = devices.filter((d) => !d.revoked)

  return (
    <div className="mt-3 rounded-lg border border-line bg-card-2 p-3">
      <div className="mb-2 flex items-center justify-between gap-2">
        <div className="text-[12px] font-medium uppercase tracking-[0.08em] text-ink-3">Paired devices</div>
        <span className="rounded bg-ink-5 px-1.5 text-[12px] tabular-nums text-ink-3">{live.length}</span>
      </div>
      <p className="mb-2 text-[13px] leading-snug text-ink-4">
        A pairing link opens the web client served by this host and signs that device in with its own revocable token. Links are one-time and expire in 10 minutes.
      </p>
      <div className="flex items-center gap-1.5">
        <Input value={label} onChange={(e) => setLabel(e.target.value)} placeholder="Device name (e.g. iPhone)" spellCheck={false} />
        <Button variant="default" size="sm" icon="plus" onClick={() => void issue()} disabled={busy || !enabled}>
          {busy ? 'Issuing…' : 'New pairing link'}
        </Button>
      </div>
      {!enabled && <p className="mt-1.5 text-[12px] text-ink-4">Enable the Orchestrator above to pair devices.</p>}
      {error && <p className="mt-1.5 text-[12px] text-danger">{error}</p>}
      {issued && (
        <div className="mt-2.5 flex items-start gap-3 rounded-md border border-line bg-card p-2.5">
          <div className="shrink-0 overflow-hidden rounded-md bg-white p-1"><QrCode value={issued.link} size={112} /></div>
          <div className="min-w-0 flex-1">
            <div className="mb-1 text-[12px] text-ink-3">Scan or open on the device — valid until {new Date(issued.expiresAt).toLocaleTimeString()}</div>
            <div className="break-all font-mono text-[12px] leading-snug text-ink-2">{issued.link}</div>
            <Button
              variant="ghost"
              size="xs"
              icon={copied ? 'check' : 'copy'}
              onClick={() => { void api.writeClipboard(issued.link).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1500) }) }}
            >
              {copied ? 'Copied' : 'Copy link'}
            </Button>
          </div>
        </div>
      )}
      {devices.length > 0 && (
        <div className="mt-2.5 space-y-1">
          {devices.map((d) => (
            <div key={d.id} className={clsx('flex items-center gap-2.5 rounded-md border border-line bg-card px-2.5 py-1.5 text-[13px]', d.revoked && 'opacity-55')}>
              <Icon name={d.revoked ? 'x' : 'monitor'} size={13} className={d.revoked ? 'text-ink-4' : 'text-ink-3'} />
              <span className="min-w-0 flex-1 truncate text-ink-2">{d.label}{d.clientInfo ? <span className="text-ink-4"> · {d.clientInfo}</span> : null}</span>
              <span className="shrink-0 text-[12px] tabular-nums text-ink-4">{d.revoked ? 'revoked' : `seen ${new Date(d.lastSeenAt).toLocaleTimeString()}`}</span>
              {!d.revoked && (
                <Button variant="ghost" size="xs" icon="x" onClick={() => void revoke(d.id)}>Revoke</Button>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
