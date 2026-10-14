import clsx from 'clsx'
import { shortPath } from '../../lib/format'
import { Icon } from '../../lib/icons'
import { Button, Field, Input, Segmented, Toggle } from '../ui'
import { ContextPressure } from '../ContextPressure'
import { updateStatusLabel } from '../../lib/update-status'
import { UpdateActions } from '../UpdateActions'
import { AdvancedSection, Section, SettingsPane, type SettingsSectionContext } from './shared'

export function RuntimeSettings({ ctx }: { ctx: SettingsSectionContext }) {
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

  return (
<SettingsPane>
        <Section title="Runtime defaults" desc="Approvals, workspace defaults, and context behavior. Room agents are configured inside each room.">
          <div className="space-y-4">
            <Field label="Default approval mode">
              <Segmented
                value={draft.approvalMode}
                onChange={(v) => patch({ approvalMode: v })}
                options={[{ value: 'suggest', label: 'Approve edits', icon: 'shield' }, { value: 'auto', label: 'Full auto', icon: 'spark' }, { value: 'readonly', label: 'Read only', icon: 'file' }]}
              />
            </Field>

            <Field label="Default workspace folder">
              <div className="flex gap-2">
                <Input readOnly value={draft.defaultCwd ? shortPath(draft.defaultCwd, 4) : ''} placeholder="No folder selected" />
                <Button variant="default" icon="folder" onClick={browseCwd}>Browse</Button>
              </div>
            </Field>

            <Field label="Auto-compact long tasks" hint={draft.autoCompact ? 'On' : 'Off'}>
              <div className="flex items-center gap-3 rounded-lg border border-line bg-card px-3 py-2.5">
                <span className="flex-1 text-[13px] leading-snug text-ink-3">Summarize earlier turns when the transcript grows, so long runs don’t hit the model’s context wall instead of stopping.</span>
                <Toggle label="Auto-compact long tasks" checked={draft.autoCompact} onChange={(v) => patch({ autoCompact: v })} />
              </div>
            </Field>

            <Field label="Improve skills after runs" hint={draft.skillSelfImprovement ? 'On' : 'Off'}>
              <div className="flex items-center gap-3 rounded-lg border border-line bg-card px-3 py-2.5">
                <span className="flex-1 text-[13px] leading-snug text-ink-3">After a successful run, let a model review the skills the agent consulted and propose a sharper version. Proposals appear as suggestions you accept or dismiss — nothing changes automatically. Uses extra tokens per run, so it’s off by default.</span>
                <Toggle label="Improve skills after runs" checked={draft.skillSelfImprovement === true} onChange={(v) => patch({ skillSelfImprovement: v })} />
              </div>
            </Field>

            <Field label="Allow network in task sandbox" hint={draft.allowTaskNetwork ? 'On' : 'Off'}>
              <div className="flex items-center gap-3 rounded-lg border border-line bg-card px-3 py-2.5">
                <span className="flex-1 text-[13px] leading-snug text-ink-3">By default the task sandbox blocks outbound network so a command can’t fetch or exfiltrate unexpectedly (package installs are still allowed). Turn on if your tasks need to reach the network — e.g. a local model on the LAN, an API, or a dev server.</span>
                <Toggle label="Allow network in task sandbox" checked={draft.allowTaskNetwork === true} onChange={(v) => patch({ allowTaskNetwork: v })} />
              </div>
            </Field>

            {/* Advanced: low-level sampling + context knobs most users never touch.
                Temperature defaults to ~0 (#49); kept here, collapsed, so the
                Runtime page stays approachable. */}
            <AdvancedSection>
              <Field label="Temperature" hint={draft.temperature.toFixed(2)}>
                <input
                  type="range" min={0} max={1} step={0.05} value={draft.temperature}
                  onChange={(e) => patch({ temperature: parseFloat(e.target.value) })}
                  className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-ink-5 accent-[var(--accent)]"
                />
                <div className="mt-1 flex justify-between text-[12px] text-ink-4"><span>0 · deterministic</span><span>higher = more creative, less reliable for tools</span><span>1</span></div>
              </Field>

              {draft.autoCompact && (
                <>
                  <Field label="Cloud compaction threshold" hint={`${Math.round(draft.contextLimit / 1000)}k tokens`}>
                    <input
                      type="range" min={16_000} max={512_000} step={4_000} value={draft.contextLimit}
                      onChange={(e) => patch({ contextLimit: parseInt(e.target.value, 10) })}
                      className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-ink-5 accent-[var(--accent)]"
                    />
                    <div className="mt-1 flex justify-between text-[12px] text-ink-4"><span>16k</span><span>cloud cap — auto-scales down to each model’s real context window</span><span>512k</span></div>
                  </Field>

                  <Field label="Local model threshold" hint={`${Math.round(draft.localContextLimit / 1000)}k tokens`}>
                    <input
                      type="range" min={4_000} max={256_000} step={4_000} value={draft.localContextLimit}
                      onChange={(e) => patch({ localContextLimit: parseInt(e.target.value, 10) })}
                      className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-ink-5 accent-[var(--accent)]"
                    />
                    <div className="mt-1 flex justify-between text-[12px] text-ink-4"><span>4k</span><span>local providers compact earlier to avoid overloaded context</span><span>256k</span></div>
                  </Field>

                  <ContextPressure />
                </>
              )}
            </AdvancedSection>

            <Field label="Run notifications" hint={draft.notifyRuns === false ? 'Off' : 'On'}>
              <div className="flex items-center gap-3 rounded-lg border border-line bg-card px-3 py-2.5">
                <span className="flex-1 text-[13px] leading-snug text-ink-3">Notify me when a run finishes, fails or needs approval while TensorFold is in the background.</span>
                <Toggle label="Run notifications" checked={draft.notifyRuns !== false} onChange={(v) => patch({ notifyRuns: v })} />
              </div>
            </Field>

            <Field label="Suggest Git on new folder projects" hint={draft.showGitInitPrompt === false ? 'Off' : 'On'}>
              <div className="flex items-center gap-3 rounded-lg border border-line bg-card px-3 py-2.5">
                <span className="flex-1 text-[13px] leading-snug text-ink-3">When you add a project from an existing folder that isn’t a Git repo, offer to initialize Git (rooms use isolated worktrees). Turn off to create folder projects without the prompt.</span>
                <Toggle label="Suggest Git on new folder projects" checked={draft.showGitInitPrompt !== false} onChange={(v) => patch({ showGitInitPrompt: v })} />
              </div>
            </Field>

            <Field label="Auto-check for app updates" hint={draft.autoCheckUpdates ? 'On' : 'Off'}>
              <div className="flex items-center gap-3 rounded-lg border border-line bg-card px-3 py-2.5">
                <span className="flex-1 text-[13px] leading-snug text-ink-3">Verify the canonical release feed against TensorFold’s embedded Ed25519 trust root and surface new versions in the workspace dock.</span>
                <Toggle label="Auto-check for app updates" checked={draft.autoCheckUpdates} onChange={(v) => patch({ autoCheckUpdates: v })} />
              </div>
            </Field>

            <div className="rounded-xl border border-line bg-card p-3.5 shadow-card">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="flex items-center gap-2 text-[13px] font-medium text-ink">
                    <Icon name={updateStatus?.state === 'available' ? 'spark' : updateStatus?.state === 'error' ? 'warning' : updateStatus?.state === 'current' && updateStatus.checkedAt != null && updateStatus.signatureVerified ? 'check' : 'clock'} size={15} className="text-ink-2" />
                    {updateStatusLabel(updateStatus)}
                  </div>
                  <div className="mt-0.5 text-[13px] text-ink-3">
                    Current {updateStatus?.currentVersion ?? 'unknown'}{updateStatus?.checkedAt ? ` · checked ${new Date(updateStatus.checkedAt).toLocaleString()}` : ''}
                  </div>
                  <div className="mt-0.5 text-[12px] text-ink-4">
                    Checks the signed TensorFold release feed at <span className="font-mono">tensorfold.dev/studio</span>.
                  </div>
                </div>
                <UpdateActions status={updateStatus} />
              </div>
              {updateStatus?.releaseName && <div className="mt-3 text-[13px] font-medium text-ink-2">{updateStatus.releaseName}</div>}
              {updateStatus?.releaseNotes && (
                <div className="scroll-thin mt-2 max-h-40 overflow-y-auto whitespace-pre-wrap rounded-lg border border-line bg-card-2 px-3 py-2 text-[13px] leading-relaxed text-ink-2">
                  {updateStatus.releaseNotes}
                </div>
              )}
              {updateStatus?.error && <div className="mt-2 rounded-lg border border-danger/30 bg-danger/10 px-3 py-2 text-[13px] text-danger">{updateStatus.error}</div>}
              {updateStatus && updateStatus.state !== 'not_configured' && (
                <div className="mt-2 flex flex-wrap gap-2 text-[12px] text-ink-4">
                  <span className="rounded-full border border-line bg-card-2 px-2 py-1">Integrity: {updateStatus.integrity.replace(/_/g, ' ')}</span>
                  {updateStatus.checksum && <span className="rounded-full border border-line bg-card-2 px-2 py-1">sha256: {updateStatus.checksum.value.slice(0, 10)}...</span>}
                  {updateStatus.signatureVerified && <span className="rounded-full border border-positive/30 bg-positive/10 px-2 py-1 text-positive">embedded-root signature verified</span>}
                  {updateStatus.integrity === 'artifact_verified' && <span className="rounded-full border border-positive/30 bg-positive/10 px-2 py-1 text-positive">downloaded bytes verified</span>}
                </div>
              )}
            </div>
          </div>
        </Section>

        <AdvancedSection title="Power & local controls" icon="coffee">
          <div className="rounded-xl border border-line bg-card p-3.5">
            <div className="flex items-start justify-between gap-3">
              <div className="flex min-w-0 items-start gap-2.5">
                <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-ink-5 text-ink-2"><Icon name="coffee" size={17} /></span>
                <div className="min-w-0">
                  <div className="text-[13px] font-medium text-ink">Keep computer alive</div>
                  <div className="mt-0.5 text-[13px] leading-snug text-ink-3">
                    On macOS, TensorFold runs <span className="font-mono text-ink-2">caffeinate -dimsu</span> while this is enabled so sleep does not interrupt active work.
                  </div>
                </div>
              </div>
              <Toggle label="Keep computer awake" checked={draft.power.keepComputerAlive} onChange={(keepComputerAlive) => commit({ ...draft, power: { ...draft.power, keepComputerAlive, lastError: null } })} />
            </div>
            <div className="mt-3 flex flex-wrap items-center gap-2 text-[13px] text-ink-3">
              <span className={clsx('rounded-full border px-2 py-1', draft.power.keepComputerAlive ? 'border-positive/30 bg-positive/10 text-positive' : 'border-line bg-card-2')}>
                {draft.power.keepComputerAlive ? 'Awake guard on' : 'Awake guard off'}
              </span>
              {draft.power.lastStartedAt && <span>Started {new Date(draft.power.lastStartedAt).toLocaleString()}</span>}
            </div>
            {draft.power.lastError && (
              <div className="mt-2.5 rounded-lg border border-warn/30 bg-warn/5 px-3 py-2 text-[13px] text-warn">
                {draft.power.lastError}
              </div>
            )}
          </div>
          <div className="rounded-xl border border-line bg-card p-3.5 shadow-card">
            <div className="mb-2 text-[13px] font-medium text-ink-2">Actor attribution, runtime gates, and memory</div>
            <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
              <Field label="Actor" hint={securityControls?.role ?? 'loading'}>
                <div className="rounded-lg border border-line bg-card-2 px-3 py-2.5 text-[13px] text-ink-2">
                  {securityControls?.actor ?? 'Checking...'}
                </div>
              </Field>
              <Field label="Command sandbox" hint={securityControls?.commandSandboxAvailable ? 'Active' : 'Unavailable'}>
                <div className="rounded-lg border border-line bg-card-2 px-3 py-2.5 text-[13px] text-ink-2">
                  {securityControls?.commandSandboxAvailable ? 'macOS sandbox-exec available' : 'Sandbox unavailable on this host'}
                </div>
              </Field>
              <Field label="Signed updates" hint="Always required">
                <div className="rounded-lg border border-line bg-card-2 px-3 py-2.5 text-[13px] text-ink-2">
                  {!securityControls?.updatePublicKeyConfigured
                    ? 'Embedded update trust root unavailable'
                    : securityControls.updateTrustRootProductionReady
                      ? 'Organization Ed25519 trust root active · downloaded bytes must match signed SHA-256'
                      : 'Development placeholder trust root active · replace it with the organization release public key before distribution'}
                </div>
              </Field>
              <Field label="MCP trust" hint={securityControls?.signedMcpRequired ? 'Signed' : 'Configured'}>
                <div className="rounded-lg border border-line bg-card-2 px-3 py-2.5 text-[13px] text-ink-2">
                  {securityControls?.mcpSandboxAvailable ? 'Process sandbox available' : 'Process sandbox unavailable'} · {securityControls?.trustedMcpExplicitlyAllowed ? 'trusted tools allowed' : 'trusted tools gated'}
                </div>
              </Field>
              <Field label="Local memory" hint={`${localMemoryTotal} record${localMemoryTotal === 1 ? '' : 's'}`}>
                <div className="rounded-lg border border-line bg-card-2 px-3 py-2.5 text-[13px] text-ink-2">
                  Chats, projects, rooms, tickets, missions, incidents, journals, browser data, usage state, summaries, archives, isolated bot workspace disks, and private Living Weights models and history
                </div>
              </Field>
              <Field label="Purge">
                <div className="flex flex-wrap gap-2">
                  <Button variant="danger" icon="trash" onClick={purgeLocalMemoryNow} disabled={securityBusy === 'purge' || securityControls?.role !== 'owner'}>
                    {securityBusy === 'purge' ? 'Purging...' : 'Purge memory'}
                  </Button>
                  <Button variant="ghost" icon="search" onClick={refreshSecurityControls} disabled={!!securityBusy}>Refresh</Button>
                </div>
              </Field>
            </div>
            {securityNotice && (
              <div className="mt-2.5 rounded-lg border border-line bg-card-2 px-3 py-2 text-[13px] text-ink-3">
                {securityNotice}
              </div>
            )}
            {securityControls && securityControls.role !== 'owner' && (
              <div className="mt-2.5 rounded-lg border border-warn/30 bg-warn/5 px-3 py-2 text-[13px] text-warn">
                Purge requires owner role.
              </div>
            )}
          </div>
        </AdvancedSection>
              </SettingsPane>

  )
}
