import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import clsx from 'clsx'
import { useStore } from '../lib/store'
import { api } from '../lib/ipc'
import type { Readiness } from '../lib/types'
import { shortPath, basename } from '../lib/format'
import { Icon } from '../lib/icons'
import { isExternalProvider } from '../../shared/provider-trust'
import { Button, Menu, MenuItem } from './ui'
import { COMPOSER_INSERT_EVENT } from '../lib/browser-preview'
import type { Attachment, ApprovalMode, ControllerSnapshot, Project, Provider, Thread, Skill, Command, SshHost, WorkspaceLocation, RemoteDirectoryListResult, TokenUsageSnapshot, ModelRouterSnapshot, ContextPressureSnapshot, PersistedProviderStatus } from '../lib/types'
import { providerStatusViewFor } from '../lib/provider-status'
import { buildTerminalReference } from '../lib/terminal-context'
import { AttachmentChips } from './AttachmentChips'
import { appendAttachments, filesToAttachments } from '../lib/attachments'
import { allowedProviders, projectDefaultModel, violatesPolicy } from '../lib/policy'
import { resolveExecutionTarget } from '../lib/execution-target'
import { resolveProviderModel } from '../lib/provider-accounts'
import { resolveSurfaceModel, settingsForSurfaceModelPick } from '../lib/assistant-model'
import { BUILTIN_CAPABILITIES, capabilityInstruction, capabilityToken, type BuiltInCapability } from '../lib/capabilities'
import { dictationSupported, startDictation } from '../lib/voice'
import { providerUsageSummary } from '../lib/provider-usage'
import { estimateComposerTokens, modelContextWindowTokens, providerLiveSteering, providerSupportsFastMode, type ContextUsageSummary } from '../lib/provider-model-metadata'
import { effectiveThreadCwd, effectiveThreadWorkspace } from '../lib/thread-workspace'
import { terminalSessionKey } from '../lib/terminal-sessions'
import { loadRecentModels, pushRecentModel, type RecentModel } from '../lib/recent-models'
import { MentionPopover } from './composer/MentionPopover'
import { SkillCommandPopover } from './composer/SkillCommandPopover'
import { SteerQueueList, type Steer } from './composer/SteerQueueList'
import { ControllerModal } from './composer/ControllerModal'
import { ApprovalMenu } from './composer/ApprovalMenu'
import { LocalEngineControls } from './composer/LocalEngineControls'
import { PhoneButton } from './phone/PhoneButton'
import { PhoneChatPanel } from './phone/PhoneChatPanel'
import { ModelPickerMenu } from './composer/ModelPickerMenu'
import { useEngineContextTokens } from '../lib/engine'
import { ModelMenuSettings } from './composer/ModelMenuSettings'
import { ComposerOverflowMenu, type OverflowWorkspace } from './composer/ComposerOverflowMenu'
import { effortDisplayLabel, providerEffortOptions } from '../lib/provider-model-metadata'

function formatControllerSnapshot(snapshot: ControllerSnapshot): string {
  const lines = [
    '<controller_window permission="explicit-user-attach">',
    `Captured: ${new Date(snapshot.capturedAt).toLocaleString()}`,
    'Permission: User clicked Controller and chose to insert this foreground-window capture.',
    'Capture scope: foreground macOS Accessibility snapshot; not passive screen scraping.',
    `App: ${snapshot.appName}${snapshot.bundleId ? ` (${snapshot.bundleId})` : ''}`,
    `Window: ${snapshot.windowTitle}`,
    `Source: ${snapshot.source}${snapshot.truncated ? ' (truncated)' : ''}`,
  ]
  if (snapshot.warning) lines.push(`Warning: ${snapshot.warning}`)
  if (snapshot.text) lines.push('', 'Accessible text:', snapshot.text)
  lines.push('</controller_window>')
  return lines.join('\n')
}

export function Composer({ mode, thread, seed }: { mode: 'home' | 'thread'; thread?: Thread; seed?: { text: string; n: number } }) {
  const settings = useStore((s) => s.settings)!
  const status = useStore((s) => s.status)
  const launch = useStore((s) => s.launch)
  const send = useStore((s) => s.send)
  const cancel = useStore((s) => s.cancel)
  const saveSettings = useStore((s) => s.saveSettings)
  const setThreadProvider = useStore((s) => s.setThreadProvider)
  const noteModelChange = useStore((s) => s.noteModelChange)
  const compactThread = useStore((s) => s.compactThread)
  const patchThread = useStore((s) => s.patchThread)
  const projects = useStore((s) => s.projects)
  const terminals = useStore((s) => s.terminals)
  const activeTerminalId = useStore((s) => s.activeTerminalId)
  const [readiness, setReadiness] = useState<Readiness | null>(null)
  const [authBusy, setAuthBusy] = useState(false)
  const setThreadProject = useStore((s) => s.setThreadProject)
  const addProject = useStore((s) => s.addProject)
  const pendingMention = useStore((s) => s.pendingMention)
  const composerSeed = useStore((s) => s.composerSeed)
  const sshHosts = useStore((s) => s.sshHosts)
  const sshStatus = useStore((s) => s.sshStatus)
  const connectSshHost = useStore((s) => s.connectSshHost)
  const setThreadWorkspace = useStore((s) => s.setThreadWorkspace)
  const setView = useStore((s) => s.setView)
  const [homeWorkspace, setHomeWorkspace] = useState<WorkspaceLocation | null>(null)
  const [homeProject, setHomeProject] = useState<string | null>(null)

  const [text, setText] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const submittingRef = useRef(false)
  const [phoneOpen, setPhoneOpen] = useState(false)
  const [attachments, setAttachments] = useState<Attachment[]>([])
  const [usageSnapshot, setUsageSnapshot] = useState<TokenUsageSnapshot | null>(null)
  const [providerStatus, setProviderStatus] = useState<PersistedProviderStatus | null>(null)
  const [modelRouterSnapshot, setModelRouterSnapshot] = useState<ModelRouterSnapshot | null>(null)
  const [contextPressureSnapshot, setContextPressureSnapshot] = useState<ContextPressureSnapshot | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const addFiles = async (files: FileList | File[]) => {
    const next = await filesToAttachments(files)
    if (next.length) setAttachments((prev) => appendAttachments(prev, next))
  }
  const [dragOver, setDragOver] = useState(false)
  const taRef = useRef<HTMLTextAreaElement>(null)
  // Steer queue: messages composed while a run is active are staged
  // here above the composer. Each carries a status lifecycle (queued → guiding →
  // guided) so the user sees the message land and get applied, not a flat row.
  // Inject live ("Guide") or let it auto-send as the next turn when the run ends.
  // The `Steer` type lives in composer/SteerQueueList.tsx (shared with that view).
  const [queued, setQueued] = useState<Steer[]>([])
  const queuedRef = useRef(queued)
  queuedRef.current = queued

  const [cwd, setCwd] = useState(settings.defaultCwd || '')
  // Incognito toggle (home composer): flags the NEXT launched thread as memory-only.
  const [incognito, setIncognito] = useState(false)
  const activeProjectId = mode === 'thread' ? (thread?.projectId ?? null) : homeProject
  const policyProject = activeProjectId ? (projects.find((p) => p.id === activeProjectId) ?? null) : null
  // Active terminal for the composer's "Reference terminal output" action (VST-039).
  const projectTerminals = activeProjectId ? (terminals[activeProjectId] ?? []) : []
  const activeTerminalSession = projectTerminals.find((s) => terminalSessionKey(s) === (activeProjectId ? activeTerminalId[activeProjectId] : null)) ?? projectTerminals[0] ?? null
  // One resolution for where new work runs (VST-041): an existing thread keeps its origin
  // stamp; otherwise project pin > localOnly > global orchestrator mode > local. The chip
  // below renders exactly this decision — a redirect is never silent.
  const executionDecision = resolveExecutionTarget(settings, policyProject)
  // An EXISTING thread routes sends to its stamped home — the resolver only applies to
  // NEW work. Without this split, a local thread under global orchestrator mode claims
  // "Runs on <host>" while the send actually stays on the thread's home machine.
  const effectiveHostId = mode === 'thread' ? (thread?.orchestratorHostId ?? null) : executionDecision.hostId
  const activeOrchestrator = effectiveHostId ? settings.remoteOrchestrators?.find((host) => host.id === effectiveHostId) ?? null : null
  const isOrchestratorProject = !!activeOrchestrator
  const orchestratorChipLabel = activeOrchestrator
    ? (mode !== 'thread' && executionDecision.source === 'global'
        ? `${activeOrchestrator.name || 'Orchestrator'} · all new work`
        : activeOrchestrator.name || 'Orchestrator')
    : mode !== 'thread' && executionDecision.unavailableReason
      ? 'Orchestrator unavailable'
      : null
  const localOnly = !!policyProject?.localOnly || !!thread?.phoneLocalOnly
  const pickerProviders = allowedProviders(settings.providers, policyProject)
  const [remoteBrowseOpen, setRemoteBrowseOpen] = useState(false)
  const [remoteBrowse, setRemoteBrowse] = useState<RemoteDirectoryListResult | null>(null)
  const [remoteBrowseBusy, setRemoteBrowseBusy] = useState(false)
  const [remoteBrowseError, setRemoteBrowseError] = useState('')

  // Per-surface default (VST-015): the home composer launches CODING tasks, so it seeds from
  // the code-surface route (Settings → Models → Coding tasks model), which folds back to the
  // globally-active route when no override is configured. Existing threads keep their own route.
  const codeSurface = resolveSurfaceModel(settings, 'code')
  const rawProviderId = thread?.bot?.providerId ?? (mode === 'thread' ? thread?.providerId ?? settings.activeProviderId : codeSurface.providerId)
  let provider = settings.providers.find((p) => p.id === rawProviderId) ?? (thread?.bot ? undefined : settings.providers[0])
  let providerAccountId = thread?.bot ? thread.bot.providerAccountId : mode === 'thread'
    ? (thread && Object.prototype.hasOwnProperty.call(thread, 'providerAccountId') ? (thread.providerAccountId ?? null) : (settings.activeProviderAccountId ?? provider?.defaultAccountId ?? null))
    : (codeSurface.providerAccountId ?? provider?.defaultAccountId ?? null)
  let model = thread?.bot?.model ?? (mode === 'thread' ? thread?.model ?? settings.activeModel : codeSurface.model)
  // Enforce the project policy: under local-only, fall back to the project's
  // default (or first local) instead of an out-of-policy cloud model.
  if (violatesPolicy(policyProject, provider)) {
    const d = projectDefaultModel(policyProject, settings.providers)
    provider = settings.providers.find((p) => p.id === d.providerId) || pickerProviders[0] || provider
    providerAccountId = d.providerAccountId ?? provider?.defaultAccountId ?? null
    model = d.model ?? provider?.models?.[0] ?? model
  }

  // Point-of-use readiness: is there a usable model? Surfaces "connect a model" /
  // "authorize" guidance instead of letting a send fail with a cryptic error.
  model = provider && !thread?.bot ? (resolveProviderModel(provider, providerAccountId, model) ?? model) : model
  const providerKey = provider?.id ?? ''
  useEffect(() => {
    let live = true
    api.getModelReadiness({ providerId: provider?.id ?? null, accountId: providerAccountId, model })
      .then((r) => { if (live) setReadiness(r) })
      .catch(() => { if (live) setReadiness(null) })
    return () => { live = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [providerKey, providerAccountId, model, settings.providers])

  const authorizeCli = async () => {
    if (!provider) return
    setAuthBusy(true)
    try {
      await api.loginCliProvider({ providerId: provider.id, accountId: providerAccountId })
      setReadiness(await api.getModelReadiness({ providerId: provider.id, accountId: providerAccountId, model }))
    } finally {
      setAuthBusy(false)
    }
  }
  const usageSummary = provider ? providerUsageSummary(provider, providerAccountId, model, usageSnapshot, modelRouterSnapshot) : null
  // Real rate-limit windows for the active provider/account (VST-037): session/weekly pace + ETA.
  const rateLimitStatus = provider ? providerStatusViewFor(providerStatus, provider.id, providerAccountId) : null
  const contextThread = thread ? contextPressureSnapshot?.threads.find((row) => row.id === thread.id) ?? null : null
  // The built-in engine's served window (max_model_len) beats a cached or name-based guess.
  const engineContextTokens = useEngineContextTokens(provider?.source === 'tensorfold', model)
  const contextWindowTokens = engineContextTokens
    ?? contextThread?.contextWindowTokens
    ?? modelContextWindowTokens(provider, model)
    ?? (provider?.kind === 'local' ? settings.localContextLimit : settings.contextLimit)
    ?? 128_000
  const contextUsedTokens = contextThread?.estimatedTokens ?? estimateComposerTokens(text, attachments)
  const contextUsage: ContextUsageSummary = {
    usedTokens: contextUsedTokens,
    windowTokens: contextWindowTokens,
    workingLimitTokens: contextThread?.workingLimitTokens ?? null,
    percent: contextWindowTokens > 0 ? contextUsedTokens / contextWindowTokens : 0,
    status: contextThread?.status ?? null,
  }
  const canConfigureFastMode = provider?.transport === 'codex-cli' && providerSupportsFastMode(provider, model)
  const workspace = mode === 'thread' && thread ? effectiveThreadWorkspace(thread, policyProject) : homeWorkspace
  const projectRoot = policyProject?.workspace?.kind === 'ssh' ? policyProject.workspace.path : (policyProject?.cwd || policyProject?.workspace?.path || '')
  // /workspace belongs to the guest; host Git/file pickers must never resolve a
  // coincidentally matching host directory for an isolated bot chat.
  const workCwd = thread?.bot?.executionTarget === 'vm' ? '' : mode === 'thread' && thread ? effectiveThreadCwd(thread, policyProject) : (workspace?.kind === 'ssh' ? workspace.path : (cwd || homeWorkspace?.path || projectRoot || ''))
  const selectLocalWorkspace = () => { if (mode === 'thread' && thread) setThreadWorkspace(thread.id, null); else setHomeWorkspace(null) }
  const selectHostWorkspace = (host: SshHost, remotePath: string) => {
    const ws: WorkspaceLocation = { kind: 'ssh', hostId: host.id, path: remotePath }
    if (mode === 'thread' && thread) setThreadWorkspace(thread.id, ws)
    else { setHomeWorkspace(ws); setCwd(remotePath) }
    connectSshHost(host.id)
  }

  const st = thread ? status[thread.id] : undefined
  const busy = st != null && ['thinking', 'streaming', 'tool', 'awaiting'].includes(st)
  const botPreset = thread?.bot ? settings.agentPresets.find((preset) => preset.id === thread.bot!.presetId) : null
  const botBlocked = !!thread?.bot && (!botPreset || !!botPreset.paused || !!botPreset.archivedAt)
  const isChat = mode === 'thread' && thread?.kind === 'chat'

  // Drain the run queue: when a run finishes, fire the next staged message as a
  // fresh turn. Skips items that are guiding/guided (already injected live). The
  // remainder stay queued and drain on the following completion, so follow-ups
  // serialize one run at a time.
  const prevBusy = useRef(false)
  useEffect(() => {
    if (prevBusy.current && !busy && st !== 'error' && !botBlocked && mode === 'thread' && thread && queuedRef.current.length) {
      const next = queuedRef.current.find((x) => x.status === 'queued')
      if (next) {
        setQueued((q) => q.filter((x) => x.id !== next.id))
        void send(next.text)
      }
    }
    prevBusy.current = busy
  }, [busy, st, botBlocked, thread?.id, mode]) // eslint-disable-line react-hooks/exhaustive-deps
  // Stale queue entries make no sense on a different thread.
  useEffect(() => { setQueued([]) }, [thread?.id])

  // git branch for the chip row
  const [branch, setBranch] = useState<string | null>(null)
  useEffect(() => {
    let live = true
    setBranch(null)
    if (workCwd) api.getBranch(workCwd, workspace).then((b) => { if (live) setBranch(b) }).catch(() => {})
    return () => { live = false }
  }, [workCwd, workspace])

  // @-mention state
  const [files, setFiles] = useState<string[]>([])
  const filesCwd = useRef('')
  const [mention, setMention] = useState<{ query: string; start: number } | null>(null)
  const [mIdx, setMIdx] = useState(0)

  // $-skill / /-command picker state
  const [skills, setSkills] = useState<Skill[]>([])
  const [commands, setCommands] = useState<Command[]>([])
  const [pick, setPick] = useState<{ kind: '/' | '$'; query: string; start: number } | null>(null)
  const [pIdx, setPIdx] = useState(0)
  const [controllerModal, setControllerModal] = useState(false)
  const [controllerPreview, setControllerPreview] = useState<ControllerSnapshot | null>(null)
  const [controllerBusy, setControllerBusy] = useState(false)
  const [controllerError, setControllerError] = useState('')
  const [dictating, setDictating] = useState(false)
  const [dictationError, setDictationError] = useState('')
  const dictationStopRef = useRef<(() => void) | null>(null)
  useEffect(() => { api.getSkills().then(setSkills).catch(() => {}); api.getCommands().then(setCommands).catch(() => {}) }, [])
  useEffect(() => {
    let live = true
    Promise.all([
      api.getTokenUsage().catch(() => null),
      api.getModelRouterSnapshot().catch(() => null),
      api.getContextPressure().catch(() => null),
      api.getProviderStatus().catch(() => null),
    ]).then(([usage, routes, pressure, status]) => {
      if (!live) return
      setUsageSnapshot(usage)
      setModelRouterSnapshot(routes)
      setContextPressureSnapshot(pressure)
      setProviderStatus(status)
    })
    // Live rate-limit refresh (VST-037): a codex turn pushes updated windows.
    const off = api.onProviderStatus?.((status) => { if (live) setProviderStatus(status) })
    const offUsage = api.onUsageCleared?.(() => { if (live) setUsageSnapshot(null) })
    return () => { live = false; off?.(); offUsage?.() }
  }, [])
  useEffect(() => () => { dictationStopRef.current?.() }, [])

  const ensureFiles = () => {
    const workspaceKey = `${workspace?.kind ?? 'local'}:${workspace?.kind === 'ssh' ? workspace.hostId : ''}:${workCwd}`
    if (filesCwd.current === workspaceKey && files.length) return
    filesCwd.current = workspaceKey
    if (!workCwd) { setFiles([]); return }
    if (workCwd) api.listFiles(workCwd, workspace)
      .then((next) => { if (filesCwd.current === workspaceKey) setFiles(next) })
      .catch(() => { if (filesCwd.current === workspaceKey) setFiles([]) })
  }

  const items = useMemo(() => {
    if (!mention || thread?.bot?.executionTarget === 'vm') return []
    const q = mention.query.toLowerCase()
    const score = (f: string) => {
      const base = (f.split('/').pop() || '').toLowerCase()
      if (!q) return 0
      if (base.startsWith(q)) return 0
      if (base.includes(q)) return 1
      if (f.toLowerCase().includes(q)) return 2
      return 3
    }
    return files
      .filter((f) => !q || f.toLowerCase().includes(q))
      .sort((a, b) => score(a) - score(b) || a.length - b.length)
      .slice(0, 8)
  }, [mention, files, thread?.bot?.executionTarget])
  const capabilityItems = useMemo(() => {
    if (!mention || thread?.bot?.executionTarget === 'vm') return []
    const q = mention.query.toLowerCase()
    return BUILTIN_CAPABILITIES
      .filter((capability) => !q || capability.name.toLowerCase().includes(q) || capability.mention.includes(q) || capability.description.toLowerCase().includes(q))
      .slice(0, 8)
  }, [mention, thread?.bot?.executionTarget])
  useEffect(() => {
    const total = capabilityItems.length + items.length
    if (total > 0 && mIdx >= total) setMIdx(0)
  }, [capabilityItems.length, items.length, mIdx])

  // Built-in slash commands (thread-scoped) shown alongside user commands.
  const builtins = useMemo<{ name: string; description: string; argHint?: string }[]>(
    () => (mode === 'thread' && !thread?.bot
      ? [{ name: 'compact', description: 'Summarize the conversation to reclaim context' },
         { name: 'goal', description: 'Set a goal the agent verifies against', argHint: '<text…>' },
         { name: 'computer-use', description: 'Let the agent operate Mac apps for this task, with approval per app', argHint: '<task… | stop>' }]
      : []),
    [mode, thread?.bot],
  )
  const pickItems = useMemo<{ name: string; description?: string; argHint?: string }[]>(() => {
    if (!pick) return []
    const q = pick.query.toLowerCase()
    if (pick.kind === '$') return skills.filter((s) => s.enabled && (!q || s.name.toLowerCase().includes(q))).slice(0, 8)
    return [...builtins, ...commands].filter((c) => !q || c.name.toLowerCase().includes(q)).slice(0, 8)
  }, [pick, skills, commands, builtins])

  useLayoutEffect(() => {
    const ta = taRef.current
    if (!ta) return
    ta.style.height = 'auto'
    ta.style.height = Math.min(ta.scrollHeight, 224) + 'px'
  }, [text])

  useEffect(() => { taRef.current?.focus() }, [mode, thread?.id])
  // Re-bind a thread when policy or provider model availability makes its stored model stale.
  useEffect(() => {
    if (mode === 'thread' && thread && !thread.bot && provider && model && (
      (localOnly && violatesPolicy(policyProject, settings.providers.find((p) => p.id === thread.providerId))) ||
      thread.providerId !== provider.id ||
      (thread.providerAccountId ?? null) !== (providerAccountId ?? null) ||
      thread.model !== model
    )) {
      void setThreadProvider(thread.id, provider.id, model || provider.models[0], providerAccountId).catch(() => {})
    }
  }, [thread?.id, provider?.id, providerAccountId, model, localOnly]) // eslint-disable-line
  useEffect(() => { if (seed && seed.text) { setText(seed.text); taRef.current?.focus() } }, [seed?.n])
  // Injected composer text (VST-040 ask-why → chat reference): insert at the caret, focus.
  useEffect(() => {
    if (!composerSeed?.text) return
    setText((t) => {
      const ta = taRef.current
      const caret = ta?.selectionStart ?? t.length
      const pad = caret > 0 && t[caret - 1] && !/\s/.test(t[caret - 1]) ? '\n' : ''
      return t.slice(0, caret) + pad + composerSeed.text + t.slice(caret)
    })
    requestAnimationFrame(() => taRef.current?.focus())
  }, [composerSeed?.n])
  // File explorer → reference a file in the composer.
  useEffect(() => {
    if (!pendingMention) return
    setText((t) => `${t}${t && !t.endsWith(' ') ? ' ' : ''}@${pendingMention.path} `)
    taRef.current?.focus()
  }, [pendingMention?.n]) // eslint-disable-line

  const pickModel = async (p: Provider, m: string, accountId: string | null = null) => {
    if (mode === 'thread' && thread) {
      // Inline "Model changed to X" notice — only on an existing thread, and only when
      // the pick is an actual change. Home / first selection stays notice-free.
      const changed = !(p.id === provider?.id && m === model && (accountId ?? null) === (providerAccountId ?? null))
      await setThreadProvider(thread.id, p.id, m, accountId)
      if (changed) await noteModelChange(thread.id, p.id, m, accountId).catch(() => {})
    } else await saveSettings(settingsForSurfaceModelPick(useStore.getState().settings ?? settings, 'code', { providerId: p.id, providerAccountId: accountId, model: m }), { silent: true })
    setRecentModels(pushRecentModel({ providerId: p.id, accountId, model: m }))
  }
  // One-click recent-model chips. Refresh from storage on mount so other threads'
  // picks propagate.
  const [recentModels, setRecentModels] = useState<RecentModel[]>(() => loadRecentModels())
  useEffect(() => { setRecentModels(loadRecentModels()) }, [thread?.id])
  const setApproval = (m: ApprovalMode) => saveSettings({ ...settings, approvalMode: m })
  const chooseDir = async () => { const d = await api.pickDirectory(); if (d) { setCwd(d); setHomeWorkspace(null) } }
  const setFolder = (dir: string) => {
    if (mode === 'thread' && thread) {
      void setThreadWorkspace(thread.id, null, dir)
    } else {
      setCwd(dir)
      setHomeWorkspace(null)
    }
  }
  const loadRemoteProjectDir = async (dir?: string | null) => {
    if (!activeOrchestrator) return
    setRemoteBrowseOpen(true)
    setRemoteBrowseBusy(true)
    setRemoteBrowseError('')
    try {
      const result = await api.listRemoteOrchestratorDir(activeOrchestrator.id, dir || projectRoot || '')
      if (!result.ok) throw new Error(result.error || 'Could not list folders on the Orchestrator.')
      setRemoteBrowse(result)
    } catch (e: any) {
      setRemoteBrowseError(e?.message ?? 'Could not list folders on the Orchestrator.')
    } finally {
      setRemoteBrowseBusy(false)
    }
  }
  const chooseRemoteProjectDir = (dir: string) => {
    setFolder(dir)
    setRemoteBrowseOpen(false)
    setRemoteBrowse(null)
    setRemoteBrowseError('')
  }

  const insertAtCaret = (insert: string) => {
    const ta = taRef.current
    const caret = ta?.selectionStart ?? text.length
    const pad = caret > 0 && !/\s/.test(text[caret - 1]) ? '\n\n' : ''
    const next = text.slice(0, caret) + pad + insert + text.slice(caret)
    setText(next)
    requestAnimationFrame(() => {
      const pos = caret + pad.length + insert.length
      taRef.current?.focus()
      taRef.current?.setSelectionRange(pos, pos)
    })
  }
  useEffect(() => {
    const onInsert = (event: Event) => {
      const detail = (event as CustomEvent<{ text?: string }>).detail
      if (typeof detail?.text === 'string' && detail.text.trim()) insertAtCaret(detail.text)
    }
    window.addEventListener(COMPOSER_INSERT_EVENT, onInsert)
    return () => window.removeEventListener(COMPOSER_INSERT_EVENT, onInsert)
  }, [text])

  const onSelectProject = (pid: string | null) => {
    if (mode === 'thread' && thread) setThreadProject(thread.id, pid)
    else {
      setHomeProject(pid)
      const p = projects.find((x) => x.id === pid)
      setCwd(pid ? (p?.cwd || p?.workspace?.path || '') : (settings.defaultCwd || ''))
      setHomeWorkspace(pid ? (p?.workspace ?? null) : null)
      setRemoteBrowseOpen(false)
      setRemoteBrowse(null)
      setRemoteBrowseError('')
    }
  }
  const handleAddProject = async (kind: 'scratch' | 'folder') => {
    if (kind === 'folder') {
      const d = await api.pickDirectory()
      if (!d) return null
      try {
        const git = await api.inspectGitRepository(d)
        if (!git.exists && settings.showGitInitPrompt !== false) {
          const ok = confirm('This folder is not initialized as a Git repo. Initialize Git now so TensorFold can use isolated room worktrees?')
          if (ok) await api.initGitRepository(d)
        }
      } catch (e: any) {
        // With the soft-suggestion off (#37), git inspection is best-effort: never
        // block project creation behind a confirm() dialog.
        if (settings.showGitInitPrompt !== false) {
          const ok = confirm(`${e?.message ?? 'Could not inspect Git for this folder.'}\n\nCreate the project anyway?`)
          if (!ok) return null
        }
      }
      return addProject(d.split('/').pop() || 'project', d)
    }
    return addProject('New project', '')
  }

  const detectMention = (v: string, caret: number) => {
    const m = /(?:^|\s)@([^\s@]*)$/.exec(v.slice(0, caret))
    if (m) { setMention({ query: m[1], start: caret - m[1].length - 1 }); setMIdx(0); ensureFiles() }
    else if (mention) setMention(null)
  }
  // $skill anywhere; /command only as the leading token.
  const detectPick = (v: string, caret: number) => {
    const before = v.slice(0, caret)
    const sm = /(?:^|\s)\$([\w-]*)$/.exec(before)
    if (sm) { setPick({ kind: '$', query: sm[1], start: caret - sm[1].length - 1 }); setPIdx(0); return }
    const cm = /^\/([\w-]*)$/.exec(before)
    if (cm) { setPick({ kind: '/', query: cm[1], start: 0 }); setPIdx(0); return }
    if (pick) setPick(null)
  }
  const onChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setText(e.target.value)
    const caret = e.target.selectionStart ?? e.target.value.length
    if (!isChat) detectMention(e.target.value, caret)
    detectPick(e.target.value, caret)
  }
  const selectMention = (file: string) => {
    if (!mention) return
    const ta = taRef.current!
    const caret = ta.selectionStart ?? text.length
    const insert = `@${file} `
    const next = text.slice(0, mention.start) + insert + text.slice(caret)
    setText(next)
    setMention(null)
    requestAnimationFrame(() => { const pos = mention.start + insert.length; ta.focus(); ta.setSelectionRange(pos, pos) })
  }
  const selectCapability = (capability: BuiltInCapability) => {
    if (!mention) return
    const ta = taRef.current!
    const caret = ta.selectionStart ?? text.length
    const insert = `${capabilityToken(capability)}\n${capabilityInstruction(capability)}\n\n`
    const next = text.slice(0, mention.start) + insert + text.slice(caret)
    setText(next)
    setMention(null)
    requestAnimationFrame(() => { const pos = mention.start + insert.length; ta.focus(); ta.setSelectionRange(pos, pos) })
  }
  // Pick a skill (inserts its body inline) or a command (completes the /name).
  const selectPick = (name: string) => {
    if (!pick) return
    const ta = taRef.current!
    const caret = ta.selectionStart ?? text.length
    let insert: string
    if (pick.kind === '$') { const s = skills.find((x) => x.name === name); insert = (s?.body?.trim() || `$${name}`) + ' ' }
    else insert = `/${name} `
    const next = text.slice(0, pick.start) + insert + text.slice(caret)
    setText(next)
    setPick(null)
    requestAnimationFrame(() => { const pos = pick.start + insert.length; ta.focus(); ta.setSelectionRange(pos, pos) })
  }
  // Reference terminal output (VST-039): pull the active terminal's backlog and insert a
  // bounded, ANSI-stripped reference at the caret so the agent can act on what just ran.
  const referenceTerminalOutput = async () => {
    if (!activeTerminalSession) return
    let backlog = ''
    // `terminalBacklog` is host-local. Never probe it for an Orchestrator session: a same-id local
    // PTY would leak the wrong output, and legacy remote rows intentionally fail closed.
    if (activeTerminalSession.hostLocation !== 'orchestrator') {
      try { backlog = await api.terminalBacklog(activeTerminalSession.id) } catch { /* keep empty → honest "no output" note */ }
    }
    const reference = buildTerminalReference(activeTerminalSession.title || 'terminal', backlog)
    const ta = taRef.current
    const caret = ta?.selectionStart ?? text.length
    const pad = caret > 0 && text[caret - 1] && !/\s/.test(text[caret - 1]) ? '\n' : ''
    setText(text.slice(0, caret) + pad + reference + text.slice(caret))
    requestAnimationFrame(() => { const pos = caret + pad.length + reference.length; ta?.focus(); ta?.setSelectionRange(pos, pos) })
  }
  const onAttach = () => {
    const ta = taRef.current
    if (!ta) return
    const caret = ta.selectionStart ?? text.length
    const pad = caret > 0 && !/\s/.test(text[caret - 1]) ? ' ' : ''
    const insert = pad + '@'
    const next = text.slice(0, caret) + insert + text.slice(caret)
    setText(next)
    ensureFiles()
    const pos = caret + insert.length
    setMention({ query: '', start: pos - 1 }); setMIdx(0)
    requestAnimationFrame(() => { ta.focus(); ta.setSelectionRange(pos, pos) })
  }
  // "Use a skill" from the overflow menu: type a $ at the caret and open the skill picker.
  const startSkillPick = () => {
    const ta = taRef.current
    if (!ta) return
    const caret = ta.selectionStart ?? text.length
    const pad = caret > 0 && !/\s/.test(text[caret - 1]) ? ' ' : ''
    const insert = pad + '$'
    setText(text.slice(0, caret) + insert + text.slice(caret))
    const pos = caret + insert.length
    setPick({ kind: '$', query: '', start: pos - 1 }); setPIdx(0)
    requestAnimationFrame(() => { ta.focus(); ta.setSelectionRange(pos, pos) })
  }
  const attachControllerSnapshot = async () => {
    setControllerBusy(true)
    setControllerError('')
    try {
      const snapshot = await api.captureController()
      setControllerPreview(snapshot)
      setControllerModal(true)
    } catch (e: any) {
      setControllerError(String(e?.message || e || 'Controller could not capture the current window.').replace(/^Error invoking remote method '[^']+':\s*/, '').replace(/^Error:\s*/, ''))
      setControllerModal(true)
    } finally {
      setControllerBusy(false)
    }
  }
  const onController = async () => {
    setControllerBusy(true)
    setControllerError('')
    try {
      const status = await api.getControllerStatus()
      if (!status.canCapture) { setControllerModal(true); return }
      await attachControllerSnapshot()
    } catch (e: any) {
      setControllerError(String(e?.message || e || 'Controller is unavailable.').replace(/^Error invoking remote method '[^']+':\s*/, '').replace(/^Error:\s*/, ''))
      setControllerModal(true)
    } finally {
      setControllerBusy(false)
    }
  }
  const enableControllerAndCapture = async () => {
    setControllerBusy(true)
    setControllerError('')
    try {
      const status = await api.enableController()
      if (!status.canCapture) {
        setControllerError(status.message)
        return
      }
    } catch (e: any) {
      setControllerError(String(e?.message || e || 'Controller could not be enabled.').replace(/^Error invoking remote method '[^']+':\s*/, '').replace(/^Error:\s*/, ''))
      return
    } finally {
      setControllerBusy(false)
    }
    await attachControllerSnapshot()
  }
  const insertControllerPreview = () => {
    if (!controllerPreview) return
    insertAtCaret(formatControllerSnapshot(controllerPreview))
    setControllerPreview(null)
    setControllerModal(false)
  }
  const toggleDictation = () => {
    if (dictating) {
      dictationStopRef.current?.()
      dictationStopRef.current = null
      setDictating(false)
      return
    }
    setDictationError('')
    if (!dictationSupported(settings.voice)) {
      setDictationError('Speech-to-text is not available in this runtime.')
      return
    }
    setDictating(true)
    dictationStopRef.current = startDictation(settings.voice, {
      onText: (spoken) => {
        setText((current) => `${current}${current && !/\s$/.test(current) ? ' ' : ''}${spoken}`)
        taRef.current?.focus()
      },
      onError: (message) => setDictationError(message),
      onEnd: () => {
        dictationStopRef.current = null
        setDictating(false)
      },
    })
  }

  // Drag a folder (or file) from Finder onto the composer to reference it as context.
  const onDrop = (e: React.DragEvent) => {
    const items = Array.from(e.dataTransfer.items || [])
    const files = Array.from(e.dataTransfer.files || [])
    if (!items.length && !files.length) return
    e.preventDefault(); setDragOver(false)
    const refs: string[] = []
    const n = Math.max(items.length, files.length)
    for (let i = 0; i < n; i++) {
      const entry = (items[i] as any)?.webkitGetAsEntry?.()
      const f = files[i] as any
      const name: string | undefined = entry?.name || f?.name
      if (!name) continue
      const abs: string | undefined = f?.path // Electron exposes the absolute path
      const isDir = !!entry?.isDirectory
      let ref = abs || name
      if (abs && workCwd && abs.startsWith(workCwd)) ref = abs.slice(workCwd.length).replace(/^[/\\]/, '') || name
      refs.push(`@${ref}${isDir ? '/' : ''}`)
    }
    if (!refs.length) return
    setText((t) => `${t}${t && !t.endsWith(' ') ? ' ' : ''}${refs.join(' ')} `)
    taRef.current?.focus()
  }

  // Expand any literal $skill tokens (for enabled skills) to their body text.
  const expandSkills = (s: string) => {
    if (!s.includes('$')) return s
    let out = s
    for (const sk of skills) {
      if (!sk.enabled || !sk.body.trim()) continue
      const esc = sk.name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
      out = out.replace(new RegExp(`(^|\\s)\\$${esc}(?=\\s|$)`, 'g'), (_m, pre) => `${pre}${sk.body.trim()}`)
    }
    return out
  }
  const dispatch = async (body: string) => {
    if (submittingRef.current) return
    if (!provider) { useStore.setState({ notice: { kind: 'error', message: 'This model is unavailable. Check Models or this bot’s configuration; your draft is still here.' } }); return }
    submittingRef.current = true
    setSubmitting(true)
    const submittedText = text
    const submittedAttachments = attachments
    const atts = attachments.length ? attachments : undefined
    try {
      if (mode === 'home') {
        const launchCwd = homeWorkspace?.kind === 'ssh' ? homeWorkspace.path : cwd || homeWorkspace?.path || projectRoot || ''
        if (!launchCwd) {
          if (isOrchestratorProject) void loadRemoteProjectDir(projectRoot)
          else void chooseDir()
          return
        }
        const result = await launch({ text: body, cwd: launchCwd, providerId: provider.id, providerAccountId, model: model || provider.models[0], approvalMode: settings.approvalMode, projectId: homeProject, workspace: homeWorkspace, attachments: atts, incognito })
        if (!result.ok) return
      } else {
        await send(body, atts)
        if (thread && useStore.getState().status[thread.id] === 'error') return
      }
      setText((current) => current === submittedText ? '' : current)
      setAttachments((current) => current === submittedAttachments ? [] : current)
    } catch (error) {
      useStore.setState({ notice: { kind: 'error', message: error instanceof Error ? error.message : 'The request could not start; your draft is still here.' } })
    } finally {
      submittingRef.current = false
      setSubmitting(false)
    }
  }
  // Start a long-running mission from a typed objective (replaces the old /goal:
  // create the mission and open the Command Centre to watch it — don't send the
  // text into the chat as a message.
  const startMission = async (objective: string) => {
    if (!provider) { setView({ name: thread?.bot ? 'bots' : 'models' }); return }
    const o = objective.trim()
    if (!o) { taRef.current?.focus(); return }
    try {
      await api.createMission({
        title: o.slice(0, 64),
        objective: o,
        projectId: mode === 'thread' ? (thread?.projectId ?? null) : homeProject,
        cwd: workCwd || cwd || null,
        providerId: provider.id,
        providerAccountId,
        model: model || provider.models[0],
      })
      setText('')
      setView({ name: 'commandCentre' })
    } catch { /* surfaced by the global IPC toast */ }
  }
  const submit = () => {
    const t = text.trim()
    if (!t) return
    if (botBlocked) { setView({ name: 'bots' }); return }
    if (readiness && !readiness.ready) return
    if (t.startsWith('/')) {
      const sp = t.indexOf(' ')
      const name = (sp === -1 ? t : t.slice(0, sp)).slice(1).toLowerCase()
      const arg = sp === -1 ? '' : t.slice(sp + 1).trim()
      // /mission <objective> — start a mission anywhere (not just in a thread).
      if (name === 'mission' && !thread?.bot) { void startMission(arg); return }
      // Built-in thread commands.
      if (mode === 'thread' && thread) {
        if (name === 'compact') { compactThread(thread.id); setText(''); return }
        if (name === 'goal' && !thread.bot) { patchThread(thread.id, arg && arg !== 'clear' ? { goal: arg, goalStatus: 'active', plan: [] } : { goal: undefined, goalStatus: undefined, plan: [] }); setText(''); return }
        if (name === 'computer-use' && !thread.bot) {
          if (arg.toLowerCase() === 'stop') { void api.cancelRun(thread.id); setText(''); return }
          const computer = BUILTIN_CAPABILITIES.find((capability) => capability.id === 'computer')!
          dispatch(`${capabilityInstruction(computer)}\n\n${arg || 'Computer use is on. Wait for my next instruction.'}`)
          return
        }
      }
      // User-defined custom command: expand its prompt, appending typed args.
      const c = commands.find((x) => x.name.toLowerCase() === name)
      if (c) { dispatch(expandSkills(arg ? `${c.prompt}\n\n${arg}` : c.prompt)); return }
    }
    dispatch(expandSkills(t))
  }

  // Stage a follow-up while a run is active: pressing Enter mid-run
  // adds the message to the queue above the composer rather than starting a new run.
  const queueMessage = () => {
    const t = text.trim()
    if (!t || !thread || botBlocked) return
    if (readiness && !readiness.ready) return
    setQueued((q) => [...q, { id: crypto.randomUUID(), text: expandSkills(t), status: 'queued' }])
    setText('')
  }
  // Inject a queued message into the live run now — drives the steer lifecycle:
  // queued → guiding (the cue is being sent) → guided (applied), then it fades out.
  const guideNow = async (id: string, body: string) => {
    if (!thread) return
    setQueued((q) => q.map((x) => x.id === id ? { ...x, status: 'guiding' } : x))
    try {
      await api.cueRun({ threadId: thread.id, userText: body })
      setQueued((q) => q.map((x) => x.id === id ? { ...x, status: 'guided' } : x))
      // Let the "guided" confirmation flash briefly, then remove the item.
      setTimeout(() => setQueued((q) => q.filter((x) => x.id !== id)), 1400)
    } catch {
      // Back to queued so the user can retry; error surfaces via the global toast.
      setQueued((q) => q.map((x) => x.id === id ? { ...x, status: 'queued' } : x))
    }
  }

  const pickFolder = async () => { const d = await api.pickDirectory(); if (d) setFolder(d) }
  // Folder choices for the overflow menu and the "Choose folder" pill; Orchestrator projects browse the host.
  const renderFolderChoices = (close: () => void) => (
    <div>
      {isOrchestratorProject ? (
        <>
          {projectRoot && <MenuItem icon="globe" active={workCwd === projectRoot} onClick={() => { chooseRemoteProjectDir(projectRoot); close() }}>Project root</MenuItem>}
          <MenuItem icon="folder" onClick={() => { void loadRemoteProjectDir(workCwd || projectRoot); setRemoteBrowseOpen(true) }}>Choose subfolder…</MenuItem>
          {remoteBrowseOpen && (
            <div className="mt-1 border-t border-line px-2 pb-2 pt-2">
              <div className="mb-1 flex items-center gap-1.5 text-[13px] text-ink-3">
                <Icon name="globe" size={14} className="text-accent" />
                <span className="min-w-0 flex-1 truncate">{remoteBrowse?.path || projectRoot || 'Loading host folders'}</span>
                {remoteBrowse?.parent && <button className="rounded px-1.5 py-0.5 text-ink-3 hover:bg-ink-5 hover:text-ink" onClick={() => void loadRemoteProjectDir(remoteBrowse.parent)}>Up</button>}
              </div>
              {remoteBrowseError && <div className="mb-1 rounded bg-danger/10 px-2 py-1 text-[13px] text-danger">{remoteBrowseError}</div>}
              <div className="scroll-thin max-h-44 overflow-y-auto">
                {remoteBrowseBusy ? (
                  <div className="px-1 py-2 text-[13px] text-ink-3">Loading folders</div>
                ) : remoteBrowse?.entries.length ? (
                  remoteBrowse.entries.map((entry) => (
                    <button key={entry.path} onClick={() => void loadRemoteProjectDir(entry.path)} className="flex w-full items-center gap-2 rounded-md px-1.5 py-1.5 text-left text-[13px] text-ink-2 hover:bg-ink-5 hover:text-ink">
                      <Icon name="folder" size={14} className="text-ink-3" />
                      <span className="min-w-0 flex-1 truncate">{entry.name}</span>
                      <Icon name="chevronRight" size={12} className="text-ink-4" />
                    </button>
                  ))
                ) : (
                  <div className="px-1 py-2 text-[13px] text-ink-3">No child folders here.</div>
                )}
              </div>
              {remoteBrowse?.path && <Button size="xs" variant="default" icon="check" onClick={() => { chooseRemoteProjectDir(remoteBrowse.path); close() }}>Use this folder</Button>}
            </div>
          )}
        </>
      ) : (
        <MenuItem icon="folder" onClick={() => { close(); void pickFolder() }}>Choose folder…</MenuItem>
      )}
      {workCwd && !isOrchestratorProject && <MenuItem icon="x" onClick={() => { setFolder(''); close() }}>No folder</MenuItem>}
    </div>
  )
  const overflowWorkspace: OverflowWorkspace | undefined = isChat ? undefined : {
    projects,
    projectId: activeProjectId,
    projectLabel: policyProject?.name ?? 'None',
    onSelectProject,
    onAddProject: handleAddProject,
    folderLabel: workCwd ? basename(workCwd) || shortPath(workCwd) : null,
    renderFolder: renderFolderChoices,
    workspace,
    hosts: sshHosts,
    statusMap: sshStatus,
    orchestratorLabel: orchestratorChipLabel,
    onLocal: selectLocalWorkspace,
    onHost: selectHostWorkspace,
    onAddHost: () => setView({ name: 'settings' }),
    branch,
  }

  const onKeyDown = (e: React.KeyboardEvent) => {
    // An IME confirmation must not choose a mention/skill or queue a message.
    if (e.nativeEvent.isComposing || e.keyCode === 229 || e.nativeEvent.keyCode === 229) return
    if (e.key === 'Enter' && e.repeat) return
    if (pick && pickItems.length) {
      if (e.key === 'ArrowDown') { e.preventDefault(); setPIdx((i) => (i + 1) % pickItems.length); return }
      if (e.key === 'ArrowUp') { e.preventDefault(); setPIdx((i) => (i - 1 + pickItems.length) % pickItems.length); return }
      if (e.key === 'Enter' || e.key === 'Tab') { e.preventDefault(); selectPick(pickItems[pIdx].name); return }
      if (e.key === 'Escape') { e.preventDefault(); setPick(null); return }
    }
    const mentionTotal = capabilityItems.length + items.length
    if (mention && mentionTotal) {
      if (e.key === 'ArrowDown') { e.preventDefault(); setMIdx((i) => (i + 1) % mentionTotal); return }
      if (e.key === 'ArrowUp') { e.preventDefault(); setMIdx((i) => (i - 1 + mentionTotal) % mentionTotal); return }
      if (e.key === 'Enter' || e.key === 'Tab') {
        e.preventDefault()
        if (mIdx < capabilityItems.length) selectCapability(capabilityItems[mIdx])
        else selectMention(items[mIdx - capabilityItems.length])
        return
      }
      if (e.key === 'Escape') { e.preventDefault(); setMention(null); return }
    }
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault()
      if (busy && thread) queueMessage()
      else submit()
    }
  }

  return (
    <div className="no-drag w-full">
      {phoneOpen && isChat && <PhoneChatPanel threadId={thread?.id} providerId={provider?.id} localAllowed={!!provider && !isExternalProvider(provider) && !thread?.orchestratorHostId && !isOrchestratorProject && workspace?.kind !== 'ssh'} onClose={() => setPhoneOpen(false)} />}
      <div className="relative">
        {/* @-mention popover */}
        {mention && (capabilityItems.length > 0 || items.length > 0) && (
          <MentionPopover
            anchor={mode}
            capabilityItems={capabilityItems}
            items={items}
            mIdx={mIdx}
            onHoverIndex={setMIdx}
            onSelectCapability={selectCapability}
            onSelectMention={selectMention}
          />
        )}

        {/* $-skill / /-command popover */}
        {pick && pickItems.length > 0 && (
          <SkillCommandPopover
            anchor={mode}
            kind={pick.kind}
            pickItems={pickItems}
            pIdx={pIdx}
            onHoverIndex={setPIdx}
            onSelect={selectPick}
          />
        )}

        {readiness && !readiness.ready && (
          <div className="mb-2 flex items-center gap-2 rounded-xl border border-line bg-card-2 px-3 py-2 text-[13px] text-ink-2">
            <Icon name="warning" size={14} className="shrink-0 text-ink-3" />
            <span className="min-w-0 flex-1">{readiness.message}</span>
            {readiness.reason === 'cli-unauthed' && (
              <button onClick={authorizeCli} disabled={authBusy} className="shrink-0 rounded-lg bg-accent px-2.5 py-1 text-[13px] font-semibold text-accent-ink transition-colors hover:bg-accent-strong disabled:opacity-50">{authBusy ? 'Authorizing…' : 'Authorize'}</button>
            )}
            <button onClick={() => setView(readiness.reason === 'needs-model' ? { name: 'models' } : { name: 'settings' })} className="shrink-0 rounded-lg border border-line px-2.5 py-1 text-[13px] font-medium text-ink-2 transition-colors hover:bg-ink-5 hover:text-ink">{readiness.reason === 'needs-model' ? 'Open Models' : 'Open Settings'}</button>
          </div>
        )}

        {/* External providers plan only (planning-guard.ts): say so before a Code task starts. */}
        {!isChat && provider && isExternalProvider(provider) && (!readiness || readiness.ready) && (
          <div className="mb-2 flex items-start gap-2 rounded-xl border border-line bg-card-2 px-3 py-2 text-[13px] leading-snug text-ink-2">
            <Icon name="shield" size={14} className="mt-px shrink-0 text-ink-3" />
            <span className="min-w-0 flex-1">
              <span className="font-medium text-ink">Planning only.</span> {provider.name || 'This provider'} will plan this task and won't touch your files. It gets a redacted transcript, and you can run its plan with the TensorFold engine.
            </span>
          </div>
        )}

        {/* Steer queue — follow-ups staged above the composer while TensorFold works.
            Each item shows its full text and a status (queued / guiding / guided).
            Guide injects it live; otherwise it auto-sends when the run finishes. */}
        {queued.length > 0 && (
          <SteerQueueList
            queued={queued}
            liveSteering={providerLiveSteering(provider)}
            onGuide={(id, body) => void guideNow(id, body)}
            onRemove={(id) => setQueued((qq) => qq.filter((x) => x.id !== id))}
          />
        )}

        <div
          onDrop={onDrop}
          onDragOver={(e) => { if (Array.from(e.dataTransfer.types).includes('Files')) { e.preventDefault(); setDragOver(true) } }}
          onDragLeave={(e) => { if (e.currentTarget === e.target) setDragOver(false) }}
          className={clsx('group rounded-[24px] border bg-card shadow-card transition-all duration-150 focus-within:border-line-strong', dragOver ? 'border-accent ring-2 ring-accent/30' : 'border-line')}
        >
          {dragOver && <div className="pointer-events-none px-4 pt-2 text-[13px] text-accent">Drop a folder or file to add it as context</div>}
          {attachments.length > 0 && (
            <AttachmentChips items={attachments} onRemove={(aid) => setAttachments((prev) => prev.filter((a) => a.id !== aid))} className="px-4 pt-3" />
          )}
          <input ref={fileInputRef} type="file" multiple className="hidden" onChange={(e) => { if (e.target.files) void addFiles(e.target.files); e.target.value = '' }} />
          <textarea
            ref={taRef}
            data-vontra-composer="true"
            value={text}
            onChange={onChange}
            onKeyDown={onKeyDown}
            onPaste={(e) => { const files = Array.from(e.clipboardData.files); if (files.length) { e.preventDefault(); void addFiles(files) } }}
            rows={mode === 'home' ? 2 : 1}
            placeholder={busy ? (st === 'awaiting' ? 'Approve above, or steer with a follow-up' : 'Press Enter to steer the running task') : mode === 'home' ? 'What should we do next?' : isChat ? 'Message TensorFold' : 'Reply to TensorFold'}
            className={clsx(
              'scroll-thin block max-h-56 w-full resize-none bg-transparent px-4 text-[14px] leading-[22px] text-ink outline-none placeholder:text-ink-3 disabled:opacity-60',
              mode === 'home' ? 'min-h-[68px] pt-4' : 'min-h-[48px] pt-3.5',
            )}
          />
          <div className="@container flex items-center gap-1 px-2 pb-2 pt-1">
            {/* "+": attach a document, reference a file or terminal output, capture a window. */}
            <Menu width={220} placement="top" trigger={(open) => (
              <button aria-label="Attach" title="Attach files or context" className={clsx('flex h-8 w-8 items-center justify-center rounded-full transition-colors', open ? 'bg-ink-5 text-ink' : 'text-ink-3 hover:bg-ink-5 hover:text-ink')}>
                <Icon name="plus" size={16} />
              </button>
            )}>
              {(close) => (
                <div>
                  <MenuItem icon="fileText" onClick={() => { close(); fileInputRef.current?.click() }}>Attach document or image</MenuItem>
                  {!isChat && <MenuItem icon="file" onClick={() => { close(); onAttach() }}>Reference a workspace file</MenuItem>}
                  {!isChat && activeTerminalSession && <MenuItem icon="terminal" onClick={() => { close(); void referenceTerminalOutput() }}>Reference terminal output</MenuItem>}
                  <MenuItem icon="monitor" onClick={() => { close(); if (!controllerBusy) onController() }}>Capture window</MenuItem>
                </div>
              )}
            </Menu>

            {!isChat && (
              <ApprovalMenu approvalMode={settings.approvalMode} onSetApproval={setApproval} planningOnly={!!provider && isExternalProvider(provider)} />
            )}

            {/* A task needs a folder: keep a small way to pick one in sight. */}
            {!isChat && !workCwd && (isOrchestratorProject ? (
              <Menu width={280} placement="top" trigger={(open) => (
                <button className={clsx('inline-flex h-8 items-center gap-1.5 rounded-full px-2.5 text-[13px] transition-colors', open ? 'bg-ink-5 text-ink' : 'text-ink-3 hover:bg-ink-5 hover:text-ink')}>
                  <Icon name="folder" size={16} />Choose folder
                </button>
              )}>
                {(close) => renderFolderChoices(close)}
              </Menu>
            ) : (
              <button onClick={() => void pickFolder()} className="inline-flex h-8 items-center gap-1.5 rounded-full px-2.5 text-[13px] text-ink-3 transition-colors hover:bg-ink-5 hover:text-ink">
                <Icon name="folder" size={16} />Choose folder
              </button>
            ))}

            {mode === 'home' && incognito && (
              <button onClick={() => setIncognito(false)} title="Incognito: this thread is not saved and vanishes on relaunch. Click to turn off." className="inline-flex h-8 items-center gap-1.5 rounded-full px-2.5 text-[13px] text-ink-2 transition-colors hover:bg-ink-5 hover:text-ink">
                <Icon name="incognito" size={16} />Incognito
              </button>
            )}

            {dictating && (
              <button onClick={toggleDictation} title="Stop voice input" className="inline-flex h-8 items-center gap-1.5 rounded-full px-2.5 text-[13px] text-positive transition-colors hover:bg-ink-5">
                <Icon name="mic" size={16} />Listening
              </button>
            )}

            <div className="min-w-0 flex-1" />

            {isChat && thread?.bot?.executionTarget !== 'vm' && <PhoneButton threadId={thread?.id} open={phoneOpen} onClick={() => setPhoneOpen((open) => !open)} />}

            {thread?.bot && <button type="button" title="This chat keeps the bot's saved model and permissions; edit the bot to configure a new chat." onClick={() => setView({ name: 'bots' })} className="inline-flex h-8 max-w-[160px] items-center gap-1.5 rounded-full px-2 text-[12px] text-ink-3 hover:bg-ink-5 hover:text-ink">
              <Icon name="shield" size={13} className="shrink-0" /><span className="truncate">{thread.bot.approvalScope === 'readonly' ? 'Read only' : 'Review actions'} · {thread.bot.executionTarget === 'vm' ? 'Bot VM' : 'This Mac'}</span>
            </button>}

            {provider?.id === 'tensorfold' && !thread?.orchestratorHostId && !isOrchestratorProject && workspace?.kind !== 'ssh' && <LocalEngineControls model={model} />}

            {provider ? <ModelPickerMenu
              provider={provider}
              model={model}
              providerAccountId={providerAccountId}
              localOnly={localOnly}
              policyProject={policyProject}
              pickerProviders={pickerProviders}
              recentModels={recentModels}
              disabled={busy || submitting || !!thread?.bot}
              usageSnapshot={usageSnapshot}
              usageSummary={usageSummary}
              rateLimitStatus={rateLimitStatus}
              onPickModel={pickModel}
              detail={providerEffortOptions(provider, model).some((option) => option.value === settings.reasoningEffort) ? effortDisplayLabel(provider, model, settings.reasoningEffort) : undefined}
              settings={(
                <ModelMenuSettings
                  provider={provider}
                  model={model}
                  reasoningEffort={settings.reasoningEffort}
                  onChangeEffort={(effort) => saveSettings({ ...settings, reasoningEffort: effort })}
                  serviceTier={canConfigureFastMode ? settings.serviceTier : null}
                  onChangeServiceTier={canConfigureFastMode ? (tier) => saveSettings({ ...settings, serviceTier: tier }) : undefined}
                  context={contextUsage}
                />
              )}
            /> : <button type="button" onClick={() => setView({ name: thread?.bot ? 'bots' : 'models' })} className="rounded-lg px-2 py-1 text-[13px] text-warn hover:bg-ink-5">Model unavailable</button>}

            <ComposerOverflowMenu
              workspace={overflowWorkspace}
              voice={{ available: !!settings.voice.enabled && !!settings.voice.sttEnabled, dictating, onToggle: toggleDictation }}
              incognito={mode === 'home' && !isChat ? { on: incognito, onToggle: () => setIncognito((v) => !v) } : undefined}
              onUseSkill={startSkillPick}
              modes={mode !== 'home' && !isChat ? {
                onMission: () => { setText('/mission '); requestAnimationFrame(() => taRef.current?.focus()) },
                onRooms: () => setView({ name: 'projects' }),
                onMonitor: () => setView({ name: 'commandCentre' }),
              } : undefined}
            />

            {busy ? (
              // One control while a run is active: stop. Follow-ups queue with Enter.
              <button onClick={() => {
                queuedRef.current = []
                setQueued([])
                if (thread) void cancel(thread.id)
              }} aria-label="Stop" title="Stop the run and clear queued follow-ups" className="ml-1 flex h-8 w-8 items-center justify-center rounded-full bg-accent text-accent-ink transition-all hover:bg-accent-strong active:scale-95">
                <Icon name="stop" size={16} />
              </button>
            ) : (
              <button onClick={submit} disabled={!text.trim() || botBlocked || submitting} aria-label={submitting ? 'Starting request' : 'Send'} className="ml-1 flex h-8 w-8 items-center justify-center rounded-full bg-accent text-accent-ink transition-all hover:bg-accent-strong active:scale-95 disabled:bg-ink-5 disabled:text-ink-4">
                <Icon name="send" size={16} strokeWidth={2} />
              </button>
            )}
          </div>
        </div>
      </div>

      {dictationError && <div className="mt-1.5 px-3 text-[13px] text-warn">{dictationError}</div>}

      <ControllerModal
        controllerModal={controllerModal}
        controllerPreview={controllerPreview}
        controllerBusy={controllerBusy}
        controllerError={controllerError || null}
        onClose={() => { setControllerPreview(null); setControllerModal(false) }}
        onEnable={enableControllerAndCapture}
        onInsert={insertControllerPreview}
      />
    </div>
  )
}
