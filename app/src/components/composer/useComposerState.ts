import { MESSAGE_QUOTE_EVENT } from '../../lib/message-quotes'
import type { MessageQuoteEvent } from '../../lib/message-quotes'
import { validateMessageQuotes, quotedUserText } from '../../../shared/message-quotes'
import { composerBudget } from '../../lib/composer-budget'
import type { InputHistoryScope } from '../../../shared/input-history'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { currentModelQualification, modelTaskFit, type ModelTaskRequirements } from '../../../shared/model-fit'
import { useModelQualifications } from '../models/qualification/useModelQualifications'
import { boundedComposerEstimate } from '../../lib/composer-estimate'
import { activeEngineModelSettings, engineProfileContextWindow } from '../../../shared/model-profiles'
import { parseThreadMemoryPolicy, threadMemoryUnsupportedReason, inspectThreadMemoryPolicy } from '../../../shared/thread-memory-policy'
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { useStore } from '../../lib/store'
import { api } from '../../lib/ipc'
import type { Readiness } from '../../lib/types'
import { isExternalProvider } from '../../../shared/provider-trust'
import type { Attachment, ApprovalMode, ControllerSnapshot, Project, Provider, Thread, Skill, Command, SshHost, WorkspaceLocation, RemoteDirectoryListResult, TokenUsageSnapshot, ModelRouterSnapshot, ContextPressureSnapshot, PersistedProviderStatus } from '../../lib/types'
import { providerStatusViewFor } from '../../lib/provider-status'
import { appendAttachments, filesToAttachments, replaceAttachment, isAttachmentReady, reviewAttachmentExtraction, acceptAttachmentExtraction } from '../../lib/attachments'
import { useConversationDraft } from '../../lib/use-conversation-draft'
import { desktopDraftPort } from '../../lib/desktop-drafts'
import { composerFileSources } from '../../lib/composer-file-sources'
import { insertTextareaText } from '../../lib/textarea-editing'
import type { ConversationDraftSendToken } from '../../lib/conversation-drafts'
import type { AttachmentTextCapability } from '../../lib/types'
import { allowedProviders, projectDefaultModel, violatesPolicy } from '../../lib/policy'
import { resolveExecutionTarget } from '../../lib/execution-target'
import { resolveProviderModel } from '../../lib/provider-accounts'
import { resolveSurfaceModel, settingsForSurfaceModelPick } from '../../lib/assistant-model'
import { BUILTIN_CAPABILITIES, capabilityInstruction, capabilityToken, type BuiltInCapability } from '../../lib/capabilities'
import { providerUsageSummary } from '../../lib/provider-usage'
import { estimateComposerTokens, modelContextWindowTokens, providerLiveSteering, providerSupportsFastMode, type ContextUsageSummary } from '../../lib/provider-model-metadata'
import { effectiveThreadCwd, effectiveThreadWorkspace } from '../../lib/thread-workspace'
import { terminalSessionKey } from '../../lib/terminal-sessions'
import { loadRecentModels, pushRecentModel, type RecentModel } from '../../lib/recent-models'
import { SteerQueueList, type Steer } from './SteerQueueList'
import { useEngineContextTokens } from '../../lib/engine'
export type ComposerProps = { mode: 'home' | 'thread'; thread?: Thread; seed?: { text: string; n: number } }

export type ComposerState = ReturnType<typeof useComposerState>

const attachmentOperations = new WeakMap<Map<string, File>, Map<string, symbol>>()
function attachmentOperation(sources: Map<string, File>, id: string): { token: symbol; current: () => boolean } {
  let operations = attachmentOperations.get(sources)
  if (!operations) { operations = new Map(); attachmentOperations.set(sources, operations) }
  const token = Symbol('attachment operation')
  operations.set(id, token)
  return { token, current: () => operations.get(id) === token }
}
export function useComposerState({ mode, thread, seed }: ComposerProps) {

  const tx = useAuthoredText()
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

  const [submitting, setSubmitting] = useState(false)
  const submittingRef = useRef(false)
  const [phoneOpen, setPhoneOpen] = useState(false)
  const [usageSnapshot, setUsageSnapshot] = useState<TokenUsageSnapshot | null>(null)
  const [providerStatus, setProviderStatus] = useState<PersistedProviderStatus | null>(null)
  const [modelRouterSnapshot, setModelRouterSnapshot] = useState<ModelRouterSnapshot | null>(null)
  const [contextPressureSnapshot, setContextPressureSnapshot] = useState<ContextPressureSnapshot | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [dragOver, setDragOver] = useState(false)
  const taRef = useRef<HTMLTextAreaElement>(null)
  // Steer queue: messages composed while a run is active are staged
  // here above the composer. Each carries a status lifecycle (queued → guiding →
  // guided) so the user sees the message land and get applied, not a flat row.
  // Inject live ("Guide") or let it auto-send as the next turn when the run ends.
  // The `Steer` type lives in composer/SteerQueueList.tsx (shared with that view).
  const [quoteEditing, setQuoteEditing] = useState(false)
  const quoteEditingRef = useRef(false)
  const [queued, setQueued] = useState<Steer[]>([])
  const queueScopeRef = useRef<string | null>(null)
  const queueEditingRef = useRef<string | null>(null)
  const queuedRef = useRef(queued)
  const queuedDraftTokens = useRef(new Map<string, ConversationDraftSendToken>())
  queuedRef.current = queued
  const updateQueue = (change: (current: Steer[]) => Steer[], expectedScope?: string) => { if (expectedScope !== undefined && queueScopeRef.current !== expectedScope) return; const next = change(queuedRef.current); queuedRef.current = next; setQueued(next) }

  const [cwd, setCwd] = useState(settings.defaultCwd || '')
  // Incognito toggle (home composer): flags the NEXT launched thread as memory-only.
  const [incognito, setIncognito] = useState(false)
  const [newMemoryPolicy, setNewMemoryPolicy] = useState(() => parseThreadMemoryPolicy(undefined))
  const threadMemory = thread ? inspectThreadMemoryPolicy(thread) : null
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
  const draft = useConversationDraft({ surface: mode === 'home' ? 'home' : 'chat', conversationId: thread?.id ?? null, hostId: effectiveHostId, incognito: thread?.incognito ?? incognito }, desktopDraftPort)
  const { text, setText, attachments, setAttachments, setQuotes } = draft
  const quotes = draft.quotes ?? []
  useEffect(() => {
    const receive = (event: Event) => {
      const detail = (event as CustomEvent<MessageQuoteEvent>).detail
      if (mode !== 'thread' || !thread || detail?.threadId !== thread.id || effectiveHostId !== null) return
      try { setQuotes(current => validateMessageQuotes([...current, detail.quote])); taRef.current?.focus() }
      catch (error) { useStore.getState().setNotice({ kind: 'error', message: error instanceof Error ? error.message : tx('This passage could not be quoted.') }) }
    }
    window.addEventListener(MESSAGE_QUOTE_EVENT, receive)
    return () => window.removeEventListener(MESSAGE_QUOTE_EVENT, receive)
  }, [mode, thread?.id, effectiveHostId, setQuotes])
  const activeDraftKey = useRef(draft.scopeKey)
  activeDraftKey.current = draft.scopeKey
  const changeQuoteEditing = useMemo(() => (editing: boolean) => {
    if (activeDraftKey.current === draft.scopeKey) { quoteEditingRef.current = editing; setQuoteEditing(editing) }
  }, [draft.scopeKey])
  const [textCapability, setTextCapability] = useState<AttachmentTextCapability | null>(null)
  const [reviewId, setReviewId] = useState<string | null>(null)
  const reselectId = useRef<string | null>(null)
  const reselectInputRef = useRef<HTMLInputElement>(null)
  const fileSelectionRef = useRef<((files: File[]) => void) | null>(null)
  const reselectSelectionRef = useRef<((files: File[]) => void) | null>(null)
  useEffect(() => { let alive = true; void api.getAttachmentTextCapability().then((value) => { if (alive) setTextCapability(value) }).catch(() => { if (alive) setTextCapability({ available: false, method: 'apple-vision', maxBytes: 20 * 1024 * 1024, maxPages: 8, reason: 'Local text extraction is unavailable in this session.' }) }); return () => { alive = false } }, [])
  useEffect(() => { setReviewId(null); reselectId.current = null }, [draft.scopeKey])
  useEffect(() => () => { dictationStopRef.current?.(); dictationStopRef.current = null }, [draft.scopeKey])
  const sourceFiles = () => composerFileSources(draft.scopeKey)
  const addFiles = async (selection: FileList | File[], replacingId?: string) => {
    const files = Array.from(selection)
    const sources = sourceFiles()
    const scopeKey = draft.scopeKey
    const introduced = new Set<string>()
    const operations = new Map<string, { current: () => boolean }>()
    await filesToAttachments(files, {
      onSource: (id, file) => { const ownedId = replacingId && !introduced.size ? replacingId : id; sources.set(ownedId, file); operations.set(ownedId, attachmentOperation(sources, ownedId)) },
      onUpdate: (row, index) => {
        if (composerFileSources(scopeKey) !== sources) return
        const id = replacingId && index === 0 ? replacingId : row.id
        if (!operations.get(id)?.current()) return
        const owned = { ...row, id }
        if (introduced.has(id) || replacingId && index === 0) { introduced.add(id); setAttachments((current) => replaceAttachment(current, owned)) }
        else { introduced.add(id); setAttachments((current) => appendAttachments(current, [owned])) }
      },
    })
  }
  const chooseFiles = (id?: string) => {
    if (id) {
      reselectSelectionRef.current = (files) => { void addFiles(files, id) }
      reselectInputRef.current?.click()
    } else {
      fileSelectionRef.current = (files) => { void addFiles(files) }
      fileInputRef.current?.click()
    }
  }
  const retryFile = (id: string) => {
    const file = sourceFiles().get(id)
    if (file) void addFiles([file], id)
    else chooseFiles(id)
  }
  const extractFile = async (id: string) => {
    const sources = sourceFiles()
    const scopeKey = draft.scopeKey
    const attachment = attachments.find((item) => item.id === id)
    const file = sourceFiles().get(id)
    if (!attachment) return
    if (!file) { setAttachments((current) => replaceAttachment(current, { ...attachment, status: 'needs-reselection', error: 'Select the original file again before extracting its text.' })); return }
    if (!file.size || file.size > (textCapability?.maxBytes ?? 20 * 1024 * 1024)) { setAttachments((current) => replaceAttachment(current, { ...attachment, status: 'error', error: 'Text extraction accepts nonempty images and PDFs up to 20 MiB.' })); return }
    const operation = attachmentOperation(sources, id)
    const current = () => composerFileSources(scopeKey) === sources && sources.get(id) === file && operation.current()
    setAttachments((current) => replaceAttachment(current, { ...attachment, status: 'processing', error: undefined }))
    try {
      const bytes = new Uint8Array(await file.arrayBuffer())
      if (!current()) return
      const result = await api.extractAttachmentText({ name: file.name, mime: attachment.mime, bytes })
      if (!current()) return
      setAttachments((current) => replaceAttachment(current, reviewAttachmentExtraction(attachment, result)))
      // The row itself opens review after navigation; never open an old conversation's modal.
    } catch (error) { if (current()) setAttachments((rows) => replaceAttachment(rows, { ...attachment, status: 'error', error: error instanceof Error ? error.message : tx("Text extraction failed; try a clearer image.") })) }
  }
  const activeOrchestrator = effectiveHostId ? settings.remoteOrchestrators?.find((host) => host.id === effectiveHostId) ?? null : null
  const isOrchestratorProject = !!activeOrchestrator
  const orchestratorChipLabel = activeOrchestrator
    ? (mode !== 'thread' && executionDecision.source === 'global'
        ? `${activeOrchestrator.name || 'Orchestrator'} · all new work`
        : activeOrchestrator.name || 'Orchestrator')
    : mode !== 'thread' && executionDecision.unavailableReason
      ? 'Orchestrator unavailable'
      : null
  const localOnly = !!policyProject?.localOnly || !!thread?.phoneLocalOnly || !!thread?.googleLocalOnly
  const pickerProviders = allowedProviders(settings.providers, policyProject)
  const [modelPickerScope, setModelPickerScope] = useState<'all' | 'local' | null>(null)
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
  const contextThread = thread ? contextPressureSnapshot?.threads.find((row) => row.id === thread.id && row.providerId === provider?.id && row.model === model) ?? null : null
  // The built-in engine's served window (max_model_len) beats a cached or name-based guess.
  const engineContextTokens = useEngineContextTokens(provider?.source === 'tensorfold', model)
  const selectedModelSettings = provider?.source === 'tensorfold' && model ? activeEngineModelSettings(settings, model) : null
  const contextWindowTokens = selectedModelSettings ? engineProfileContextWindow(selectedModelSettings, engineContextTokens) : engineContextTokens
    ?? contextThread?.contextWindowTokens
    ?? modelContextWindowTokens(provider, model)
    ?? (provider?.kind === 'local' ? settings.localContextLimit : settings.contextLimit)
    ?? 128_000
  const composerEstimate = useMemo(() => boundedComposerEstimate(text, attachments), [text, attachments])
  const { reports: qualifications } = useModelQualifications(api, effectiveHostId ?? null)
  const contextUsedTokens = contextThread?.estimatedTokens ?? composerEstimate.tokens
  const contextUsage: ContextUsageSummary = {
    usedTokens: contextUsedTokens,
    windowTokens: contextWindowTokens,
    workingLimitTokens: contextThread?.workingLimitTokens ?? null,
    percent: contextWindowTokens > 0 ? contextUsedTokens / contextWindowTokens : 0,
    status: contextThread?.status ?? (composerEstimate.partial ? 'Partial draft estimate; some attachment text exceeds the preview limit' : 'Approximate draft estimate'),
  }
  const canConfigureFastMode = provider?.transport === 'codex-cli' && providerSupportsFastMode(provider, model)
  const workspace = mode === 'thread' && thread ? effectiveThreadWorkspace(thread, policyProject) : homeWorkspace
  const projectRoot = policyProject?.workspace?.kind === 'ssh' ? policyProject.workspace.path : (policyProject?.cwd || policyProject?.workspace?.path || '')
  // /workspace belongs to the guest; host Git/file pickers must never resolve a
  // coincidentally matching host directory for an isolated bot chat.
  const workCwd = thread?.bot?.executionTarget === 'vm' ? '' : mode === 'thread' && thread ? effectiveThreadCwd(thread, policyProject) : (workspace?.kind === 'ssh' ? workspace.path : (cwd || homeWorkspace?.path || projectRoot || ''))
  const inputHistoryScope = useMemo<InputHistoryScope>(() => ({
    draft: { surface: mode === 'home' ? 'home' : 'chat', conversationId: thread?.id ?? null, hostId: effectiveHostId, incognito: thread?.incognito ?? incognito },
    projectId: activeProjectId, cwd: workCwd, providerId: provider?.id ?? null, accountId: providerAccountId ?? null,
  }), [mode, thread?.id, effectiveHostId, thread?.incognito, incognito, activeProjectId, workCwd, provider?.id, providerAccountId])
  const inputHistoryKey = JSON.stringify(inputHistoryScope)
  const historyAllowed = !!desktopDraftPort && !!provider && !inputHistoryScope.draft.incognito && !thread?.bot && !thread?.phoneLocalOnly && !thread?.googleLocalOnly && !thread?.workspaceBoundary && effectiveHostId === null && workspace?.kind !== 'ssh'
  const preSendBudget = useMemo(() => composerBudget({
    text, attachments, quotedText: quotes.length ? quotedUserText('', quotes) : '', messages: thread?.apiMessages ?? [], windowTokens: contextWindowTokens,
    windowVerified: !!(selectedModelSettings || engineContextTokens || contextThread || modelContextWindowTokens(provider, model)),
    outputReserve: selectedModelSettings?.maxOutputTokens ?? null,
    snapshotAt: contextPressureSnapshot?.generatedAt, updatedAt: thread?.updatedAt, running: thread?.status === 'running',
  }), [text, attachments, quotes, thread?.apiMessages, contextWindowTokens, selectedModelSettings, engineContextTokens, contextThread, provider, model, contextPressureSnapshot?.generatedAt, thread?.updatedAt, thread?.status])
  const taskRequirements: ModelTaskRequirements = {
    images: attachments.some(attachment => attachment.kind === 'image' && !attachment.text),
    tools: mode === 'thread' && thread?.kind === 'task' && !!workCwd && !!provider && !isExternalProvider(provider),
    localOnly,
    ...(thread?.bot ? { privateContext: 'bot' as const } : thread?.phoneLocalOnly ? { privateContext: 'phone' as const } : thread?.googleLocalOnly ? { privateContext: 'google' as const } : thread?.incognito ? { privateContext: 'incognito' as const } : {}),
  }
  const currentModelFit = modelTaskFit({ requirements: taskRequirements, local: !!provider && !isExternalProvider(provider), report: provider && model ? currentModelQualification(qualifications, { providerId: provider.id, accountId: providerAccountId, model }) : null, toolPolicyAllowed: !!provider && !isExternalProvider(provider) })
  const selectLocalWorkspace = () => { if (mode === 'thread' && thread) setThreadWorkspace(thread.id, null); else setHomeWorkspace(null) }
  const selectHostWorkspace = (host: SshHost, remotePath: string) => {
    const ws: WorkspaceLocation = { kind: 'ssh', hostId: host.id, path: remotePath }
    if (mode === 'thread' && thread) setThreadWorkspace(thread.id, ws)
    else { setHomeWorkspace(ws); setCwd(remotePath) }
    connectSshHost(host.id)
  }

  const st = thread ? status[thread.id] : undefined
  const busy = st != null && ['thinking', 'streaming', 'tool', 'awaiting'].includes(st)
  const modelPickerDisabled = busy || submitting || !!thread?.bot
  useEffect(() => { setModelPickerScope(null) }, [thread?.id, mode, modelPickerDisabled])
  const botPreset = thread?.bot ? settings.agentPresets.find((preset) => preset.id === thread.bot!.presetId) : null
  const recoveryBlocked = !!thread?.runRecovery && (!thread.runRecovery.canStartNewRun && !busy || thread.runRecovery.retry === 'requires_review' && thread.runRecoveryReviewedId !== thread.runRecovery.runId)
  const botBlocked = !!thread?.bot && (!botPreset || !!botPreset.paused || !!botPreset.archivedAt)
  const isChat = mode === 'thread' && thread?.kind === 'chat'

  // Drain the run queue: when a run finishes, fire the next staged message as a
  // fresh turn. Skips items that are guiding/guided (already injected live). The
  // remainder stay queued and drain on the following completion, so follow-ups
  // serialize one run at a time.
  const prevBusy = useRef(false)
  useEffect(() => {
    if (queueScopeRef.current === draft.scopeKey && activeDraftKey.current === draft.scopeKey && prevBusy.current && !busy && !queueEditingRef.current && (!thread?.runRecovery || thread.runRecovery.state === 'completed') && st !== 'error' && !botBlocked && mode === 'thread' && thread && queuedRef.current.length) {
      const next = queuedRef.current.find((x) => x.status === 'queued')
      if (next) {
        const token = queuedDraftTokens.current.get(next.id)
        const ownerScope = draft.scopeKey
        updateQueue((q) => q.map((item) => item.id === next.id ? { ...item, status: 'guiding' } : item))
        void send(next.text, undefined, thread.id).then(async (accepted) => {
          if (accepted && token) {
            try { await draft.clearAccepted(token) }
            catch { useStore.setState({ notice: { kind: 'error', message: tx("The model accepted this follow-up, but its saved draft could not be cleared; retry saving without sending it again.") } }) }
          }
          updateQueue((q) => accepted ? q.filter((item) => item.id !== next.id) : q.map((item) => item.id === next.id ? { ...item, status: 'queued' } : item), ownerScope)
        }).catch(() => updateQueue((q) => q.map((item) => item.id === next.id ? { ...item, status: 'queued' } : item), ownerScope))
      }
    }
    prevBusy.current = busy
  }, [busy, st, botBlocked, thread?.id, mode]) // eslint-disable-line react-hooks/exhaustive-deps
  // Stale queue entries make no sense on a different thread.
  useEffect(() => { quoteEditingRef.current = false; setQuoteEditing(false); queueScopeRef.current = draft.scopeKey; queuedRef.current = []; setQueued([]); queuedDraftTokens.current.clear(); queueEditingRef.current = null; prevBusy.current = false }, [draft.scopeKey])

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
      ? [{ name: 'compact', description: tx("Summarize the conversation to reclaim context") },
         { name: 'goal', description: tx("Set a goal the agent verifies against"), argHint: '<text…>' },
         { name: 'computer-use', description: tx("Let the agent operate Mac apps for this task, with approval per app"), argHint: '<task… | stop>' }]
      : []),
    [mode, thread?.bot, tx],
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
    const ta = taRef.current
    if (ta) setText(insertTextareaText(ta, composerSeed.text))
    useStore.setState({ composerSeed: null })
  }, [composerSeed?.n])
  // File explorer → reference a file in the composer.
  useEffect(() => {
    if (!pendingMention) return
    const ta = taRef.current
    if (ta) setText(insertTextareaText(ta, `@${pendingMention.path} `))
    useStore.setState({ pendingMention: null })
  }, [pendingMention?.n]) // eslint-disable-line

  // One-click recent-model chips. Refresh from storage on mount so other threads'
  // picks propagate.
  const [recentModels, setRecentModels] = useState<RecentModel[]>(() => loadRecentModels())
  useEffect(() => { setRecentModels(loadRecentModels()) }, [thread?.id])
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
  return {
    mode, thread, tx, settings, status, launch, send, cancel,
    saveSettings, setThreadProvider, noteModelChange, compactThread, patchThread, projects,
    terminals, activeTerminalId, readiness, setReadiness, authBusy, setAuthBusy,
    setThreadProject, addProject, pendingMention, composerSeed, sshHosts, sshStatus,
    connectSshHost, setThreadWorkspace, setView, homeWorkspace, setHomeWorkspace, homeProject,
    setHomeProject, submitting, setSubmitting, submittingRef, phoneOpen, setPhoneOpen,
    usageSnapshot, setUsageSnapshot, providerStatus, setProviderStatus, modelRouterSnapshot, setModelRouterSnapshot,
    contextPressureSnapshot, setContextPressureSnapshot, fileInputRef, dragOver, setDragOver, taRef,
    quoteEditing, setQuoteEditing, quoteEditingRef, queued, setQueued, queueScopeRef,
    queueEditingRef, queuedRef, queuedDraftTokens, updateQueue, cwd, setCwd,
    incognito, setIncognito, newMemoryPolicy, setNewMemoryPolicy, threadMemory, activeProjectId,
    policyProject, projectTerminals, activeTerminalSession, executionDecision, effectiveHostId, draft,
    text, setText, attachments, setAttachments, setQuotes, quotes,
    activeDraftKey, changeQuoteEditing, textCapability, setTextCapability, reviewId, setReviewId,
    reselectId, reselectInputRef, fileSelectionRef, reselectSelectionRef, sourceFiles, addFiles,
    chooseFiles, retryFile, extractFile, activeOrchestrator, isOrchestratorProject, orchestratorChipLabel,
    localOnly, pickerProviders, modelPickerScope, setModelPickerScope, remoteBrowseOpen, setRemoteBrowseOpen,
    remoteBrowse, setRemoteBrowse, remoteBrowseBusy, setRemoteBrowseBusy, remoteBrowseError, setRemoteBrowseError,
    codeSurface, rawProviderId, provider, providerAccountId, model, providerKey,
    authorizeCli, usageSummary, rateLimitStatus, contextThread, engineContextTokens, selectedModelSettings,
    contextWindowTokens, composerEstimate, qualifications, contextUsedTokens, contextUsage, canConfigureFastMode,
    workspace, projectRoot, workCwd, inputHistoryScope, inputHistoryKey, historyAllowed,
    preSendBudget, taskRequirements, currentModelFit, selectLocalWorkspace, selectHostWorkspace, st,
    busy, modelPickerDisabled, botPreset, recoveryBlocked, botBlocked, isChat,
    prevBusy, branch, setBranch, files, setFiles, filesCwd,
    mention, setMention, mIdx, setMIdx, skills, setSkills,
    commands, setCommands, pick, setPick, pIdx, setPIdx,
    controllerModal, setControllerModal, controllerPreview, setControllerPreview, controllerBusy, setControllerBusy,
    controllerError, setControllerError, dictating, setDictating, dictationError, setDictationError,
    dictationStopRef, ensureFiles, items, capabilityItems, builtins, pickItems,
    recentModels, setRecentModels, pickModel
  }
}
