import { DraftMessageQuotes } from './composer/MessageQuotes'
import { editQueuedFollowUp, capturedDraftMatches } from '../lib/queued-followups'
import { MESSAGE_QUOTE_EVENT } from '../lib/message-quotes'
import type { MessageQuoteEvent } from '../lib/message-quotes'
import { validateMessageQuotes, quotedUserText } from '../../shared/message-quotes'
import { isComposerSendKey } from '../lib/composer-keyboard'
import { composerBudget } from '../lib/composer-budget'
import { ContextBudget } from './composer/ContextBudget'
import { PromptHistory } from './composer/PromptHistory'
import type { InputHistoryScope } from '../../shared/input-history'
import { currentIntlLocale } from '../lib/preferences/current-locale'
import { modelFitReason } from '../lib/preferences/display'
import { OriginalDiagnostic } from './OriginalDiagnostic'
import { useAuthoredText } from '../lib/preferences/useAuthoredText'
import { currentModelQualification, modelTaskFit, type ModelTaskRequirements } from '../../shared/model-fit'
import { useModelQualifications } from './models/qualification/useModelQualifications'
import { boundedComposerEstimate } from '../lib/composer-estimate'
import { DraftSaveIndicator } from './composer/DraftSaveIndicator'
import { activeEngineModelSettings, engineProfileContextWindow } from '../../shared/model-profiles'
import { mergeThreadRecord } from '../lib/store/shared/merge'
import { parseThreadMemoryPolicy, threadMemoryUnsupportedReason, inspectThreadMemoryPolicy } from '../../shared/thread-memory-policy'
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
import { AttachmentChips, AttachmentTextReview } from './AttachmentChips'
import { appendAttachments, filesToAttachments, replaceAttachment, isAttachmentReady, reviewAttachmentExtraction, acceptAttachmentExtraction } from '../lib/attachments'
import { useConversationDraft } from '../lib/use-conversation-draft'
import { desktopDraftPort } from '../lib/desktop-drafts'
import { composerFileSources } from '../lib/composer-file-sources'
import { insertTextareaText } from '../lib/textarea-editing'
import type { ConversationDraftSendToken } from '../lib/conversation-drafts'
import type { AttachmentTextCapability } from '../lib/types'
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
import { TASK_STARTERS, TaskStarters, type TaskStarterId } from './composer/TaskStarters'
import { useEngineContextTokens } from '../lib/engine'
import { ModelMenuSettings } from './composer/ModelMenuSettings'
import { ComposerOverflowMenu, type OverflowWorkspace } from './composer/ComposerOverflowMenu'
import { effortDisplayLabel, providerEffortOptions } from '../lib/provider-model-metadata'

const attachmentOperations = new WeakMap<Map<string, File>, Map<string, symbol>>()
function attachmentOperation(sources: Map<string, File>, id: string): { token: symbol; current: () => boolean } {
  let operations = attachmentOperations.get(sources)
  if (!operations) { operations = new Map(); attachmentOperations.set(sources, operations) }
  const token = Symbol('attachment operation')
  operations.set(id, token)
  return { token, current: () => operations.get(id) === token }
}

function formatControllerSnapshot(snapshot: ControllerSnapshot): string {
  const lines = [
    '<controller_window permission="explicit-user-attach">',
    `Captured: ${new Date(snapshot.capturedAt).toLocaleString(currentIntlLocale())}`,
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
      if (!result.ok) throw new Error(result.error || tx("Could not list folders on the Orchestrator."))
      setRemoteBrowse(result)
    } catch (e: any) {
      setRemoteBrowseError(e?.message ?? tx("Could not list folders on the Orchestrator."))
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
    if (!ta) return
    const caret = ta.selectionStart
    const pad = caret > 0 && !/\s/.test(text[caret - 1]) ? '\n\n' : ''
    setText(insertTextareaText(ta, pad + insert))
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
          const ok = confirm(tx("This folder is not initialized as a Git repo. Initialize Git now so TensorFold can use isolated room worktrees?"))
          if (ok) await api.initGitRepository(d)
        }
      } catch (e: any) {
        // With the soft-suggestion off (#37), git inspection is best-effort: never
        // block project creation behind a confirm() dialog.
        if (settings.showGitInitPrompt !== false) {
          const ok = confirm(tx("{value0}\n\nCreate the project anyway?", { value0: e?.message ?? tx("Could not inspect Git for this folder.") }))
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
    setText(insertTextareaText(ta, insert, mention.start, Math.max(caret, ta.selectionEnd)))
    setMention(null)
    requestAnimationFrame(() => { const pos = mention.start + insert.length; ta.focus(); ta.setSelectionRange(pos, pos) })
  }
  const selectCapability = (capability: BuiltInCapability) => {
    if (!mention) return
    const ta = taRef.current!
    const caret = ta.selectionStart ?? text.length
    const insert = `${capabilityToken(capability)}\n${capabilityInstruction(capability)}\n\n`
    setText(insertTextareaText(ta, insert, mention.start, Math.max(caret, ta.selectionEnd)))
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
    setText(insertTextareaText(ta, insert, pick.start, Math.max(caret, ta.selectionEnd)))
    setPick(null)
    requestAnimationFrame(() => { const pos = pick.start + insert.length; ta.focus(); ta.setSelectionRange(pos, pos) })
  }
  // Reference terminal output (VST-039): pull the active terminal's backlog and insert a
  // bounded, ANSI-stripped reference at the caret so the agent can act on what just ran.
  const referenceTerminalOutput = async () => {
    if (!activeTerminalSession) return
    const originalEditor = taRef.current
    let backlog = ''
    // `terminalBacklog` is host-local. Never probe it for an Orchestrator session: a same-id local
    // PTY would leak the wrong output, and legacy remote rows intentionally fail closed.
    if (activeTerminalSession.hostLocation !== 'orchestrator') {
      try { backlog = await api.terminalBacklog(activeTerminalSession.id) } catch { /* keep empty → honest "no output" note */ }
    }
    const reference = buildTerminalReference(activeTerminalSession.title || 'terminal', backlog)
    const ta = taRef.current
    if (!ta || ta !== originalEditor) return
    const caret = ta?.selectionStart ?? text.length
    const pad = caret > 0 && text[caret - 1] && !/\s/.test(text[caret - 1]) ? '\n' : ''
    setText(insertTextareaText(ta, pad + reference))
    requestAnimationFrame(() => { const pos = caret + pad.length + reference.length; ta?.focus(); ta?.setSelectionRange(pos, pos) })
  }
  const onAttach = () => {
    const ta = taRef.current
    if (!ta) return
    const caret = ta.selectionStart ?? text.length
    const pad = caret > 0 && !/\s/.test(text[caret - 1]) ? ' ' : ''
    const insert = pad + '@'
    setText(insertTextareaText(ta, insert))
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
    setText(insertTextareaText(ta, insert))
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
      setDictationError(tx("Speech-to-text is not available in this runtime."))
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

  // Files are read as explicit attachments; directories retain the workspace-reference flow.
  const onDrop = (e: React.DragEvent) => {
    const items = Array.from(e.dataTransfer.items || [])
    const files = Array.from(e.dataTransfer.files || [])
    if (!items.length && !files.length) return
    e.preventDefault(); setDragOver(false)
    const refs: string[] = []
    const selectedFiles: File[] = []
    const n = Math.max(items.length, files.length)
    for (let i = 0; i < n; i++) {
      const entry = (items[i] as any)?.webkitGetAsEntry?.()
      const f = files[i] as any
      const name: string | undefined = entry?.name || f?.name
      if (!name) continue
      const abs: string | undefined = f?.path // Electron exposes the absolute path
      const isDir = !!entry?.isDirectory
      if (!isDir && f) { selectedFiles.push(f); continue }
      let ref = abs || name
      if (abs && workCwd && abs.startsWith(workCwd)) ref = abs.slice(workCwd.length).replace(/^[/\\]/, '') || name
      refs.push(`@${ref}${isDir ? '/' : ''}`)
    }
    if (selectedFiles.length) void addFiles(selectedFiles)
    if (refs.length) setText((t) => `${t}${t && !t.endsWith(' ') ? ' ' : ''}${refs.join(' ')} `)
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
    if (submittingRef.current || quoteEditingRef.current) return
    if (!draft.ready || attachments.some((attachment) => !isAttachmentReady(attachment))) {
      useStore.setState({ notice: { kind: 'error', message: tx("Finish reviewing, retrying or removing the files that need attention before sending; your draft is preserved.") } }); return
    }
    if (!provider) { useStore.setState({ notice: { kind: 'error', message: tx("This model is unavailable. Check Models or this bot’s configuration; your draft is still here.") } }); return }
    submittingRef.current = true
    setSubmitting(true)
    const submitted = draft.capture()
    const atts = submitted.attachments.length ? submitted.attachments : undefined
    try {
      if (mode === 'home') {
        const launchCwd = homeWorkspace?.kind === 'ssh' ? homeWorkspace.path : cwd || homeWorkspace?.path || projectRoot || ''
        if (!launchCwd) {
          if (isOrchestratorProject) void loadRemoteProjectDir(projectRoot)
          else void chooseDir()
          return
        }
        const result = await launch({ text: body, cwd: launchCwd, providerId: provider.id, providerAccountId, model: model || provider.models[0], approvalMode: settings.approvalMode, projectId: homeProject, workspace: homeWorkspace, attachments: atts, incognito, memoryPolicy: newMemoryPolicy })
        if (!result.ok) return
      } else {
        if (!thread || !await send(body, atts, thread.id, submitted.quotes)) return
      }
      await draft.clearAccepted(submitted)
    } catch (error) {
      useStore.setState({ notice: { kind: 'error', message: error instanceof Error ? error.message : tx("The request could not start; your draft is still here.") } })
    } finally {
      submittingRef.current = false
      setSubmitting(false)
    }
  }
  // Start a long-running mission from a typed objective (replaces the old /goal:
  // create the mission and open the Command Centre to watch it — don't send the
  // text into the chat as a message.
  const startMission = async (objective: string) => {
    if (attachments.length || quotes.length) { useStore.setState({ notice: { kind: 'error', message: tx("Mission commands do not attach files; remove them or send them in this conversation first. Your draft is preserved.") } }); return }
    if (!provider) { setView({ name: thread?.bot ? 'bots' : 'models' }); return }
    const o = objective.trim()
    if (!o) { taRef.current?.focus(); return }
    const submitted = draft.capture()
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
      await draft.clearAccepted(submitted)
      setView({ name: 'commandCentre' })
    } catch { /* surfaced by the global IPC toast */ }
  }
  const submit = () => {
    if (quoteEditingRef.current) return
    const t = text.trim()
    if (!t) return
    if (!draft.ready || attachments.some((attachment) => !isAttachmentReady(attachment))) { useStore.setState({ notice: { kind: 'error', message: tx("Some files need review, retry or reselection; finish them or remove them before sending.") } }); return }
    if (currentModelFit.disabledReason) { useStore.getState().setNotice({ kind: 'error', message: modelFitReason(currentModelFit.disabledReason, tx) }); return }
    if (recoveryBlocked) { useStore.getState().setNotice({ kind: 'error', message: tx("Review the previous run outcome before sending another request.") }); return }
    if (botBlocked) { setView({ name: 'bots' }); return }
    if (readiness && !readiness.ready) return
    if (t.startsWith('/')) {
      if (quotes.length) { useStore.getState().setNotice({ kind: 'error', message: tx('Send quoted passages as a conversation message; remove them before using a command.') }); return }
      const sp = t.indexOf(' ')
      const name = (sp === -1 ? t : t.slice(0, sp)).slice(1).toLowerCase()
      const arg = sp === -1 ? '' : t.slice(sp + 1).trim()
      if (attachments.length && ['mission', 'compact', 'goal'].includes(name)) { useStore.setState({ notice: { kind: 'error', message: tx("This command does not accept files; send or remove your attachments first.") } }); return }
      // /mission <objective> — start a mission anywhere (not just in a thread).
      if (name === 'mission' && !thread?.bot) { void startMission(arg); return }
      // Built-in thread commands.
      if (mode === 'thread' && thread) {
        if (name === 'compact') { const submitted = draft.capture(); void compactThread(thread.id).then(() => draft.clearAccepted(submitted)).catch(() => {}); return }
        if (name === 'goal' && !thread.bot) { const submitted = draft.capture(); void patchThread(thread.id, arg && arg !== 'clear' ? { goal: arg, goalStatus: 'active', plan: [] } : { goal: undefined, goalStatus: undefined, plan: [] }).then(() => draft.clearAccepted(submitted)).catch(() => {}); return }
        if (name === 'computer-use' && !thread.bot) {
          if (arg.toLowerCase() === 'stop') { const submitted = draft.capture(); void api.cancelRun(thread.id).then(() => draft.clearAccepted(submitted)).catch(() => {}); return }
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
    if (!t || !thread || botBlocked || queueEditingRef.current) return
    if (quotes.length) { useStore.getState().setNotice({ kind: 'error', message: tx('Quotes stay in your draft until the current run finishes.') }); return }
    if (attachments.length) { useStore.setState({ notice: { kind: 'error', message: tx("Files stay in your draft while the model is running; wait for it to finish before sending them.") } }); return }
    if (readiness && !readiness.ready) return
    const waiting = queuedRef.current.find((item) => item.status === 'queued')
    if (waiting && waiting.text === expandSkills(t)) return
    const id = waiting?.id ?? crypto.randomUUID()
    queueScopeRef.current = draft.scopeKey
    queuedDraftTokens.current.set(id, draft.capture())
    const nextQueue: Steer[] = waiting ? queuedRef.current.map(item => item.id === id ? { ...item, text: expandSkills(t) } : item) : [...queuedRef.current, { id, text: expandSkills(t), status: 'queued' }]
    queuedRef.current = nextQueue; setQueued(nextQueue)
    useStore.setState({ notice: { kind: 'info', message: tx("Follow-up queued for this session; its draft stays saved until the model accepts it.") } })
  }
  // Inject a queued message into the live run now — drives the steer lifecycle:
  // queued → guiding (the cue is being sent) → guided (applied), then it fades out.
  const editQueued = (id: string, expectedText: string, body: string): boolean => {
    if (!thread || queueScopeRef.current !== draft.scopeKey || activeDraftKey.current !== draft.scopeKey || queueEditingRef.current !== id) return false
    const result = editQueuedFollowUp(queuedRef.current, { id, expectedText, text: body })
    if (!result.ok) return false
    const oldToken = queuedDraftTokens.current.get(id), current = draft.capture()
    if (capturedDraftMatches(oldToken, current)) { setText(body); queuedDraftTokens.current.set(id, draft.capture()) }
    else queuedDraftTokens.current.delete(id)
    queuedRef.current = result.queued; setQueued(result.queued)
    return true
  }
  const guideNow = async (id: string, body: string) => {
    const current = queuedRef.current.find(row => row.id === id)
    if (!thread || queueScopeRef.current !== draft.scopeKey || queueEditingRef.current || !current || current.status !== 'queued' || current.text !== body || recoveryBlocked || botBlocked || activeDraftKey.current !== draft.scopeKey) return
    queuedRef.current = queuedRef.current.map(row => row.id === id ? { ...row, status: 'guiding' } : row)
    updateQueue((q) => q.map((x) => x.id === id ? { ...x, status: 'guiding' } : x))
    const submitted = queuedDraftTokens.current.get(id)
    const ownerScope = draft.scopeKey
    try {
      const result = !busy ? { queued: await send(body, undefined, thread.id) } : thread.orchestratorHostId
        ? await api.guideRemoteOrchestratorThread(thread.orchestratorHostId, { threadId: thread.id, userText: body })
        : await api.cueRun({ threadId: thread.id, userText: body })
      if (!result.queued) {
        updateQueue((q) => q.map((x) => x.id === id ? { ...x, status: 'queued' } : x), ownerScope)
        useStore.setState({ notice: { kind: 'info', message: tx("The run has finished; this follow-up remains in your saved draft so you can send it as the next turn.") } })
        return
      }
      if (submitted) {
        try { await draft.clearAccepted(submitted) }
        catch { useStore.setState({ notice: { kind: 'error', message: tx("The model accepted this follow-up, but clearing its saved draft failed; retry saving the draft without sending the same follow-up again.") } }) }
      }
      updateQueue((q) => q.map((x) => x.id === id ? { ...x, status: 'guided' } : x), ownerScope)
      // Let the "guided" confirmation flash briefly, then remove the item.
      setTimeout(() => updateQueue((q) => q.filter((x) => x.id !== id), ownerScope), 1400)
    } catch {
      // Back to queued so the user can retry; error surfaces via the global toast.
      updateQueue((q) => q.map((x) => x.id === id ? { ...x, status: 'queued' } : x), ownerScope)
    }
  }

  const pickFolder = async () => { const d = await api.pickDirectory(); if (d) setFolder(d) }
  // Folder choices for the overflow menu and the "Choose folder" pill; Orchestrator projects browse the host.
  const renderFolderChoices = (close: () => void) => (
    <div>
      {isOrchestratorProject ? (
        <>
          {projectRoot && <MenuItem icon="globe" active={workCwd === projectRoot} onClick={() => { chooseRemoteProjectDir(projectRoot); close() }}>{tx("Project root")}</MenuItem>}
          <MenuItem icon="folder" onClick={() => { void loadRemoteProjectDir(workCwd || projectRoot); setRemoteBrowseOpen(true) }}>{tx("Choose subfolder…")}</MenuItem>
          {remoteBrowseOpen && (
            <div className="mt-1 border-t border-line px-2 pb-2 pt-2">
              <div className="mb-1 flex items-center gap-1.5 text-[13px] text-ink-3">
                <Icon name="globe" size={14} className="text-accent" />
                <span className="min-w-0 flex-1 truncate">{remoteBrowse?.path || projectRoot || tx("Loading host folders")}</span>
                {remoteBrowse?.parent && <button className="rounded px-1.5 py-0.5 text-ink-3 hover:bg-ink-5 hover:text-ink" onClick={() => void loadRemoteProjectDir(remoteBrowse.parent)}>{tx("Up")}</button>}
              </div>
              {remoteBrowseError && <div className="mb-1 rounded bg-danger/10 px-2 py-1 text-[13px] text-danger">{remoteBrowseError}</div>}
              <div className="scroll-thin max-h-44 overflow-y-auto">
                {remoteBrowseBusy ? (
                  <div className="px-1 py-2 text-[13px] text-ink-3">{tx("Loading folders")}</div>
                ) : remoteBrowse?.entries.length ? (
                  remoteBrowse.entries.map((entry) => (
                    <button key={entry.path} onClick={() => void loadRemoteProjectDir(entry.path)} className="flex w-full items-center gap-2 rounded-md px-1.5 py-1.5 text-left text-[13px] text-ink-2 hover:bg-ink-5 hover:text-ink">
                      <Icon name="folder" size={14} className="text-ink-3" />
                      <span className="min-w-0 flex-1 truncate">{entry.name}</span>
                      <Icon name="chevronRight" size={12} className="text-ink-4" />
                    </button>
                  ))
                ) : (
                  <div className="px-1 py-2 text-[13px] text-ink-3">{tx("No child folders here.")}</div>
                )}
              </div>
              {remoteBrowse?.path && <Button size="xs" variant="default" icon="check" onClick={() => { chooseRemoteProjectDir(remoteBrowse.path); close() }}>{tx("Use this folder")}</Button>}
            </div>
          )}
        </>
      ) : (
        <MenuItem icon="folder" onClick={() => { close(); void pickFolder() }}>{tx("Choose folder…")}</MenuItem>
      )}
      {workCwd && !isOrchestratorProject && <MenuItem icon="x" onClick={() => { setFolder(''); close() }}>{tx("No folder")}</MenuItem>}
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
      if ((e.key === 'Enter' && !e.shiftKey && !e.altKey) || e.key === 'Tab') { e.preventDefault(); selectPick(pickItems[pIdx].name); return }
      if (e.key === 'Escape') { e.preventDefault(); setPick(null); return }
    }
    const mentionTotal = capabilityItems.length + items.length
    if (mention && mentionTotal) {
      if (e.key === 'ArrowDown') { e.preventDefault(); setMIdx((i) => (i + 1) % mentionTotal); return }
      if (e.key === 'ArrowUp') { e.preventDefault(); setMIdx((i) => (i - 1 + mentionTotal) % mentionTotal); return }
      if ((e.key === 'Enter' && !e.shiftKey && !e.altKey) || e.key === 'Tab') {
        e.preventDefault()
        if (mIdx < capabilityItems.length) selectCapability(capabilityItems[mIdx])
        else selectMention(items[mIdx - capabilityItems.length])
        return
      }
      if (e.key === 'Escape') { e.preventDefault(); setMention(null); return }
    }
    if (isComposerSendKey(e, settings.uiPreferences?.sendKey)) {
      e.preventDefault()
      if (busy && thread) queueMessage()
      else submit()
    }
  }

  const reviewingAttachment = attachments.find((item) => item.id === reviewId && item.extraction)
  const startTask = (id: TaskStarterId) => {
    if (busy || submittingRef.current || !draft.ready || desktopDraftPort && draft.error || activeDraftKey.current !== draft.scopeKey) return
    const current = draft.capture()
    if (current.text.length || current.attachments.length) return
    const starter = TASK_STARTERS.find((item) => item.id === id)
    if (!starter) return
    setText(starter.prompt)
    taRef.current?.focus()
    if (starter.attachFile) chooseFiles()
  }
  const showTaskStarters = draft.ready && !(desktopDraftPort && draft.error) && !busy && !submitting && text.length === 0 && attachments.length === 0 && (mode === 'home' || !!thread && thread.timeline.length === 0 && thread.apiMessages.length === 0)
  return (
    <div className="no-drag w-full">
      {showTaskStarters && <TaskStarters onSelect={startTask} />}
      {!!quotes.length && thread && <DraftMessageQuotes key={draft.scopeKey} threadId={thread.id} onEditingChange={changeQuoteEditing} quotes={quotes} disabled={submitting} onChange={(id, value) => setQuotes(current => current.map(quote => quote.id === id ? { ...quote, text: value } : quote))} onRemove={id => setQuotes(current => current.filter(quote => quote.id !== id))} />}
      {reviewingAttachment && <AttachmentTextReview key={`${draft.scopeKey}:${reviewingAttachment.id}`} attachment={reviewingAttachment} onCancel={() => setReviewId(null)} onAccept={(edited) => { setAttachments((current) => replaceAttachment(current, acceptAttachmentExtraction(reviewingAttachment, edited))); setReviewId(null) }} />}
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
              <button onClick={authorizeCli} disabled={authBusy} className="shrink-0 rounded-lg bg-accent px-2.5 py-1 text-[13px] font-semibold text-accent-ink transition-colors hover:bg-accent-strong disabled:opacity-50">{authBusy ? tx("Authorizing…") : tx("Authorize")}</button>
            )}
            <button onClick={() => setView(readiness.reason === 'needs-model' ? { name: 'models' } : { name: 'settings' })} className="shrink-0 rounded-lg border border-line px-2.5 py-1 text-[13px] font-medium text-ink-2 transition-colors hover:bg-ink-5 hover:text-ink">{readiness.reason === 'needs-model' ? tx("Open Models") : tx("Open Settings")}</button>
          </div>
        )}

        {/* External providers plan only (planning-guard.ts): say so before a Code task starts. */}
        {!isChat && provider && isExternalProvider(provider) && (!readiness || readiness.ready) && (
          <div className="mb-2 flex items-start gap-2 rounded-xl border border-line bg-card-2 px-3 py-2 text-[13px] leading-snug text-ink-2">
            <Icon name="shield" size={14} className="mt-px shrink-0 text-ink-3" />
            <span className="min-w-0 flex-1">
              <span className="font-medium text-ink">{tx("Planning only.")}</span> {provider.name || tx("This provider")} {tx("will plan this task and won't touch your files. It gets a redacted transcript, and you can run its plan with the TensorFold engine.")}</span>
          </div>
        )}

        {/* Steer queue — follow-ups staged above the composer while TensorFold works.
            Each item shows its full text and a status (queued / guiding / guided).
            Guide injects it live; otherwise it auto-sends when the run finishes. */}
        {queued.length > 0 && (
          <SteerQueueList
            key={draft.scopeKey}
            running={busy}
            queued={queued}
            onEdit={editQueued}
            onEditingChange={(id) => { queueEditingRef.current = id }}
            liveSteering={providerLiveSteering(provider)}
            onGuide={(id, body) => void guideNow(id, body)}
            onRemove={(id) => { queuedRef.current = queuedRef.current.filter(row => row.id !== id); setQueued(queuedRef.current); queuedDraftTokens.current.delete(id); if (queueEditingRef.current === id) queueEditingRef.current = null }}
          />
        )}

        <div
          onDrop={onDrop}
          onDragOver={(e) => { if (Array.from(e.dataTransfer.types).includes('Files')) { e.preventDefault(); setDragOver(true) } }}
          onDragLeave={(e) => { if (e.currentTarget === e.target) setDragOver(false) }}
          className={clsx('group rounded-[24px] border bg-card shadow-card transition-all duration-150 focus-within:border-line-strong', dragOver ? 'border-accent ring-2 ring-accent/30' : 'border-line')}
        >
          {dragOver && <div className="pointer-events-none px-4 pt-2 text-[13px] text-accent">{tx("Drop a folder or file to add it as context")}</div>}
          {attachments.length > 0 && (
            <AttachmentChips items={attachments} onRemove={(aid) => { sourceFiles().delete(aid); setAttachments((prev) => prev.filter((a) => a.id !== aid)) }} onRetry={retryFile} onReselect={chooseFiles} onExtractText={(id) => void extractFile(id)} onReviewText={setReviewId} extractionAvailable={textCapability?.available === true} extractionUnavailableReason={textCapability?.reason ?? 'Checking local text extraction…'} className="px-4 pt-3" />
          )}
          <input ref={fileInputRef} type="file" multiple className="hidden" onChange={(e) => { const files = Array.from(e.target.files ?? []); fileSelectionRef.current?.(files); fileSelectionRef.current = null; e.target.value = '' }} />
          <input ref={reselectInputRef} type="file" className="hidden" onChange={(e) => { const files = Array.from(e.target.files ?? []); reselectSelectionRef.current?.(files); reselectSelectionRef.current = null; e.target.value = '' }} />
          {draft.error && <div role="alert" className="px-4 pt-2 text-[12px] text-danger">{tx("Draft not saved: ")}{<OriginalDiagnostic>{draft.error}</OriginalDiagnostic>} <button type="button" onClick={() => void draft.retry()} className="underline">{tx("Retry save")}</button></div>}
          {currentModelFit.disabledReason && <p role="status" className="px-4 pt-2 text-[12px] text-warn">{modelFitReason(currentModelFit.disabledReason, tx)}</p>}
          {!currentModelFit.disabledReason && taskRequirements.images && currentModelFit.unknown.includes('vision') && <p role="status" className="px-4 pt-2 text-[12px] text-ink-3">{tx("Image support has not been checked for this model. Check its capabilities in Models, use local text extraction, or send to try it.")}</p>}
          <textarea
            key={draft.scopeKey}
            ref={taRef}
            aria-label={tx("Message composer")}
            disabled={!draft.ready}
            data-vontra-composer="true"
            value={text}
            onChange={onChange}
            onKeyDown={onKeyDown}
            onPaste={(e) => { const files = Array.from(e.clipboardData.files); if (files.length) { e.preventDefault(); void addFiles(files) } }}
            rows={mode === 'home' ? 2 : 1}
            placeholder={busy ? (st === 'awaiting' ? tx("Approve above, or steer with a follow-up") : (settings.uiPreferences?.sendKey === 'modified-enter' ? tx("Press Command or Control + Enter to queue a follow-up") : tx("Press Enter to steer the running task"))) : mode === 'home' ? tx("What should we do next?") : isChat ? tx("Message TensorFold") : tx("Reply to TensorFold")}
            className={clsx(
              'scroll-thin block max-h-56 w-full resize-none bg-transparent px-4 text-[14px] leading-[22px] text-ink outline-none placeholder:text-ink-3 disabled:opacity-60',
              mode === 'home' ? 'min-h-[68px] pt-4' : 'min-h-[48px] pt-3.5',
            )}
          />
          <ContextBudget budget={preSendBudget} />
          {/* Keep Send/Stop anchored while the secondary controls can wrap on narrow panes. */}
          <div className="@container flex items-end gap-1 px-2 pb-2 pt-1">
            <div className="flex min-w-0 flex-1 flex-wrap items-center gap-1">
              {/* "+": attach a document, reference a file or terminal output, capture a window. */}
              <Menu width={220} placement="top" trigger={(open) => (
                <button aria-label={tx("Attach")} title={tx("Attach files or context")} className={clsx('flex h-8 w-8 items-center justify-center rounded-full transition-colors', open ? 'bg-ink-5 text-ink' : 'text-ink-3 hover:bg-ink-5 hover:text-ink')}>
                  <Icon name="plus" size={16} />
                </button>
              )}>
                {(close) => (
                  <div>
                    <MenuItem icon="fileText" onClick={() => { close(); chooseFiles() }}>{tx("Attach document or image")}</MenuItem>
                    {!isChat && <MenuItem icon="file" onClick={() => { close(); onAttach() }}>{tx("Reference a workspace file")}</MenuItem>}
                    {!isChat && activeTerminalSession && <MenuItem icon="terminal" onClick={() => { close(); void referenceTerminalOutput() }}>{tx("Reference terminal output")}</MenuItem>}
                    <MenuItem icon="monitor" onClick={() => { close(); if (!controllerBusy) onController() }}>{tx("Capture window")}</MenuItem>
                  </div>
                )}
              </Menu>

              {!isChat && (
                <ApprovalMenu approvalMode={settings.approvalMode} onSetApproval={setApproval} planningOnly={!!provider && isExternalProvider(provider)} />
              )}

              {/* A task needs a folder: keep a small way to pick one in sight. */}
              {!isChat && !workCwd && (isOrchestratorProject ? (
                <Menu width={280} placement="top" trigger={(open) => (
                  <button aria-label={tx("Choose folder")} title={tx("Choose folder")} className={clsx('inline-flex h-8 shrink-0 items-center gap-1.5 rounded-full px-2.5 text-[13px] transition-colors', open ? 'bg-ink-5 text-ink' : 'text-ink-3 hover:bg-ink-5 hover:text-ink')}>
                    <Icon name="folder" size={16} className="shrink-0" /><span className="hidden @lg:inline">{tx("Choose folder")}</span>
                  </button>
                )}>
                  {(close) => renderFolderChoices(close)}
                </Menu>
              ) : (
                <button onClick={() => void pickFolder()} aria-label={tx("Choose folder")} title={tx("Choose folder")} className="inline-flex h-8 shrink-0 items-center gap-1.5 rounded-full px-2.5 text-[13px] text-ink-3 transition-colors hover:bg-ink-5 hover:text-ink">
                  <Icon name="folder" size={16} className="shrink-0" /><span className="hidden @lg:inline">{tx("Choose folder")}</span>
                </button>
              ))}

              {mode === 'home' && incognito && (
                <button onClick={() => setIncognito(false)} title={tx("Incognito: this thread is not saved and vanishes on relaunch. Click to turn off.")} className="inline-flex h-8 items-center gap-1.5 rounded-full px-2.5 text-[13px] text-ink-2 transition-colors hover:bg-ink-5 hover:text-ink">
                  <Icon name="incognito" size={16} />{tx("Incognito")}</button>
              )}

              {dictating && (
                <button onClick={toggleDictation} title={tx("Stop voice input")} className="inline-flex h-8 items-center gap-1.5 rounded-full px-2.5 text-[13px] text-positive transition-colors hover:bg-ink-5">
                  <Icon name="mic" size={16} />{tx("Listening")}</button>
              )}

              {isChat && thread?.bot?.executionTarget !== 'vm' && <PhoneButton threadId={thread?.id} open={phoneOpen} onClick={() => setPhoneOpen((open) => !open)} />}

              {thread?.bot && <button type="button" title={tx("This chat keeps the bot's saved model and permissions; edit the bot to configure a new chat.")} onClick={() => setView({ name: 'bots' })} className="inline-flex h-8 max-w-[160px] items-center gap-1.5 rounded-full px-2 text-[12px] text-ink-3 hover:bg-ink-5 hover:text-ink">
                <Icon name="shield" size={13} className="shrink-0" /><span className="truncate">{thread.bot.approvalScope === 'readonly' ? tx("Read only") : tx("Review actions")} · {thread.bot.executionTarget === 'vm' ? tx("Bot VM") : tx("This Mac")}</span>
              </button>}

              {provider?.id === 'tensorfold' && !thread?.orchestratorHostId && !isOrchestratorProject && workspace?.kind !== 'ssh' && <LocalEngineControls model={model} />}

              <div className={clsx('ml-auto max-w-[280px] flex-[1_1_120px]', provider && isExternalProvider(provider) ? 'min-w-[164px] @md:min-w-[224px]' : 'min-w-[120px]')}>
                {provider ? <ModelPickerMenu
                  provider={provider}
                  model={model}
                  providerAccountId={providerAccountId}
                  ownerId={effectiveHostId}
                  localOnly={localOnly}
                  policyProject={policyProject}
                  qualifications={qualifications}
                  requirements={taskRequirements}
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
                /> : <button type="button" onClick={() => setView({ name: thread?.bot ? 'bots' : 'models' })} className="rounded-lg px-2 py-1 text-[13px] text-warn hover:bg-ink-5">{tx("Model unavailable")}</button>}
              </div>

              <ComposerOverflowMenu
                memory={thread?.bot ? undefined : {
                  policy: threadMemory ? threadMemory.policy : newMemoryPolicy,
                  problem: threadMemory?.problem,
                  revision: threadMemory?.revision ?? undefined,
                  incognito: thread ? thread.incognito === true : incognito,
                  busy: busy || submitting,
                  unsupported: threadMemory?.revision === null ? 'This damaged revision cannot be repaired through the memory menu; start a new chat.' : threadMemoryUnsupportedReason(thread ?? { orchestratorHostId: effectiveHostId }, provider),
                  isNew: !thread,
                  onSave: async (policy) => {
                    if (!thread) { setNewMemoryPolicy(policy); return }
                    const updated = await api.setThreadMemoryPolicy(thread.id, policy, thread.memoryPolicyRevision ?? 0)
                    useStore.setState((state) => ({ threads: state.threads.map((entry) => entry.id === updated.id ? mergeThreadRecord(entry, updated) : entry) }))
                  },
                }}
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

            </div>
            {historyAllowed && <PromptHistory key={inputHistoryKey} scope={inputHistoryScope} port={api} disabled={!draft.ready || submitting || busy} onInsert={prompt => {
              if (activeDraftKey.current !== draft.scopeKey || !taRef.current) return
              const area = taRef.current
              const addition = `${area.value && !area.value.endsWith('\n') ? '\n\n' : ''}${prompt}`
              setText(insertTextareaText(area, addition, area.value.length, area.value.length))
            }} />}
            <DraftSaveIndicator ready={draft.ready} saving={draft.saving || draft.dirty} privateDraft={!!(thread?.incognito || incognito)} sessionOnly={!desktopDraftPort} problem={!!draft.error} label={draft.error ? tx("Draft not saved: {value0}", { value0: draft.error }) : undefined} />
            {busy ? (
              // One control while a run is active: stop. Follow-ups queue with Enter.
              <button onClick={() => {
                queueEditingRef.current = null
                queuedDraftTokens.current.clear()
                queuedRef.current = []
                setQueued([])
                if (thread) void cancel(thread.id)
              }} aria-label={tx("Stop")} title={tx("Stop the run and clear queued follow-ups")} className="ml-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-accent text-accent-ink transition-all hover:bg-accent-strong active:scale-95">
                <Icon name="stop" size={16} />
              </button>
            ) : (
              <button onClick={submit} disabled={!text.trim() || quoteEditing || botBlocked || recoveryBlocked || !!currentModelFit.disabledReason || submitting} aria-label={submitting ? tx("Starting request") : tx("Send")} title={settings.uiPreferences?.sendKey === 'modified-enter' ? tx("Command or Control + Enter to send") : tx("Enter to send")} className="ml-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-accent text-accent-ink transition-all hover:bg-accent-strong active:scale-95 disabled:bg-ink-5 disabled:text-ink-4">
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
