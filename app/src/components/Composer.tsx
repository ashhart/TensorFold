import { DraftMessageQuotes } from './composer/MessageQuotes'
import { editQueuedFollowUp, capturedDraftMatches } from '../lib/queued-followups'
import { MESSAGE_QUOTE_EVENT, type MessageQuoteEvent } from '../lib/message-quotes'
import { validateMessageQuotes } from '../../shared/message-quotes'
import { isComposerSendKey } from '../lib/composer-keyboard'
import { ContextBudget } from './composer/ContextBudget'
import { PromptHistory } from './composer/PromptHistory'
import type { InputHistoryScope } from '../../shared/input-history'
import { modelFitReason } from '../lib/preferences/display'
import { DraftPersistenceNotice, isSessionOnlyDraftNotice } from './composer/DraftPersistenceNotice'
import { ComposerWorkspaceBar } from './composer/ComposerWorkspaceBar'
import { useAuthoredText } from '../lib/preferences/useAuthoredText'
import { DraftSaveIndicator } from './composer/DraftSaveIndicator'
import { parseThreadMemoryPolicy } from '../../shared/thread-memory-policy'
import { useEffect, useMemo, useRef, useState } from 'react'
import clsx from 'clsx'
import { useStore } from '../lib/store'
import { api } from '../lib/ipc'
import { Icon } from '../lib/icons'
import { isExternalProvider } from '../../shared/provider-trust'
import type { Thread, AttachmentTextCapability } from '../lib/types'
import { AttachmentChips, AttachmentTextReview } from './AttachmentChips'
import { appendAttachments, filesToAttachments, replaceAttachment, isAttachmentReady, reviewAttachmentExtraction, acceptAttachmentExtraction } from '../lib/attachments'
import { useConversationDraft } from '../lib/use-conversation-draft'
import { desktopDraftPort } from '../lib/desktop-drafts'
import { composerFileSources } from '../lib/composer-file-sources'
import { insertTextareaText } from '../lib/textarea-editing'
import type { ConversationDraftSendToken } from '../lib/conversation-drafts'
import { BUILTIN_CAPABILITIES, capabilityInstruction } from '../lib/capabilities'
import { providerLiveSteering } from '../lib/provider-model-metadata'
import { MentionPopover } from './composer/MentionPopover'
import { SkillCommandPopover } from './composer/SkillCommandPopover'
import { SteerQueueList, type Steer } from './composer/SteerQueueList'
import { ControllerModal } from './composer/ControllerModal'
import { PhoneChatPanel } from './phone/PhoneChatPanel'
import { TASK_STARTERS, TaskStarters, type TaskStarterId } from './composer/TaskStarters'
import { useComposerWorkspace } from './composer/useComposerWorkspace'
import { useComposerModel } from './composer/useComposerModel'
import { useComposerInput } from './composer/useComposerInput'
import { ComposerToolbar } from './composer/ComposerToolbar'

const attachmentOperations = new WeakMap<Map<string, File>, Map<string, symbol>>()
function attachmentOperation(sources: Map<string, File>, id: string): { token: symbol; current: () => boolean } {
  let operations = attachmentOperations.get(sources)
  if (!operations) { operations = new Map(); attachmentOperations.set(sources, operations) }
  const token = Symbol('attachment operation')
  operations.set(id, token)
  return { token, current: () => operations.get(id) === token }
}

export function Composer({ mode, thread, seed }: { mode: 'home' | 'thread'; thread?: Thread; seed?: { text: string; n: number } }) {
  const tx = useAuthoredText()
  const settings = useStore((s) => s.settings)!
  const workspaceState = useComposerWorkspace({ mode, thread, settings })
  const { homeWorkspace, homeProject, cwd, activeProjectId, effectiveHostId, isOrchestratorProject,
    workspace, projectRoot, workCwd, chooseDir, loadRemoteProjectDir } = workspaceState
  const status = useStore((s) => s.status)
  const launch = useStore((s) => s.launch)
  const send = useStore((s) => s.send)
  const cancel = useStore((s) => s.cancel)
  const compactThread = useStore((s) => s.compactThread)
  const patchThread = useStore((s) => s.patchThread)
  const setView = useStore((s) => s.setView)

  const [submitting, setSubmitting] = useState(false)
  const submittingRef = useRef(false)
  const [phoneOpen, setPhoneOpen] = useState(false)
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

  // Incognito toggle (home composer): flags the NEXT launched thread as memory-only.
  const [incognito, setIncognito] = useState(false)
  const [newMemoryPolicy, setNewMemoryPolicy] = useState(() => parseThreadMemoryPolicy(undefined))
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

  const modelState = useComposerModel({ mode, thread, settings, draft, work: workspaceState })
  const { provider, providerAccountId, model, readiness, authBusy, authorizeCli, taskRequirements,
    currentModelFit, preSendBudget } = modelState
  const input = useComposerInput({ mode, thread, seed, settings, draft, work: workspaceState, taRef, addFiles, setDragOver })
  const { skills, commands, mention, setMention, mIdx, setMIdx, items, capabilityItems,
    pick, setPick, pIdx, setPIdx, pickItems, selectMention, selectCapability, selectPick, onChange, onDrop,
    controllerModal, setControllerModal, controllerPreview, setControllerPreview, controllerBusy, controllerError,
    enableControllerAndCapture, insertControllerPreview, dictationError } = input

  const inputHistoryScope = useMemo<InputHistoryScope>(() => ({
    draft: { surface: mode === 'home' ? 'home' : 'chat', conversationId: thread?.id ?? null, hostId: effectiveHostId, incognito: thread?.incognito ?? incognito },
    projectId: activeProjectId, cwd: workCwd, providerId: provider?.id ?? null, accountId: providerAccountId ?? null,
  }), [mode, thread?.id, effectiveHostId, thread?.incognito, incognito, activeProjectId, workCwd, provider?.id, providerAccountId])
  const inputHistoryKey = JSON.stringify(inputHistoryScope)
  const historyAllowed = !!desktopDraftPort && !!provider && !inputHistoryScope.draft.incognito && !thread?.bot && !thread?.phoneLocalOnly && !thread?.googleLocalOnly && !thread?.workspaceBoundary && effectiveHostId === null && workspace?.kind !== 'ssh'

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
  // A new scope drops stale queue entries and owns its current busy state for the next completion.
  useEffect(() => { quoteEditingRef.current = false; setQuoteEditing(false); queueScopeRef.current = draft.scopeKey; queuedRef.current = []; setQueued([]); queuedDraftTokens.current.clear(); queueEditingRef.current = null; prevBusy.current = busy }, [draft.scopeKey]) // eslint-disable-line react-hooks/exhaustive-deps

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
      {mode === 'home' && <ComposerWorkspaceBar work={workspaceState} disabled={busy || submitting} />}
      {showTaskStarters && mode !== 'home' && <TaskStarters onSelect={startTask} />}
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
          <DraftPersistenceNotice error={draft.error} sessionOnly={!desktopDraftPort} onRetry={() => void draft.retry()} />
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
            <ComposerToolbar mode={mode} thread={thread} settings={settings} draft={draft}
              work={workspaceState} models={modelState} input={input} busy={busy} submitting={submitting}
              incognito={incognito} setIncognito={setIncognito} phoneOpen={phoneOpen} setPhoneOpen={setPhoneOpen}
              newMemoryPolicy={newMemoryPolicy} setNewMemoryPolicy={setNewMemoryPolicy} chooseFiles={chooseFiles} taRef={taRef} />
            {historyAllowed && <PromptHistory key={inputHistoryKey} scope={inputHistoryScope} port={api} disabled={!draft.ready || submitting || busy} onInsert={prompt => {
              if (activeDraftKey.current !== draft.scopeKey || !taRef.current) return
              const area = taRef.current
              const addition = `${area.value && !area.value.endsWith('\n') ? '\n\n' : ''}${prompt}`
              setText(insertTextareaText(area, addition, area.value.length, area.value.length))
            }} />}
            <DraftSaveIndicator ready={draft.ready} saving={draft.saving || draft.dirty} privateDraft={!!(thread?.incognito || incognito)} sessionOnly={!desktopDraftPort} problem={!!draft.error && !isSessionOnlyDraftNotice(draft.error, !desktopDraftPort)} label={draft.error && !isSessionOnlyDraftNotice(draft.error, !desktopDraftPort) ? tx("Draft not saved: {value0}", { value0: draft.error }) : undefined} />
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
      {showTaskStarters && mode === 'home' && <div className="mt-3"><TaskStarters onSelect={startTask} /></div>}

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
