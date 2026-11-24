import { editQueuedFollowUp, capturedDraftMatches } from '../../lib/queued-followups'
import { isComposerSendKey } from '../../lib/composer-keyboard'
import { modelFitReason } from '../../lib/preferences/display'
import { useStore } from '../../lib/store'
import { api } from '../../lib/ipc'
import { shortPath, basename } from '../../lib/format'
import { Icon } from '../../lib/icons'
import { Button, Menu, MenuItem } from '../ui'
import type { Attachment, ApprovalMode, ControllerSnapshot, Project, Provider, Thread, Skill, Command, SshHost, WorkspaceLocation, RemoteDirectoryListResult, TokenUsageSnapshot, ModelRouterSnapshot, ContextPressureSnapshot, PersistedProviderStatus } from '../../lib/types'
import { appendAttachments, filesToAttachments, replaceAttachment, isAttachmentReady, reviewAttachmentExtraction, acceptAttachmentExtraction } from '../../lib/attachments'
import { desktopDraftPort } from '../../lib/desktop-drafts'
import { BUILTIN_CAPABILITIES, capabilityInstruction, capabilityToken, type BuiltInCapability } from '../../lib/capabilities'
import { SteerQueueList, type Steer } from './SteerQueueList'
import { TASK_STARTERS, TaskStarters, type TaskStarterId } from './TaskStarters'
import { ComposerOverflowMenu, type OverflowWorkspace } from './ComposerOverflowMenu'
import { type ComposerState } from './useComposerState'
import { type ComposerWorkspace } from './useComposerWorkspace'

export type ComposerSubmit = ReturnType<typeof useComposerSubmit>
export function useComposerSubmit(state: ComposerState & ComposerWorkspace) {
  const {
    tx, settings, status, launch, send, cancel,
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
    recentModels, setRecentModels, pickModel, setApproval, chooseDir, setFolder,
    loadRemoteProjectDir, chooseRemoteProjectDir, insertAtCaret, onSelectProject, handleAddProject, detectMention,
    detectPick, onChange, selectMention, selectCapability, selectPick, referenceTerminalOutput,
    onAttach, startSkillPick, attachControllerSnapshot, onController, enableControllerAndCapture, insertControllerPreview,
    toggleDictation, onDrop
  } = state
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
          useStore.setState({ notice: { kind: 'info', persistent: true, message: tx("Choose a folder for this conversation. Your draft and attachments stay here if you cancel.") } })
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
    if (!t && !attachments.length) return
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
  return {
    expandSkills, dispatch, startMission, submit, queueMessage, editQueued,
    guideNow, pickFolder, renderFolderChoices, overflowWorkspace, onKeyDown, reviewingAttachment,
    startTask, showTaskStarters
  }
}
