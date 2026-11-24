import { localExecutionSource } from '../../lib/bot-labels'
import { DraftMessageQuotes } from './MessageQuotes'
import { ContextBudget } from './ContextBudget'
import { PromptHistory } from './PromptHistory'
import { modelFitReason } from '../../lib/preferences/display'
import { OriginalDiagnostic } from '../OriginalDiagnostic'
import { DraftSaveIndicator } from './DraftSaveIndicator'
import { mergeThreadRecord } from '../../lib/store/shared/merge'
import { parseThreadMemoryPolicy, threadMemoryUnsupportedReason, inspectThreadMemoryPolicy } from '../../../shared/thread-memory-policy'
import clsx from 'clsx'
import { useStore } from '../../lib/store'
import { api } from '../../lib/ipc'
import { Icon } from '../../lib/icons'
import { isExternalProvider } from '../../../shared/provider-trust'
import { Button, Menu, MenuItem } from '../ui'
import type { Attachment, ApprovalMode, ControllerSnapshot, Project, Provider, Thread, Skill, Command, SshHost, WorkspaceLocation, RemoteDirectoryListResult, TokenUsageSnapshot, ModelRouterSnapshot, ContextPressureSnapshot, PersistedProviderStatus } from '../../lib/types'
import { AttachmentChips, AttachmentTextReview } from '../AttachmentChips'
import { appendAttachments, filesToAttachments, replaceAttachment, isAttachmentReady, reviewAttachmentExtraction, acceptAttachmentExtraction } from '../../lib/attachments'
import { desktopDraftPort } from '../../lib/desktop-drafts'
import { insertTextareaText } from '../../lib/textarea-editing'
import { estimateComposerTokens, modelContextWindowTokens, providerLiveSteering, providerSupportsFastMode, type ContextUsageSummary } from '../../lib/provider-model-metadata'
import { MentionPopover } from './MentionPopover'
import { SkillCommandPopover } from './SkillCommandPopover'
import { SteerQueueList, type Steer } from './SteerQueueList'
import { ControllerModal } from './ControllerModal'
import { ApprovalMenu } from './ApprovalMenu'
import { LocalEngineControls } from './LocalEngineControls'
import { PhoneButton } from '../phone/PhoneButton'
import { PhoneChatPanel } from '../phone/PhoneChatPanel'
import { ModelPickerMenu } from './ModelPickerMenu'
import { TASK_STARTERS, TaskStarters, type TaskStarterId } from './TaskStarters'
import { ModelMenuSettings } from './ModelMenuSettings'
import { ComposerOverflowMenu, type OverflowWorkspace } from './ComposerOverflowMenu'
import { effortDisplayLabel, providerEffortOptions } from '../../lib/provider-model-metadata'
import { type ComposerState } from './useComposerState'
import { type ComposerWorkspace } from './useComposerWorkspace'
import { type ComposerSubmit } from './useComposerSubmit'

export function ComposerView(state: ComposerState & ComposerWorkspace & ComposerSubmit) {
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
    toggleDictation, onDrop, expandSkills, dispatch, startMission, submit,
    queueMessage, editQueued, guideNow, pickFolder, renderFolderChoices, overflowWorkspace,
    onKeyDown, reviewingAttachment, startTask, showTaskStarters
  } = state
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
                <ApprovalMenu approvalMode={settings.approvalMode} onSetApproval={setApproval} planningOnly={!!provider && isExternalProvider(provider)} onChooseModel={() => setModelPickerScope('local')} modelPickerDisabled={modelPickerDisabled} modelPickerOpen={!modelPickerDisabled && modelPickerScope !== null} />
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
                <Icon name="shield" size={13} className="shrink-0" /><span className="truncate">{thread.bot.approvalScope === 'readonly' ? tx("Read only") : tx("Review actions")} · {thread.bot.executionTarget === 'vm' ? tx("Bot VM") : tx(localExecutionSource(api.platform))}</span>
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
                  disabled={modelPickerDisabled}
                  open={!modelPickerDisabled && modelPickerScope !== null}
                  onOpenChange={(open) => setModelPickerScope(open ? 'all' : null)}
                  initialScope={modelPickerScope ?? 'all'}
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
              <button onClick={submit} disabled={(!text.trim() && !attachments.length) || quoteEditing || botBlocked || recoveryBlocked || !!currentModelFit.disabledReason || submitting} aria-label={submitting ? tx("Starting request") : tx("Send")} title={settings.uiPreferences?.sendKey === 'modified-enter' ? tx("Command or Control + Enter to send") : tx("Enter to send")} className="ml-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-accent text-accent-ink transition-all hover:bg-accent-strong active:scale-95 disabled:bg-ink-5 disabled:text-ink-4">
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
