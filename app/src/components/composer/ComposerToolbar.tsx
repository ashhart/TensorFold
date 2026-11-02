import type { Dispatch, RefObject, SetStateAction } from 'react'
import clsx from 'clsx'
import { localExecutionSource } from '../../lib/bot-labels'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { useStore } from '../../lib/store'
import { api } from '../../lib/ipc'
import type { Settings, Thread, ApprovalMode } from '../../lib/types'
import { isExternalProvider } from '../../../shared/provider-trust'
import { parseThreadMemoryPolicy, threadMemoryUnsupportedReason, inspectThreadMemoryPolicy } from '../../../shared/thread-memory-policy'
import { mergeThreadRecord } from '../../lib/store/shared/merge'
import { effortDisplayLabel, providerEffortOptions } from '../../lib/provider-model-metadata'
import { Icon } from '../../lib/icons'
import { Menu, MenuItem } from '../ui'
import { ApprovalMenu } from './ApprovalMenu'
import { LocalEngineControls } from './LocalEngineControls'
import { PhoneButton } from '../phone/PhoneButton'
import { ModelPickerMenu } from './ModelPickerMenu'
import { ModelMenuSettings } from './ModelMenuSettings'
import { ComposerOverflowMenu } from './ComposerOverflowMenu'
import type { useConversationDraft } from '../../lib/use-conversation-draft'
import type { ComposerWorkspace } from './useComposerWorkspace'
import type { ComposerModel } from './useComposerModel'
import type { ComposerInput } from './useComposerInput'

interface ComposerToolbarProps {
  mode: 'home' | 'thread'
  thread?: Thread
  settings: Settings
  draft: ReturnType<typeof useConversationDraft>
  work: ComposerWorkspace
  models: ComposerModel
  input: ComposerInput
  busy: boolean
  submitting: boolean
  incognito: boolean
  setIncognito: Dispatch<SetStateAction<boolean>>
  phoneOpen: boolean
  setPhoneOpen: Dispatch<SetStateAction<boolean>>
  newMemoryPolicy: ReturnType<typeof parseThreadMemoryPolicy>
  setNewMemoryPolicy: Dispatch<SetStateAction<ReturnType<typeof parseThreadMemoryPolicy>>>
  chooseFiles: (id?: string) => void
  taRef: RefObject<HTMLTextAreaElement>
}

export function ComposerToolbar({ mode, thread, settings, draft, work, models, input, busy, submitting,
  incognito, setIncognito, phoneOpen, setPhoneOpen, newMemoryPolicy, setNewMemoryPolicy, chooseFiles, taRef }: ComposerToolbarProps) {
  const tx = useAuthoredText()
  const saveSettings = useStore((s) => s.saveSettings)
  const setView = useStore((s) => s.setView)
  const isChat = mode === 'thread' && thread?.kind === 'chat'
  const threadMemory = thread ? inspectThreadMemoryPolicy(thread) : null
  const { setText } = draft
  const { workCwd, workspace, activeTerminalSession, isOrchestratorProject, renderFolderChoices,
    pickFolder, policyProject, effectiveHostId, overflowWorkspace } = work
  const { provider, model, providerAccountId, localOnly, qualifications, taskRequirements, pickerProviders,
    recentModels, usageSnapshot, usageSummary, rateLimitStatus, pickModel, canConfigureFastMode, contextUsage } = models
  const { onAttach, referenceTerminalOutput, controllerBusy, onController, dictating, toggleDictation, startSkillPick } = input
  const setApproval = (m: ApprovalMode) => saveSettings({ ...settings, approvalMode: m })
  return (
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
      {mode !== 'home' && !isChat && !workCwd && (isOrchestratorProject ? (
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

      {provider?.id === 'tensorfold' && !thread?.orchestratorHostId && !isOrchestratorProject && workspace?.kind !== 'ssh' && <LocalEngineControls model={model} compact />}

      <div className="ml-auto min-w-[140px] max-w-[360px] flex-[1_1_180px]">
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
          showPlanningBadge={isChat}
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
  )
}
