import { currentIntlLocale } from '../../lib/preferences/current-locale'
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { api } from '../../lib/ipc'
import { useStore } from '../../lib/store'
import { isPickerGuidance, PICKER_GUIDANCE_MESSAGE } from '../../lib/store/slices/uiSlice'
import { COMPOSER_INSERT_EVENT } from '../../lib/browser-preview'
import type { Attachment, ApprovalMode, ControllerSnapshot, Project, Provider, Thread, Skill, Command, SshHost, WorkspaceLocation, RemoteDirectoryListResult, TokenUsageSnapshot, ModelRouterSnapshot, ContextPressureSnapshot, PersistedProviderStatus } from '../../lib/types'
import { buildTerminalReference } from '../../lib/terminal-context'
import { insertTextareaText } from '../../lib/textarea-editing'
import { BUILTIN_CAPABILITIES, capabilityInstruction, capabilityToken, type BuiltInCapability } from '../../lib/capabilities'
import { dictationSupported, startDictation } from '../../lib/voice'
import { type ComposerState } from './useComposerState'

export type ComposerWorkspace = ReturnType<typeof useComposerWorkspace>

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

export function useComposerWorkspace(state: ComposerState) {
  const {
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
  } = state
  const setApproval = (m: ApprovalMode) => saveSettings({ ...settings, approvalMode: m })
  const chooseDir = async () => {
    const d = await api.pickDirectory()
    if (!d) return
    setCwd(d)
    setHomeWorkspace(null)
    if (isPickerGuidance(useStore.getState().notice, tx(PICKER_GUIDANCE_MESSAGE))) useStore.setState({ notice: null })
  }
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
  return {
    setApproval, chooseDir, setFolder, loadRemoteProjectDir, chooseRemoteProjectDir, insertAtCaret,
    onSelectProject, handleAddProject, detectMention, detectPick, onChange, selectMention,
    selectCapability, selectPick, referenceTerminalOutput, onAttach, startSkillPick, attachControllerSnapshot,
    onController, enableControllerAndCapture, insertControllerPreview, toggleDictation, onDrop
  }
}
