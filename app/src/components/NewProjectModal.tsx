import { OriginalDiagnostic } from './OriginalDiagnostic'
import { useAuthoredText } from '../lib/preferences/useAuthoredText'
import { useEffect, useMemo, useRef, useState } from 'react'
import clsx from 'clsx'
import { useStore } from '../lib/store'
import { resolveExecutionTarget } from '../lib/execution-target'
import { api } from '../lib/ipc'
import { shortPath } from '../lib/format'
import { Icon } from '../lib/icons'
import { Button, Field, Input, Segmented } from './ui'
import { TAG_COLORS, TAG_PRESETS, addTag, collectTags, normalizeTag, tagTone, toggleTag } from '../lib/project-tags'
import type { GitRepositoryState, ProjectExecutionTarget, RemoteDirectoryListResult, RemoteOrchestratorSnapshotResult } from '../lib/types'

export function NewProjectModal() {
  const tx = useAuthoredText()
  const open = useStore((s) => s.newProjectOpen)
  const modalIntent = useStore((s) => s.newProjectIntent)
  const close = useStore((s) => s.closeNewProject)
  const newProject = useStore((s) => s.newProject)
  const projects = useStore((s) => s.projects)
  const settings = useStore((s) => s.settings)!
  const saveSettings = useStore((s) => s.saveSettings)
  // Soft-suggestion (#37): when off, never block creation behind a Git prompt.
  const gitPromptEnabled = settings.showGitInitPrompt !== false

  const [name, setName] = useState('')
  const [cwd, setCwd] = useState('')
  const [busy, setBusy] = useState(false)
  const creating = useRef(false)
  const [error, setError] = useState('')
  const [gitState, setGitState] = useState<GitRepositoryState | null>(null)
  const [gitBusy, setGitBusy] = useState(false)
  const [gitInitBusy, setGitInitBusy] = useState(false)
  const [gitPromptDismissed, setGitPromptDismissed] = useState(false)
  const [selectedTags, setSelectedTags] = useState<string[]>([])
  const [tagAdding, setTagAdding] = useState(false)
  const [tagDraft, setTagDraft] = useState('')
  const [tagColor, setTagColor] = useState('neutral')
  const [pendingTagColors, setPendingTagColors] = useState<Record<string, string>>({})
  // #51 — pick where this project runs up front: this Mac, or a paired
  // Orchestrator (e.g. the Mini) while you keep working from the local folder.
  const [executionTarget, setExecutionTarget] = useState<ProjectExecutionTarget>('local')
  const [orchestratorId, setOrchestratorId] = useState<string | null>(null)
  const [remoteBrowseOpen, setRemoteBrowseOpen] = useState(false)
  const [remoteBrowse, setRemoteBrowse] = useState<RemoteDirectoryListResult | null>(null)
  const [remoteBrowseBusy, setRemoteBrowseBusy] = useState(false)
  const [remoteBrowseError, setRemoteBrowseError] = useState('')
  const [remoteCheck, setRemoteCheck] = useState<RemoteOrchestratorSnapshotResult | null>(null)
  const [remoteCheckBusy, setRemoteCheckBusy] = useState(false)
  const orchestrators = (settings.remoteOrchestrators ?? []).filter((h) => h.enabled)
  const selectedOrchestrator = orchestrators.find((host) => host.id === orchestratorId) ?? orchestrators[0] ?? null
  const tagColors = { ...(settings.tagColors ?? {}), ...pendingTagColors }
  const knownTags = useMemo(() => {
    const seen = new Set<string>()
    return [...TAG_PRESETS, ...collectTags(projects)].filter((tag) => {
      if (seen.has(tag)) return false
      seen.add(tag)
      return true
    })
  }, [projects])

  useEffect(() => {
    if (!open) return
    setName('')
    setCwd('')
    setBusy(creating.current)
    setError('')
    setGitState(null)
    setGitBusy(false)
    setGitInitBusy(false)
    setGitPromptDismissed(false)
    setSelectedTags([])
    setTagAdding(false)
    setTagDraft('')
    setTagColor('neutral')
    setPendingTagColors({})
    // Global orchestrator mode (VST-041): new projects preselect the default host so the
    // modal matches where work will actually run. Still freely switchable to Local.
    const globalDecision = resolveExecutionTarget(settings, null)
    setExecutionTarget(globalDecision.hostId ? 'orchestrator' : 'local')
    setOrchestratorId(globalDecision.hostId)
    setRemoteBrowseOpen(false)
    setRemoteBrowse(null)
    setRemoteBrowseBusy(false)
    setRemoteBrowseError('')
    setRemoteCheck(null)
    setRemoteCheckBusy(false)
    // Only a fresh opening/intent resets the form: its target is a snapshot of that request —
    // a settings save mid-open must not yank a choice the user already made here.
  }, [open, modalIntent]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!open) return
    setGitState(null)
    setGitPromptDismissed(false)
    setError('')
    if (executionTarget !== 'local' || !cwd) {
      setGitBusy(false)
      return
    }
    let live = true
    setGitBusy(true)
    api.inspectGitRepository(cwd)
      .then((state) => { if (live) setGitState(state) })
      .catch((e) => {
        if (!live) return
        setGitState({
          cwd,
          exists: false,
          insideWorkTree: false,
          topLevel: null,
          branch: null,
          hasCommits: false,
          message: e?.message ?? 'Could not inspect Git repository.',
        })
      })
      .finally(() => { if (live) setGitBusy(false) })
    return () => { live = false }
  }, [open, cwd, executionTarget])

  useEffect(() => {
    if (!open || executionTarget !== 'orchestrator' || !selectedOrchestrator) {
      setRemoteCheck(null)
      setRemoteCheckBusy(false)
      return
    }
    let live = true
    setRemoteCheck(null)
    setRemoteCheckBusy(true)
    api.getRemoteSnapshot(selectedOrchestrator.id)
      .then((result) => { if (live) setRemoteCheck(result) })
      .catch((e) => {
        if (!live) return
        setRemoteCheck({
          ok: false,
          hostId: selectedOrchestrator.id,
          name: selectedOrchestrator.name,
          serverUrl: selectedOrchestrator.serverUrl,
          error: e?.message ?? 'Could not reach this Orchestrator.',
        })
      })
      .finally(() => { if (live) setRemoteCheckBusy(false) })
    return () => { live = false }
  }, [open, executionTarget, selectedOrchestrator?.id])

  if (!open) return null

  const refreshSelectedOrchestrator = async () => {
    if (!selectedOrchestrator || remoteCheckBusy) return
    setRemoteCheckBusy(true)
    setRemoteCheck(null)
    try {
      setRemoteCheck(await api.getRemoteSnapshot(selectedOrchestrator.id))
    } catch (e: any) {
      setRemoteCheck({
        ok: false,
        hostId: selectedOrchestrator.id,
        name: selectedOrchestrator.name,
        serverUrl: selectedOrchestrator.serverUrl,
        error: e?.message ?? 'Could not reach this Orchestrator.',
      })
    } finally {
      setRemoteCheckBusy(false)
    }
  }

  const loadRemoteDir = async (dir?: string | null) => {
    if (!selectedOrchestrator) {
      setError(tx("Choose an Orchestrator before browsing host folders."))
      return
    }
    setRemoteBrowseOpen(true)
    setRemoteBrowseBusy(true)
    setRemoteBrowseError('')
    setError('')
    try {
      const result = await api.listRemoteOrchestratorDir(selectedOrchestrator.id, dir ?? '')
      if (!result.ok) throw new Error(result.error || tx("Could not list folders on the Orchestrator."))
      setRemoteBrowse(result)
      if (dir) {
        setCwd(result.path)
        if (!name) setName(result.path.split('/').filter(Boolean).pop() || 'Remote project')
      }
    } catch (e: any) {
      setRemoteBrowseError(e?.message ?? tx("Could not list folders on the Orchestrator."))
    } finally {
      setRemoteBrowseBusy(false)
    }
  }

  const browse = async () => {
    if (executionTarget === 'orchestrator') {
      await loadRemoteDir(cwd || remoteBrowse?.path || '')
      return
    }
    const dir = await api.pickDirectory()
    if (!dir) return
    setCwd(dir)
    if (!name) setName(dir.split('/').pop() || '')
  }

  const chooseRemoteDir = () => {
    const path = remoteBrowse?.path
    if (!path) return
    setCwd(path)
    if (!name) setName(path.split('/').filter(Boolean).pop() || 'Remote project')
    setRemoteBrowseOpen(false)
  }

  const initializeGit = async () => {
    if (!cwd || gitInitBusy) return
    setGitInitBusy(true)
    setError('')
    try {
      const state = await api.initGitRepository(cwd)
      setGitState(state)
      setGitPromptDismissed(false)
    } catch (e: any) {
      setError(e?.message ?? tx("Could not initialize Git."))
    } finally {
      setGitInitBusy(false)
    }
  }

  const commitTag = () => {
    const tag = normalizeTag(tagDraft)
    if (!tag) return
    setSelectedTags((current) => addTag(current, tag))
    setPendingTagColors((current) => ({ ...current, [tag]: tagColor }))
    setTagAdding(false)
    setTagDraft('')
    setTagColor('neutral')
  }

  const create = async () => {
    if (busy || creating.current) return
    const finalName = name.trim()
    if (!finalName) {
      setError(tx("Enter a project name."))
      return
    }
    const requestIntent = modalIntent
    const finalCwd = executionTarget === 'orchestrator' ? cwd.trim() : cwd
    if (executionTarget === 'orchestrator' && !finalCwd) {
      setError(tx("Choose a folder on the Orchestrator before creating the project."))
      return
    }
    const needsGitSetup = executionTarget === 'local' && !!cwd && !!gitState && (!gitState.exists || !gitState.hasCommits)
    if (executionTarget === 'local' && cwd && (gitBusy || !gitState)) {
      setError(tx("Still checking this folder. Wait for the Git check to finish before creating the project."))
      return
    }
    if (gitPromptEnabled && needsGitSetup && !gitPromptDismissed) {
      setError(tx("Initialize Git first, or choose Not now before creating the project."))
      return
    }
    creating.current = true
    setBusy(true)
    setError('')
    try {
      if (Object.keys(pendingTagColors).length) {
        await saveSettings({ ...settings, tagColors: { ...(settings.tagColors ?? {}), ...pendingTagColors } }, { silent: true })
      }
      await newProject({ name: finalName, cwd: finalCwd, mode: 'solo', executionTarget, orchestratorId: executionTarget === 'orchestrator' ? (orchestratorId || orchestrators[0]?.id || null) : null, tags: selectedTags }, requestIntent)
    } catch (e: any) {
      if (useStore.getState().newProjectOpen && useStore.getState().newProjectIntent === requestIntent) setError(e?.message ?? tx("Could not create project."))
    } finally {
      creating.current = false
      setBusy(false)
    }
  }

  const needsGitSetup = executionTarget === 'local' && !!cwd && !!gitState && (!gitState.exists || !gitState.hasCommits)
  const gitCreateBlocked = executionTarget === 'local' && !!cwd && gitBusy
  const remoteFolderVerified = executionTarget === 'orchestrator' && !!cwd && remoteBrowse?.path === cwd && !remoteBrowseError

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-6 fade-in" onClick={close}>
      <div className="absolute inset-0 bg-black/40 backdrop-blur-[2px]" />
      <div onClick={(e) => e.stopPropagation()} className="relative w-full max-w-[430px] rounded-2xl border border-line bg-card p-5 shadow-pop">
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-[15px] font-semibold tracking-[-0.01em] text-ink">{tx("New project")}</h2>
          <button onClick={close} className="flex h-7 w-7 items-center justify-center rounded-lg text-ink-3 transition-colors hover:bg-ink-5 hover:text-ink"><Icon name="x" size={15} /></button>
        </div>

        <div className="space-y-4">
          <Field label={tx("Name")}>
            <Input autoFocus value={name} onChange={(e) => setName(e.target.value)} placeholder={tx("Secure Automations")} onKeyDown={(e) => { if (e.key === 'Enter') void create() }} />
          </Field>

          <Field label={tx("Tags")} hint={tx("optional")}>
            <div className="flex flex-wrap items-center gap-1.5">
              {knownTags.map((tag) => {
                const active = selectedTags.includes(tag)
                return (
                  <button
                    key={tag}
                    type="button"
                    onClick={() => setSelectedTags((current) => toggleTag(current, tag))}
                    className={clsx('rounded-md px-2 py-1 text-[12px] font-medium capitalize transition-colors', active ? tagTone(tag, tagColors) : 'text-ink-3 hover:bg-ink-5 hover:text-ink')}
                  >
                    {tag}
                  </button>
                )
              })}
              {tagAdding ? (
                <span className="flex min-w-0 flex-wrap items-center gap-1.5 rounded-md border border-line bg-card-2 px-1.5 py-1">
                  <input
                    autoFocus
                    value={tagDraft}
                    onChange={(e) => setTagDraft(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') { e.preventDefault(); commitTag() }
                      else if (e.key === 'Escape') { e.preventDefault(); setTagAdding(false); setTagDraft('') }
                    }}
                    placeholder={tx("tag name")}
                    className="h-5 w-24 bg-transparent text-[12px] text-ink outline-none placeholder:text-ink-4"
                  />
                  {TAG_COLORS.map((color) => (
                    <button
                      key={color.name}
                      type="button"
                      aria-label={color.label}
                      title={color.label}
                      onClick={() => setTagColor(color.name)}
                      className={clsx('h-4 w-4 rounded-full transition-transform', color.swatch, tagColor === color.name ? 'ring-2 ring-ink/70 ring-offset-1 ring-offset-card' : 'opacity-60 hover:opacity-100')}
                    />
                  ))}
                  <button type="button" onClick={commitTag} disabled={!normalizeTag(tagDraft)} className="rounded bg-accent px-1.5 py-0.5 text-[12px] font-medium text-accent-ink transition-colors hover:bg-accent-strong disabled:opacity-40">{tx("Add")}</button>
                </span>
              ) : (
                <button type="button" onClick={() => { setTagAdding(true); setTagDraft(''); setTagColor('neutral') }} className="flex items-center gap-1 rounded-md px-2 py-1 text-[12px] font-medium text-ink-3 transition-colors hover:bg-ink-5 hover:text-ink">
                  <Icon name="plus" size={11} />{tx("Tag")}</button>
              )}
            </div>
          </Field>

          <Field label={tx("Folder")} hint={tx("optional")}>
            <div className="flex gap-2">
              <Input readOnly value={cwd ? shortPath(cwd, 3) : ''} placeholder={executionTarget === 'orchestrator' ? tx("Choose a host folder") : tx("Start from scratch")} />
              <Button variant="default" icon="folder" onClick={browse}>{tx("Browse")}</Button>
            </div>
            {executionTarget === 'orchestrator' && (
              <div className="mt-2 rounded-lg border border-line bg-card-2/60 px-2.5 py-2 text-[13px] leading-snug text-ink-3">
                <div className="flex items-center justify-between gap-2 text-ink-2">
                  <div className="flex min-w-0 items-center gap-1.5">
                    <Icon name="globe" size={12} className="shrink-0 text-accent" />
                    <span className="truncate">{selectedOrchestrator ? tx("Folders on {value0}", { value0: selectedOrchestrator.name }) : tx("Choose an Orchestrator first")}</span>
                  </div>
                  {cwd && <span className={clsx('shrink-0 rounded-full border px-1.5 py-0.5 text-[11px]', remoteFolderVerified ? 'border-positive/30 bg-positive/10 text-positive' : 'border-line bg-card text-ink-4')}>{remoteFolderVerified ? tx("Reachable") : tx("Selected")}</span>}
                </div>
                {cwd && <div className="mt-1 truncate font-mono text-[12px] text-ink-4">{cwd}</div>}
                {remoteFolderVerified && <div className="mt-1 text-[12px] text-ink-4">{tx("Child folders visible: {count}", { count: remoteBrowse?.entries.length ?? 0 })}</div>}
              </div>
            )}
            {remoteBrowseOpen && executionTarget === 'orchestrator' && (
              <div className="mt-2 overflow-hidden rounded-xl border border-line bg-card-2">
                <div className="flex items-center gap-2 border-b border-line px-2.5 py-2">
                  <Icon name="globe" size={13} className="shrink-0 text-accent" />
                  <div className="min-w-0 flex-1 truncate font-mono text-[12px] text-ink-3">{remoteBrowse?.path || tx("Loading host folders...")}</div>
                  <Button size="xs" variant="ghost" icon="back" onClick={() => remoteBrowse?.parent && void loadRemoteDir(remoteBrowse.parent)} disabled={!remoteBrowse?.parent || remoteBrowseBusy}>{tx("Up")}</Button>
                  <Button size="xs" variant="default" icon="check" onClick={chooseRemoteDir} disabled={!remoteBrowse?.path || remoteBrowseBusy}>{tx("Use")}</Button>
                </div>
                {remoteBrowseError && <div className="border-b border-line px-2.5 py-2 text-[13px] text-danger">{remoteBrowseError}</div>}
                <div className="scroll-thin max-h-48 overflow-y-auto p-1.5">
                  {remoteBrowseBusy ? (
                    <div className="px-2 py-3 text-[13px] text-ink-3">{tx("Loading folders...")}</div>
                  ) : remoteBrowse?.entries.length ? (
                    remoteBrowse.entries.map((entry) => (
                      <button
                        key={entry.path}
                        onClick={() => void loadRemoteDir(entry.path)}
                        className="flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left text-[13px] text-ink-2 transition-colors hover:bg-ink-5 hover:text-ink"
                      >
                        <Icon name="folder" size={13} className="shrink-0 text-ink-3" />
                        <span className="min-w-0 flex-1 truncate">{entry.name}</span>
                        <Icon name="chevronRight" size={12} className="shrink-0 text-ink-4" />
                      </button>
                    ))
                  ) : (
                    <div className="px-2 py-3 text-[13px] text-ink-3">{tx("No child folders here. Use this folder or go up.")}</div>
                  )}
                </div>
              </div>
            )}
          </Field>

          <Field label={tx("Run location")} hint={tx("where rooms & tasks execute")}>
            <Segmented<ProjectExecutionTarget>
              value={executionTarget}
              onChange={(target) => {
                setExecutionTarget(target)
                setCwd('')
                // Default to the first paired host so we never persist
                // executionTarget='orchestrator' with no host (invalid state).
                if (target === 'orchestrator') setOrchestratorId((cur) => cur ?? orchestrators[0]?.id ?? null)
                else {
                  setRemoteBrowseOpen(false)
                  setRemoteBrowse(null)
                  setRemoteBrowseError('')
                }
              }}
              options={[{ value: 'local', label: tx("This Mac"), icon: 'monitor' }, { value: 'orchestrator', label: tx("Orchestrator"), icon: 'globe' }]}
            />
            {executionTarget === 'orchestrator' && (
              orchestrators.length === 0
                ? <div className="mt-2 rounded-lg border border-warn/30 bg-warn/10 px-2.5 py-2 text-[13px] leading-snug text-warn">{tx("No paired Orchestrator yet — add one in Settings → Orchestrator. The folder stays on this Mac; jobs spool up on the host and you drive the room from here.")}</div>
                : <select
                    value={orchestratorId ?? orchestrators[0].id}
                    onChange={(e) => {
                      setOrchestratorId(e.target.value)
                      setCwd('')
                      setRemoteBrowseOpen(false)
                      setRemoteBrowse(null)
                      setRemoteBrowseError('')
                    }}
                    className="no-drag mt-2 h-9 w-full cursor-pointer rounded-lg border border-line bg-card px-2.5 text-[13px] text-ink outline-none focus:border-accent"
                  >
                    {orchestrators.map((h) => <option key={h.id} value={h.id}>{h.name} · {h.serverUrl}</option>)}
                  </select>
            )}
            {executionTarget === 'orchestrator' && selectedOrchestrator && (
              <div className={clsx('mt-2 rounded-lg border px-2.5 py-2 text-[13px] leading-snug', remoteCheck?.ok ? 'border-positive/30 bg-positive/10 text-positive' : remoteCheck && !remoteCheck.ok ? 'border-danger/30 bg-danger/5 text-danger' : 'border-line bg-card-2/60 text-ink-3')}>
                <div className="flex items-center justify-between gap-2">
                  <div className="flex min-w-0 items-center gap-1.5">
                    <Icon name={remoteCheckBusy ? 'clock' : remoteCheck?.ok ? 'check' : remoteCheck && !remoteCheck.ok ? 'warning' : 'globe'} size={12} className="shrink-0" />
                    <span className="truncate">
                      {remoteCheckBusy
                        ? tx("Checking {value0}...", { value0: selectedOrchestrator.name })
                        : remoteCheck?.ok
                          ? tx("{value0} is reachable", { value0: remoteCheck.name || selectedOrchestrator.name })
                          : remoteCheck && !remoteCheck.ok
                            ? remoteCheck.error || tx("Connection failed")
                            : tx("Host not checked yet")}
                    </span>
                  </div>
                  <Button size="xs" variant="ghost" icon="refresh" onClick={() => void refreshSelectedOrchestrator()} disabled={remoteCheckBusy}>
                    {tx("Check")}</Button>
                </div>
                {remoteCheck?.ok && (
                  <div className="mt-1 text-[12px]">
                    {tx("Projects: {projects} · Rooms: {rooms} · Active tasks: {tasks}", { projects: remoteCheck.totalProjects ?? 0, rooms: remoteCheck.totalRooms ?? 0, tasks: remoteCheck.activeThreadIds?.length ?? 0 })}
                  </div>
                )}
              </div>
            )}
          </Field>

          {executionTarget === 'local' && cwd && (
            <div className={clsx('rounded-xl border p-3', needsGitSetup ? 'border-warn/40 bg-warn/10' : 'border-line bg-card-2/40')}>
              <div className="flex items-start gap-2.5">
                <span className={clsx('mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg', needsGitSetup ? 'bg-warn/15 text-warn' : 'bg-ink-5 text-ink-2')}>
                  <Icon name={gitBusy ? 'clock' : needsGitSetup ? 'warning' : 'branch'} size={15} />
                </span>
                <div className="min-w-0 flex-1">
                  <div className="text-[13px] font-medium text-ink">
                    {gitBusy ? tx("Checking Git...") : needsGitSetup ? tx("Git setup needed") : tx("Git ready")}
                  </div>
                  <div className="mt-0.5 text-[13px] leading-snug text-ink-3">
                    {gitState?.exists && gitState.hasCommits
                      ? `${gitState.branch ? `Branch ${gitState.branch}` : 'Detached HEAD'}`
                      : tx("Rooms use isolated worktrees when this folder is a Git repo with an initial commit.")}
                  </div>
                  {gitPromptEnabled && needsGitSetup && !gitPromptDismissed && (
                    <div className="mt-3 flex flex-wrap gap-2">
                      <Button size="sm" variant="primary" icon="branch" onClick={initializeGit} disabled={gitInitBusy}>{gitInitBusy ? tx("Initializing...") : tx("Initialize Git")}</Button>
                      <Button size="sm" variant="ghost" onClick={() => { setGitPromptDismissed(true); setError('') }}>{tx("Not now")}</Button>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>

        {error && <div className="mt-4 rounded-lg border border-danger/30 bg-danger/10 px-3 py-2 text-[13px] leading-snug text-danger">{<OriginalDiagnostic>{error}</OriginalDiagnostic>}</div>}

        <div className="mt-5 flex justify-end gap-2">
          <Button variant="ghost" onClick={close}>{tx("Cancel")}</Button>
          <Button variant="primary" icon="plus" onClick={create} disabled={busy || !name.trim() || gitCreateBlocked || (executionTarget === 'orchestrator' && (orchestrators.length === 0 || !cwd.trim()))}>{busy ? tx("Creating...") : tx("Create project")}</Button>
        </div>
      </div>
    </div>
  )
}
