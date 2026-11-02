import { useEffect, useRef, useState } from 'react'
import { useStore } from '../../lib/store'
import { api } from '../../lib/ipc'
import type { Settings, Thread, SshHost, WorkspaceLocation, RemoteDirectoryListResult } from '../../lib/types'
import { shortPath, basename } from '../../lib/format'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { resolveExecutionTarget } from '../../lib/execution-target'
import { effectiveThreadCwd, effectiveThreadWorkspace } from '../../lib/thread-workspace'
import { terminalSessionKey } from '../../lib/terminal-sessions'
import { Icon } from '../../lib/icons'
import { Button, MenuItem } from '../ui'
import type { OverflowWorkspace } from './ComposerOverflowMenu'

export function useComposerWorkspace({ mode, thread, settings }: { mode: 'home' | 'thread'; thread?: Thread; settings: Settings }) {
  const tx = useAuthoredText()
  const setView = useStore((s) => s.setView)
  const isChat = mode === 'thread' && thread?.kind === 'chat'
  const projects = useStore((s) => s.projects)
  const terminals = useStore((s) => s.terminals)
  const activeTerminalId = useStore((s) => s.activeTerminalId)
  const setThreadProject = useStore((s) => s.setThreadProject)
  const addProject = useStore((s) => s.addProject)
  const sshHosts = useStore((s) => s.sshHosts)
  const sshStatus = useStore((s) => s.sshStatus)
  const connectSshHost = useStore((s) => s.connectSshHost)
  const setThreadWorkspace = useStore((s) => s.setThreadWorkspace)
  const remembered = useStore.getState().homeComposerWorkspace
  const [homeWorkspace, setHomeWorkspace] = useState<WorkspaceLocation | null>(remembered?.workspace ?? null)
  const [homeProject, setHomeProject] = useState<string | null>(remembered?.projectId ?? null)
  const [cwd, setCwd] = useState(remembered?.cwd ?? settings.defaultCwd ?? '')
  useEffect(() => {
    if (mode === 'home') useStore.setState({ homeComposerWorkspace: { workspace: homeWorkspace, projectId: homeProject, cwd } })
  }, [mode, homeWorkspace, homeProject, cwd])
  useEffect(() => {
    if (homeProject && !projects.some(project => project.id === homeProject)) setHomeProject(null)
  }, [homeProject, projects])
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
  const [remoteBrowseOpen, setRemoteBrowseOpen] = useState(false)
  const [remoteBrowse, setRemoteBrowse] = useState<RemoteDirectoryListResult | null>(null)
  const [remoteBrowseBusy, setRemoteBrowseBusy] = useState(false)
  const [remoteBrowseError, setRemoteBrowseError] = useState('')
  const workspace = mode === 'thread' && thread ? effectiveThreadWorkspace(thread, policyProject) : homeWorkspace
  const projectRoot = policyProject?.workspace?.kind === 'ssh' ? policyProject.workspace.path : (policyProject?.cwd || policyProject?.workspace?.path || '')
  // /workspace belongs to the guest; host Git/file pickers must never resolve a
  // coincidentally matching host directory for an isolated bot chat.
  const workCwd = thread?.bot?.executionTarget === 'vm' ? '' : mode === 'thread' && thread ? effectiveThreadCwd(thread, policyProject) : (workspace?.kind === 'ssh' ? workspace.path : (cwd || homeWorkspace?.path || projectRoot || ''))
  const scope = JSON.stringify([mode, thread?.id, activeProjectId, effectiveHostId, workspace?.kind,
    workspace?.kind === 'ssh' ? workspace.hostId : null, workCwd])
  const currentScope = useRef(scope)
  currentScope.current = scope
  const generation = useRef(0)
  const folderRequest = useRef(0)
  const browseRequest = useRef(0)
  useEffect(() => () => { generation.current++ }, [])
  const captureScope = () => {
    const owner = currentScope.current, revision = generation.current
    return () => currentScope.current === owner && generation.current === revision
  }
  const captureFolderRequest = () => {
    const current = captureScope(), request = ++folderRequest.current
    return () => current() && request === folderRequest.current
  }
  const selectLocalWorkspace = () => { if (mode === 'thread' && thread) setThreadWorkspace(thread.id, null); else setHomeWorkspace(null) }
  const selectHostWorkspace = (host: SshHost, remotePath: string) => {
    const ws: WorkspaceLocation = { kind: 'ssh', hostId: host.id, path: remotePath }
    if (mode === 'thread' && thread) setThreadWorkspace(thread.id, ws)
    else { setHomeWorkspace(ws); setCwd(remotePath) }
    connectSshHost(host.id)
  }
  const [branch, setBranch] = useState<string | null>(null)
  useEffect(() => {
    let live = true
    setBranch(null)
    if (workCwd) api.getBranch(workCwd, workspace).then((b) => { if (live) setBranch(b) }).catch(() => {})
    return () => { live = false }
  }, [workCwd, workspace])
  const chooseDir = async () => {
    const current = captureFolderRequest()
    const d = await api.pickDirectory()
    if (d && current()) { setCwd(d); setHomeWorkspace(null) }
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
    const ownsScope = captureScope(), request = ++browseRequest.current
    const current = () => ownsScope() && request === browseRequest.current
    setRemoteBrowseOpen(true)
    setRemoteBrowseBusy(true)
    setRemoteBrowseError('')
    try {
      const result = await api.listRemoteOrchestratorDir(activeOrchestrator.id, dir || projectRoot || '')
      if (!current()) return
      if (!result.ok) throw new Error(result.error || tx("Could not list folders on the Orchestrator."))
      setRemoteBrowse(result)
    } catch (e: any) {
      if (current()) setRemoteBrowseError(e?.message ?? tx("Could not list folders on the Orchestrator."))
    } finally {
      if (current()) setRemoteBrowseBusy(false)
    }
  }
  const chooseRemoteProjectDir = (dir: string) => {
    setFolder(dir)
    setRemoteBrowseOpen(false)
    setRemoteBrowse(null)
    setRemoteBrowseError('')
  }
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
      setRemoteBrowseBusy(false)
    }
  }
  const handleAddProject = async (kind: 'scratch' | 'folder') => {
    const current = captureFolderRequest()
    if (kind === 'folder') {
      const d = await api.pickDirectory()
      if (!d || !current()) return null
      try {
        const git = await api.inspectGitRepository(d)
        if (!current()) return null
        if (!git.exists && settings.showGitInitPrompt !== false) {
          const ok = confirm(tx("This folder is not initialized as a Git repo. Initialize Git now so TensorFold can use isolated room worktrees?"))
          if (ok) await api.initGitRepository(d)
        }
      } catch (e: any) {
        if (!current()) return null
        // With the soft-suggestion off (#37), git inspection is best-effort: never
        // block project creation behind a confirm() dialog.
        if (settings.showGitInitPrompt !== false) {
          const ok = confirm(tx("{value0}\n\nCreate the project anyway?", { value0: e?.message ?? tx("Could not inspect Git for this folder.") }))
          if (!ok) return null
        }
      }
      if (!current()) return null
      const added = await addProject(d.split('/').pop() || 'project', d)
      return current() ? added : null
    }
    const added = await addProject('New project', '')
    return current() ? added : null
  }
  const pickFolder = async () => {
    const current = captureFolderRequest()
    const d = await api.pickDirectory()
    if (d && current()) setFolder(d)
  }
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

  return { homeWorkspace, homeProject, cwd, activeProjectId, policyProject, effectiveHostId, isOrchestratorProject,
    workspace, projectRoot, workCwd, activeTerminalSession, chooseDir, loadRemoteProjectDir,
    pickFolder, renderFolderChoices, overflowWorkspace }
}

export type ComposerWorkspace = ReturnType<typeof useComposerWorkspace>
