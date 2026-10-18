import { api } from '../../ipc'
import { findTerminalSession, nextActiveKeyAfterClose, reconcileSessionsForOrigin, removeSessionByKey, renameSessionByKey, setSessions, stampOrchestratorSession, terminalSessionKey, terminalSessionRoute } from '../../terminal-sessions'
import { remoteHostIdForProjectId } from '../shared/remote-host'
import { restoredProjects } from '../shared/app-actions'
import type { Slice } from '../types'
import type { CliDefinition, CliSession } from '../../types'

export const terminalsSlice: Slice = (set, get) => ({
  terminals: {} as Record<string, CliSession[]>,
  activeTerminalId: {} as Record<string, string | null>,

  async listTerminalClis(projectId): Promise<CliDefinition[]> {
    try {
      const hostId = remoteHostIdForProjectId(get(), projectId)
      return hostId ? await api.remoteListClis(hostId) : await api.listClis()
    } catch {
      return []
    }
  },

  async loadTerminals(projectId) {
    const applyOriginList = (sessions: CliSession[], orchestratorHostId: string | null) => {
      set((st) => {
        const terminals = reconcileSessionsForOrigin(st.terminals, projectId, sessions, orchestratorHostId)
        const aggregate = terminals[projectId] ?? []
        const active = st.activeTerminalId[projectId]
        const stillThere = active && aggregate.some((s) => terminalSessionKey(s) === active)
        return {
          terminals,
          activeTerminalId: { ...st.activeTerminalId, [projectId]: stillThere ? active : (aggregate[0] ? terminalSessionKey(aggregate[0]) : null) },
        }
      })
    }
    try {
      // Local PTYs and Orchestrator PTYs are independent origins. Always refresh local, then each
      // relevant remote origin, and reconcile each list separately so changing the new-work target
      // cannot make existing terminals disappear.
      const current = get()
      const hostIds = new Set<string>()
      const newWorkHostId = remoteHostIdForProjectId(current, projectId)
      if (newWorkHostId) hostIds.add(newWorkHostId)
      for (const session of current.terminals[projectId] ?? []) {
        const route = terminalSessionRoute(session)
        if (route.kind === 'orchestrator') hostIds.add(route.hostId)
      }
      const loads: Promise<void>[] = [
        api.listTerminals(projectId).then((sessions) => applyOriginList(sessions, null)),
        ...[...hostIds].map((hostId) => api.remoteTerminalList(hostId, projectId)
          .then((sessions) => applyOriginList(sessions, hostId))),
      ]
      await Promise.allSettled(loads)
      // Bring back any terminals that were running when the app last quit, RESUMED to their prior CLI
      // session (terminal:restore types resumeString into the new shell). Guarded to fire once per
      // project per app session so we never re-restore something the user has just killed. The
      // onTerminalSessions broadcast (triggered by each restore) remains the source of truth for the
      // live list, so we don't merge the restored sessions in by hand here.
      //
      // Built-ins only: a custom CLI's `command` is an arbitrary string the user once typed, typed
      // character-for-character into a real login shell on restore (every character is
      // shell-interpreted there, unlike an argv-array spawn) — auto-replaying it with no
      // confirmation is a command-injection risk if it ever contained shell metacharacters. A
      // session with no `builtin` flag (older persisted state, pre-dating this field) is treated
      // the same as a custom CLI: not auto-restored.
      if (!restoredProjects.has(projectId)) {
        restoredProjects.add(projectId)
        const restorable = await api.restorableTerminals(projectId)
        for (const s of restorable) { if (s.builtin) void api.restoreTerminal(s.id).catch(() => {}) }
      }
    } catch { /* surfaced by surfaceIpcErrors */ }
  },
  async startTerminal(projectId, cliId, opts): Promise<CliSession | null> {
    try {
      const hostId = remoteHostIdForProjectId(get(), projectId)
      if (hostId && (opts?.command || cliId.startsWith('custom:'))) {
        set({ notice: { kind: 'error', message: 'Custom terminal commands are not available on Orchestrator projects.', authored: true } })
        return null
      }
      const raw = hostId
        ? await api.remoteTerminalStart(hostId, { projectId, cliId, label: opts?.label })
        : await api.startTerminal(projectId, cliId, opts)
      const s = hostId
        ? stampOrchestratorSession(raw, hostId)
        : { ...raw, hostLocation: 'local' as const, orchestratorHostId: undefined }
      // The terminal:sessions broadcast will also add it; set it active now for instant focus.
      set((st) => ({
        terminals: setSessions(st.terminals, projectId, [...(st.terminals[projectId] ?? []).filter((x) => terminalSessionKey(x) !== terminalSessionKey(s)), s]),
        activeTerminalId: { ...st.activeTerminalId, [projectId]: terminalSessionKey(s) },
      }))
      return s
    } catch { return null }
  },
  async killTerminal(projectId, sessionId) {
    const session = findTerminalSession(get().terminals[projectId] ?? [], sessionId)
    if (!session) {
      set({ notice: { kind: 'error', message: 'TensorFold cannot safely identify that terminal session.', authored: true } })
      return
    }
    const route = terminalSessionRoute(session)
    if (route.kind === 'unresolved') {
      set({ notice: { kind: 'error', message: 'This remote terminal predates origin tracking, so TensorFold cannot safely determine which host owns it. Reopen it from its Orchestrator.', authored: true } })
      return
    }
    try {
      if (route.kind === 'orchestrator') await api.remoteTerminalKill(route.hostId, session.id)
      else await api.killTerminal(session.id)
    } catch {
      return
    }
    set((st) => {
      const list = st.terminals[projectId] ?? []
      const sessionKey = terminalSessionKey(session)
      const nextActive = st.activeTerminalId[projectId] === sessionKey ? nextActiveKeyAfterClose(list, sessionKey) : st.activeTerminalId[projectId]
      return {
        terminals: removeSessionByKey(st.terminals, projectId, sessionKey),
        activeTerminalId: { ...st.activeTerminalId, [projectId]: nextActive },
      }
    })
  },
  async renameTerminal(projectId, sessionId, title) {
    const clean = title.trim()
    if (!clean) return
    const session = findTerminalSession(get().terminals[projectId] ?? [], sessionId)
    if (!session) {
      set({ notice: { kind: 'error', message: 'TensorFold cannot safely identify that terminal session.', authored: true } })
      return
    }
    const route = terminalSessionRoute(session)
    if (route.kind === 'unresolved') {
      set({ notice: { kind: 'error', message: 'This remote terminal predates origin tracking, so TensorFold cannot safely determine which host owns it. Reopen it from its Orchestrator.', authored: true } })
      return
    }
    try {
      if (route.kind === 'orchestrator') await api.remoteTerminalRename(route.hostId, session.id, clean)
      else await api.renameTerminal(session.id, clean)
      set((st) => ({ terminals: renameSessionByKey(st.terminals, projectId, terminalSessionKey(session), clean) }))
    } catch { /* surfaceIpcErrors reports the failure; keep the durable title */ }
  },
  setActiveTerminal(projectId, sessionId) {
    set((st) => {
      const session = sessionId ? findTerminalSession(st.terminals[projectId] ?? [], sessionId) : null
      return { activeTerminalId: { ...st.activeTerminalId, [projectId]: session ? terminalSessionKey(session) : sessionId } }
    })
  },
  openTerminal(projectId, sessionId) {
    set((st) => {
      const session = findTerminalSession(st.terminals[projectId] ?? [], sessionId)
      return { view: { name: 'project', id: projectId, tab: 'terminals' }, activeTerminalId: { ...st.activeTerminalId, [projectId]: session ? terminalSessionKey(session) : sessionId } }
    })
  },
})
