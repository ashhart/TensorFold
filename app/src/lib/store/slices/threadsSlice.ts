import { api } from '../../ipc'
import { modelDisplayName, resolveProviderModel } from '../../provider-accounts'
import { addInflightAppAction, cancelThreadRunApi, createThreadForProject, deleteInflightAppAction, guideThreadRunApi, isInflightAppAction, isAppActionApplied, markAppActionApplied, startThreadRunApi } from '../shared/app-actions'
import { remoteHostIdForThread, owningRemoteHostIdForTicket, remoteHostIdForProject } from '../shared/remote-host'
import { mergeThreadLists, mergeThreadRecord } from '../shared/merge'
import { applyProjectModel, projectDefaults, runErrorMessage, selectedAccountId } from '../shared/model-policy'
import { clearPendingAgentEvents, drainPendingAgentEvents, projectedHostId, projectionOriginsMatch, quarantineProjectionCollision, queuePendingAgentEvent } from '../shared/proxy-state'
import { BUSY_RUN, threadStatusFromRun } from '../types'
import type { RunStatus } from '../types'
import type { AgentEvent, Attachment, Thread, WorkspaceLocation } from '../../types'
import type { GetState, SetState, Slice, View } from '../types'

export const threadsSlice: Slice = (set, get) => ({
  threads: [] as Thread[],
  status: {} as Record<string, RunStatus>,
  approvals: {} as Record<string, { callId: string; itemId: string }[]>,

  async newThread(projectId = null) {
    const s = get().settings
    const project = projectId ? get().projects.find((p) => p.id === projectId) ?? null : null
    const defaults = projectDefaults(project)
    const cwd = projectId ? defaults.cwd : (s?.defaultCwd || '')
    const { thread: t, remoteHostId } = await createThreadForProject(get, cwd, 'task', projectId)
    const patch: Partial<Thread> = {}
    if (project && !t.cwd && defaults.cwd) patch.cwd = defaults.cwd
    if (project && !t.workspace && defaults.workspace) patch.workspace = defaults.workspace
    if (!remoteHostId && Object.keys(patch).length) {
      Object.assign(t, patch)
      await api.setThreadWorkspace(t.id, patch.workspace ?? null, patch.cwd ?? null).catch(() => {})
    }
    if (!remoteHostId) applyProjectModel(get, s, projectId, t)
    set((st) => ({ threads: [t, ...st.threads.filter((x) => x.id !== t.id)], view: { name: 'thread', id: t.id } }))
  },

  async newChat(projectId = null) {
    const s = get().settings
    const { thread: t, remoteHostId } = await createThreadForProject(get, '', 'chat', projectId)
    if (!remoteHostId) applyProjectModel(get, s, projectId, t)
    set((st) => ({ threads: [t, ...st.threads.filter((x) => x.id !== t.id)], view: { name: 'thread', id: t.id } }))
  },

  async forkThreadFromTurn(threadId, turn, options) {
    try {
      const r = await api.checkpointForkTurn({
        threadId,
        turn,
        ...(options?.worktree ? { worktree: true } : {}),
      })
      if (!r.ok || !r.thread) return { ok: false, error: r.error || 'Could not fork from this turn.' }
      const t = r.thread
      // Idempotent with the host's thread:update push — whichever lands second is a no-op.
      set((st) => ({ threads: [t, ...st.threads.filter((x) => x.id !== t.id)], view: { name: 'thread', id: t.id } }))
      return { ok: true, worktreePath: r.worktreePath ?? undefined, worktreeError: r.worktreeError ?? undefined }
    } catch (e) {
      return { ok: false, error: e instanceof Error ? e.message : 'Could not fork from this turn.' }
    }
  },

  // Cross-engine handoff (VST-036): continue this conversation on a different engine.
  // The host flattens the transcript, creates the new thread, and auto-runs the seed.
  async handoffThread(threadId, toProviderId, toModel, intent) {
    try {
      const t = await api.handoffThread({ threadId, toProviderId, toModel: toModel ?? null, ...(intent ? { intent } : {}) })
      set((st) => ({ threads: [t, ...st.threads.filter((x) => x.id !== t.id)], view: { name: 'thread', id: t.id }, status: { ...st.status, [t.id]: 'thinking' } }))
      return { ok: true }
    } catch (e) {
      return { ok: false, error: e instanceof Error ? e.message : 'Handoff failed.' }
    }
  },

  async setThreadProject(threadId, projectId) {
    const project = projectId ? get().projects.find((p) => p.id === projectId) ?? null : null
    const defaults = projectDefaults(project)
    const prev = get().threads.find((t) => t.id === threadId)
    set((st) => ({
      threads: st.threads.map((t) => (t.id === threadId ? {
        ...t,
        projectId,
        ...(project ? { cwd: defaults.cwd, workspace: defaults.workspace } : {}),
      } : t)),
    }))
    try {
      await api.setThreadProject(threadId, projectId)
    } catch (e) {
      if (prev) set((st) => ({ threads: st.threads.map((t) => (t.id === threadId ? prev : t)) }))
      throw e
    }
  },

  async addProject(name, cwd) {
    const p = await api.createProject({ name, cwd })
    set((st) => ({ projects: [p, ...st.projects] }))
    return p
  },

  async launch({ text, cwd, providerId, model, approvalMode, providerAccountId, projectId = null, workspace = null, attachments, sourceThreadKey, persistActiveModel = true, incognito = false }) {
    const s = get().settings
    const provider = s?.providers.find((p) => p.id === providerId)
    const accountId = selectedAccountId(provider, providerAccountId)
    const safeModel = provider ? (resolveProviderModel(provider, accountId, model) ?? model) : model
    if (s && persistActiveModel) await get().saveSettings({ ...s, activeProviderId: providerId, activeProviderAccountId: accountId, activeModel: safeModel, defaultCwd: cwd, approvalMode }, { silent: true })
    const project = projectId ? get().projects.find((candidate) => candidate.id === projectId) ?? null : null
    const remoteHostId = remoteHostIdForProject(get(), project)
    const { thread: t } = await createThreadForProject(get, cwd, 'task', projectId)
    if (incognito) {
      // Memory-only thread: the host's saveThreads/spine seams skip it once flagged.
      t.incognito = true
      await api.patchThread(t.id, { incognito: true })
    }
    if (!remoteHostId) await api.configureThread(t.id, providerId, safeModel, accountId)
    if (!remoteHostId) {
      t.providerId = providerId
      t.providerAccountId = accountId
      t.model = safeModel
    }
    if (!remoteHostId && workspace) {
      t.workspace = workspace
      if (!t.cwd && workspace.path) t.cwd = workspace.path
      await api.setThreadWorkspace(t.id, workspace, t.cwd || null)
    }
    set((st) => ({ threads: [t, ...st.threads.filter((x) => x.id !== t.id)], view: { name: 'thread', id: t.id }, status: { ...st.status, [t.id]: 'thinking' } }))
    // When this run was launched from a connector conversation, carry its threadKey so the
    // host can inject that source's per-channel ephemeral prompt for this run only.
    try {
      await startThreadRunApi(remoteHostId, { threadId: t.id, userText: text, attachments, sourceThreadKey })
    } catch (e) {
      const message = runErrorMessage(e)
      set((st) => ({ status: { ...st.status, [t.id]: 'error' }, notice: { kind: 'error', message } }))
    }
  },

  async setThreadWorkspace(threadId, workspace, cwd) {
    const prior = get().threads.find((t) => t.id === threadId)
    const localFallback = prior?.workspace?.kind === 'ssh' ? (get().settings?.defaultCwd ?? '') : (prior?.cwd ?? '')
    const targetCwd = workspace?.path ?? cwd ?? localFallback
    set((st) => ({ threads: st.threads.map((t) => (t.id === threadId ? { ...t, workspace, cwd: targetCwd } : t)) }))
    try {
      const updated = await api.setThreadWorkspace(threadId, workspace, targetCwd)
      set((st) => ({ threads: st.threads.map((t) => (t.id === threadId ? updated : t)) }))
    } catch (error) {
      if (prior) set((st) => ({ threads: st.threads.map((t) => (t.id === threadId ? prior : t)) }))
      throw error
    }
  },

  async deleteThread(id) {
    await get().archiveItem('thread', id)
  },

  async send(text, attachments?: Attachment[]) {
    const v = get().view
    if (v.name !== 'thread') return
    const id = v.id
    const st = get().status[id]
    const remoteHostId = remoteHostIdForThread(get(), id)
    if (st && BUSY_RUN.has(st)) {
      const result = await guideThreadRunApi(remoteHostId, { threadId: id, userText: text, attachments })
      if (result.queued) return
    }
    set((st) => ({ status: { ...st.status, [id]: 'thinking' } }))
    try {
      await startThreadRunApi(remoteHostId, { threadId: id, userText: text, attachments })
    } catch (e) {
      const message = runErrorMessage(e)
      set((st) => ({ status: { ...st.status, [id]: 'error' }, notice: { kind: 'error', message } }))
    }
  },

  async cancel(threadId) {
    const hostId = remoteHostIdForThread(get(), threadId)
    const cancelling = hostId
      ? get().tickets.filter((ticket) => ticket.execution?.target === 'remote' && ticket.execution.threadId === threadId && owningRemoteHostIdForTicket(get(), ticket) === hostId)
      : []
    if (cancelling.length) {
      for (const ticket of cancelling) {
        const updated = await api.transitionRemoteOrchestratorTicket(hostId!, ticket.id, { action: 'cancel' })
        get().applyTicketUpdate(updated)
      }
    } else {
      await cancelThreadRunApi(hostId, threadId)
    }
    set((st) => {
      const approvals = { ...st.approvals }
      delete approvals[threadId]
      return { status: { ...st.status, [threadId]: 'idle' }, approvals }
    })
  },

  async approve(threadId, callId, decision) {
    const hostId = remoteHostIdForThread(get(), threadId)
    await (hostId ? api.approveRemoteOrchestrator(hostId, callId, decision) : api.approve(callId, decision))
    set((st) => ({ approvals: { ...st.approvals, [threadId]: (st.approvals[threadId] || []).filter((a) => a.callId !== callId) } }))
  },

  async setThreadProvider(threadId, providerId, model, providerAccountId) {
    const provider = get().settings?.providers.find((p) => p.id === providerId)
    const accountId = selectedAccountId(provider, providerAccountId)
    const safeModel = provider ? (resolveProviderModel(provider, accountId, model) ?? model) : model
    set((st) => ({ threads: st.threads.map((t) => (t.id === threadId ? { ...t, providerId, providerAccountId: accountId, model: safeModel } : t)) }))
    await api.configureThread(threadId, providerId, safeModel, accountId)
    const s = get().settings
    if (s) await get().saveSettings({ ...s, activeProviderId: providerId, activeProviderAccountId: accountId, activeModel: safeModel }, { silent: true })
  },
  async patchThread(id, patch) {
    const prev = get().threads.find((t) => t.id === id)
    set((st) => ({ threads: st.threads.map((t) => (t.id === id ? { ...t, ...patch } : t)) }))
    try {
      await api.patchThread(id, patch)
    } catch (e) {
      if (prev) set((st) => ({ threads: st.threads.map((t) => (t.id === id ? prev : t)) }))
      throw e
    }
  },
  async noteModelChange(threadId, providerId, model, providerAccountId) {
    const thread = get().threads.find((t) => t.id === threadId)
    if (!thread) return
    const provider = get().settings?.providers.find((p) => p.id === providerId)
    const accountId = selectedAccountId(provider, providerAccountId)
    const safeModel = provider ? (resolveProviderModel(provider, accountId, model) ?? model) : model
    // Append the notice in MAIN (the timeline owner) which echoes it back via item_add — instead of
    // round-tripping our whole timeline, which could clobber main's copy if a stream is in flight.
    await api.appendThreadNotice(threadId, `Model changed to ${modelDisplayName(provider, safeModel)}`)
  },
  async renameThread(id, title) {
    const finalTitle = title.trim() || 'Untitled'
    set((st) => ({ threads: st.threads.map((t) => (t.id === id ? { ...t, title: finalTitle } : t)) }))
    await api.renameThread(id, finalTitle)
  },
  async setThreadEngine(id, engine) {
    const previousEngine = get().threads.find((thread) => thread.id === id)?.engine
    const nextEngine = engine === 'native' ? undefined : engine
    set((st) => ({ threads: st.threads.map((t) => (t.id === id ? { ...t, engine: nextEngine } : t)) }))
    try {
      await api.setThreadEngine(id, engine)
    } catch (error) {
      set((st) => ({ threads: st.threads.map((t) => (t.id === id && t.engine === nextEngine ? { ...t, engine: previousEngine } : t)) }))
      throw error
    }
  },
  async compactThread(id) {
    const result = await api.compactThread(id)
    if (result.thread) set((st) => ({ threads: st.threads.map((t) => (t.id === id ? result.thread : t)) }))
  },
  applyThreadUpdate(t) {
    const existingThread = get().threads.find((thread) => thread.id === t.id)
    if (existingThread && !projectionOriginsMatch(projectedHostId(existingThread), projectedHostId(t))) {
      const message = quarantineProjectionCollision('thread', t.id, projectedHostId(existingThread), projectedHostId(t))
      if (message) set({ notice: { kind: 'error', message } })
      return
    }
    if (t.archivedAt) {
      clearPendingAgentEvents(t.id, projectedHostId(t))
      set((st) => {
        const status = { ...st.status }
        const approvals = { ...st.approvals }
        delete status[t.id]
        delete approvals[t.id]
        return {
          threads: st.threads.filter((thread) => thread.id !== t.id),
          status,
          approvals,
          view: st.view.name === 'thread' && st.view.id === t.id ? { name: 'home' } : st.view,
        }
      })
      return
    }
    set((st) => ({
      threads: [st.threads.find((thread) => thread.id === t.id) ? mergeThreadRecord(t, st.threads.find((thread) => thread.id === t.id)!) : t, ...st.threads.filter((thread) => thread.id !== t.id)].sort((a, b) => b.updatedAt - a.updatedAt),
    }))
    const events = drainPendingAgentEvents(t.id, projectedHostId(t))
    for (const event of events) get().applyEvent(event)
    // Ask the owning host to project its own thread state onto bound remote tickets.
    // The renderer never supplies runState/execution patches.
    const mirrored = get().threads.find((thread) => thread.id === t.id) ?? t
    if (mirrored.orchestratorHostId) {
      for (const ticket of get().tickets) {
        if (ticket.execution?.target !== 'remote' || ticket.execution.threadId !== mirrored.id) continue
        const ownerHostId = owningRemoteHostIdForTicket(get(), ticket)
        if (!ownerHostId) continue
        void api.transitionRemoteOrchestratorTicket(ownerHostId, ticket.id, { action: 'sync' }).then((next) => get().applyTicketUpdate(next)).catch(() => {})
      }
    }
  },

  applyEvent(e: AgentEvent) {
    const targetThread = get().threads.find((thread) => thread.id === e.threadId)
    if (!targetThread) {
      queuePendingAgentEvent(e)
      return
    }
    if (!projectionOriginsMatch(projectedHostId(targetThread), e.orchestratorHostId)) {
      const message = quarantineProjectionCollision('agent event', e.threadId, projectedHostId(targetThread), e.orchestratorHostId ?? null)
      if (message) set({ notice: { kind: 'error', message } })
      return
    }
    set((st) => {
      let threads = st.threads
      const status = { ...st.status }
      const approvals = { ...st.approvals }

      const patchThread = (fn: (t: Thread) => Thread) => {
        threads = threads.map((t) => (t.id === e.threadId ? fn(t) : t))
      }

      switch (e.type) {
        case 'item_add':
          patchThread((t) => (t.timeline.some((i) => i.id === e.item.id) ? t : { ...t, timeline: [...t.timeline, { ...e.item }], updatedAt: Date.now() }))
          break
        case 'item_delta':
          patchThread((t) => ({
            ...t,
            timeline: t.timeline.map((i) => (i.id === e.id ? { ...i, ...(e.trace && !i.trace ? { trace: e.trace } : {}), [e.field]: ((i as any)[e.field] ?? '') + e.chunk } : i)),
          }))
          break
        case 'item_update':
          patchThread((t) => ({ ...t, timeline: t.timeline.map((i) => (i.id === e.id ? { ...i, ...(e.trace && !i.trace ? { trace: e.trace } : {}), ...e.patch } : i)) }))
          break
        case 'thread_update':
          patchThread((t) => ({ ...t, ...e.patch, updatedAt: Date.now() }))
          break
        case 'title':
          patchThread((t) => ({ ...t, title: e.title }))
          break
        case 'plan':
          patchThread((t) => ({ ...t, plan: e.plan, goalStatus: e.goalStatus ?? t.goalStatus }))
          break
        case 'status': {
          status[e.threadId] = e.status as RunStatus
          const next = threadStatusFromRun(e.status as RunStatus)
          patchThread((t) => (t.status === next ? t : { ...t, status: next }))
          break
        }
        case 'approval_request':
          approvals[e.threadId] = [...(approvals[e.threadId] || []), { callId: e.callId, itemId: e.itemId }]
          break
        case 'done':
          status[e.threadId] = 'idle'
          delete approvals[e.threadId]
          patchThread((t) => (t.status === 'idle' ? t : { ...t, status: 'idle' }))
          break
        case 'error':
          status[e.threadId] = 'error'
          delete approvals[e.threadId]
          patchThread((t) => (t.status === 'error' ? t : { ...t, status: 'error' }))
          break
      }

      // bubble active thread to top of the list on updates
      if (e.type === 'item_add') {
        const idx = threads.findIndex((t) => t.id === e.threadId)
        if (idx > 0) { const [t] = threads.splice(idx, 1); threads = [t, ...threads] }
      }

      return { threads, status, approvals }
    })

    // #42 — auto-apply app-control actions the agent requested (creates +
    // navigation). change_setting waits for the user's confirm chip. Guarded so a
    // persisted timeline replaying on reload never re-runs an action.
    const action = e.type === 'item_add' ? e.item.appAction : e.type === 'item_update' ? (e.patch as any)?.appAction : null
    const actionItemId = e.type === 'item_add' ? e.item.id : e.type === 'item_update' ? e.id : null
    if (action && actionItemId && !isAppActionApplied(actionItemId) && !isInflightAppAction(actionItemId) && action.kind !== 'change_setting' && action.state !== 'applied' && action.state !== 'dismissed') {
      addInflightAppAction(actionItemId)
      const s = get()
      const v = s.view
      const currentProjectId = v.name === 'project' ? v.id
        : v.name === 'room' ? (s.rooms.find((r) => r.id === v.id)?.projectId ?? null)
        : v.name === 'thread' ? (s.threads.find((t) => t.id === v.id)?.projectId ?? null)
        : null
      const p = action.payload || {}
      const threadId = e.threadId
      // Mark applied (and reflect it in the in-session timeline) ONLY once the create lands —
      // otherwise a failed IPC create would stay marked-applied in localStorage forever, and the
      // !appliedAppActions guard above would block every retry, silently dropping the request.
      const markApplied = () => {
        deleteInflightAppAction(actionItemId)
        markAppActionApplied(actionItemId)
        set((st) => ({ threads: st.threads.map((t) => (t.id !== threadId ? t : { ...t, timeline: t.timeline.map((i) => (i.id !== actionItemId ? i : { ...i, appAction: i.appAction ? { ...i.appAction, state: 'applied' } : i.appAction })) })) }))
      }
      const onFail = (e: unknown) => {
        deleteInflightAppAction(actionItemId) // un-block so a reload / re-emit can retry
        console.error('[app-control] failed to apply', action.kind, e)
        set({ notice: { kind: 'error', message: `Couldn't ${action.kind.replace('_', ' ')} — ${String((e as any)?.message ?? e)}` } })
      }
      const run = (work: Promise<unknown>) => { void work.then(markApplied, onFail) }
      try {
        if (action.kind === 'create_project') run(s.addProject(String(p.name), p.folder ? String(p.folder) : ''))
        else if (action.kind === 'create_room') { const pid = p.projectId || currentProjectId || s.projects[0]?.id; if (pid) run(s.createRoom(pid, String(p.name || 'New room'))); else deleteInflightAppAction(actionItemId) }
        else if (action.kind === 'create_kanban') { const pid = p.projectId || currentProjectId || s.projects[0]?.id; if (pid) run(s.createTicket({ projectId: pid, title: String(p.title), body: String(p.body || ''), type: p.type || 'task', priority: p.priority || 'medium', status: 'todo' })); else deleteInflightAppAction(actionItemId) }
        else if (action.kind === 'open_view') { const view = String(p.view); if (['home', 'projects', 'settings', 'plugins', 'automations', 'commandCentre', 'assistant'].includes(view)) { s.setView({ name: view } as View); markApplied() } else deleteInflightAppAction(actionItemId) }
        else deleteInflightAppAction(actionItemId)
      } catch (e) { onFail(e) }
    }
  },
  async resolveAppAction(threadId, itemId, decision) {
    const s = get()
    const item = s.threads.find((t) => t.id === threadId)?.timeline.find((i) => i.id === itemId)
    const action = item?.appAction
    if (!action || action.kind !== 'change_setting') return
    if (decision === 'apply' && s.settings) {
      const key = String(action.payload?.key), value = action.payload?.value
      const next: any = { ...s.settings }
      if (key === 'theme' && ['system', 'light', 'dark'].includes(value)) next.theme = value
      else if (key === 'approvalMode' && ['suggest', 'auto', 'readonly'].includes(value)) next.approvalMode = value
      else if (key === 'reasoningEffort' && ['none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max', 'ultrathinks', 'ultracode', 'dynamic', '512'].includes(value)) next.reasoningEffort = value
      else return // unknown key/value — leave the chip pending rather than mark it applied
      // Only mark applied once the save actually lands (saveSettings surfaces its own error).
      try { await s.saveSettings(next) } catch { return }
    }
    set((st) => ({ threads: st.threads.map((t) => (t.id !== threadId ? t : { ...t, timeline: t.timeline.map((i) => (i.id !== itemId ? i : { ...i, appAction: { ...action, state: decision === 'apply' ? 'applied' : 'dismissed' } })) })) }))
  },
})
