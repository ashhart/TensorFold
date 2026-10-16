import { api } from '../../ipc'
import { enqueueProjectUpdate } from '../shared/queue'
import { projectRecordHostId, remoteHostIdForProject } from '../shared/remote-host'
import { mutableProjectConfigPatch, mutableProjectProxyPatch, projectedProjectHostId, projectionOriginsMatch, quarantineProjectionCollision, removeProjectOwnedProxyState } from '../shared/proxy-state'
import { projectUpdateState } from '../shared/queue'
import type { Slice } from '../types'
import type { Project } from '../../types'

export const projectsSlice: Slice = (set, get) => ({
  projects: [] as Project[],

  async createProject(name) {
    const p = await api.createProject({ name: name || 'New project', cwd: '' })
    set((st) => ({ projects: [p, ...st.projects], view: { name: 'project', id: p.id } }))
  },
  // Full creation flow (New Project modal): create the project container first.
  // Rooms, Kanban boards, and solo operations are added inside the project.
  async newProject(opts, modalIntent) {
    const intent = modalIntent ?? get().newProjectIntent
    const base = {
      name: opts.name || 'New project', cwd: opts.cwd || '', mode: opts.mode,
      localOnly: opts.localOnly, executionTarget: opts.executionTarget ?? 'local', orchestratorId: opts.orchestratorId ?? null,
      defaultProviderId: opts.defaultProviderId ?? null, defaultProviderAccountId: opts.defaultProviderAccountId ?? null, defaultModel: opts.defaultModel ?? null,
      tags: opts.tags ?? [],
    }
    let p: Project
    if (base.executionTarget === 'orchestrator') {
      const hostId = base.orchestratorId ?? get().settings?.remoteOrchestrators?.find((host) => host.enabled)?.id ?? null
      if (!hostId) throw new Error('Choose an Orchestrator before creating a remote project.')
      if (!base.cwd.trim()) throw new Error('Choose a folder on the Orchestrator before creating the project.')
      const remote = await api.createRemoteOrchestratorProject(hostId, base)
      p = await api.createProject({
        id: remote.id,
        name: remote.name || base.name,
        cwd: remote.cwd || base.cwd,
        workspace: remote.workspace ?? null,
        memory: remote.memory ?? '',
        team: remote.team ?? '',
        mode: remote.mode || base.mode,
        localOnly: remote.localOnly ?? base.localOnly,
        executionTarget: 'orchestrator',
        orchestratorId: hostId,
        defaultProviderId: remote.defaultProviderId ?? base.defaultProviderId,
        defaultProviderAccountId: remote.defaultProviderAccountId ?? base.defaultProviderAccountId,
        defaultModel: remote.defaultModel ?? base.defaultModel,
        kanban: remote.kanban ?? { modelMode: 'single' },
        tags: remote.tags ?? base.tags,
        createdAt: remote.createdAt,
      })
    } else {
      p = await api.createProject(base)
    }
    set((st) => ({
      projects: [p, ...st.projects.filter((project) => project.id !== p.id)],
      ...(st.newProjectIntent === intent ? {
        newProjectOpen: false,
        view: { name: 'project' as const, id: p.id, ...(p.executionTarget === 'orchestrator' ? { tab: 'terminals' as const } : {}) },
      } : {}),
    }))
    return p
  },
  async syncRemoteProjects(hostId, remoteProjects) {
    const proxies = remoteProjects
      .filter((project) => !project.archivedAt)
      .map((project) => ({
        id: project.id,
        name: project.name || 'Remote project',
        cwd: project.cwd || '',
        workspace: project.workspace ?? null,
        memory: project.memory ?? '',
        team: project.team ?? '',
        mode: project.mode ?? 'solo',
        localOnly: project.localOnly ?? false,
        executionTarget: 'orchestrator' as const,
        orchestratorId: hostId,
        defaultProviderId: project.defaultProviderId ?? null,
        defaultProviderAccountId: project.defaultProviderAccountId ?? null,
        defaultModel: project.defaultModel ?? null,
        kanban: project.kanban ?? { modelMode: 'single' as const },
        tags: project.tags ?? [],
        createdAt: project.createdAt,
      }))
    if (!proxies.length) return
    const persisted = await Promise.all(proxies.map((project) => api.createProject(project)))
    set((st) => {
      const remoteIds = new Set(persisted.map((project) => project.id))
      return { projects: [...persisted, ...st.projects.filter((project) => !remoteIds.has(project.id))] }
    })
  },
  applyProjectUpdate(p) {
    const existingProject = get().projects.find((project) => project.id === p.id)
    if (existingProject && !projectionOriginsMatch(projectedProjectHostId(existingProject), projectedProjectHostId(p))) {
      const message = quarantineProjectionCollision('project', p.id, projectedProjectHostId(existingProject), projectedProjectHostId(p))
      if (message) set({ notice: { kind: 'error', message } })
      return
    }
    if (p.archivedAt) {
      set((st) => removeProjectOwnedProxyState(st, p.id, { name: 'projects' }, projectedProjectHostId(p)))
      return
    }
    set((st) => ({
      projects: [p, ...st.projects.filter((project) => project.id !== p.id)].sort((a, b) => b.updatedAt - a.updatedAt),
    }))
  },
  async updateProject(id, patch) {
    const prev = get().projects.find((p) => p.id === id)
    if (!prev) throw new Error('Project not found.')
    const version = projectUpdateState.bumpSequence()
    const optimisticPatch: Partial<Project> = { ...patch }
    if (patch.executionTarget === 'local' && !Object.prototype.hasOwnProperty.call(patch, 'orchestratorId')) optimisticPatch.orchestratorId = null
    const touched = new Set<keyof Project>(Object.keys(optimisticPatch) as (keyof Project)[])
    if (Object.prototype.hasOwnProperty.call(patch, 'executionTarget') || Object.prototype.hasOwnProperty.call(patch, 'orchestratorId')) {
      touched.add('executionTarget'); touched.add('orchestratorId')
    }
    for (const field of touched) {
      const key = `${id}:${String(field)}`
      if (!projectUpdateState.committedValues.has(key)) projectUpdateState.committedValues.set(key, prev[field])
      projectUpdateState.fieldVersions.set(key, version)
    }
    projectUpdateState.latestVersions.set(id, version)
    const desired = { ...prev, ...optimisticPatch }
    set((st) => ({ projects: st.projects.map((p) => (p.id === id ? { ...p, ...optimisticPatch } : p)) }))

    return enqueueProjectUpdate(id, async () => {
      try {
      const previousHostId = projectRecordHostId(prev)
      const optimistic = desired
      const targetHostId = optimistic?.executionTarget === 'orchestrator'
        ? optimistic.orchestratorId ?? get().settings?.remoteOrchestrators?.find((host) => host.enabled)?.id ?? null
        : null
      let committed: Project
      if (targetHostId) {
        const configPatch = mutableProjectConfigPatch(patch)
        const remoteProject = previousHostId === targetHostId
          ? await api.updateRemoteOrchestratorProject(targetHostId, id, configPatch)
          : await api.createRemoteOrchestratorProject(targetHostId, {
              ...optimistic,
              id,
              name: optimistic?.name ?? prev?.name ?? 'Remote project',
              cwd: optimistic?.cwd ?? prev?.cwd ?? '',
            })
        const proxy = { ...remoteProject, executionTarget: 'orchestrator' as const, orchestratorId: targetHostId }
        await api.setProjectExecutionTarget(id, 'orchestrator', targetHostId, mutableProjectProxyPatch(proxy))
        if (previousHostId && previousHostId !== targetHostId) await api.deleteRemoteOrchestratorProject(previousHostId, id).catch(() => {})
        committed = proxy
      } else {
        if (previousHostId) await api.deleteRemoteOrchestratorProject(previousHostId, id)
        if (previousHostId || Object.prototype.hasOwnProperty.call(patch, 'executionTarget') || Object.prototype.hasOwnProperty.call(patch, 'orchestratorId')) {
          committed = await api.setProjectExecutionTarget(id, 'local', null, mutableProjectConfigPatch(patch))
        } else {
          const configPatch = mutableProjectConfigPatch(patch)
          committed = Object.keys(configPatch).length ? await api.updateProject(id, configPatch) : desired
        }
      }
      set((st) => ({ projects: st.projects.map((project) => {
        if (project.id !== id) return project
        const applied: Partial<Project> = {}
        for (const field of touched) {
          if (projectUpdateState.fieldVersions.get(`${id}:${String(field)}`) === version) (applied as any)[field] = committed[field]
        }
        if (projectUpdateState.latestVersions.get(id) === version) applied.updatedAt = committed.updatedAt
        return { ...project, ...applied }
      }) }))
      for (const field of touched) projectUpdateState.committedValues.set(`${id}:${String(field)}`, committed[field])
      } catch (error) {
        // Roll back only fields still owned by this request. A newer keystroke/update
        // remains visible and will run next in the per-project queue.
        set((st) => ({ projects: st.projects.map((project) => {
          if (project.id !== id) return project
          const rollback: Partial<Project> = {}
          for (const field of touched) {
            const key = `${id}:${String(field)}`
            if (projectUpdateState.fieldVersions.get(key) === version) (rollback as any)[field] = projectUpdateState.committedValues.get(key)
          }
          return { ...project, ...rollback }
        }) }))
        throw error
      } finally {
        for (const field of touched) {
          const key = `${id}:${String(field)}`
          if (projectUpdateState.fieldVersions.get(key) === version) {
            projectUpdateState.fieldVersions.delete(key)
            projectUpdateState.committedValues.delete(key)
          }
        }
        if (projectUpdateState.latestVersions.get(id) === version) projectUpdateState.latestVersions.delete(id)
      }
    })
  },
  async renameProject(id, name) {
    const finalName = name.trim() || 'New project'
    await get().updateProject(id, { name: finalName })
  },
  async deleteProject(id) {
    await get().archiveItem('project', id)
  },
})
