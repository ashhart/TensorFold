// Global orchestrator mode resolution (VST-041): decide WHERE something the user starts
// (thread, chat, terminal) should execute. One pure function so the store, the composer
// chip, and the terminals tab all agree — the effective target is a visible contract, not
// a hidden redirect.
//
// Resolution order:
// 1. A project explicitly targeting an orchestrator keeps ITS host (its own pin wins over
//    the global default host).
// 2. A `localOnly` project is a hard local pin — its cwd/providers may not exist on the
//    orchestrator, so global mode never overrides it.
// 3. Global mode: settings.defaultExecutionTarget === 'orchestrator' routes everything
//    else (including project-less chats/tasks) to the default orchestrator host.
// 4. Otherwise: local.

import type { Project, RemoteOrchestratorProfile, Settings } from './types'

export type ExecutionTargetSource = 'project' | 'local_only' | 'global' | 'local'

export interface ExecutionTargetDecision {
  hostId: string | null
  // Why this target was chosen — surfaced in the composer chip so a redirect is never silent.
  source: ExecutionTargetSource
  // Set when a mode asked for an orchestrator but no usable host exists (disabled, missing,
  // or none configured): the caller must surface this instead of silently running local.
  unavailableReason: string | null
}

type SettingsSlice = Pick<Settings, 'remoteOrchestrators' | 'defaultExecutionTarget' | 'defaultOrchestratorId'>
type ProjectSlice = Pick<Project, 'executionTarget' | 'orchestratorId' | 'localOnly'> | null | undefined

function usableHost(hosts: RemoteOrchestratorProfile[] | undefined, id: string | null | undefined): RemoteOrchestratorProfile | null {
  const list = (hosts ?? []).filter((host) => host.enabled)
  if (id) return list.find((host) => host.id === id) ?? null
  return list[0] ?? null
}

export function resolveExecutionTarget(settings: SettingsSlice | null | undefined, project: ProjectSlice): ExecutionTargetDecision {
  if (project?.executionTarget === 'orchestrator') {
    const host = usableHost(settings?.remoteOrchestrators, project.orchestratorId)
    return host
      ? { hostId: host.id, source: 'project', unavailableReason: null }
      : { hostId: null, source: 'project', unavailableReason: 'This project targets an Orchestrator, but no enabled host matches — check Settings → Orchestrator.' }
  }
  if (project?.localOnly) {
    return { hostId: null, source: 'local_only', unavailableReason: null }
  }
  if (settings?.defaultExecutionTarget === 'orchestrator') {
    const host = usableHost(settings.remoteOrchestrators, settings.defaultOrchestratorId)
    return host
      ? { hostId: host.id, source: 'global', unavailableReason: null }
      : { hostId: null, source: 'global', unavailableReason: 'Orchestrator mode is on, but no enabled host is configured — check Settings → Orchestrator.' }
  }
  return { hostId: null, source: 'local', unavailableReason: null }
}

// Short label for the composer chip / terminal badges: names the host and, when global
// mode drove the decision, says so.
export function executionTargetLabel(decision: ExecutionTargetDecision, hosts: RemoteOrchestratorProfile[] | undefined): string | null {
  if (!decision.hostId) return decision.unavailableReason ? 'Orchestrator unavailable' : null
  const name = (hosts ?? []).find((host) => host.id === decision.hostId)?.name || 'Orchestrator'
  return decision.source === 'global' ? `${name} · all new work` : name
}
