import type { Project, Provider, WorkspaceLocation } from '../../agent/types'
import type { KanbanPhaseModel, ProjectKanbanSettings } from '../../agent/types'

export const PROJECT_UPDATE_FIELDS = [
  'name',
  'tags',
  'kanban',
  'localOnly',
  'defaultProviderId',
  'defaultProviderAccountId',
  'defaultModel',
  'memory',
  'team',
] as const satisfies readonly (keyof Project)[]

const PROJECT_CREATE_FIELDS = [
  'id', 'name', 'cwd', 'workspace', 'memory', 'team', 'mode', 'localOnly',
  'executionTarget', 'orchestratorId', 'defaultProviderId', 'defaultProviderAccountId',
  'defaultModel', 'kanban', 'tags', 'createdAt',
] as const satisfies readonly (keyof Project)[]

/**
 * Copy only declared own properties into a fresh patch. Rejecting unknown keys is
 * intentional: Object.assign(target, JSON.parse('{"__proto__":...}')) invokes the
 * prototype setter and can otherwise smuggle archive/origin authority past deny lists.
 */
export function pickStrictOwnPatch<T extends object, K extends keyof T>(
  input: unknown,
  allowedFields: readonly K[],
  label: string,
): Partial<Pick<T, K>> {
  if (!input || typeof input !== 'object' || Array.isArray(input)) throw new Error(`${label} patch must be an object.`)
  const allowed = new Set<string>(allowedFields as readonly string[])
  for (const key of Object.keys(input)) {
    if (!allowed.has(key)) throw new Error(`${key} cannot be changed through ${label} updates.`)
  }
  const source = input as T
  const out: Partial<Pick<T, K>> = {}
  for (const field of allowedFields) {
    if (Object.prototype.hasOwnProperty.call(source, field)) {
      Object.defineProperty(out, field, { value: source[field], enumerable: true, configurable: true, writable: true })
    }
  }
  return out
}

export function sanitizeProjectUpdatePatch(input: unknown): Partial<Project> {
  const patch = pickStrictOwnPatch<Project, typeof PROJECT_UPDATE_FIELDS[number]>(input, PROJECT_UPDATE_FIELDS, 'project')
  if (patch.name !== undefined && (typeof patch.name !== 'string' || !patch.name.trim() || patch.name.length > 200)) throw new Error('Project name must be a non-empty string of at most 200 characters.')
  if (patch.memory !== undefined && (typeof patch.memory !== 'string' || patch.memory.length > 1_000_000)) throw new Error('Project memory must be text no larger than 1,000,000 characters.')
  if (patch.team !== undefined && (typeof patch.team !== 'string' || patch.team.length > 1_000_000)) throw new Error('Project team guidance must be text no larger than 1,000,000 characters.')
  if (patch.localOnly !== undefined && typeof patch.localOnly !== 'boolean') throw new Error('Project localOnly must be a boolean.')
  for (const field of ['defaultProviderId', 'defaultProviderAccountId', 'defaultModel'] as const) {
    const value = patch[field]
    if (value !== undefined && value !== null && (typeof value !== 'string' || value.length > 1_000)) throw new Error(`Project ${field} must be a string no longer than 1,000 characters or null.`)
  }
  if (patch.tags !== undefined) {
    if (!Array.isArray(patch.tags) || patch.tags.length > 64 || patch.tags.some((tag) => typeof tag !== 'string' || tag.length > 80)) throw new Error('Project tags must be an array of at most 64 short strings.')
    patch.tags = [...new Set(patch.tags.map((tag) => tag.trim()).filter(Boolean))]
  }
  if (patch.kanban !== undefined) patch.kanban = sanitizeKanbanSettings(patch.kanban)
  return patch
}

function sanitizeWorkspaceLocation(value: unknown): WorkspaceLocation | null | undefined {
  if (value === undefined || value === null) return value
  const workspace = pickStrictOwnPatch<WorkspaceLocation, keyof WorkspaceLocation>(value, ['kind', 'hostId', 'path'], 'project workspace')
  if (workspace.kind !== 'local' && workspace.kind !== 'ssh') throw new Error('Project workspace kind must be local or ssh.')
  if (typeof workspace.path !== 'string' || !workspace.path.trim() || workspace.path.length > 4_096) throw new Error('Project workspace path must be non-empty text no longer than 4,096 characters.')
  if (workspace.kind === 'ssh') {
    if (typeof workspace.hostId !== 'string' || !workspace.hostId.trim() || workspace.hostId.length > 200) throw new Error('SSH project workspaces require a short host id.')
  } else if (workspace.hostId !== undefined && workspace.hostId !== null) {
    throw new Error('Local project workspaces cannot name an SSH host.')
  }
  return { kind: workspace.kind, path: workspace.path.trim(), ...(workspace.kind === 'ssh' ? { hostId: workspace.hostId!.trim() } : {}) }
}

export function sanitizeProjectCreateInput(input: unknown, options: { allowId?: boolean; allowCreatedAt?: boolean } = {}): Partial<Project> {
  const raw = pickStrictOwnPatch<Project, typeof PROJECT_CREATE_FIELDS[number]>(input, PROJECT_CREATE_FIELDS, 'project creation')
  if (raw.id !== undefined && (!options.allowId || typeof raw.id !== 'string' || !raw.id.trim() || raw.id.length > 200)) throw new Error('Project creation cannot choose this id.')
  if (raw.cwd !== undefined && (typeof raw.cwd !== 'string' || raw.cwd.length > 4_096)) throw new Error('Project cwd must be text no longer than 4,096 characters.')
  if (raw.mode !== undefined && raw.mode !== 'solo' && raw.mode !== 'room') throw new Error('Project mode must be solo or room.')
  if (raw.executionTarget !== undefined && raw.executionTarget !== 'local' && raw.executionTarget !== 'orchestrator') throw new Error('Project executionTarget must be local or orchestrator.')
  if (raw.orchestratorId !== undefined && raw.orchestratorId !== null && (typeof raw.orchestratorId !== 'string' || !raw.orchestratorId.trim() || raw.orchestratorId.length > 200)) throw new Error('Project orchestratorId must be a short string or null.')
  if (raw.createdAt !== undefined && (!options.allowCreatedAt || !Number.isFinite(raw.createdAt) || raw.createdAt < 0)) throw new Error('Project creation cannot choose this createdAt value.')
  raw.workspace = sanitizeWorkspaceLocation(raw.workspace)
  const config = sanitizeProjectUpdatePatch({
    ...(Object.prototype.hasOwnProperty.call(raw, 'name') ? { name: raw.name } : {}),
    ...(Object.prototype.hasOwnProperty.call(raw, 'memory') ? { memory: raw.memory } : {}),
    ...(Object.prototype.hasOwnProperty.call(raw, 'team') ? { team: raw.team } : {}),
    ...(Object.prototype.hasOwnProperty.call(raw, 'localOnly') ? { localOnly: raw.localOnly } : {}),
    ...(Object.prototype.hasOwnProperty.call(raw, 'defaultProviderId') ? { defaultProviderId: raw.defaultProviderId } : {}),
    ...(Object.prototype.hasOwnProperty.call(raw, 'defaultProviderAccountId') ? { defaultProviderAccountId: raw.defaultProviderAccountId } : {}),
    ...(Object.prototype.hasOwnProperty.call(raw, 'defaultModel') ? { defaultModel: raw.defaultModel } : {}),
    ...(Object.prototype.hasOwnProperty.call(raw, 'kanban') ? { kanban: raw.kanban } : {}),
    ...(Object.prototype.hasOwnProperty.call(raw, 'tags') ? { tags: raw.tags } : {}),
  })
  return { ...raw, ...config }
}

const KANBAN_FIELDS = ['modelMode', 'single', 'draft', 'ticketReview', 'review', 'build', 'finalGate'] as const satisfies readonly (keyof ProjectKanbanSettings)[]
const PHASE_MODEL_FIELDS = ['providerId', 'providerAccountId', 'model'] as const satisfies readonly (keyof KanbanPhaseModel)[]

function sanitizePhaseModel(input: unknown, label: string): KanbanPhaseModel {
  const model = pickStrictOwnPatch<KanbanPhaseModel, typeof PHASE_MODEL_FIELDS[number]>(input, PHASE_MODEL_FIELDS, label)
  for (const field of PHASE_MODEL_FIELDS) {
    const value = model[field]
    if (value !== undefined && value !== null && (typeof value !== 'string' || value.length > 1_000)) throw new Error(`${label} ${field} must be a string no longer than 1,000 characters or null.`)
  }
  return model
}

function sanitizeKanbanSettings(input: unknown): ProjectKanbanSettings {
  const settings = pickStrictOwnPatch<ProjectKanbanSettings, typeof KANBAN_FIELDS[number]>(input, KANBAN_FIELDS, 'project kanban')
  if (settings.modelMode !== undefined && settings.modelMode !== 'single' && settings.modelMode !== 'phased') throw new Error('Project kanban modelMode must be single or phased.')
  for (const field of ['single', 'draft', 'ticketReview', 'review', 'build', 'finalGate'] as const) {
    if (settings[field] !== undefined) settings[field] = sanitizePhaseModel(settings[field], `project kanban ${field}`)
  }
  return settings
}

export function validateProjectProviderDefaults(current: Project, patch: Partial<Project>, providers: readonly Provider[]): void {
  const touchesRoute = ['localOnly', 'defaultProviderId', 'defaultProviderAccountId', 'defaultModel', 'kanban'].some((field) => Object.prototype.hasOwnProperty.call(patch, field))
  if (!touchesRoute) return
  const next = { ...current, ...patch }
  if (!next.defaultProviderId) {
    if (next.defaultProviderAccountId) throw new Error('Choose a project provider before choosing its account.')
    // A model-only override intentionally follows the active/first eligible provider
    // in resolveProjectProvider; preserve that supported legacy/default behavior.
  } else {
    const provider = providers.find((candidate) => candidate.id === next.defaultProviderId)
    if (!provider) throw new Error('Choose a configured project provider.')
    if (next.localOnly && provider.kind !== 'local') throw new Error('A local-only project must use a local provider.')
    const account = next.defaultProviderAccountId ? provider.accounts?.find((candidate) => candidate.id === next.defaultProviderAccountId) : null
    if (next.defaultProviderAccountId && !account) throw new Error('Choose an account that belongs to the project provider.')
    const models = account?.models?.length ? account.models : provider.models
    if (next.defaultModel && models.length && !models.includes(next.defaultModel)) throw new Error('Choose a model offered by the project provider/account.')
  }
  validateKanbanProviderRoutes(next, providers)
}

function validateKanbanProviderRoutes(project: Project, providers: readonly Provider[]): void {
  const kanban = project.kanban
  if (!kanban) return
  for (const field of ['single', 'draft', 'ticketReview', 'review', 'build', 'finalGate'] as const) {
    const route = kanban[field]
    if (!route) continue
    const hasRoute = !!route.providerId || !!route.providerAccountId || !!route.model
    if (!hasRoute) continue
    if (!route.providerId || !route.model) throw new Error(`Project kanban ${field} must choose both a provider and model.`)
    const provider = providers.find((candidate) => candidate.id === route.providerId)
    if (!provider) throw new Error(`Choose a configured provider for project kanban ${field}.`)
    if (project.localOnly && provider.kind !== 'local') throw new Error(`A local-only project must use a local provider for project kanban ${field}.`)
    const account = route.providerAccountId ? provider.accounts?.find((candidate) => candidate.id === route.providerAccountId) : null
    if (route.providerAccountId && !account) throw new Error(`Choose an account that belongs to the project kanban ${field} provider.`)
    const models = account?.models?.length ? account.models : provider.models
    if (models.length && !models.includes(route.model)) throw new Error(`Choose a model offered by the project kanban ${field} provider/account.`)
  }
}

export function validateProjectExecutionTargetInput(
  executionTarget: unknown,
  orchestratorId: unknown,
  hosts: readonly { id: string; enabled?: boolean }[],
): { executionTarget: 'local' | 'orchestrator'; orchestratorId: string | null } {
  if (executionTarget !== 'local' && executionTarget !== 'orchestrator') throw new Error('Choose Local or Orchestrator execution.')
  if (executionTarget === 'local') return { executionTarget, orchestratorId: null }
  const hostId = typeof orchestratorId === 'string' ? orchestratorId.trim() : ''
  if (!hosts.some((candidate) => candidate.id === hostId && candidate.enabled)) throw new Error('Choose an enabled Orchestrator before routing project work remotely.')
  return { executionTarget, orchestratorId: hostId }
}
