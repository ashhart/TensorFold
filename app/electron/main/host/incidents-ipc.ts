import { randomUUID } from 'node:crypto'
import { runAgent } from '../../agent/loop'
import {
  createMission,
  startMission,
} from '../../agent/missions'
import type { AgentEvent, Mission, Provider, Thread } from '../../agent/types'
import type { CreateIncidentInput, Incident, IncidentEventKind, IncidentMitigation } from '../../agent/types'
import { assertProviderEndpoint, auditPreview } from '../../agent/security'
import { selectedProviderAccountId } from './provider-runtime'
import { store } from '../state'
import { auditEvent, requireSecurityRole } from './audit'
import { handle } from './ipc'
import { persistIncidents, persistMissions } from './persistence'
import { resolveWorkspaceCwd } from './workspace'
import { sanitizeIncidentCreate, sanitizeIncidentPatch } from './public-config-policy'

interface IncidentIpcDeps {
  effectiveContextLimit(provider: Provider, model: string | null): number
  fullToolContext(run?: { thread?: Thread; provider?: Provider; model?: string; signal?: AbortSignal }): unknown
  applyEventToThread(thread: Thread, event: AgentEvent): void
  replaceMission(mission: Mission): Mission
  getMission(id: string): Mission
  runMission(mission: Mission): Promise<void>
  emitIncident(incident: Incident): void
}

export const newIncidentEventId = () => 'ie_' + randomUUID().slice(0, 8)

function findIncident(id: string): Incident {
  const inc = store.incidents.find((i) => i.id === id)
  if (!inc) throw new Error('Incident not found.')
  return inc
}

function touchIncident(inc: Incident, emitIncident: (incident: Incident) => void) {
  inc.updatedAt = Date.now()
  persistIncidents()
  emitIncident(inc)
}

async function runReadonlyAgentScan(deps: IncidentIpcDeps, cwd: string, objective: string, opts?: { maxSteps?: number; timeoutMs?: number }): Promise<string> {
  const provider = store.settings.providers.find((p) => p.id === store.settings.activeProviderId)
  const model = store.settings.activeModel
  if (!provider || !model) throw new Error('No provider/model configured.')
  assertProviderEndpoint(provider)
  const safeCwd = await resolveWorkspaceCwd(cwd, 'incident scan')
  const controller = new AbortController()
  const timeoutMs = opts?.timeoutMs ?? 4 * 60_000
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  const thread: Thread = {
    id: randomUUID(), title: 'Incident scan', kind: 'task', projectId: null,
    cwd: safeCwd, providerId: provider.id, model, createdAt: Date.now(), updatedAt: Date.now(),
    status: 'running', apiMessages: [{ role: 'user', content: objective }], timeline: [],
  }
  try {
    await runAgent({
      thread, provider, model,
      approvalMode: 'readonly',
      temperature: store.settings.temperature, reasoningEffort: store.settings.reasoningEffort,
      autoCompact: store.settings.autoCompact, contextLimit: deps.effectiveContextLimit(provider, model),
      toolContext: deps.fullToolContext({ thread, provider, model, signal: controller.signal }) as any,
      maxSteps: opts?.maxSteps ?? 24, maxWallMs: timeoutMs,
      emit: (e) => deps.applyEventToThread(thread, e),
      signal: controller.signal,
      requestApproval: async () => 'deny',
      persist: () => {},
    })
  } finally { clearTimeout(timer) }
  const texts = thread.timeline.filter((i) => i.kind === 'assistant' && i.text).map((i) => i.text!.trim()).filter(Boolean)
  return texts.join('\n\n').trim() || 'No findings produced.'
}

const INCIDENT_SCAN_PROMPT = (inc: Incident) => `You are running a STRICT READ-ONLY first-pass triage for a production incident. You have no write or command-execution permission — any such attempt will be denied, so do not rely on it.

Incident: ${inc.title} (${inc.severity.toUpperCase()})${inc.summary ? `\nReported: ${inc.summary}` : ''}
Working directory: ${inc.cwd}

Do this:
1. Read logs, config, and recent changes that are visible here.
2. Identify likely symptoms, error signatures, and suspicious recent edits.
3. Propose — but DO NOT apply — candidate mitigations, each with a one-line rationale and a rollback note.

Return concise markdown with sections: Symptoms, Evidence, Likely cause, Candidate mitigations (numbered). Cite file paths / log lines. If something cannot be read, say so rather than guessing.`

export function registerIncidentIpc(deps: IncidentIpcDeps) {
  handle('incidents:get', async () => store.incidents)
  handle('incidents:create', async (_e, requestedInput: CreateIncidentInput) => {
    requireSecurityRole('operator', 'create incidents')
    const input = sanitizeIncidentCreate(requestedInput)
    const now = Date.now()
    const cwd = await resolveWorkspaceCwd(input.cwd ?? store.settings.defaultCwd, 'incident workspace')
    const inc: Incident = {
      id: 'inc_' + randomUUID().slice(0, 8), title: input.title || 'Untitled incident',
      severity: input.severity ?? 'sev3', status: 'open', cwd, summary: input.summary || undefined,
      createdAt: now, updatedAt: now,
      timeline: [{ id: newIncidentEventId(), kind: 'note', text: 'Incident opened.', at: now, source: 'user' }],
      mitigations: [], scan: { status: 'idle' },
    }
    store.incidents = [inc, ...store.incidents].slice(0, 500)
    persistIncidents()
    await auditEvent('incident.create', { id: inc.id, severity: inc.severity, cwd: auditPreview(cwd) })
    return inc
  })
  handle('incidents:update', async (_e, id: string, requestedPatch: Partial<Incident>) => {
    requireSecurityRole('operator', 'update incidents')
    const inc = findIncident(id)
    const patch = sanitizeIncidentPatch(requestedPatch)
    if (Object.prototype.hasOwnProperty.call(patch, 'title')) inc.title = patch.title!
    if (Object.prototype.hasOwnProperty.call(patch, 'severity')) inc.severity = patch.severity!
    if (Object.prototype.hasOwnProperty.call(patch, 'status')) inc.status = patch.status!
    if (Object.prototype.hasOwnProperty.call(patch, 'summary')) inc.summary = patch.summary
    touchIncident(inc, deps.emitIncident)
    return inc
  })
  handle('incidents:delete', async (_e, id: string) => {
    requireSecurityRole('operator', 'delete incidents')
    store.incidents = store.incidents.filter((i) => i.id !== id)
    persistIncidents()
  })
  handle('incidents:addEvent', async (_e, id: string, kind: IncidentEventKind, text: string) => {
    requireSecurityRole('operator', 'add incident events')
    const inc = findIncident(id)
    inc.timeline.push({ id: newIncidentEventId(), kind: kind ?? 'note', text: String(text || '').slice(0, 4000), at: Date.now(), source: 'user' })
    touchIncident(inc, deps.emitIncident)
    return inc
  })
  handle('incidents:proposeMitigation', async (_e, id: string, text: string, rationale?: string) => {
    requireSecurityRole('operator', 'propose incident mitigations')
    const inc = findIncident(id)
    const m: IncidentMitigation = { id: 'im_' + randomUUID().slice(0, 8), text: String(text || '').slice(0, 2000), rationale: rationale ? String(rationale).slice(0, 1000) : undefined, status: 'proposed', proposedAt: Date.now() }
    inc.mitigations.push(m)
    inc.timeline.push({ id: newIncidentEventId(), kind: 'mitigation', text: `Proposed: ${m.text}`, at: m.proposedAt, source: 'user' })
    touchIncident(inc, deps.emitIncident)
    return inc
  })
  handle('incidents:decideMitigation', async (_e, id: string, mitigationId: string, decision: IncidentMitigation['status']) => {
    requireSecurityRole('operator', 'decide incident mitigations')
    const inc = findIncident(id)
    const m = inc.mitigations.find((x) => x.id === mitigationId)
    if (!m) throw new Error('Mitigation not found.')
    m.status = decision
    m.decidedAt = Date.now()
    inc.timeline.push({ id: newIncidentEventId(), kind: 'mitigation', text: `Mitigation ${decision}: ${m.text}`, at: m.decidedAt, source: 'user' })
    if (decision === 'approved' && inc.status === 'open') inc.status = 'mitigating'
    touchIncident(inc, deps.emitIncident)
    await auditEvent('incident.mitigation', { id: inc.id, mitigationId, decision })
    return inc
  })
  handle('incidents:executeMitigation', async (_e, id: string, mitigationId: string) => {
    requireSecurityRole('operator', 'execute incident mitigations')
    const inc = findIncident(id)
    const m = inc.mitigations.find((x) => x.id === mitigationId)
    if (!m) throw new Error('Mitigation not found.')
    if (m.status !== 'approved') throw new Error('Approve the mitigation before executing it.')
    if (m.missionId) throw new Error('This mitigation is already being executed.')
    const provider = store.settings.providers.find((p) => p.id === store.settings.activeProviderId)
    const objective = `Apply the following APPROVED incident mitigation to the workspace, then report exactly what you changed.

Incident: ${inc.title} (${inc.severity.toUpperCase()})
Mitigation: ${m.text}${m.rationale ? `\nRationale: ${m.rationale}` : ''}

Make the SMALLEST change that implements the mitigation; do no unrelated work. After editing, summarize the diff and how to verify and roll it back. If the change looks unsafe or out of scope, stop and explain instead of proceeding.`
    const mission = createMission({
      title: `Mitigate: ${inc.title}`, objective, cwd: inc.cwd,
      providerId: store.settings.activeProviderId,
      providerAccountId: provider ? selectedProviderAccountId(provider, undefined) : null,
      model: store.settings.activeModel,
      destructiveApprovalRequired: true,
      metadata: { allowMutations: true, incidentId: inc.id, mitigationId },
    })
    store.missions.unshift(mission)
    persistMissions()
    const started = deps.replaceMission(startMission(mission))
    if (started.status === 'running') {
      void deps.runMission(started).then(() => {
        try {
          const cur = store.incidents.find((i) => i.id === id)
          if (!cur) return
          const mm = cur.mitigations.find((x) => x.id === mitigationId)
          if (!mm) return
          const fin = deps.getMission(started.id)
          if (fin.status === 'completed') { mm.status = 'applied'; cur.timeline.push({ id: newIncidentEventId(), kind: 'mitigation', text: `Mitigation applied via mission ${started.id}.`, at: Date.now(), source: 'agent' }) }
          else { cur.timeline.push({ id: newIncidentEventId(), kind: 'mitigation', text: `Mitigation mission ${fin.status}.`, at: Date.now(), source: 'agent' }) }
          touchIncident(cur, deps.emitIncident)
        } catch { /* mission record gone */ }
      })
    }
    m.missionId = started.id
    inc.timeline.push({ id: newIncidentEventId(), kind: 'mitigation', text: `Execution started as guarded mission ${started.id} — mutating tools require per-step approval.`, at: Date.now(), source: 'user' })
    if (inc.status === 'open' || inc.status === 'investigating') inc.status = 'mitigating'
    await auditEvent('incident.mitigation.execute', { id: inc.id, mitigationId, missionId: started.id })
    touchIncident(inc, deps.emitIncident)
    return inc
  })
  handle('incidents:scan', async (_e, id: string) => {
    requireSecurityRole('operator', 'scan incidents')
    const inc = findIncident(id)
    if (inc.scan.status === 'running') return inc
    inc.scan = { status: 'running' }
    if (inc.status === 'open') inc.status = 'investigating'
    touchIncident(inc, deps.emitIncident)
    try {
      const findings = await runReadonlyAgentScan(deps, inc.cwd, INCIDENT_SCAN_PROMPT(inc), { maxSteps: 24 })
      const at = Date.now()
      inc.scan = { status: 'done', findings, ranAt: at }
      inc.timeline.push({ id: newIncidentEventId(), kind: 'scan', text: findings, at, source: 'scan' })
      await auditEvent('incident.scan', { id: inc.id, ok: true })
    } catch (e: any) {
      inc.scan = { status: 'error', error: String(e?.message ?? e), ranAt: Date.now() }
      await auditEvent('incident.scan', { id: inc.id, ok: false })
    }
    touchIncident(inc, deps.emitIncident)
    return inc
  })
  handle('incidents:postmortem', async (_e, id: string) => {
    requireSecurityRole('operator', 'generate incident postmortems')
    const inc = findIncident(id)
    const timelineText = inc.timeline.map((ev) => `- [${ev.kind}] ${ev.text}`).join('\n').slice(0, 8000)
    const mitigationsText = inc.mitigations.map((m) => `- (${m.status}) ${m.text}${m.rationale ? ` — ${m.rationale}` : ''}`).join('\n') || '- none recorded'
    const prompt = `Write a blameless postmortem in markdown for this incident. Be concise and factual; use only the information given, do not invent specifics.

Title: ${inc.title}
Severity: ${inc.severity.toUpperCase()}
Status: ${inc.status}${inc.summary ? `\nSummary: ${inc.summary}` : ''}

Timeline:
${timelineText}

Mitigations:
${mitigationsText}

Sections: Summary, Impact, Timeline, Root cause, Resolution, Action items (each with an [owner] placeholder), Lessons learned.`
    const postmortem = await runReadonlyAgentScan(deps, inc.cwd, prompt, { maxSteps: 8, timeoutMs: 3 * 60_000 })
    inc.postmortem = postmortem
    inc.postmortemAt = Date.now()
    if (inc.status === 'resolved') inc.status = 'closed'
    touchIncident(inc, deps.emitIncident)
    return inc
  })
}
