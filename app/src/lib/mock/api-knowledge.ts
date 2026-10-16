import type { Command, ConnectorEvent, ProjectMemoryCandidate, ProjectMemoryEntry, Skill, WebhookDeliveryRecord, WebhookEndpoint } from '../types'
import type { Incident, RunExportBundle } from '../types'
import type { TensorFoldStudioApi } from '../ipc'
import { ago, mockState, uid } from './state'
import { mockWebhookSnapshot } from './runtime'

type KnowledgeMockApi = Pick<
  TensorFoldStudioApi,
  | 'getSkills'
  | 'saveSkill'
  | 'deleteSkill'
  | 'getProjectSkills'
  | 'trustProjectSkill'
  | 'suggestionDiff'
  | 'listArtifacts'
  | 'previewArtifact'
  | 'redactArtifact'
  | 'exportArtifact'
  | 'deleteArtifact'
  | 'getObserverStatus'
  | 'getProjectMemory'
  | 'getProjectMemoryCandidates'
  | 'proposeProjectMemory'
  | 'approveProjectMemoryCandidate'
  | 'rejectProjectMemoryCandidate'
  | 'memoryCandidateDiff'
  | 'rollbackMemoryEntry'
  | 'saveProjectMemory'
  | 'deleteProjectMemory'
  | 'getWebhooks'
  | 'saveWebhook'
  | 'approveWebhook'
  | 'deleteWebhook'
  | 'buildExport'
  | 'saveExport'
  | 'importBundle'
  | 'getIncidents'
  | 'createIncident'
  | 'updateIncident'
  | 'deleteIncident'
  | 'addIncidentEvent'
  | 'proposeMitigation'
  | 'decideMitigation'
  | 'executeMitigation'
  | 'scanIncident'
  | 'generatePostmortem'
  | 'testWebhook'
  | 'getCommands'
  | 'saveCommand'
  | 'deleteCommand'
  | 'getSuggestions'
  | 'acceptSuggestion'
  | 'dismissSuggestion'
  | 'onSuggestionsUpdate'
>

function strictMockConfig<T extends object>(input: unknown, allowed: readonly string[], label: string): Partial<T> {
  if (!input || typeof input !== 'object' || Array.isArray(input)) throw new Error(`${label} must be a plain object.`)
  const prototype = Object.getPrototypeOf(input)
  if (prototype !== Object.prototype && prototype !== null) throw new Error(`${label} must not inherit fields.`)
  const allowedSet = new Set(allowed)
  const out = Object.create(null) as Partial<T>
  for (const key of Reflect.ownKeys(input)) {
    if (typeof key !== 'string' || !allowedSet.has(key)) throw new Error(`${String(key)} cannot be changed through ${label}.`)
    const descriptor = Object.getOwnPropertyDescriptor(input, key)
    if (!descriptor?.enumerable || !('value' in descriptor)) throw new Error(`${key} must be an ordinary field.`)
    if (descriptor.value === undefined) continue
    Object.defineProperty(out, key, { value: descriptor.value, enumerable: true, configurable: true, writable: true })
  }
  return out
}

function assertMockUniqueName(records: readonly { id: string; name: string }[], name: string, excludeId?: string): void {
  const normalized = name.trim().toLocaleLowerCase()
  if (records.some((record) => record.id !== excludeId && record.name.trim().toLocaleLowerCase() === normalized)) throw new Error('Name already exists.')
}

function assertMockCommandName(name: string): void {
  if (!/^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/.test(name) || ['mission', 'compact', 'goal'].includes(name.toLocaleLowerCase())) throw new Error('Command name is invalid or reserved.')
}

export function createMockKnowledgeApi(): KnowledgeMockApi {
  return {
    getSkills: async () => mockState.mockSkills,
    saveSkill: async (requested) => {
      const s = strictMockConfig<Skill>(requested, ['id', 'name', 'description', 'body', 'enabled', 'trigger'], 'skill save')
      if (s.id) { const ex = mockState.mockSkills.find((x) => x.id === s.id); if (!ex) throw new Error('Skill not found.'); if (ex.scope === 'project' || ex.projectId) throw new Error('Project skills are repo-owned.'); Object.assign(ex, s, { id: ex.id, createdAt: ex.createdAt, updatedAt: Date.now() }); return ex }
      const skill: Skill = { id: uid(), name: s.name || 'skill', description: s.description || '', body: s.body || '', enabled: s.enabled ?? true, ...(s.trigger?.trim() ? { trigger: s.trigger.trim() } : {}), createdAt: Date.now(), updatedAt: Date.now() }
      mockState.mockSkills = [skill, ...mockState.mockSkills]; return skill
    },
    deleteSkill: async (id) => { mockState.mockSkills = mockState.mockSkills.filter((s) => s.id !== id) },
    // VST-010: repo-authored project skills with hash-pinned trust (simulated).
    getProjectSkills: async (projectId) => mockState.mockProjectSkills?.filter((v) => v.skill.projectId === projectId) ?? [],
    trustProjectSkill: async ({ projectId, skillId, trusted }) => {
      mockState.mockProjectSkills = (mockState.mockProjectSkills ?? []).map((v) =>
        v.skill.projectId === projectId && (v.skill.id === skillId || v.skill.id.endsWith(`_${skillId}`))
          ? { ...v, trust: trusted ? 'trusted' as const : 'untrusted' as const, skill: { ...v.skill, enabled: trusted } }
          : v)
      return mockState.mockProjectSkills.filter((v) => v.skill.projectId === projectId)
    },
    suggestionDiff: async (id) => {
      const suggestion = (await Promise.resolve(mockState.mockSuggestions ?? [])).find((s) => s.id === id)
      if (!suggestion) throw new Error('suggestion not found')
      const target = suggestion.targetSkillId ? mockState.mockSkills.find((s) => s.id === suggestion.targetSkillId) ?? null : null
      const fields = [
        { field: 'name' as const, before: target?.name ?? null, after: suggestion.name },
        { field: 'description' as const, before: target?.description ?? null, after: suggestion.description },
        { field: 'body' as const, before: target?.body ?? null, after: suggestion.body },
        { field: 'argHint' as const, before: null, after: suggestion.argHint ?? null },
      ]
      return {
        suggestionId: suggestion.id,
        action: target ? 'update' as const : 'create' as const,
        targetSkillId: target?.id ?? null,
        changes: fields.filter(({ before, after }) => (target ? before !== after : after != null && after !== '')),
      }
    },
    // First-class artifacts (VST-011) — demo parity.
    listArtifacts: async (threadId) => threadId ? mockState.mockArtifacts.filter((a) => a.provenance.threadId === threadId) : mockState.mockArtifacts,
    previewArtifact: async (id) => {
      const a = mockState.mockArtifacts.find((x) => x.id === id)
      if (!a) throw new Error('artifact not found')
      const meta = [a.kind, a.sizeBytes != null ? `${a.sizeBytes} B` : 'size unknown', a.path ?? 'inline', a.redaction === 'redacted' ? 'redacted' : null].filter(Boolean).join(' · ')
      const textual = ['document', 'report', 'log', 'file', 'spreadsheet', 'export'].includes(a.kind)
      const text = a.inlineText
      if (!textual || text == null) return { kind: a.kind, title: a.title, excerpt: null, truncated: false, meta }
      return { kind: a.kind, title: a.title, excerpt: text.slice(0, 1200), truncated: text.length > 1200, meta }
    },
    redactArtifact: async (id) => {
      const at = mockState.mockArtifacts.findIndex((x) => x.id === id)
      if (at < 0) throw new Error('artifact not found')
      const a = mockState.mockArtifacts[at]
      const textual = ['document', 'report', 'log', 'file', 'spreadsheet', 'export'].includes(a.kind)
      const masked = (a.inlineText ?? '').replace(/[\w.+-]+@[\w-]+\.[\w.]+/g, '[email]').replace(/\bsk-[A-Za-z0-9_-]{16,}\b/g, '[secret]')
      const next = textual && a.inlineText != null
        ? { ...a, inlineText: masked, redaction: 'redacted' as const, redactionReport: [{ kind: 'email' as const, count: 1 }], updatedAt: Date.now() }
        : { ...a, redaction: 'not_required' as const, redactionReport: [], updatedAt: Date.now() }
      mockState.mockArtifacts[at] = next
      return next
    },
    exportArtifact: async (id) => {
      const a = mockState.mockArtifacts.find((x) => x.id === id)
      if (!a) throw new Error('artifact not found')
      return { artifact: a, content: a.inlineText }
    },
    deleteArtifact: async (id) => { mockState.mockArtifacts = mockState.mockArtifacts.filter((x) => x.id !== id) },
    getObserverStatus: async () => ({ hooks: [], recent: [] }),
    getProjectMemory: async (projectId) => mockState.mockProjectMemory.filter((m) => m.projectId === projectId),
    getProjectMemoryCandidates: async (projectId) => mockState.mockProjectMemoryCandidates.filter((m) => m.projectId === projectId && m.status === 'pending'),
    proposeProjectMemory: async (m) => {
      const now = Date.now()
      const candidate: ProjectMemoryCandidate = { id: m.id || uid(), projectId: m.projectId!, kind: m.kind || 'summary', text: (m.text || '').trim(), sourceType: m.sourceType || 'manual', sourceRef: m.sourceRef ?? null, confidence: m.confidence || 'medium', privacyScope: m.privacyScope || 'project', owner: m.owner ?? null, reason: m.reason ?? null, status: 'pending', targetEntryId: m.targetEntryId ?? null, reviewedAt: null, reviewedBy: null, durableEntryId: null, createdAt: now, updatedAt: now }
      mockState.mockProjectMemoryCandidates = [candidate, ...mockState.mockProjectMemoryCandidates]
      return candidate
    },
    approveProjectMemoryCandidate: async (id) => {
      const now = Date.now()
      const candidate = mockState.mockProjectMemoryCandidates.find((m) => m.id === id)
      if (!candidate || candidate.status !== 'pending') throw new Error('memory candidate not found')
      const existing = candidate.targetEntryId ? mockState.mockProjectMemory.find((m) => m.id === candidate.targetEntryId) : null
      const entry: ProjectMemoryEntry = existing
        ? Object.assign(existing, {
          kind: candidate.kind, text: candidate.text, sourceType: candidate.sourceType, sourceRef: candidate.sourceRef, confidence: candidate.confidence, privacyScope: candidate.privacyScope || 'project', owner: candidate.owner, updatedAt: now,
          // Reversible overwrite (VST-009): the superseded state becomes a revision.
          revisions: [{ text: existing.text, kind: existing.kind, confidence: existing.confidence, privacyScope: existing.privacyScope || 'project', sourceRef: existing.sourceRef, replacedAt: now, replacedBy: 'user' }, ...(existing.revisions ?? [])].slice(0, 5),
        })
        : { id: uid(), projectId: candidate.projectId, kind: candidate.kind, text: candidate.text, sourceType: candidate.sourceType, sourceRef: candidate.sourceRef, confidence: candidate.confidence, privacyScope: candidate.privacyScope || 'project', owner: candidate.owner, lastVerifiedAt: candidate.sourceRef ? now : null, createdAt: now, updatedAt: now }
      if (!existing) mockState.mockProjectMemory = [entry, ...mockState.mockProjectMemory]
      mockState.mockProjectMemoryCandidates = mockState.mockProjectMemoryCandidates.map((m) => m.id === id ? { ...m, status: 'approved', reviewedAt: now, reviewedBy: 'user', durableEntryId: entry.id, updatedAt: now } : m)
      return entry
    },
    // VST-009: what approving would DO — create vs per-field update (demo parity).
    memoryCandidateDiff: async (id) => {
      const candidate = mockState.mockProjectMemoryCandidates.find((m) => m.id === id)
      if (!candidate) throw new Error('memory candidate not found')
      const target = candidate.targetEntryId ? mockState.mockProjectMemory.find((m) => m.id === candidate.targetEntryId) ?? null : null
      const fields = [
        { field: 'text' as const, before: target?.text ?? null, after: candidate.text },
        { field: 'kind' as const, before: target?.kind ?? null, after: candidate.kind },
        { field: 'confidence' as const, before: target?.confidence ?? null, after: candidate.confidence },
        { field: 'privacyScope' as const, before: (target?.privacyScope ?? null) as string | null, after: (candidate.privacyScope || 'project') as string },
        { field: 'sourceRef' as const, before: target?.sourceRef ?? null, after: candidate.sourceRef },
      ]
      return {
        candidateId: candidate.id,
        action: target ? 'update' as const : 'create' as const,
        targetEntryId: target?.id ?? null,
        changes: fields.filter(({ before, after }) => (target ? before !== after : after != null)),
      }
    },
    rollbackMemoryEntry: async (id) => {
      const now = Date.now()
      const entry = mockState.mockProjectMemory.find((m) => m.id === id)
      if (!entry) throw new Error('memory entry not found')
      mockState.mockProjectMemoryCandidates = mockState.mockProjectMemoryCandidates.map((m) =>
        m.durableEntryId === id && m.status === 'approved' ? { ...m, status: 'rolled_back', reviewedAt: now, updatedAt: now } : m)
      const [previous, ...older] = entry.revisions ?? []
      if (!previous) {
        mockState.mockProjectMemory = mockState.mockProjectMemory.filter((m) => m.id !== id)
        return { action: 'removed' as const, entry: null }
      }
      Object.assign(entry, { text: previous.text, kind: previous.kind, confidence: previous.confidence, privacyScope: previous.privacyScope, sourceRef: previous.sourceRef, revisions: older, lastVerifiedAt: null, updatedAt: now })
      return { action: 'restored_previous' as const, entry }
    },
    rejectProjectMemoryCandidate: async (id) => {
      const now = Date.now()
      const candidate = mockState.mockProjectMemoryCandidates.find((m) => m.id === id)
      if (!candidate || candidate.status !== 'pending') throw new Error('memory candidate not found')
      const rejected: ProjectMemoryCandidate = { ...candidate, status: 'rejected', reviewedAt: now, reviewedBy: 'user', updatedAt: now }
      mockState.mockProjectMemoryCandidates = mockState.mockProjectMemoryCandidates.map((m) => m.id === id ? rejected : m)
      return rejected
    },
    saveProjectMemory: async (m) => {
      const existing = m.id ? mockState.mockProjectMemory.find((x) => x.id === m.id) : null
      const now = Date.now()
      const candidate: ProjectMemoryCandidate = { id: uid(), projectId: existing?.projectId ?? m.projectId!, kind: m.kind ?? existing?.kind ?? 'summary', text: m.text ?? existing?.text ?? '', sourceType: m.sourceType ?? existing?.sourceType ?? 'manual', sourceRef: Object.prototype.hasOwnProperty.call(m, 'sourceRef') ? m.sourceRef ?? null : existing?.sourceRef ?? null, confidence: m.confidence ?? existing?.confidence ?? 'medium', privacyScope: m.privacyScope ?? existing?.privacyScope ?? 'project', owner: Object.prototype.hasOwnProperty.call(m, 'owner') ? m.owner ?? null : existing?.owner ?? null, reason: 'Manual save reviewed in demo mode.', status: 'pending', targetEntryId: existing?.id ?? null, reviewedAt: null, reviewedBy: null, durableEntryId: null, createdAt: now, updatedAt: now }
      mockState.mockProjectMemoryCandidates = [candidate, ...mockState.mockProjectMemoryCandidates]
      const entry: ProjectMemoryEntry = existing
        ? Object.assign(existing, {
          kind: candidate.kind, text: candidate.text, sourceType: candidate.sourceType, sourceRef: candidate.sourceRef, confidence: candidate.confidence, privacyScope: candidate.privacyScope || 'project', owner: candidate.owner, updatedAt: now,
          // Same reversibility as the host path (VST-009): manual edits are rollbackable too.
          revisions: [{ text: existing.text, kind: existing.kind, confidence: existing.confidence, privacyScope: existing.privacyScope || 'project', sourceRef: existing.sourceRef, replacedAt: now, replacedBy: 'user' }, ...(existing.revisions ?? [])].slice(0, 5),
        })
        : { id: uid(), projectId: candidate.projectId, kind: candidate.kind, text: candidate.text, sourceType: candidate.sourceType, sourceRef: candidate.sourceRef, confidence: candidate.confidence, privacyScope: candidate.privacyScope || 'project', owner: candidate.owner, lastVerifiedAt: candidate.sourceRef ? now : null, createdAt: now, updatedAt: now }
      if (!existing) mockState.mockProjectMemory = [entry, ...mockState.mockProjectMemory]
      mockState.mockProjectMemoryCandidates = mockState.mockProjectMemoryCandidates.map((item) => item.id === candidate.id ? { ...item, status: 'approved', reviewedAt: now, reviewedBy: 'user', durableEntryId: entry.id, updatedAt: now } : item)
      return entry
    },
    deleteProjectMemory: async (id) => { mockState.mockProjectMemory = mockState.mockProjectMemory.filter((m) => m.id !== id) },
    getWebhooks: async () => mockWebhookSnapshot(),
    saveWebhook: async (requested) => {
      const w = strictMockConfig<WebhookEndpoint>(requested, ['id', 'name', 'direction', 'enabled', 'scope', 'url', 'path', 'secretRef', 'replayWindowSeconds', 'capturePolicy', 'triggerPolicy', 'runAutomationId', 'eventKinds', 'approvalRequired'], 'webhook save')
      if (w.id) {
        const ex = mockState.mockWebhooks.find((x) => x.id === w.id)
        if (!ex) throw new Error('Webhook not found.')
        {
          const previousUrl = ex.url
          const previousDirection = ex.direction
          Object.assign(ex, w, { id: ex.id, createdAt: ex.createdAt, deliveries: ex.deliveries, signatureMode: ex.signatureMode, secretRotations: ex.secretRotations, secretRef: w.secretRef === '••••••••' ? ex.secretRef : (w.secretRef ?? ex.secretRef), updatedAt: Date.now() })
          if (ex.direction === 'outbound') {
            ex.eventKinds = ex.eventKinds ?? []
            ex.approvalRequired = Object.prototype.hasOwnProperty.call(w, 'approvalRequired') ? w.approvalRequired : previousDirection === 'outbound' ? (ex.approvalRequired ?? true) : true
            if (previousDirection !== 'outbound' || (w.url && w.url !== previousUrl)) ex.approvedAt = null
            ex.path = undefined
            ex.runAutomationId = null
          } else {
            ex.url = undefined
            ex.eventKinds = []
            ex.approvalRequired = false
            ex.approvedAt = null
          }
        }
      }
      else {
        const direction = w.direction || 'inbound'
        mockState.mockWebhooks = [{
          id: uid(), name: w.name || 'Webhook', direction, enabled: w.enabled ?? false, scope: w.scope || {},
          path: direction === 'inbound' ? (w.path || `/hooks/${uid().slice(0, 6)}`) : undefined,
          url: direction === 'outbound' ? (w.url || 'https://example.com/vontra') : undefined,
          secretRef: w.secretRef === '••••••••' ? '' : (w.secretRef || ''),
          signatureMode: 'hmac-sha256', replayWindowSeconds: w.replayWindowSeconds ?? 300,
          capturePolicy: w.capturePolicy || 'redacted_payload', triggerPolicy: w.triggerPolicy || 'notify',
          runAutomationId: direction === 'inbound' ? (w.runAutomationId ?? null) : null,
          eventKinds: direction === 'outbound' ? (w.eventKinds ?? []) : [],
          approvalRequired: direction === 'outbound' ? (w.approvalRequired ?? true) : false,
          approvedAt: null,
          deliveries: [],
          createdAt: Date.now(), updatedAt: Date.now(),
        }, ...mockState.mockWebhooks]
      }
      return mockWebhookSnapshot()
    },
    approveWebhook: async (id) => {
      const endpoint = mockState.mockWebhooks.find((candidate) => candidate.id === id)
      if (!endpoint) throw new Error('Webhook not found.')
      if (endpoint.direction !== 'outbound' || endpoint.approvalRequired === false) throw new Error('This webhook does not require outbound destination approval.')
      const now = Date.now()
      endpoint.approvedAt = now
      endpoint.updatedAt = now
      return mockWebhookSnapshot()
    },
    deleteWebhook: async (id) => { mockState.mockWebhooks = mockState.mockWebhooks.filter((w) => w.id !== id); return mockWebhookSnapshot() },
    buildExport: async (threadId, opts) => {
      const now = Date.now()
      const mode = opts?.mode ?? 'full'
      const redacted = opts?.redact ?? true
      const anonymized = opts?.anonymize ?? (mode === 'share' || mode === 'eval')
      const bundle: RunExportBundle = {
        schema: 'vontra.run-export/1', mode, exportedAt: now, redacted, anonymized,
        thread: { id: anonymized ? 'anon-demo' : threadId, title: 'Demo run', kind: 'task', cwd: anonymized ? undefined : '/repo', provider: anonymized ? undefined : 'openai', model: 'gpt-demo', createdAt: ago(20), updatedAt: now, status: 'idle' },
        messages: [ { role: 'user', content: 'Fix the failing checkout test.' }, { role: 'assistant', content: 'Found the bug in cart.ts and patched it.' } ],
        timeline: [ { kind: 'tool', tool: 'read', title: 'read cart.ts', state: 'ok', at: ago(19) }, { kind: 'tool', tool: 'edit', title: 'edit cart.ts', state: 'ok', at: ago(18) } ],
        stats: { messages: 2, toolCalls: 2, diffs: 1, denied: 0, errors: 0 },
        redactionReport: redacted ? [{ kind: 'email', count: 1 }] : [],
      }
      return { bundle, json: JSON.stringify(bundle, null, 2) }
    },
    saveExport: async () => ({ saved: false, canceled: true }),
    importBundle: async () => ({ bundle: null, error: 'Importing a bundle needs the desktop app.' }),
    getIncidents: async () => mockState.mockIncidents,
    createIncident: async (input) => {
      const now = Date.now()
      const inc: Incident = { id: 'inc_' + uid().slice(0, 6), title: input.title || 'Untitled incident', severity: input.severity ?? 'sev3', status: 'open', cwd: input.cwd || '/repo', summary: input.summary, createdAt: now, updatedAt: now, timeline: [{ id: uid(), kind: 'note', text: 'Incident opened.', at: now, source: 'user' }], mitigations: [], scan: { status: 'idle' } }
      mockState.mockIncidents = [inc, ...mockState.mockIncidents]; return inc
    },
    updateIncident: async (id, requested) => {
      const patch = strictMockConfig<Incident>(requested, ['title', 'severity', 'status', 'summary'], 'incident update')
      if (patch.severity && !['sev1', 'sev2', 'sev3', 'sev4'].includes(patch.severity)) throw new Error('Incident severity has an invalid value.')
      if (patch.status && !['open', 'investigating', 'mitigating', 'resolved', 'closed'].includes(patch.status)) throw new Error('Incident status has an invalid value.')
      const inc = mockState.mockIncidents.find((i) => i.id === id)!
      Object.assign(inc, patch, { updatedAt: Date.now() })
      return inc
    },
    deleteIncident: async (id) => { mockState.mockIncidents = mockState.mockIncidents.filter((i) => i.id !== id) },
    addIncidentEvent: async (id, kind, text) => { const inc = mockState.mockIncidents.find((i) => i.id === id)!; inc.timeline.push({ id: uid(), kind, text, at: Date.now(), source: 'user' }); inc.updatedAt = Date.now(); return inc },
    proposeMitigation: async (id, text, rationale) => { const inc = mockState.mockIncidents.find((i) => i.id === id)!; inc.mitigations.push({ id: uid(), text, rationale, status: 'proposed', proposedAt: Date.now() }); inc.timeline.push({ id: uid(), kind: 'mitigation', text: `Proposed: ${text}`, at: Date.now(), source: 'user' }); inc.updatedAt = Date.now(); return inc },
    decideMitigation: async (id, mitigationId, decision) => { const inc = mockState.mockIncidents.find((i) => i.id === id)!; const m = inc.mitigations.find((x) => x.id === mitigationId); if (m) { m.status = decision; m.decidedAt = Date.now() } inc.updatedAt = Date.now(); return inc },
    executeMitigation: async (id, mitigationId) => { const inc = mockState.mockIncidents.find((i) => i.id === id)!; const m = inc.mitigations.find((x) => x.id === mitigationId); if (m) { m.missionId = 'mis_demo'; } inc.timeline.push({ id: uid(), kind: 'mitigation', text: 'Execution started as guarded mission (demo).', at: Date.now(), source: 'user' }); inc.status = 'mitigating'; inc.updatedAt = Date.now(); return inc },
    scanIncident: async (id) => { const inc = mockState.mockIncidents.find((i) => i.id === id)!; const findings = '## Symptoms\n- Demo scan (browser mock).\n\n## Likely cause\n- N/A in demo mode.'; inc.scan = { status: 'done', findings, ranAt: Date.now() }; inc.timeline.push({ id: uid(), kind: 'scan', text: findings, at: Date.now(), source: 'scan' }); inc.updatedAt = Date.now(); return inc },
    generatePostmortem: async (id) => { const inc = mockState.mockIncidents.find((i) => i.id === id)!; inc.postmortem = '# Postmortem (demo)\n\nGenerated in browser mock mode.'; inc.postmortemAt = Date.now(); inc.updatedAt = Date.now(); return inc },
    testWebhook: async (id) => {
      const w = mockState.mockWebhooks.find((x) => x.id === id)
      if (w?.direction === 'outbound') {
        const record: WebhookDeliveryRecord = {
          id: uid(), direction: 'outbound', eventKind: 'test', urlOrPath: w.url || '', status: w.approvalRequired !== false && !w.approvedAt ? 'failed' : 'delivered',
          attempts: w.approvalRequired !== false && !w.approvedAt ? 0 : 1, httpStatus: w.approvalRequired !== false && !w.approvedAt ? null : 202, error: w.approvalRequired !== false && !w.approvedAt ? 'Destination needs approval before outbound delivery.' : null, idempotencyKey: 'demo_' + uid(),
          payloadPreview: '{"type":"test","source":"vontra"}', createdAt: Date.now(), finishedAt: Date.now(),
        }
        w.deliveries = [record, ...(w.deliveries ?? [])].slice(0, 30)
      } else if (w?.direction === 'inbound') {
        const now = Date.now()
        const record: WebhookDeliveryRecord = {
          id: uid(),
          direction: 'inbound',
          eventKind: 'test',
          urlOrPath: w.path || '',
          status: 'delivered',
          attempts: 1,
          httpStatus: 200,
          error: null,
          idempotencyKey: 'demo_' + uid(),
          payloadPreview: '{"title":"Test delivery","source":"vontra"}',
          createdAt: now,
          finishedAt: now,
        }
        w.deliveries = [record, ...(w.deliveries ?? [])].slice(0, 30)
        const event: ConnectorEvent = { id: uid(), connectorId: 'webhook', pluginName: w.name, action: 'webhook_test', direction: 'inbound', classification: 'communication', priority: 'normal', trustLevel: 'suggest', requiresApproval: false, approvalId: null, title: 'Test delivery', summary: `Sample inbound payload for ${w.name}`, actor: 'Webhook', threadKey: `webhook:test:${w.id}`, payloadKeys: ['title', 'source'], createdAt: now }
        mockState.mockConnectorEvents = [event, ...mockState.mockConnectorEvents].slice(0, 80)
      }
      return mockWebhookSnapshot()
    },
    getCommands: async () => mockState.mockCommands,
    saveCommand: async (requested) => {
      const c = strictMockConfig<Command>(requested, ['id', 'name', 'description', 'prompt', 'argHint'], 'command save')
      if (c.id) { const ex = mockState.mockCommands.find((x) => x.id === c.id); if (!ex) throw new Error('Command not found.'); const name = c.name ?? ex.name; assertMockCommandName(name); assertMockUniqueName(mockState.mockCommands, name, ex.id); Object.assign(ex, c, { id: ex.id, createdAt: ex.createdAt, updatedAt: Date.now() }); return ex }
      const name = c.name || 'command'; assertMockCommandName(name); assertMockUniqueName(mockState.mockCommands, name)
      const cmd: Command = { id: uid(), name, description: c.description || '', prompt: c.prompt || '', argHint: c.argHint, createdAt: Date.now(), updatedAt: Date.now() }
      mockState.mockCommands = [cmd, ...mockState.mockCommands]; return cmd
    },
    deleteCommand: async (id) => { mockState.mockCommands = mockState.mockCommands.filter((c) => c.id !== id) },
    getSuggestions: async () => mockState.mockSuggestions ?? [],
    acceptSuggestion: async () => ({ ok: false, skill: null, command: null }),
    dismissSuggestion: async () => {},
    onSuggestionsUpdate: () => () => {},
  }
}
