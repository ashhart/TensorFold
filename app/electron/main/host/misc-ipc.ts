import type { ProjectSkillView } from '../../agent/types'
import type { CueRunPayload, CueRunResult, StartRunPayload } from '../../agent/types'
import { auditPreview } from '../../agent/security'
import { auditEvent, requireSecurityRole } from './audit'
import { handle } from './ipc'
import type { ProjectSkillsRuntime } from './project-skills-runtime'
import type { ObserverHub } from '../../agent/observer-hooks'
import { skillSuggestionDiff } from '../../agent/suggestions'
import { artifactPathAccessKind, artifactPreview, artifactsForThread, redactArtifactRecord, MAX_INLINE_ARTIFACT_CHARS } from '../../agent/artifacts'
import { readArtifactTextBounded, readArtifactTextPrefix } from '../../agent/artifact-files'
import { store } from '../state'
import type { StartLoopInput } from './loop-host'

interface LoopHostLike {
  startLoop(input: StartLoopInput): unknown
  getLoops(): unknown
  stopLoop(id: string): unknown
}

interface ProviderStatusLike {
  status(): unknown
}

interface MiscIpcDeps {
  projectSkills: ProjectSkillsRuntime
  observerHub: ObserverHub
  providerStatus: ProviderStatusLike
  cancelThreadRun(threadId: string): void
  guideThreadRun(payload: CueRunPayload): Promise<CueRunResult>
  startThreadRun(payload: StartRunPayload): Promise<{ runId: string }>
  loopHost: LoopHostLike
  persistArtifacts(): void
}

/**
 * A grab-bag of renderer-invocable handlers that don't belong to a dedicated runtime:
 * repo-authored project skills (VST-010), first-class artifacts (VST-011), observer
 * health (VST-012), the provider-rate-limit read, the bare agent start/cue/cancel
 * controls, loop controls, and the window minimize/maximize/close actions.
 */
export function registerMiscIpc(deps: MiscIpcDeps) {
  const { projectSkills, observerHub, providerStatus, loopHost } = deps

  // Trust UI + review flow (VST-010): live scan with trust states, and the trust toggle —
  // operator-gated + audited (trusting repo content is a durable-behavior decision).
  handle('skills:project', async (_e, projectId: string): Promise<ProjectSkillView[]> => {
    const project = store.projects.find((p) => p.id === projectId)
    if (!project) return []
    return projectSkills.scanProjectSkills(project)
  })
  handle('skills:trustProjectSkill', async (_e, input: { projectId: string; skillId: string; trusted: boolean }): Promise<ProjectSkillView[]> => {
    requireSecurityRole('operator', 'trust project skills')
    const project = store.projects.find((p) => p.id === String(input?.projectId ?? ''))
    if (!project) throw new Error('project not found')
    const views = await projectSkills.setProjectSkillTrust(project, String(input?.skillId ?? ''), input?.trusted === true)
    await auditEvent('skills.project_trust', { projectId: auditPreview(project.id), skillId: auditPreview(String(input?.skillId ?? '')), trusted: input?.trusted === true })
    return views
  })
  // What accepting a suggestion would DO (VST-010) — the diff behind the review UI.
  handle('suggestions:diff', async (_e, id: string) => {
    const suggestion = store.suggestions.find((s) => s.id === id)
    if (!suggestion) throw new Error('suggestion not found')
    return skillSuggestionDiff(store.skills, suggestion)
  })

  // ---- first-class artifacts (VST-011): list / preview / redact / export / delete ----
  handle('artifacts:list', async (_e, threadId?: string) =>
    threadId ? artifactsForThread(store.artifacts, threadId) : store.artifacts)
  handle('artifacts:preview', async (_e, id: string) => {
    const artifact = store.artifacts.find((a) => a.id === id)
    if (!artifact) throw new Error('artifact not found')
    let content: string | null = null
    let unavailable: string | null = null
    if (artifact.path && !artifact.inlineText) {
      const access = artifactPathAccessKind(artifact)
      if (access === 'local') {
        // Bounded read; a missing/unreadable/binary file degrades to the metadata card.
        try { content = await readArtifactTextPrefix(artifact.path, MAX_INLINE_ARTIFACT_CHARS) } catch { content = null }
      } else {
        unavailable = access === 'ssh' ? 'remote content unavailable' : 'source workspace unavailable'
      }
    }
    const preview = artifactPreview(artifact, content)
    return unavailable ? { ...preview, meta: `${preview.meta} · ${unavailable}` } : preview
  })
  // Redaction produces a SAFE COPY inside the record (the original file is never edited):
  // content is masked via the shared export rules, the record becomes inline-backed with
  // the file kept as sourceRef, and the report lands on the record for the receipt.
  handle('artifacts:redact', async (_e, id: string) => {
    requireSecurityRole('operator', 'redact artifacts')
    const at = store.artifacts.findIndex((a) => a.id === id)
    if (at < 0) throw new Error('artifact not found')
    const artifact = store.artifacts[at]
    let content: string | null = artifact.inlineText
    if (artifact.path && content == null) {
      const access = artifactPathAccessKind(artifact)
      if (access !== 'local') {
        throw new Error(access === 'ssh'
          ? 'Remote path-backed artifacts cannot be redacted on this host. Save bounded inline content or copy the artifact locally first.'
          : 'The artifact source workspace is unavailable; refusing to read a same-named local path.')
      }
      try { content = await readArtifactTextPrefix(artifact.path, MAX_INLINE_ARTIFACT_CHARS) } catch { content = null }
    }
    const result = redactArtifactRecord(artifact, content)
    const next = result.content != null && artifact.path
      ? { ...result.artifact, inlineText: result.content.slice(0, MAX_INLINE_ARTIFACT_CHARS), path: null, provenance: { ...result.artifact.provenance, sourceRef: result.artifact.provenance.sourceRef ?? artifact.path } }
      : result.artifact
    store.artifacts[at] = next
    deps.persistArtifacts()
    await auditEvent('artifact.redacted', { artifactId: auditPreview(id), status: next.redaction, hits: next.redactionReport.reduce((sum, r) => sum + r.count, 0) })
    return next
  })
  handle('artifacts:export', async (_e, id: string) => {
    const artifact = store.artifacts.find((a) => a.id === id)
    if (!artifact) throw new Error('artifact not found')
    let content: string | null = artifact.inlineText
    if (artifact.path && content == null) {
      const access = artifactPathAccessKind(artifact)
      if (access !== 'local') {
        throw new Error(access === 'ssh'
          ? 'Remote path-backed artifacts cannot be exported on this host. Save bounded inline content or copy the artifact locally first.'
          : 'The artifact source workspace is unavailable; refusing to read a same-named local path.')
      }
      content = await readArtifactTextBounded(artifact.path)
    }
    await auditEvent('artifact.exported', { artifactId: auditPreview(id), redaction: artifact.redaction })
    return { artifact, content }
  })
  handle('artifacts:delete', async (_e, id: string) => {
    requireSecurityRole('operator', 'delete artifacts')
    store.artifacts = store.artifacts.filter((a) => a.id !== id)
    deps.persistArtifacts()
  })

  // Observer hooks (VST-012): per-hook health + the recent-event ring for the doctor.
  handle('observers:status', async () => ({ hooks: observerHub.status(), recent: observerHub.recent(50) }))

  // Provider rate-limit status (VST-037): the picker + doctor read the persisted per-provider
  // windows; live updates push over 'providerStatus:update'.
  handle('providerStatus:get', async () => providerStatus.status())
  handle('agent:cancel', async (_e, threadId: string) => deps.cancelThreadRun(threadId))
  handle('agent:cue', async (_e, payload: CueRunPayload): Promise<CueRunResult> => deps.guideThreadRun(payload))
  handle('agent:start', async (_e, payload: StartRunPayload) => deps.startThreadRun(payload))

  handle('loop:start', async (_e, input: StartLoopInput) => loopHost.startLoop(input))
  handle('loop:list', async () => loopHost.getLoops())
  handle('loop:stop', async (_e, id: string) => loopHost.stopLoop(id))

  handle('window:action', async (_e, action: string) => {
    if (!store.win) return
    if (action === 'minimize') store.win.minimize()
    else if (action === 'maximize') store.win.isMaximized() ? store.win.unmaximize() : store.win.maximize()
    else if (action === 'close') store.win.close()
  })
}
