import type { Mission, Thread, TimelineItem } from './types'
import { formatMissionScratchMemory } from './missions'
import { validateMissionStep, missionStatusForStepAction } from './mission-invariants'
import type { MissionInvariantError } from './orchestration-errors'
import { recordMissionEvent, type MissionJournalEventType } from './orchestration-journal'
import { emitMissionLifecycle } from '../main/host/orchestration-events'
import { UI_SMOKE_RECIPES } from './ui-smoke-recipes'

export interface MissionVerifierResult {
  done: boolean
  note: string
  evidenceCount?: number
}

export type MissionStepAction = 'continue' | 'complete' | 'pause'

export interface MissionStepDecision {
  action: MissionStepAction
  reason: string
  nextPrompt?: string
  // Set when the decided action would be an illegal status transition from the
  // mission's current status (e.g. the mission is already terminal). The runtime
  // surfaces this via its error journal path; legal decisions leave it undefined,
  // so existing happy-path behavior is unchanged.
  invariantError?: MissionInvariantError
}

// Guard a step decision against the mission status invariants. Returns the typed
// error when the action maps to an illegal transition from `mission.status`, or
// null otherwise. Pure — does not throw, so callers can surface it on the existing
// error event path without disrupting the loop.
export function guardMissionStep(mission: Pick<Mission, 'status'>, action: MissionStepAction): MissionInvariantError | null {
  return validateMissionStep(mission.status, action)
}

const DEFAULT_AUTONOMOUS_RUN_LIMIT = 15
const FRONTEND_MISSION_RE = /\b(frontend|front-end|ui|browser preview|localhost|dev server|vite|screenshot|click)\b/i
const UI_SMOKE_WORKFLOWS = new Set(['frontend_ui_smoke', 'ui_smoke', 'ui-smoke-localhost', 'ui-smoke-interaction', 'ui-smoke-responsive'])
const LOCALHOST_RE = /\bhttps?:\/\/(?:localhost|127\.0\.0\.1|\[::1\])(?::\d+)?(?:\/[^\s)]*)?/i
const DEV_SERVER_RE = /\b(?:npm|pnpm|yarn|bun)\s+(?:run\s+)?(?:dev|start|serve|preview)\b|\bvite\b|\bnext\s+dev\b/i
const SCREENSHOT_PATH_RE = /(?:^|\s)(\/[^\s]+?\.(?:png|jpe?g|webp|gif))(?=\s|$|\))/i

export interface FrontendMissionEvidenceRequirements {
  devServer: boolean
  localhost: boolean
  accessibilitySnapshot: boolean
  interaction: boolean
  screenshot: boolean
}

export interface FrontendMissionEvidence {
  complete: boolean
  requirements: FrontendMissionEvidenceRequirements
  missing: string[]
  sourceItemIds: string[]
  artifactRefs: string[]
  summary: string
}

export function missionRunCount(mission: Mission): number {
  const metadataCount = typeof mission.metadata?.runCount === 'number' && Number.isFinite(mission.metadata.runCount)
    ? Math.max(0, Math.floor(mission.metadata.runCount))
    : 0
  const checkpointCount = mission.checkpoints.reduce((max, checkpoint) => {
    const runIndex = typeof checkpoint.data?.runIndex === 'number' && Number.isFinite(checkpoint.data.runIndex)
      ? Math.max(0, Math.floor(checkpoint.data.runIndex))
      : 0
    return Math.max(max, runIndex)
  }, 0)
  return Math.max(metadataCount, checkpointCount)
}

export function missionRunLimit(mission: Mission): number {
  if (typeof mission.maxRuns === 'number' && Number.isFinite(mission.maxRuns) && mission.maxRuns > 0) {
    return Math.max(1, Math.floor(mission.maxRuns))
  }
  if (mission.repeatMode === 'self_paced' || mission.repeatMode === 'interval') return 1
  if (mission.repeatMode === 'fixed_count') return 1
  return DEFAULT_AUTONOMOUS_RUN_LIMIT
}

export function shouldStartFreshMissionThread(mission: Mission, runIndex: number): boolean {
  return runIndex <= 1 || mission.contextMode !== 'build_on_previous'
}

export function missionNeedsFrontendUiSmoke(mission: Pick<Mission, 'title' | 'objective' | 'stopCondition' | 'metadata'>): boolean {
  const workflow = typeof mission.metadata?.workflow === 'string' ? mission.metadata.workflow.trim().toLowerCase() : ''
  const recipeId = typeof mission.metadata?.recipeId === 'string' ? mission.metadata.recipeId.trim().toLowerCase() : ''
  if (UI_SMOKE_WORKFLOWS.has(workflow) || UI_SMOKE_WORKFLOWS.has(recipeId)) return true
  return FRONTEND_MISSION_RE.test([mission.title, mission.objective, mission.stopCondition].filter(Boolean).join('\n'))
}

export function buildFrontendMissionEvidence(thread: Pick<Thread, 'timeline'>): FrontendMissionEvidence {
  const requirements: FrontendMissionEvidenceRequirements = {
    devServer: false,
    localhost: false,
    accessibilitySnapshot: false,
    interaction: false,
    screenshot: false,
  }
  const sourceItemIds: string[] = []
  const artifactRefs: string[] = []

  for (const item of thread.timeline) {
    if (item.kind !== 'tool' || item.state !== 'ok') continue
    const output = item.output ?? ''
    const command = item.command ?? ''
    const path = item.path ?? ''
    const title = item.title ?? ''
    const tool = item.tool ?? ''
    const haystack = [tool, title, command, path, output].filter(Boolean).join('\n')
    let used = false

    if (tool === 'run_command' && (DEV_SERVER_RE.test(command) || LOCALHOST_RE.test(output) || /\bready in\b/i.test(output))) {
      requirements.devServer = true
      used = true
    }

    if (tool === 'browser_automation') {
      if (/\bAction:\s*(?:navigate|tab_new)\b/i.test(output) && LOCALHOST_RE.test(haystack)) {
        requirements.localhost = true
        used = true
      }
      if (/Accessibility snapshot:/i.test(output)) {
        requirements.accessibilitySnapshot = true
        used = true
      }
      if (/\bAction:\s*(?:click|type|fill_form)\b/i.test(output) || /\bBrowser\s+(?:click|type|fill_form)\b/i.test(title)) {
        requirements.interaction = true
        used = true
      }
      if (isScreenshotEvidence(item)) {
        requirements.screenshot = true
        used = true
      }
    } else if (isScreenshotEvidence(item)) {
      requirements.screenshot = true
      used = true
    }

    const screenshot = screenshotArtifactRef(item)
    if (screenshot && !artifactRefs.includes(screenshot)) artifactRefs.push(screenshot)
    if (used && !sourceItemIds.includes(item.id)) sourceItemIds.push(item.id)
  }

  const missing = [
    requirements.devServer ? null : 'dev-server command',
    requirements.localhost ? null : 'localhost navigation',
    requirements.accessibilitySnapshot ? null : 'accessibility snapshot',
    requirements.interaction ? null : 'browser interaction',
    requirements.screenshot ? null : 'screenshot',
  ].filter(Boolean) as string[]
  const complete = missing.length === 0
  const summary = complete
    ? `Frontend UI smoke evidence complete: dev-server command, localhost navigation, accessibility snapshot, browser interaction, and screenshot attached from ${sourceItemIds.length} tool item${sourceItemIds.length === 1 ? '' : 's'}.`
    : `Frontend UI smoke evidence incomplete; missing ${missing.join(', ')}.`

  return {
    complete,
    requirements,
    missing,
    sourceItemIds,
    artifactRefs,
    summary,
  }
}

function appendFrontendMissionInstructions(lines: string[]): void {
  const recipes = UI_SMOKE_RECIPES
    .filter((recipe) => recipe.id === 'ui-smoke-localhost' || recipe.id === 'ui-smoke-interaction')
    .map((recipe) => recipe.name)
    .join(' + ')
  lines.push(
    '',
    `Frontend UI smoke evidence (${recipes}):`,
    '- Start or reuse the local dev server with run_command and keep the localhost URL/output as evidence.',
    '- Open localhost with browser_automation navigate, then inspect the UI with an accessibility snapshot.',
    '- Exercise at least one real click, type, or form-fill path with browser_automation.',
    '- Capture a screenshot and check browser console/network evidence after the interaction.',
    '- Attach evidence by naming the run_command item, browser_automation actions, screenshot path, and any console/network findings before claiming completion.',
  )
}

function isScreenshotEvidence(item: TimelineItem): boolean {
  if (/\.(?:png|jpe?g|webp|gif)$/i.test(item.path ?? '')) return true
  if (/\bScreenshot:/i.test(item.output ?? '')) return true
  if (/\bscreenshot\b/i.test([item.tool, item.title].filter(Boolean).join(' '))) return true
  return false
}

function screenshotArtifactRef(item: TimelineItem): string | null {
  if (/\.(?:png|jpe?g|webp|gif)$/i.test(item.path ?? '')) return item.path ?? null
  const match = (item.output ?? '').match(SCREENSHOT_PATH_RE)
  return match?.[1] ?? null
}

export function buildMissionRunPrompt(mission: Mission, runIndex: number, notes: string[] = []): string {
  const limit = missionRunLimit(mission)
  const lines = [
    `Mission: ${mission.title}`,
    '',
    'Objective:',
    mission.objective,
  ]
  if (mission.stopCondition.trim()) lines.push('', 'Stop condition:', mission.stopCondition.trim())
  lines.push('', `Run: ${runIndex} of ${limit}`)
  lines.push(`Repeat mode: ${mission.repeatMode}`)
  lines.push(`Context mode: ${mission.contextMode}`)
  if (notes.length && mission.contextMode !== 'fresh') {
    lines.push('', 'Previous mission notes:')
    for (const note of notes.slice(-8)) lines.push(`- ${note}`)
  }
  const scratchMemory = formatMissionScratchMemory(mission)
  if (scratchMemory && mission.contextMode !== 'fresh') {
    lines.push('', 'Mission scratch memory (mission-scoped, not project memory):')
    lines.push(scratchMemory)
  }
  if (missionNeedsFrontendUiSmoke(mission)) appendFrontendMissionInstructions(lines)
  lines.push('', 'Work until the current pass has made concrete progress. If you believe the stop condition is met, include the evidence that proves it.')
  return lines.join('\n')
}

export function buildMissionContinuationPrompt(mission: Mission, nextRunIndex: number, reason: string): string {
  const stop = mission.stopCondition.trim()
  return [
    `Mission: ${mission.title}`,
    '',
    'Objective:',
    mission.objective,
    '',
    `Mission pass ${nextRunIndex - 1} did not finish the mission: ${reason}`,
    stop ? `Stop condition: ${stop}` : null,
    'Continue with the next pass. Build on proven work only, gather fresh evidence, and stop only when the condition is actually satisfied.',
  ].filter(Boolean).join('\n')
}

// Durable-journal recorders for the mission lifecycle points the supervisor/runtime
// drives but that are not the per-pass decision (created at queue time, started at each
// run, and checkpoint at each verified pass). All fire-and-forget + error-tolerant.

/** Record the durable 'created' event for a mission (at queue/creation time). */
export function recordMissionCreated(mission: Pick<Mission, 'id' | 'status' | 'title' | 'repeatMode'>): void {
  void recordMissionEvent(mission.id, {
    type: 'created',
    actor: 'system',
    payload: { status: mission.status, title: mission.title, repeatMode: mission.repeatMode },
  })
  // Live push (additive, alongside the durable journal): let the Command Centre learn a
  // mission was queued without waiting for its next snapshot poll.
  emitMissionLifecycle('created', mission.id, { status: mission.status, title: mission.title, repeatMode: mission.repeatMode })
}

/** Record the durable 'started' event for a mission run (status -> running). */
export function recordMissionStarted(mission: Pick<Mission, 'id' | 'status'>, runIndex: number): void {
  void recordMissionEvent(mission.id, {
    type: 'started',
    actor: 'supervisor',
    payload: { from: mission.status, to: 'running', runIndex },
  })
  // Live push: a pass began; the running state should light up immediately.
  emitMissionLifecycle('started', mission.id, { status: 'running', runIndex })
}

/** Record the durable 'checkpoint' event for a verified mission pass. */
export function recordMissionCheckpoint(
  missionId: string,
  checkpoint: { id?: string; label?: string; sequence?: number; runIndex?: number },
): void {
  void recordMissionEvent(missionId, {
    type: 'checkpoint',
    actor: 'supervisor',
    payload: {
      ...(checkpoint.id !== undefined ? { checkpointId: checkpoint.id } : {}),
      ...(checkpoint.label !== undefined ? { label: checkpoint.label } : {}),
      ...(checkpoint.sequence !== undefined ? { sequence: checkpoint.sequence } : {}),
      ...(checkpoint.runIndex !== undefined ? { runIndex: checkpoint.runIndex } : {}),
    },
  })
  // Live push: a verified pass landed a checkpoint — surface progress without a poll.
  emitMissionLifecycle('checkpoint', missionId, { runIndex: checkpoint.runIndex, reason: checkpoint.label })
}

// Map a supervisor step action to its durable mission journal event type. 'continue'
// queues another pass (requeued); 'complete' finalizes; 'pause' is a status change.
function journalEventForStepAction(action: MissionStepAction): MissionJournalEventType {
  switch (action) {
    case 'complete':
      return 'finalized'
    case 'continue':
      return 'requeued'
    case 'pause':
    default:
      return 'status_changed'
  }
}

export function decideNextMissionStep(
  mission: Mission,
  runIndex: number,
  verifier: MissionVerifierResult,
  now = Date.now(),
): MissionStepDecision {
  const decision = decideNextMissionStepCore(mission, runIndex, verifier, now)
  // Annotate (do not alter) the decision with any status-invariant violation so the
  // runtime can surface it on the error path. Legal transitions get `null` here, so
  // the returned decision is byte-for-byte the same as before for the happy path.
  const invariantError = guardMissionStep(mission, decision.action)
  // Durable on-disk journal: record this lifecycle transition alongside the in-memory
  // mission journal so the mission is reconstructable after a crash. Fire-and-forget +
  // error-tolerant (recordMissionEvent never throws), so the decision path is unchanged.
  void recordMissionEvent(mission.id, {
    type: journalEventForStepAction(decision.action),
    actor: 'supervisor',
    ts: now,
    payload: {
      action: decision.action,
      from: mission.status,
      to: missionStatusForStepAction(decision.action),
      runIndex,
      reason: decision.reason,
      ...(invariantError ? { invariantError: invariantError.toFields() } : {}),
    },
  })
  // Live push: surface the terminal/parked transition the moment the supervisor decides
  // it. 'complete' finalizes the mission; 'pause' is a status change. 'continue' just
  // queues another pass — the next recordMissionStarted emits 'started', so it needs no
  // event here (avoids a noisy duplicate per pass).
  if (decision.action === 'complete') {
    emitMissionLifecycle('finalized', mission.id, { status: missionStatusForStepAction(decision.action), runIndex, reason: decision.reason, ts: now })
  } else if (decision.action === 'pause') {
    emitMissionLifecycle('status_changed', mission.id, { status: missionStatusForStepAction(decision.action), runIndex, reason: decision.reason, ts: now })
  }
  return invariantError ? { ...decision, invariantError } : decision
}

function decideNextMissionStepCore(
  mission: Mission,
  runIndex: number,
  verifier: MissionVerifierResult,
  now: number,
): MissionStepDecision {
  if (mission.expiresAt && now >= mission.expiresAt) {
    return { action: 'pause', reason: 'Mission expiry reached before completion.' }
  }

  const limit = missionRunLimit(mission)
  const stopCondition = mission.stopCondition.trim()
  const evidenceCount = typeof verifier.evidenceCount === 'number' && Number.isFinite(verifier.evidenceCount)
    ? Math.max(0, Math.floor(verifier.evidenceCount))
    : 0
  const verifierNote = verifier.note || 'No verifier note.'
  const evidenceBackedVerifierNote = verifier.done && stopCondition && evidenceCount <= 0
    ? `Stop condition was claimed complete without recorded evidence: ${verifierNote}`
    : verifierNote

  if (mission.repeatMode === 'fixed_count') {
    if (runIndex >= limit) return { action: 'complete', reason: `Completed ${runIndex} fixed mission pass${runIndex === 1 ? '' : 'es'}.` }
    return {
      action: 'continue',
      reason: `Fixed-count mission has ${limit - runIndex} pass${limit - runIndex === 1 ? '' : 'es'} remaining.`,
      nextPrompt: buildMissionContinuationPrompt(mission, runIndex + 1, `fixed-count run ${runIndex}/${limit} complete`),
    }
  }

  if (mission.repeatMode === 'self_paced') {
    return { action: 'pause', reason: 'Self-paced mission completed one pass and is waiting for the next manual start.' }
  }

  if (mission.repeatMode === 'interval') {
    return { action: 'pause', reason: 'Interval mission completed one pass and is waiting for the next scheduled wake.' }
  }

  if (!stopCondition && mission.repeatMode !== 'forever') {
    return { action: 'complete', reason: 'Mission has no stop condition, so one completed pass is enough.' }
  }

  if (stopCondition && verifier.done && evidenceCount > 0) {
    return { action: 'complete', reason: verifierNote }
  }

  if (runIndex >= limit) {
    return { action: 'pause', reason: `Run limit reached before the stop condition was verified: ${evidenceBackedVerifierNote}` }
  }

  return {
    action: 'continue',
    reason: stopCondition ? evidenceBackedVerifierNote : `Forever mission pass ${runIndex} complete.`,
    nextPrompt: buildMissionContinuationPrompt(mission, runIndex + 1, stopCondition ? evidenceBackedVerifierNote : `forever run ${runIndex}/${limit} complete`),
  }
}
