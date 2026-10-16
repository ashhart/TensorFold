// Run: npx tsx electron/agent/mission-supervisor.test.ts
import assert from 'node:assert/strict'
import {
  buildFrontendMissionEvidence,
  buildMissionContinuationPrompt,
  buildMissionRunPrompt,
  decideNextMissionStep,
  missionNeedsFrontendUiSmoke,
  missionRunCount,
  missionRunLimit,
  shouldStartFreshMissionThread,
} from './mission-supervisor'
import type { Mission } from './types'

function mission(overrides: Partial<Mission> = {}): Mission {
  return {
    id: 'm1',
    title: 'Daily brief',
    objective: 'Collect source-backed AI news.',
    stopCondition: 'Five source-backed items are ready.',
    repeatMode: 'until_condition',
    contextMode: 'build_on_previous',
    roleProfileId: 'operator',
    fanOutBudget: 'single',
    maxRuns: 3,
    costCapUsd: null,
    expiresAt: null,
    destructiveApprovalRequired: true,
    status: 'running',
    runBudgetClass: 'mission',
    budget: { executionClass: 'mission', label: 'Mission', maxWallSeconds: 3600, heartbeatIntervalSeconds: 30, stallGraceSeconds: 120, requiresHeartbeat: true, startedAt: 1_000, deadlineAt: 3_601_000 },
    heartbeat: { intervalSeconds: 30, stallGraceSeconds: 120, required: true, count: 1, lastBeatAt: 1_000, nextDueAt: 31_000, missedSince: null },
    checkpoints: [],
    journal: [],
    scratchMemory: [],
    pendingApprovals: [],
    threadId: null,
    projectId: null,
    cwd: null,
    providerId: null,
    providerAccountId: null,
    model: null,
    createdAt: 1_000,
    updatedAt: 1_000,
    startedAt: 1_000,
    finishedAt: null,
    ...overrides,
  }
}

{
  const m = mission({
    metadata: { runCount: 1 },
    checkpoints: [{ id: 'c1', sequence: 1, label: 'Pass 2', summary: 'done', status: 'running', createdAt: 2_000, data: { runIndex: 2 } }],
  })
  assert.equal(missionRunCount(m), 2)
}

{
  assert.equal(missionRunLimit(mission({ maxRuns: 0 })), 15)
  assert.equal(missionRunLimit(mission({ repeatMode: 'interval', maxRuns: null })), 1)
  assert.equal(missionRunLimit(mission({ repeatMode: 'fixed_count', maxRuns: 4 })), 4)
}

{
  const m = mission()
  assert.equal(shouldStartFreshMissionThread(m, 1), true)
  assert.equal(shouldStartFreshMissionThread(m, 2), false)
  assert.equal(shouldStartFreshMissionThread(mission({ contextMode: 'fresh_with_notes' }), 2), true)
}

{
  const m = mission()
  const prompt = buildMissionRunPrompt(m, 2, ['Found three sources.'])
  assert.match(prompt, /Run: 2 of 3/)
  assert.match(prompt, /Previous mission notes/)
  assert.match(prompt, /Five source-backed items/)
}

{
  const prompt = buildMissionRunPrompt(mission({
    title: 'Frontend smoke mission',
    objective: 'Start the dev server, open localhost, inspect UI, click the primary action, screenshot, and attach evidence.',
    stopCondition: 'The localhost UI smoke evidence is attached.',
    metadata: { workflow: 'frontend_ui_smoke' },
  }), 1)
  assert.match(prompt, /\$ui-smoke-localhost/)
  assert.match(prompt, /run_command/)
  assert.match(prompt, /browser_automation/)
  assert.match(prompt, /localhost/)
  assert.match(prompt, /accessibility snapshot/)
  assert.match(prompt, /click/)
  assert.match(prompt, /screenshot/)
  assert.match(prompt, /attach evidence/i)
  assert.equal(missionNeedsFrontendUiSmoke(mission({ metadata: { workflow: 'frontend_ui_smoke' } })), true)
}

{
  const evidence = buildFrontendMissionEvidence({
    timeline: [
      { id: 'dev', kind: 'tool', tool: 'run_command', title: 'Run dev server', command: 'npm run dev -- --host 127.0.0.1', output: 'VITE ready in 100ms\nLocal: http://localhost:5173/', state: 'ok', createdAt: 10 },
      { id: 'nav', kind: 'tool', tool: 'browser_automation', title: 'Browser navigate', output: 'Action: navigate\nURL: http://localhost:5173/\nAccessibility snapshot:\n- button "New project" [ref=e1]', state: 'ok', createdAt: 20 },
      { id: 'click', kind: 'tool', tool: 'browser_automation', title: 'Browser click', path: '/Users/example/.vontra/browser-automation/frontend-smoke.png', output: 'Action: click\nNote: Clicked target.\nScreenshot: /Users/example/.vontra/browser-automation/frontend-smoke.png (frontend-smoke)', state: 'ok', createdAt: 30 },
    ],
  })
  assert.equal(evidence.complete, true)
  assert.deepEqual(evidence.requirements, {
    devServer: true,
    localhost: true,
    accessibilitySnapshot: true,
    interaction: true,
    screenshot: true,
  })
  assert.deepEqual(evidence.missing, [])
  assert.deepEqual(evidence.sourceItemIds, ['dev', 'nav', 'click'])
  assert.deepEqual(evidence.artifactRefs, ['/Users/example/.vontra/browser-automation/frontend-smoke.png'])
  assert.match(evidence.summary, /Frontend UI smoke evidence complete/)

  const incomplete = buildFrontendMissionEvidence({
    timeline: [
      { id: 'dev', kind: 'tool', tool: 'run_command', command: 'npm run dev', output: 'Local: http://localhost:5173/', state: 'ok', createdAt: 10 },
    ],
  })
  assert.equal(incomplete.complete, false)
  assert.deepEqual(incomplete.missing, ['localhost navigation', 'accessibility snapshot', 'browser interaction', 'screenshot'])
}

{
  const prompt = buildMissionRunPrompt(mission({
    scratchMemory: [{
      id: 'scratch-1',
      kind: 'finding',
      text: 'Keep the candidate source list local to this mission.',
      source: 'agent',
      runIndex: 1,
      checkpointId: null,
      createdAt: 1_000,
      updatedAt: 1_000,
    }],
  }), 2)
  assert.match(prompt, /Mission scratch memory \(mission-scoped, not project memory\):/)
  assert.match(prompt, /Keep the candidate source list local to this mission/)

  const freshPrompt = buildMissionRunPrompt(mission({
    contextMode: 'fresh',
    scratchMemory: [{
      id: 'scratch-2',
      kind: 'finding',
      text: 'Do not carry into fresh context.',
      source: 'agent',
      runIndex: 1,
      checkpointId: null,
      createdAt: 1_000,
      updatedAt: 1_000,
    }],
  }), 2)
  assert.doesNotMatch(freshPrompt, /Mission scratch memory/)
}

{
  const keepGoing = decideNextMissionStep(mission(), 1, { done: false, note: 'Only two source-backed items.' }, 2_000)
  assert.equal(keepGoing.action, 'continue')
  assert.match(keepGoing.nextPrompt ?? '', /Continue with the next pass/)

  const done = decideNextMissionStep(mission(), 2, { done: true, note: 'All five items have links.', evidenceCount: 5 }, 2_000)
  assert.equal(done.action, 'complete')
  assert.match(done.reason, /All five/)

  const capped = decideNextMissionStep(mission(), 3, { done: false, note: 'Still missing sources.' }, 2_000)
  assert.equal(capped.action, 'pause')
  assert.match(capped.reason, /Run limit/)
}

{
  const fixed = mission({ repeatMode: 'fixed_count', maxRuns: 2, stopCondition: '' })
  assert.equal(decideNextMissionStep(fixed, 1, { done: false, note: '' }, 2_000).action, 'continue')
  assert.equal(decideNextMissionStep(fixed, 2, { done: false, note: '' }, 2_000).action, 'complete')

  const selfPaced = mission({ repeatMode: 'self_paced' })
  assert.equal(decideNextMissionStep(selfPaced, 1, { done: false, note: '' }, 2_000).action, 'pause')
}

{
  const m = mission({ maxRuns: 4 })
  const pass1 = decideNextMissionStep(m, 1, { done: false, note: 'Only two source-backed items.', evidenceCount: 2 }, 2_000)
  const pass2 = decideNextMissionStep(m, 2, { done: false, note: 'Four items found; one missing a link.', evidenceCount: 4 }, 2_000)
  const pass3 = decideNextMissionStep(m, 3, { done: true, note: 'Five source-backed items are linked.', evidenceCount: 5 }, 2_000)
  assert.equal(pass1.action, 'continue')
  assert.equal(pass2.action, 'continue')
  assert.equal(pass3.action, 'complete')
  assert.match(pass2.nextPrompt ?? '', /Continue with the next pass/)
}

{
  // Fresh-thread continuations (contextMode 'fresh_with_notes') start a brand-new thread
  // every pass, so the continuation prompt is the *only* context that thread ever sees —
  // it must still carry the mission's objective, not just the pass-failure reason and
  // stop condition.
  const m = mission({ contextMode: 'fresh_with_notes', objective: 'Collect source-backed AI news from five distinct outlets.' })
  assert.equal(shouldStartFreshMissionThread(m, 2), true)
  const prompt = buildMissionContinuationPrompt(m, 2, 'Only two source-backed items.')
  assert.match(prompt, /Objective:/)
  assert.match(prompt, /Collect source-backed AI news from five distinct outlets\./)
  assert.match(prompt, /Stop condition:/)
  assert.match(prompt, /Only two source-backed items\./)
}

{
  const plainDone = decideNextMissionStep(mission(), 1, { done: true, note: 'done', evidenceCount: 0 }, 2_000)
  assert.equal(plainDone.action, 'continue')
  assert.match(plainDone.reason, /without recorded evidence/)

  const missingEvidenceCount = decideNextMissionStep(mission(), 1, { done: true, note: 'the stop condition is met' }, 2_000)
  assert.equal(missingEvidenceCount.action, 'continue')
  assert.match(missingEvidenceCount.reason, /without recorded evidence/)

  const cappedWithoutEvidence = decideNextMissionStep(mission({ maxRuns: 1 }), 1, { done: true, note: 'done', evidenceCount: 0 }, 2_000)
  assert.equal(cappedWithoutEvidence.action, 'pause')
  assert.match(cappedWithoutEvidence.reason, /without recorded evidence/)
}

{
  const expired = mission({ expiresAt: 1_500 })
  assert.equal(decideNextMissionStep(expired, 1, { done: true, note: 'done', evidenceCount: 1 }, 2_000).action, 'pause')
}

console.log('mission supervisor tests passed')
