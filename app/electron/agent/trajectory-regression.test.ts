import assert from 'node:assert/strict'
import {
  captureTrajectory,
  captureGolden,
  freezeGolden,
  compareTrajectory,
  isTrajectoryStable,
  formatComparison,
} from './trajectory-regression'

let checks = 0
function ok(cond: unknown, msg: string) {
  assert.ok(cond, msg)
  checks++
}

function main() {
  // --- capture: derives path/set/count and drops non-tool rows ---
  const raw = [
    { tool: 'list_dir', phase: 'explore', note: 'root' },
    { tool: 'read_file' },
    { thinking: 'no tool here' }, // dropped — no tool name
    { tool: '  search  ' }, // trimmed
    { tool: '' }, // dropped — empty
    { tool: 'apply_patch', phase: 'edit' },
  ]
  const t = captureTrajectory('fix-bug', raw)
  assert.deepEqual(t.toolPath, ['list_dir', 'read_file', 'search', 'apply_patch'])
  ok(t.stepCount === 4, 'non-tool + empty rows dropped from step count')
  assert.deepEqual(t.toolSet, ['apply_patch', 'list_dir', 'read_file', 'search'])
  ok(t.steps[0].phase === 'explore' && t.steps[0].note === 'root', 'phase/note preserved')
  assert.equal(t.taskId, 'fix-bug')
  checks++

  // taskId is required
  assert.throws(() => captureTrajectory('', [{ tool: 'x' }]), /taskId is required/)
  checks++

  // --- freeze: golden stamps injected time + label, and is a deep copy ---
  const golden = captureGolden('fix-bug', raw, { now: 1700, label: 'abc123' })
  ok(golden.capturedAt === 1700, 'capture time injected, not Date.now()')
  ok(golden.label === 'abc123', 'label recorded on golden')
  golden.toolPath.push('MUTATED')
  assert.deepEqual(captureTrajectory('fix-bug', raw).toolPath, ['list_dir', 'read_file', 'search', 'apply_patch'])
  checks++
  // re-freeze a clean golden for the comparisons below
  const g = captureGolden('fix-bug', raw, { now: 1700, label: 'abc123' })

  // freezeGolden without a label leaves label undefined
  const noLabel = freezeGolden(captureTrajectory('t', [{ tool: 'a' }]), { now: 5 })
  ok(noLabel.label === undefined && noLabel.capturedAt === 5, 'optional label omitted')
  checks++

  // --- identical candidate: no findings, stable ---
  const same = captureTrajectory('fix-bug', raw)
  const rSame = compareTrajectory(g, same)
  ok(rSame.regressed === false && rSame.findings.length === 0, 'identical trajectory has no findings')
  ok(isTrajectoryStable(rSame), 'isTrajectoryStable true when not regressed')
  checks++

  // --- step inflation: 4 -> 12 is a regression ("used to take 4 steps, now 12") ---
  const twelve = captureTrajectory(
    'fix-bug',
    ['list_dir', 'read_file', 'search', 'apply_patch', 'read_file', 'read_file', 'search', 'search', 'read_file', 'search', 'read_file', 'apply_patch'].map((tool) => ({ tool })),
  )
  const rInflate = compareTrajectory(g, twelve)
  ok(rInflate.regressed === true, '4 -> 12 steps flagged as regression')
  const infl = rInflate.findings.find((f) => f.kind === 'step_inflation')
  ok(infl?.severity === 'regression' && /now 12/.test(infl.message), 'step_inflation message reports new count')
  ok(rInflate.goldenSteps === 4 && rInflate.candidateSteps === 12, 'result carries both step counts')
  checks++

  // --- mild growth 4 -> 6 (+50%) is a warning, not a regression (>25%, <=100%). The
  // extra steps extend the golden path (prefix-preserving) and reuse existing tools, so
  // step growth is the only signal — it must not escalate to a regression. ---
  const six = captureTrajectory('fix-bug', ['list_dir', 'read_file', 'search', 'apply_patch', 'read_file', 'search'].map((tool) => ({ tool })))
  const rWarn = compareTrajectory(g, six, { regressionStepGrowth: 1.0, warnStepGrowth: 0.25 })
  const inflWarn = rWarn.findings.find((f) => f.kind === 'step_inflation')
  ok(inflWarn?.severity === 'warning', '4 -> 6 (+50%) is a warning, not a regression')
  ok(rWarn.regressed === false, 'mild prefix-preserving growth does not regress')
  checks++

  // --- wrong-tool pick: introduces a tool the golden never used => regression ---
  const wrongTool = captureTrajectory('fix-bug', [
    { tool: 'list_dir' },
    { tool: 'read_file' },
    { tool: 'web_search' }, // was 'search' — a different/new tool
    { tool: 'apply_patch' },
  ])
  const rWrong = compareTrajectory(g, wrongTool)
  ok(rWrong.regressed === true, 'newly-introduced tool is a regression')
  const setChange = rWrong.findings.find((f) => f.kind === 'tool_set_changed')
  ok(setChange?.severity === 'regression' && Array.isArray(setChange.actual) && setChange.actual.includes('web_search'), 'introduced tool named in finding')
  const div = rWrong.findings.find((f) => f.kind === 'path_divergence')
  ok(div?.atIndex === 2 && div.expected === 'search' && div.actual === 'web_search', 'path divergence pinpoints the wrong step')
  // most-severe first
  ok(rWrong.findings[0].severity === 'regression', 'findings sorted most-severe first')
  checks++

  // --- dropping a tool (fewer steps) is a warning, not a regression ---
  const dropped = captureTrajectory('fix-bug', [{ tool: 'list_dir' }, { tool: 'read_file' }, { tool: 'apply_patch' }])
  const rDrop = compareTrajectory(g, dropped)
  const dropFinding = rDrop.findings.find((f) => f.kind === 'tool_set_changed')
  ok(dropFinding?.severity === 'warning' && Array.isArray(dropFinding.expected) && dropFinding.expected.includes('search'), 'dropped tool is a warning')
  // path still diverges at index 2 (apply_patch vs search) -> that is the regression driver
  ok(rDrop.findings.some((f) => f.kind === 'path_divergence' && f.severity === 'regression'), 'reordered path still flags divergence')
  checks++

  // --- path divergence can be down-ranked to a warning via options ---
  const reordered = captureTrajectory('fix-bug', [{ tool: 'read_file' }, { tool: 'list_dir' }, { tool: 'search' }, { tool: 'apply_patch' }])
  const rSoft = compareTrajectory(g, reordered, { pathDivergenceIsRegression: false })
  ok(rSoft.regressed === false, 'pathDivergenceIsRegression:false demotes reorder to warning')
  ok(rSoft.findings.some((f) => f.kind === 'path_divergence' && f.severity === 'warning'), 'divergence present as warning')
  checks++

  // --- task mismatch is always a regression ---
  const other = captureTrajectory('other-task', raw)
  const rMismatch = compareTrajectory(g, other)
  ok(rMismatch.regressed === true, 'task mismatch regresses')
  ok(rMismatch.findings[0].kind === 'task_mismatch', 'task mismatch surfaces')
  checks++

  // --- prefix path (candidate is exact prefix of golden) does NOT diverge ---
  const prefix = captureTrajectory('fix-bug', [{ tool: 'list_dir' }, { tool: 'read_file' }])
  const rPrefix = compareTrajectory(g, prefix)
  ok(!rPrefix.findings.some((f) => f.kind === 'path_divergence'), 'a strict prefix does not count as divergence')
  checks++

  // --- empty golden, non-empty candidate: warning for new behaviour ---
  const emptyGolden = captureGolden('t2', [], { now: 1 })
  const rEmpty = compareTrajectory(emptyGolden, captureTrajectory('t2', [{ tool: 'a' }]))
  ok(rEmpty.findings.some((f) => f.kind === 'step_inflation' && f.severity === 'warning'), 'empty golden vs steps warns')
  checks++

  // --- formatComparison renders a readable summary ---
  const text = formatComparison(rWrong)
  ok(/REGRESSED/.test(text) && /regression:/.test(text), 'formatComparison shows verdict + findings')
  const okText = formatComparison(rSame)
  ok(/ok \(4 vs 4 steps\)/.test(okText), 'clean comparison formats as ok one-liner')
  checks++

  console.log(`trajectory-regression: all ${checks} checks passed`)
}

main()
