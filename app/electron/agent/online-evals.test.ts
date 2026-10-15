import assert from 'node:assert/strict'
import {
  runOnlineEval,
  pickSamples,
  substanceScorer,
  tokenBudgetScorer,
  judgeScorer,
  type RunSample,
  type JudgeFn,
  type ThresholdRule,
} from './online-evals'

let ok = 0
const t = (label: string, cond: boolean) => {
  assert.ok(cond, label)
  ok++
}

// ---- pickSamples --------------------------------------------------------

const mk = (n: number): RunSample[] =>
  Array.from({ length: n }, (_, i) => ({ index: i, text: 'x'.repeat(500) }))

// rate=1 -> every turn
t('rate=1 scores all', pickSamples(mk(4), { sampleRate: 1 }).length === 4)
// rate=0 -> only index 0 (always-first guarantee)
assert.deepEqual(pickSamples(mk(4), { sampleRate: 0 }), [0])
t('rate=0 keeps only index 0', true)
// deterministic rng gate
// rng is drawn once per sample (idx0 still drawn, but idx0 is kept via always-first).
// seq: idx0->0.9(kept anyway), idx1->0.1(kept), idx2->0.9(no), idx3->0.1(kept).
const seq = [0.9, 0.1, 0.9, 0.1]
let i = 0
const rng = () => seq[i++ % seq.length]
assert.deepEqual(pickSamples(mk(4), { sampleRate: 0.5, rng }), [0, 1, 3])
t('rng gate is deterministic (idx0 always, then rng<0.5)', true)
// sampleEvery cadence is deterministic regardless of rng
i = 0
assert.deepEqual(pickSamples(mk(6), { sampleRate: 0, sampleEvery: 3, rng: () => 0.99 }), [0, 3])
t('sampleEvery=3 picks 0 and 3', true)

// ---- heuristic scorers --------------------------------------------------

const sub = substanceScorer()
assert.equal(sub.score({ index: 0, text: '' }), null) // abstain on empty
t('substance abstains on empty', true)
{
  const short = sub.score({ index: 0, text: 'hi' })!
  const long = sub.score({ index: 0, text: 'x'.repeat(400) })!
  t('substance rewards longer text', long.score > short.score)
  t('substance caps at 1', long.score === 1)
  const refusal = sub.score({ index: 0, text: "I can't help with that " + 'x'.repeat(400) })!
  t('substance penalises refusals', refusal.score <= 0.2)
}

const budget = tokenBudgetScorer(1000)
assert.equal(budget.score({ index: 0, text: 'a' }), null) // no token usage -> abstain
t('tokenBudget abstains without tokens', true)
t('tokenBudget=1 under budget', budget.score({ index: 0, text: 'a', totalTokens: 500 })!.score === 1)
t('tokenBudget decays over budget', budget.score({ index: 0, text: 'a', totalTokens: 4000 })!.score === 0.25)

// ---- judge scorer (DI) --------------------------------------------------

const calls: number[] = []
const fakeJudge: JudgeFn = async ({ index }) => {
  calls.push(index)
  return { score: index === 1 ? 0.1 : 0.9, rationale: 'judged' }
}
const judge = judgeScorer(fakeJudge)
{
  const r = await judge.score({ index: 2, text: 'hello' })
  t('judge returns injected score', r!.score === 0.9)
  const abstain = await judge.score({ index: 0, text: '   ' })
  t('judge abstains on blank (no judge call)', abstain === null)
}
// judge score is clamped
{
  const wild = judgeScorer(async () => ({ score: 5 }))
  const r = await wild.score({ index: 0, text: 'x' })
  t('judge clamps out-of-range score', r!.score === 1)
}

// ---- full pass: quality alert -------------------------------------------

const badRun: RunSample[] = [
  { index: 0, text: 'ok '.repeat(200), totalTokens: 100 }, // long+cheap => good
  { index: 1, text: 'no', totalTokens: 100 }, // judge scores 0.1 => bad
]
const qualityRule: ThresholdRule = {
  kind: 'quality',
  metric: 'llm_judge',
  op: 'below',
  threshold: 0.6,
  severity: 'critical',
  message: 'judge mean {value} < {threshold}',
}
{
  const rep = await runOnlineEval(badRun, {
    scorers: [judgeScorer(fakeJudge), substanceScorer()],
    rules: [qualityRule],
    sampleRate: 1,
    now: () => 12345,
  })
  t('all turns sampled', rep.sampledIndices.length === 2)
  const jAgg = rep.aggregates.find((a) => a.scorer === 'llm_judge')!
  t('judge mean = (0.9+0.1)/2 = 0.5', Math.abs(jAgg.mean! - 0.5) < 1e-9)
  t('quality alert fired', rep.alerts.length === 1)
  t('alert kind quality', rep.alerts[0].kind === 'quality')
  t('alert severity critical', rep.alerts[0].severity === 'critical')
  t('alert message substituted', rep.alerts[0].message === 'judge mean 0.5 < 0.6')
  t('alert uses injected clock', rep.alerts[0].at === 12345)
}

// ---- full pass: cost alert (total cost over threshold) ------------------

const costRun: RunSample[] = [
  { index: 0, text: 'a', costUsd: 0.4, totalTokens: 1000 },
  { index: 1, text: 'b', costUsd: 0.4, totalTokens: 1000 },
]
const costRule: ThresholdRule = {
  kind: 'cost',
  metric: 'cost',
  op: 'above',
  threshold: 0.5,
  severity: 'warn',
  message: 'run cost ${value} over ${threshold}',
}
{
  const rep = await runOnlineEval(costRun, { scorers: [], rules: [costRule], now: () => 1 })
  t('total cost summed over ALL samples', Math.abs(rep.totalCostUsd - 0.8) < 1e-9)
  t('total tokens summed', rep.totalTokens === 2000)
  t('cost alert fired', rep.alerts.length === 1 && rep.alerts[0].kind === 'cost')
  t('cost value substituted', rep.alerts[0].message === 'run cost $0.8 over $0.5')
}

// ---- rule on unavailable metric is skipped ------------------------------
{
  const rep = await runOnlineEval([{ index: 0, text: '' }], {
    scorers: [substanceScorer()], // abstains -> mean null
    rules: [{ kind: 'quality', metric: 'substance', op: 'below', threshold: 0.9, severity: 'warn', message: 'x' }],
    now: () => 1,
  })
  t('substance mean is null (all abstained)', rep.aggregates.find((a) => a.scorer === 'substance')!.mean === null)
  t('rule on null metric does not fire', rep.alerts.length === 0)
}

// ---- a throwing scorer abstains, never sinks the run --------------------
{
  const boom = { id: 'boom', kind: 'quality' as const, score() { throw new Error('boom') } }
  const rep = await runOnlineEval([{ index: 0, text: 'hello world', totalTokens: 10 }], {
    scorers: [boom, tokenBudgetScorer(1000)],
    rules: [],
    now: () => 1,
  })
  t('throwing scorer produces no scores', rep.aggregates.find((a) => a.scorer === 'boom')!.count === 0)
  t('other scorers still run past a throw', rep.aggregates.find((a) => a.scorer === 'token_budget')!.count === 1)
}

// ---- 'above' vs 'below' semantics + no false positives ------------------
{
  const rep = await runOnlineEval([{ index: 0, text: 'x'.repeat(500), totalTokens: 100 }], {
    scorers: [substanceScorer()],
    rules: [{ kind: 'quality', metric: 'substance', op: 'below', threshold: 0.5, severity: 'warn', message: 'x' }],
    now: () => 1,
  })
  t('healthy run fires no alert', rep.alerts.length === 0)
}

console.log(`online-evals tests passed (${ok} assertions)`)
