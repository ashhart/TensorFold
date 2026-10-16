// Run: npx jiti electron/agent/best-of-n.test.ts
import assert from 'node:assert/strict'
import {
  runBestOfN,
  scoreAttempt,
  rankScored,
  type AttemptDescriptor,
  type AttemptOutcome,
  type VerifierVerdict,
  type ScoredAttempt,
  type BestOfNDeps,
} from './best-of-n'

let passed = 0
function ok(name: string) {
  passed++
  console.log(`  ✓ ${name}`)
}

function d(id: string, spec = id): AttemptDescriptor {
  return { id, spec }
}

function baseDeps(overrides: Partial<BestOfNDeps> = {}): BestOfNDeps {
  return {
    runAttempt: async (desc) => ({ ok: true, output: `out:${desc.id}` }),
    verify: async () => ({ testsPassed: true, vote: 0.5 }),
    ...overrides,
  }
}

async function run() {
  // --- scoreAttempt: crashed is always lowest, below any red build --------------
  {
    const crashedTopVote = scoreAttempt({ ok: false, output: '' }, { vote: 1, testsPassed: true })
    const redZeroVote = scoreAttempt({ ok: true, output: 'x' }, { testsPassed: false, vote: 0 })
    assert.ok(crashedTopVote < redZeroVote, 'a crashed attempt must rank below even a red-build attempt')
    ok('scoreAttempt: crashed attempt is strictly below a failing-tests attempt')
  }

  // --- scoreAttempt: green > unknown > red, regardless of vote ------------------
  {
    const green = scoreAttempt({ ok: true, output: 'x' }, { testsPassed: true, vote: 0 })
    const unknown = scoreAttempt({ ok: true, output: 'x' }, { vote: 1 }) // no test signal, max vote
    const red = scoreAttempt({ ok: true, output: 'x' }, { testsPassed: false, vote: 1 })
    assert.ok(green > unknown, 'green build (vote 0) must beat unknown (vote 1) — tests dominate vote')
    assert.ok(unknown > red, 'unknown (vote 1) must beat red (vote 1)')
    ok('scoreAttempt: testsPassed tier strictly dominates the vote')
  }

  // --- scoreAttempt: vote breaks ties WITHIN a tier ----------------------------
  {
    const hi = scoreAttempt({ ok: true, output: 'x' }, { testsPassed: true, vote: 0.9 })
    const lo = scoreAttempt({ ok: true, output: 'x' }, { testsPassed: true, vote: 0.1 })
    assert.ok(hi > lo, 'within the green tier, higher vote wins')
    ok('scoreAttempt: vote orders attempts within the same tier')
  }

  // --- scoreAttempt: vote is clamped, NaN/negative/>1 tolerated -----------------
  {
    const over = scoreAttempt({ ok: true, output: 'x' }, { testsPassed: true, vote: 5 })
    const at1 = scoreAttempt({ ok: true, output: 'x' }, { testsPassed: true, vote: 1 })
    assert.equal(over, at1, 'vote > 1 clamps to 1')
    const neg = scoreAttempt({ ok: true, output: 'x' }, { testsPassed: true, vote: -3 })
    const at0 = scoreAttempt({ ok: true, output: 'x' }, { testsPassed: true, vote: 0 })
    assert.equal(neg, at0, 'negative vote clamps to 0')
    const nan = scoreAttempt({ ok: true, output: 'x' }, { testsPassed: true, vote: NaN as any })
    assert.equal(nan, at0, 'NaN vote treated as 0')
    ok('scoreAttempt: vote is clamped to [0,1] and tolerates NaN/negative/overflow')
  }

  // --- Happy path: N attempts, verifier picks the green + highest vote ----------
  {
    const descriptors = [d('a'), d('b'), d('c')]
    const verify = async (desc: AttemptDescriptor): Promise<VerifierVerdict> => {
      if (desc.id === 'a') return { testsPassed: true, vote: 0.3 }
      if (desc.id === 'b') return { testsPassed: true, vote: 0.9 } // best
      return { testsPassed: false, vote: 1.0 } // red, loses despite top vote
    }
    const res = await runBestOfN(descriptors, baseDeps({ verify }))
    assert.equal(res.ranked.length, 3)
    assert.equal(res.winner?.descriptor.id, 'b')
    assert.deepEqual(res.ranked.map((r) => r.descriptor.id), ['b', 'a', 'c'])
    ok('runBestOfN: picks the green attempt with the highest vote as winner')
  }

  // --- Every descriptor appears exactly once in ranked --------------------------
  {
    const descriptors = [d('a'), d('b'), d('c'), d('e')]
    const res = await runBestOfN(descriptors, baseDeps())
    const ids = res.ranked.map((r) => r.descriptor.id).sort()
    assert.deepEqual(ids, ['a', 'b', 'c', 'e'])
    ok('runBestOfN: every descriptor appears exactly once in the ranking')
  }

  // --- A crashed attempt is scored as a loss, not thrown ------------------------
  {
    const descriptors = [d('good'), d('boom')]
    const runAttempt = async (desc: AttemptDescriptor): Promise<AttemptOutcome> => {
      if (desc.id === 'boom') return { ok: false, output: '', error: 'exploded' }
      return { ok: true, output: 'ok' }
    }
    const res = await runBestOfN(descriptors, baseDeps({ runAttempt }))
    assert.equal(res.winner?.descriptor.id, 'good')
    const boom = res.ranked.find((r) => r.descriptor.id === 'boom')!
    assert.equal(boom.outcome.ok, false)
    assert.equal(boom.outcome.error, 'exploded')
    ok('runBestOfN: a crashed attempt becomes a low-scored loss (winner is the survivor)')
  }

  // --- A throwing runAttempt is caught per-attempt, not batch-fatal -------------
  {
    const descriptors = [d('throws'), d('fine')]
    const runAttempt = async (desc: AttemptDescriptor): Promise<AttemptOutcome> => {
      if (desc.id === 'throws') throw new Error('unexpected reject')
      return { ok: true, output: 'fine' }
    }
    const res = await runBestOfN(descriptors, baseDeps({ runAttempt }))
    assert.equal(res.winner?.descriptor.id, 'fine')
    const t = res.ranked.find((r) => r.descriptor.id === 'throws')!
    assert.equal(t.outcome.ok, false)
    assert.match(t.outcome.error ?? '', /unexpected reject/)
    ok('runBestOfN: a throwing runAttempt is caught and does not sink the batch')
  }

  // --- A throwing verifier => neutral verdict, rationale surfaced, not fatal -----
  {
    const descriptors = [d('a'), d('b')]
    const verify = async (desc: AttemptDescriptor): Promise<VerifierVerdict> => {
      if (desc.id === 'a') throw new Error('judge died')
      return { testsPassed: true, vote: 0.2 }
    }
    const res = await runBestOfN(descriptors, baseDeps({ verify }))
    // b has a real green verdict; a has neutral (unknown tier, vote 0). b should win.
    assert.equal(res.winner?.descriptor.id, 'b')
    const a = res.ranked.find((r) => r.descriptor.id === 'a')!
    assert.match(a.verdict.rationale ?? '', /verifier failed: judge died/)
    ok('runBestOfN: a throwing verifier yields a neutral verdict with surfaced rationale')
  }

  // --- Crashed attempts are NOT verified ---------------------------------------
  {
    const descriptors = [d('boom')]
    let verifyCalls = 0
    const deps = baseDeps({
      runAttempt: async () => ({ ok: false, output: '', error: 'no' }),
      verify: async () => { verifyCalls++; return { testsPassed: true, vote: 1 } },
    })
    await runBestOfN(descriptors, deps)
    assert.equal(verifyCalls, 0, 'a crashed attempt should never be handed to the verifier')
    ok('runBestOfN: does not verify a crashed attempt')
  }

  // --- Deterministic tie-break by original index (independent of settle order) --
  {
    // Two green attempts with identical vote; the SECOND settles first. Winner must still be
    // the earlier descriptor by original index.
    const descriptors = [d('first'), d('second')]
    const runAttempt = async (desc: AttemptDescriptor): Promise<AttemptOutcome> => {
      // second finishes first
      await new Promise((r) => setTimeout(r, desc.id === 'second' ? 5 : 40))
      return { ok: true, output: desc.id }
    }
    const verify = async (): Promise<VerifierVerdict> => ({ testsPassed: true, vote: 0.5 })
    const res = await runBestOfN(descriptors, baseDeps({ runAttempt, verify, maxConcurrent: 2 }))
    assert.equal(res.winner?.descriptor.id, 'first', 'tie must break to the earlier original index, not settle order')
    assert.deepEqual(res.ranked.map((r) => r.descriptor.id), ['first', 'second'])
    ok('runBestOfN: ties break deterministically by original index, not completion order')
  }

  // --- Bounded concurrency: never more than maxConcurrent in flight -------------
  {
    const descriptors = Array.from({ length: 6 }, (_, i) => d(`t${i}`))
    let inFlight = 0
    let maxInFlight = 0
    const runAttempt = async (desc: AttemptDescriptor): Promise<AttemptOutcome> => {
      inFlight++
      maxInFlight = Math.max(maxInFlight, inFlight)
      await new Promise((r) => setTimeout(r, 25))
      inFlight--
      return { ok: true, output: desc.id }
    }
    await runBestOfN(descriptors, baseDeps({ runAttempt, maxConcurrent: 2 }))
    assert.equal(maxInFlight, 2, `expected at most 2 attempts in flight, saw ${maxInFlight}`)
    ok('runBestOfN: bounds concurrency to maxConcurrent')
  }

  // --- Default maxConcurrent is a small sane value ------------------------------
  {
    const descriptors = Array.from({ length: 5 }, (_, i) => d(`t${i}`))
    let inFlight = 0
    let maxInFlight = 0
    const runAttempt = async (desc: AttemptDescriptor): Promise<AttemptOutcome> => {
      inFlight++
      maxInFlight = Math.max(maxInFlight, inFlight)
      await new Promise((r) => setTimeout(r, 15))
      inFlight--
      return { ok: true, output: desc.id }
    }
    await runBestOfN(descriptors, baseDeps({ runAttempt })) // no maxConcurrent
    assert.ok(maxInFlight > 1 && maxInFlight <= 3, `default should be a small (<=3) sane value, saw ${maxInFlight}`)
    ok('runBestOfN: defaults maxConcurrent to a small sane value')
  }

  // --- Abort before start launches nothing, all attempts are losses -------------
  {
    const descriptors = [d('a'), d('b')]
    const controller = new AbortController()
    controller.abort()
    let runCalls = 0
    const deps = baseDeps({
      runAttempt: async () => { runCalls++; return { ok: true, output: 'x' } },
      signal: controller.signal,
    })
    const res = await runBestOfN(descriptors, deps)
    assert.equal(runCalls, 0, 'an already-aborted signal must launch zero attempts')
    assert.equal(res.ranked.length, 2)
    assert.ok(res.ranked.every((r) => !r.outcome.ok), 'all attempts are losses when aborted before start')
    ok('runBestOfN: an already-aborted signal launches nothing and marks all attempts as losses')
  }

  // --- Abort mid-run stops launching new attempts -------------------------------
  {
    const descriptors = Array.from({ length: 6 }, (_, i) => d(`t${i}`))
    const controller = new AbortController()
    let launched = 0
    const runAttempt = async (desc: AttemptDescriptor): Promise<AttemptOutcome> => {
      launched++
      await new Promise((r) => setTimeout(r, 20))
      return { ok: true, output: desc.id }
    }
    const deps = baseDeps({ runAttempt, maxConcurrent: 2, signal: controller.signal })
    setTimeout(() => controller.abort(), 5)
    const res = await runBestOfN(descriptors, deps)
    assert.equal(res.ranked.length, 6, 'still one ranked entry per descriptor')
    assert.ok(launched < descriptors.length, `expected abort to prevent some launches, launched=${launched}`)
    ok('runBestOfN: aborting mid-run stops launching new attempts but still returns every descriptor')
  }

  // --- Empty descriptors => empty ranking, null winner --------------------------
  {
    const res = await runBestOfN([], baseDeps({ runAttempt: async () => { throw new Error('nope') } }))
    assert.deepEqual(res.ranked, [])
    assert.equal(res.winner, null)
    ok('runBestOfN: empty descriptor list gives an empty ranking and a null winner')
  }

  // --- Single attempt: it wins even if red (only candidate) ---------------------
  {
    const res = await runBestOfN([d('solo')], baseDeps({
      verify: async () => ({ testsPassed: false, vote: 0 }),
    }))
    assert.equal(res.winner?.descriptor.id, 'solo')
    assert.equal(res.ranked.length, 1)
    ok('runBestOfN: a lone attempt wins even with failing tests (it is the only candidate)')
  }

  // --- winner === ranked[0] invariant -------------------------------------------
  {
    const res = await runBestOfN([d('a'), d('b'), d('c')], baseDeps())
    assert.equal(res.winner, res.ranked[0])
    ok('runBestOfN: winner is exactly ranked[0]')
  }

  // --- meta is carried through untouched ----------------------------------------
  {
    const desc: AttemptDescriptor = { id: 'm', spec: 's', meta: { model: 'opus', temperature: 0.7 } }
    const res = await runBestOfN([desc], baseDeps())
    assert.deepEqual(res.winner?.descriptor.meta, { model: 'opus', temperature: 0.7 })
    ok('runBestOfN: descriptor.meta is carried through to the ranked result untouched')
  }

  // --- rankScored: pure, does not mutate input, deterministic -------------------
  {
    const mk = (id: string, score: number): ScoredAttempt => ({
      descriptor: d(id),
      outcome: { ok: true, output: '' },
      verdict: {},
      score,
    })
    const input = [mk('a', 1), mk('b', 3), mk('c', 3)]
    const snapshot = input.map((s) => s.descriptor.id)
    const ranked = rankScored(input, (_s, i) => i)
    // b and c tie at 3; b came first => b before c. a last.
    assert.deepEqual(ranked.map((r) => r.descriptor.id), ['b', 'c', 'a'])
    assert.deepEqual(input.map((s) => s.descriptor.id), snapshot, 'rankScored must not mutate its input array order')
    ok('rankScored: sorts best-first, breaks ties by index, and does not mutate input')
  }

  console.log(`\n${passed} checks passed.`)
}

run().catch((e) => {
  console.error(e)
  process.exit(1)
})
