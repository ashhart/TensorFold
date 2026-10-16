// Run: jiti electron/agent/env-snapshots.test.ts
import assert from 'node:assert/strict'
import {
  CACHE_KEY_SCHEMA_VERSION,
  computeCacheKey,
  isBakeableStep,
  isSnapshotStale,
  planRestore,
  planSnapshotDescriptor,
  type SnapshotInputFile,
} from './env-snapshots.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('env-snapshots tests:')

const f = (path: string, content?: string): SnapshotInputFile => ({ path, content });

// --- descriptor: node + lock ------------------------------------------------
{
  const d = planSnapshotDescriptor({
    files: [f('package.json', '{"name":"x"}'), f('package-lock.json', '{"lockfileVersion":3}')],
  })
  assert.equal(d.layers.length, 1)
  assert.equal(d.layers[0].manager, 'npm')
  assert.equal(d.layers[0].command, 'npm ci') // locked => reproducible install
  assert.deepEqual(d.layers[0].inputs, ['package-lock.json', 'package.json'])
  assert.ok(d.toolchains.some((t) => t.kind === 'node'))
  ok('npm lock present => npm ci layer + node toolchain')
}

// --- manifest with no lock => unlocked command ------------------------------
{
  const d = planSnapshotDescriptor({ files: [f('package.json', '{}')] })
  assert.equal(d.layers[0].command, 'npm install')
  ok('package.json only => npm install (unlocked)')
}

// --- pnpm preferred over npm when both signatures could match ---------------
{
  const d = planSnapshotDescriptor({
    files: [f('package.json', '{}'), f('pnpm-lock.yaml', 'lockfileVersion: 9')],
  })
  const managers = d.layers.map((l) => l.manager)
  assert.deepEqual(managers, ['pnpm']) // one node layer, pnpm wins (listed first)
  assert.equal(d.layers[0].command, 'pnpm install --frozen-lockfile')
  ok('pnpm lock => single pnpm layer (no duplicate node install)')
}

// --- multi-toolchain: node + python -----------------------------------------
{
  const d = planSnapshotDescriptor({
    files: [
      f('package-lock.json', 'a'),
      f('package.json', '{}'),
      f('poetry.lock', 'b'),
      f('pyproject.toml', '[tool.poetry]'),
      f('.python-version', '3.11.4\n'),
    ],
  })
  assert.equal(d.layers.length, 2)
  const py = d.toolchains.find((t) => t.kind === 'python')
  assert.deepEqual(py, { kind: 'python', version: '3.11.4' })
  // toolchains sorted by kind
  assert.deepEqual(d.toolchains.map((t) => t.kind), ['node', 'python'])
  ok('node + python layers with explicit python version pin')
}

// --- .nvmrc version parsing strips leading v and comments -------------------
{
  const d = planSnapshotDescriptor({ files: [f('.nvmrc', '# comment\nv20.11.0\n')] })
  assert.deepEqual(d.toolchains, [{ kind: 'node', version: '20.11.0' }])
  assert.equal(d.layers.length, 0)
  ok('.nvmrc => node@20.11.0, no install layer')
}

// --- setup script split into bakeable vs runtime ----------------------------
{
  const d = planSnapshotDescriptor({
    files: [f('package.json', '{}')],
    setup: { steps: ['npm ci', 'npm run db:migrate', 'npm run codegen'] },
  })
  assert.ok(!d.runtimeSteps.includes('npm ci'))
  assert.deepEqual(d.runtimeSteps, ['npm run db:migrate', 'npm run codegen'])
  ok('setup steps: install baked, migrate/codegen deferred to runtime')
}

{
  assert.equal(isBakeableStep('  npm ci  '), true)
  assert.equal(isBakeableStep('pnpm install --frozen-lockfile'), true)
  assert.equal(isBakeableStep('npm run test'), false)
  assert.equal(isBakeableStep('echo hi'), false)
  ok('isBakeableStep classifies install vs non-install')
}

// --- absent file marker: content undefined => not present -------------------
{
  const d = planSnapshotDescriptor({ files: [f('package.json'), f('.nvmrc')] })
  assert.equal(d.layers.length, 0)
  assert.equal(d.toolchains.length, 0)
  ok('files with no content are treated as absent')
}

// --- cache key: determinism + invalidation ----------------------------------
{
  const files = [f('package.json', '{"v":1}'), f('package-lock.json', 'LOCK-A')]
  const d = planSnapshotDescriptor({ files })
  const k1 = computeCacheKey({ descriptor: d, files })
  const k2 = computeCacheKey({ descriptor: d, files })
  assert.equal(k1, k2) // deterministic
  assert.ok(k1.startsWith('snap_'))

  // changing the lock content flips the key
  const files2 = [f('package.json', '{"v":1}'), f('package-lock.json', 'LOCK-B')]
  const d2 = planSnapshotDescriptor({ files: files2 })
  const k3 = computeCacheKey({ descriptor: d2, files: files2 })
  assert.notEqual(k1, k3)
  ok('cache key deterministic; lock edit invalidates')
}

// --- cache key: salt + schema version affect the key ------------------------
{
  const files = [f('package.json', '{}')]
  const d = planSnapshotDescriptor({ files })
  const base = computeCacheKey({ descriptor: d, files })
  const salted = computeCacheKey({ descriptor: d, files, salt: 'linux/arm64' })
  const versioned = computeCacheKey({ descriptor: d, files, schemaVersion: CACHE_KEY_SCHEMA_VERSION + 1 })
  assert.notEqual(base, salted)
  assert.notEqual(base, versioned)
  ok('salt and schemaVersion both invalidate the key')
}

// --- cache key: unrelated files do NOT affect key ---------------------------
{
  const core = [f('package.json', '{}'), f('package-lock.json', 'L')]
  const d = planSnapshotDescriptor({ files: core })
  const k1 = computeCacheKey({ descriptor: d, files: core })
  const k2 = computeCacheKey({ descriptor: d, files: [...core, f('README.md', 'hello world')] })
  assert.equal(k1, k2) // README isn't referenced by any layer
  ok('files not referenced by a layer/toolchain do not change the key')
}

// --- isSnapshotStale ---------------------------------------------------------
{
  const files = [f('package.json', '{}'), f('package-lock.json', 'L1')]
  const d = planSnapshotDescriptor({ files })
  const key = computeCacheKey({ descriptor: d, files })
  assert.deepEqual(isSnapshotStale({ cachedKey: key, descriptor: d, files }), { stale: false, freshKey: key })

  const files2 = [f('package.json', '{}'), f('package-lock.json', 'L2')]
  const d2 = planSnapshotDescriptor({ files: files2 })
  const r = isSnapshotStale({ cachedKey: key, descriptor: d2, files: files2 })
  assert.equal(r.stale, true)
  assert.notEqual(r.freshKey, key)
  ok('isSnapshotStale: same inputs fresh, changed lock stale')
}

// --- restore plan: cache hit => reuse ---------------------------------------
{
  const files = [f('package.json', '{}'), f('package-lock.json', 'L')]
  const d = planSnapshotDescriptor({ files, setup: { steps: ['npm ci', 'npm run migrate'] } })
  const key = computeCacheKey({ descriptor: d, files })
  const plan = planRestore({ descriptor: d, targetKey: key, present: { cachedKeys: [key] } })
  assert.equal(plan.verdict, 'reuse')
  assert.equal(plan.cacheHit, true)
  // only the runtime step runs on a hit
  assert.deepEqual(plan.steps.map((s) => s.kind), ['runtime'])
  assert.equal(plan.steps[0].command, 'npm run migrate')
  ok('cache hit => reuse, only runtime steps run')
}

// --- restore plan: cold cache => full rebuild -------------------------------
{
  const files = [f('package.json', '{}'), f('package-lock.json', 'L')]
  const d = planSnapshotDescriptor({ files })
  const key = computeCacheKey({ descriptor: d, files })
  const plan = planRestore({ descriptor: d, targetKey: key, present: {} })
  assert.equal(plan.verdict, 'rebuild')
  assert.equal(plan.cacheHit, false)
  const kinds = plan.steps.map((s) => s.kind)
  assert.ok(kinds.includes('toolchain'))
  assert.ok(kinds.includes('layer'))
  // toolchain before layer (order matters for a real bake)
  assert.ok(kinds.indexOf('toolchain') < kinds.indexOf('layer'))
  ok('cold cache => rebuild with toolchain-before-layer ordering')
}

// --- restore plan: warm env => partial --------------------------------------
{
  const files = [f('package.json', '{}'), f('package-lock.json', 'L')]
  const d = planSnapshotDescriptor({ files })
  const key = computeCacheKey({ descriptor: d, files })
  const plan = planRestore({
    descriptor: d,
    targetKey: key,
    present: { toolchains: [{ kind: 'node' }], installedManagers: [] },
  })
  assert.equal(plan.verdict, 'partial')
  // node toolchain satisfied => skipped; npm layer still runs
  assert.ok(!plan.steps.some((s) => s.kind === 'toolchain'))
  assert.ok(plan.steps.some((s) => s.kind === 'layer'))
  assert.ok(plan.reasons.length > 0)
  ok('warm env (toolchain present) => partial restore skips toolchain step')
}

// --- restore plan: everything present but key differs => partial ------------
{
  const files = [f('package.json', '{}'), f('package-lock.json', 'L')]
  const d = planSnapshotDescriptor({ files })
  const plan = planRestore({
    descriptor: d,
    targetKey: 'snap_stale',
    present: { toolchains: [{ kind: 'node' }], installedManagers: ['npm'] },
  })
  assert.equal(plan.verdict, 'partial')
  assert.equal(plan.steps.length, 0) // nothing to run, but not a clean reuse
  ok('all baked present, key differs => partial with no steps')
}

// --- toolchain version mismatch is NOT satisfied ----------------------------
{
  const d = planSnapshotDescriptor({ files: [f('.nvmrc', '20.11.0')] })
  const plan = planRestore({
    descriptor: d,
    targetKey: 'snap_x',
    present: { toolchains: [{ kind: 'node', version: '18.0.0' }] },
  })
  assert.ok(plan.steps.some((s) => s.kind === 'toolchain'))
  ok('pinned toolchain version mismatch => toolchain step still required')
}

console.log(`\nenv-snapshots: ${passed} passed`)
