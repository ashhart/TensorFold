import assert from 'node:assert/strict'
import { mkdtemp, mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'

const root = await mkdtemp(path.join(tmpdir(), 'vontra-managed-output-'))
const workspace = path.join(root, 'workspace')
const stateRoot = path.join(root, 'app-state', 'outputs')
process.env.VONTRA_HOME = path.join(root, 'vontra-home')
const { boundToolOutput, toolOutputSpillPath } = await import('../../agent/context-governance')
const { executeTool } = await import('../../agent/tools')
const { createManagedOutputStore } = await import('./managed-outputs')
await mkdir(workspace, { recursive: true })

try {
  const store = createManagedOutputStore(stateRoot)
  const secretMarker = 'MANAGED-OUTPUT-PLAINTEXT-SECRET'
  const output = ['first', 'second', 'third', secretMarker, 'x'.repeat(20_000)].join('\n')
  const requestedRef = toolOutputSpillPath('readonly-run', 1, 'read_file')
  const ref = await store.write(requestedRef, output)
  const bounded = boundToolOutput({ tool: 'read_file', output, spillRelPath: ref, spillWritten: true })
  assert.match(bounded.content, /vontra-output:\/\//)

  let remoteBackendTouched = false
  const remoteBackend = new Proxy({ kind: 'ssh', rootCwd: '/remote/project' } as any, {
    get(target, property) {
      if (property in target) return target[property]
      return async () => { remoteBackendTouched = true; throw new Error('remote backend must not serve managed output') }
    },
  })
  const page = await executeTool('read_file', { path: ref, offset: 2, limit: 1 }, {
    cwd: '/remote/project', backend: remoteBackend, readManagedOutput: store.read,
  })
  assert.equal(page.ok, true)
  assert.match(page.output, /second/)
  assert.equal(remoteBackendTouched, false, 'opaque output paging is host-local even for an SSH workspace')
  assert.deepEqual(await readdir(workspace), [], 'read-only spill never creates an untracked workspace file')
  assert.equal((await readdir(stateRoot)).length, 1, 'full output lives under protected app state')
  const rawFile = await readFile(path.join(stateRoot, requestedRef.slice('vontra-output://'.length)), 'utf8')
  assert.match(rawFile, /safe-json:v1:/, 'managed output uses the protected JSON envelope')
  assert.doesNotMatch(rawFile, new RegExp(secretMarker), 'raw managed-output file contains no plaintext tool secret')

  const legacyRef = 'vontra-output://legacy.txt'
  const legacyPath = path.join(stateRoot, 'legacy.txt')
  await writeFile(legacyPath, 'LEGACY-RAW-OUTPUT-SECRET', { mode: 0o600 })
  assert.equal(await store.read(legacyRef), 'LEGACY-RAW-OUTPUT-SECRET', 'legacy plaintext output remains readable during migration')
  assert.doesNotMatch(await readFile(legacyPath, 'utf8'), /LEGACY-RAW-OUTPUT-SECRET/, 'legacy output is replaced by a protected envelope on first read')
  await assert.rejects(() => store.write('vontra-output://../escape.txt', 'nope'), /invalid managed output reference/)
} finally {
  delete process.env.VONTRA_HOME
  await rm(root, { recursive: true, force: true })
}

console.log('managed output read-only spill tests passed')
