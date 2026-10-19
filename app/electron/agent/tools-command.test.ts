import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { executeTool } from './tools.ts'
import { canUseMacSandbox } from './sandbox.ts'

const nativeSandbox = canUseMacSandbox()
const localOptIn = 'VONTRA_ALLOW_UNSANDBOXED_COMMANDS'
const ok = (name: string) => console.log(`PASS ${name}`)
const shellQuote = (value: string) => `'${value.replace(/'/g, "'\\''")}'`

function restoreOptIn(value: string | undefined) {
  if (value === undefined) delete process.env[localOptIn]
  else process.env[localOptIn] = value
}

async function withCommandWorkspace(run: (cwd: string) => Promise<void>) {
  const cwd = await fs.mkdtemp(path.join(os.tmpdir(), 'vontra-tools-command-'))
  const previous = process.env[localOptIn]
  try {
    delete process.env[localOptIn]
    if (!nativeSandbox) process.env[localOptIn] = '1'
    await fs.mkdir(path.join(cwd, 'src/nested'), { recursive: true })
    await fs.writeFile(path.join(cwd, 'README.md'), '# Synthetic workspace\n')
    await run(cwd)
  } finally {
    restoreOptIn(previous)
    await fs.rm(cwd, { recursive: true, force: true })
  }
}

if (!nativeSandbox) {
  const cwd = await fs.mkdtemp(path.join(os.tmpdir(), 'vontra-tools-command-denied-'))
  const previous = process.env[localOptIn]
  try {
    delete process.env[localOptIn]
    const result = await executeTool('run_command', { command: 'printf unexpected > should-not-exist.txt' }, { cwd })
    assert.equal(result.ok, false)
    assert.match(result.output, /No supported command sandbox is available/)
    await assert.rejects(fs.access(path.join(cwd, 'should-not-exist.txt')))
    ok('default command execution fails closed without a supported sandbox')
  } finally {
    restoreOptIn(previous)
    await fs.rm(cwd, { recursive: true, force: true })
  }
  console.log('INFO command fixtures use scoped local-development opt-in in disposable workspaces')
}

await withCommandWorkspace(async (cwd) => {
  const result = await executeTool('run_command', { command: 'printf ok' }, { cwd })
  assert.strictEqual(result.ok, true, result.output)
  assert.match(result.meta ?? '', nativeSandbox ? /sandbox=build-test/ : /sandbox=unsandboxed/)
  assert.match(result.meta ?? '', nativeSandbox ? /macos-sandbox-exec/ : /unsandboxed/)
  assert.match(result.meta ?? '', /artifacts=sbox-/)
  assert.doesNotMatch(result.output, /Sandbox artifacts/)
  assert.equal(result.output, 'ok')
  assert.equal(result.artifacts?.id, result.sandboxArtifacts?.id)
  assert.match(result.artifacts?.id ?? '', /^sbox-/)
  ok('run_command keeps sandbox metadata out of the model-visible output')
})

await withCommandWorkspace(async (cwd) => {
  const vendorSecret = `ghp_${'A'.repeat(36)}`
  const opaqueSecret = 'p9Ks_4LzN7qR2vX8mC5dT1wB6hJ3sF0a'
  const ordinaryHash = '0123456789abcdef'.repeat(4)
  await fs.writeFile(path.join(cwd, 'package.json'), JSON.stringify({
    name: 'malicious-output-fixture',
    scripts: { test: 'node leak.cjs' },
  }))
  await fs.writeFile(path.join(cwd, 'leak.cjs'), [
    `console.log(${JSON.stringify(`vendor ${vendorSecret}`)})`,
    `console.error(${JSON.stringify(`opaque ${opaqueSecret}`)})`,
    `console.log(${JSON.stringify(`ordinary log checksum ${ordinaryHash}`)})`,
  ].join('\n'))
  const fixtureBin = path.join(cwd, 'fixture-bin')
  await fs.mkdir(fixtureBin)
  await fs.writeFile(path.join(fixtureBin, 'npm'), `#!/bin/sh\nexec ${shellQuote(process.execPath)} leak.cjs\n`)
  await fs.chmod(path.join(fixtureBin, 'npm'), 0o755)
  const previousPath = process.env.PATH
  process.env.PATH = `${fixtureBin}:/usr/bin:/bin`
  let result: Awaited<ReturnType<typeof executeTool>>
  try {
    result = await executeTool('run_command', { command: 'npm test --silent' }, { cwd })
  } finally {
    if (previousPath === undefined) delete process.env.PATH
    else process.env.PATH = previousPath
  }
  assert.strictEqual(result.ok, true, result.output)
  assert.doesNotMatch(result.output, new RegExp(vendorSecret))
  assert.doesNotMatch(result.output, new RegExp(opaqueSecret))
  assert.match(result.output, /TensorFold(?: Studio)? redacted 2 sensitive values/)
  assert.match(result.output, new RegExp(ordinaryHash), 'ordinary SHA-256-like hashes remain readable')
  assert.match(result.meta ?? '', /redacted=2/)
  assert.doesNotMatch(JSON.stringify(result.sandboxArtifacts?.logs ?? []), new RegExp(vendorSecret))
  assert.doesNotMatch(JSON.stringify(result.sandboxArtifacts?.logs ?? []), new RegExp(opaqueSecret))
  ok('malicious npm test output is value-redacted while ordinary logs and hashes survive')
})

await withCommandWorkspace(async (cwd) => {
  const result = await executeTool('run_command', { command: 'mkdir -p .vontra-artifacts && printf patchable > .vontra-artifacts/result.txt' }, { cwd })
  assert.strictEqual(result.ok, true)
  assert.strictEqual(result.artifacts?.patch, true)
  assert.ok(result.artifacts?.diffs.some((entry) => entry.path === '.vontra-artifacts/result.txt' && entry.status === 'added'))
  assert.match(result.sandboxArtifacts?.patch ?? '', /diff --git a\/\.vontra-artifacts\/result.txt b\/\.vontra-artifacts\/result.txt/)
  assert.match(result.sandboxArtifacts?.patch ?? '', /\+patchable/)
  ok('run_command returns sandbox patch evidence to the caller')
})
