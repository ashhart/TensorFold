import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { spawn } from 'node:child_process'
import { createRequire } from 'node:module'
import { installWorkspacePackages, type IsolatedPackageRunner } from './package-install'
import { parsePackageInstallCommand } from './package-install-command'
import { sandboxCommandEnv } from './sandbox-process'
import { commandClass, approvalPolicyScope, requiresSavedApprovalPolicy } from './approval-policy'
import { commandApprovalReason } from './command-policy'
import type { SandboxedCommandResult } from './sandbox-types'

const success = (): SandboxedCommandResult => ({
  code: 0, stdout: '', stderr: '', sandboxed: false, profile: 'package-install', implementation: 'unsandboxed',
})

const actualNpm: IsolatedPackageRunner = ({ command, cwd }) => new Promise((resolve) => {
  const child = spawn('/bin/sh', ['-c', command], {
    cwd,
    env: sandboxCommandEnv(cwd),
    stdio: ['ignore', 'pipe', 'pipe'],
  })
  let stdout = ''
  let stderr = ''
  child.stdout.on('data', (value) => { stdout += value })
  child.stderr.on('data', (value) => { stderr += value })
  child.on('error', (error) => resolve({ ...success(), code: 127, stderr: error.message }))
  child.on('close', (code) => resolve({ ...success(), code: code ?? 1, stdout, stderr }))
})

async function writeJson(target: string, value: unknown): Promise<void> {
  await fs.mkdir(path.dirname(target), { recursive: true })
  await fs.writeFile(target, JSON.stringify(value, null, 2) + '\n')
}

async function fakePreparedInstall(stage: string, name = 'published-fixture'): Promise<void> {
  const target = path.join(stage, 'node_modules', name)
  await fs.mkdir(target, { recursive: true })
  await writeJson(path.join(target, 'package.json'), { name, version: '1.0.0', main: 'index.js' })
  await fs.writeFile(path.join(target, 'index.js'), 'module.exports = 42\n')
  const manifest = JSON.parse(await fs.readFile(path.join(stage, 'package.json'), 'utf8'))
  manifest.dependencies = { [name]: '1.0.0' }
  await writeJson(path.join(stage, 'package.json'), manifest)
  await writeJson(path.join(stage, 'package-lock.json'), {
    name: manifest.name, version: '1.0.0', lockfileVersion: 3,
    packages: { '': { name: manifest.name, version: '1.0.0', dependencies: manifest.dependencies } },
  })
}

async function noInstallScratch(root: string): Promise<void> {
  assert.deepEqual((await fs.readdir(root)).filter((name) => name.startsWith('.tensorfold-')), [])
}

async function main() {
  for (const command of ['npm ci', 'npm i', 'npm install', 'npm add example@^1.2.3']) {
    assert.equal(commandClass(command), 'package-install')
    assert.ok(commandApprovalReason(command))
    assert.ok(requiresSavedApprovalPolicy(approvalPolicyScope('tool', 'run_command', { command })))
    assert.ok(parsePackageInstallCommand(command).args.includes('--ignore-scripts'))
  }
  for (const command of [
    'npm install --ignore-scripts=false', 'npm install --prefix /elsewhere', 'npm install -g example',
    'npm install && node dangerous.js', 'npm exec create-app', 'pnpm install', 'pip install package',
    'npm install file:../../escape', 'npm install https://example.com/package.tgz',
    'npm install $(command)', 'npm install --dry-run', 'npm install --package-lock-only',
    'npm ci example', 'npm install --registry=https://example.com', 'npm install --omit=dev=other',
  ]) assert.throws(() => parsePackageInstallCommand(command), Error, command)

  const base = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorfold-dependency-test-')))
  try {
    const root = path.join(base, 'monorepo')
    await writeJson(path.join(root, 'package.json'), {
      name: 'dependency-fixture', version: '1.0.0', private: true,
      workspaces: ['packages/*'],
      dependencies: { 'file-fixture': 'file:vendor/local' },
      scripts: { preinstall: 'node -e "require(\'fs\').writeFileSync(\'lifecycle-ran\',\'bad\')"' },
    })
    for (const [directory, name, answer] of [
      ['packages/workspace', 'workspace-fixture', 21], ['vendor/local', 'file-fixture', 42],
    ] as const) {
      await writeJson(path.join(root, directory, 'package.json'), { name, version: '1.0.0', main: 'index.js' })
      await fs.writeFile(path.join(root, directory, 'index.js'), `module.exports = ${answer}\n`)
    }
    await fs.writeFile(path.join(root, '.npmrc'), 'registry=https://invalid.example\n//invalid.example/:_authToken=synthetic\n')
    let prepared = 0
    const checkedNpm: IsolatedPackageRunner = async (input) => {
      prepared++
      assert.equal(path.dirname(input.cwd), path.dirname(root), 'the isolated stage is a sibling of the denied project')
      assert.equal(await fs.stat(input.cwd).then((value) => value.mode & 0o777), 0o700)
      await assert.rejects(fs.access(path.join(input.cwd, '.npmrc')))
      await assert.rejects(fs.access(path.join(input.cwd, 'packages/workspace/index.js')))
      const result = await actualNpm(input)
      await assert.rejects(fs.access(path.join(input.cwd, 'lifecycle-ran')))
      return result
    }
    const installed = await installWorkspacePackages('npm install --offline', root, checkedNpm)
    assert.equal(installed.code, 0, installed.stderr)
    assert.match(installed.stdout, /Dependencies were installed/)
    assert.match(installed.stdout, /Lifecycle scripts were skipped/)
    const requireFixture = createRequire(path.join(root, 'package.json'))
    assert.equal(requireFixture('workspace-fixture'), 21)
    assert.equal(requireFixture('file-fixture'), 42)
    assert.ok(installed.changedPaths?.includes('node_modules'))
    assert.ok(installed.changedPaths?.includes('package-lock.json'))
    await assert.rejects(fs.access(path.join(root, 'lifecycle-ran')))
    await noInstallScratch(root)

    const clean = await installWorkspacePackages('npm ci --offline', root, checkedNpm)
    assert.equal(clean.code, 0, clean.stderr)
    assert.equal(prepared, 2)
    assert.equal(requireFixture.resolve('file-fixture'), path.join(root, 'vendor/local/index.js'))
    await noInstallScratch(root)

    const modulesBefore = await fs.stat(path.join(root, 'node_modules'))
    const edited = await installWorkspacePackages('npm ci --offline', root, async (input) => {
      const result = await checkedNpm(input)
      const manifestPath = path.join(root, 'package.json')
      const manifest = JSON.parse(await fs.readFile(manifestPath, 'utf8'))
      manifest.description = 'A newer user edit that must survive'
      await writeJson(manifestPath, manifest)
      return result
    })
    assert.notEqual(edited.code, 0)
    assert.match(edited.stderr, /Manifest changed/)
    assert.equal(JSON.parse(await fs.readFile(path.join(root, 'package.json'), 'utf8')).description,
      'A newer user edit that must survive')
    assert.equal((await fs.stat(path.join(root, 'node_modules'))).ino, modulesBefore.ino)
    await noInstallScratch(root)

    const lateManifest = await installWorkspacePackages('npm ci --offline', root, checkedNpm, {
      beforePublish: async (relative) => {
        if (relative !== 'node_modules') return
        const manifestPath = path.join(root, 'package.json')
        const manifest = JSON.parse(await fs.readFile(manifestPath, 'utf8'))
        manifest.dependencies['not-in-the-prepared-tree'] = '1.0.0'
        await writeJson(manifestPath, manifest)
      },
    })
    assert.notEqual(lateManifest.code, 0, 'an unchanged manifest is still an input when node_modules is published')
    assert.match(lateManifest.stderr, /Manifest changed/)
    assert.equal((await fs.stat(path.join(root, 'node_modules'))).ino, modulesBefore.ino)
    assert.equal(JSON.parse(await fs.readFile(path.join(root, 'package.json'), 'utf8')).dependencies['not-in-the-prepared-tree'], '1.0.0')
    await noInstallScratch(root)

    const rollbackRoot = path.join(base, 'rollback')
    await writeJson(path.join(rollbackRoot, 'package.json'), { name: 'rollback-fixture', version: '1.0.0' })
    await fs.mkdir(path.join(rollbackRoot, 'node_modules', 'old'), { recursive: true })
    await fs.writeFile(path.join(rollbackRoot, 'node_modules', 'old', 'keep.txt'), 'original dependency')
    const manifestBefore = await fs.readFile(path.join(rollbackRoot, 'package.json'), 'utf8')
    const rolledBack = await installWorkspacePackages('npm install published-fixture', rollbackRoot, async ({ cwd }) => {
      await fakePreparedInstall(cwd)
      return success()
    }, { beforePublish: (relative) => { if (relative === 'node_modules') throw new Error('simulated disk failure') } })
    assert.notEqual(rolledBack.code, 0)
    assert.match(rolledBack.stderr, /Previous dependencies were restored/)
    assert.equal(await fs.readFile(path.join(rollbackRoot, 'package.json'), 'utf8'), manifestBefore)
    assert.equal(await fs.readFile(path.join(rollbackRoot, 'node_modules/old/keep.txt'), 'utf8'), 'original dependency')
    await assert.rejects(fs.access(path.join(rollbackRoot, 'package-lock.json')))
    await noInstallScratch(rollbackRoot)

    const escaped = await installWorkspacePackages('npm install published-fixture', rollbackRoot, async ({ cwd }) => {
      await fakePreparedInstall(cwd)
      await fs.symlink(base, path.join(cwd, 'node_modules', 'escape'))
      return success()
    })
    assert.notEqual(escaped.code, 0)
    assert.match(escaped.stderr, /symlink leaving/)
    assert.equal(await fs.readFile(path.join(rollbackRoot, 'package.json'), 'utf8'), manifestBefore)
    await noInstallScratch(rollbackRoot)

    const finalRaceRoot = path.join(base, 'commit-manifest-race')
    await writeJson(path.join(finalRaceRoot, 'package.json'), { name: 'commit-race', version: '1.0.0' })
    await fs.mkdir(path.join(finalRaceRoot, 'node_modules/old'), { recursive: true })
    await fs.writeFile(path.join(finalRaceRoot, 'node_modules/old/keep.txt'), 'original dependency')
    const rename = fs.rename
    let changedAtPublication = false
    fs.rename = async (source, destination) => {
      await rename(source, destination)
      if (!changedAtPublication && String(source).includes('.tensorfold-dependency-next-') && String(destination) === path.join(finalRaceRoot, 'node_modules')) {
        changedAtPublication = true
        await writeJson(path.join(finalRaceRoot, 'package.json'), { name: 'commit-race', version: '1.0.0', description: 'Typed after the last publication' })
      }
    }
    let commitRace: SandboxedCommandResult
    try {
      commitRace = await installWorkspacePackages('npm install', finalRaceRoot, async ({ cwd }) => {
        await fs.mkdir(path.join(cwd, 'node_modules/prepared'), { recursive: true })
        await fs.writeFile(path.join(cwd, 'node_modules/prepared/index.js'), 'module.exports = 1')
        return success()
      })
    } finally { fs.rename = rename }
    assert.equal(changedAtPublication, true)
    assert.notEqual(commitRace.code, 0, 'a post-publication manifest edit must be caught before success and backup disposal')
    assert.match(commitRace.stderr, /Manifest changed/)
    assert.equal(await fs.readFile(path.join(finalRaceRoot, 'node_modules/old/keep.txt'), 'utf8'), 'original dependency')
    assert.equal(JSON.parse(await fs.readFile(path.join(finalRaceRoot, 'package.json'), 'utf8')).description, 'Typed after the last publication')
    await noInstallScratch(finalRaceRoot)

    const cancelled = new AbortController()
    const aborted = await installWorkspacePackages('npm install published-fixture', rollbackRoot, async ({ cwd }) => {
      await fakePreparedInstall(cwd)
      cancelled.abort()
      return success()
    }, { signal: cancelled.signal })
    assert.equal(aborted.code, 130)
    assert.equal(await fs.readFile(path.join(rollbackRoot, 'package.json'), 'utf8'), manifestBefore)
    await noInstallScratch(rollbackRoot)

    const badRoot = path.join(base, 'unsupported')
    await writeJson(path.join(badRoot, 'package.json'), { name: 'bad-fixture', dependencies: { forbidden: 'file:../../outside' } })
    let ran = false
    const notRun = await installWorkspacePackages('npm install', badRoot, async () => { ran = true; return success() })
    assert.notEqual(notRun.code, 0)
    assert.equal(ran, false)
    assert.match(notRun.stderr, /leaves the selected workspace/)
    await noInstallScratch(badRoot)

    const changedModules = await installWorkspacePackages('npm install published-fixture', rollbackRoot, async ({ cwd }) => {
      await fakePreparedInstall(cwd)
      await fs.writeFile(path.join(rollbackRoot, 'node_modules', 'concurrent.txt'), 'newer work')
      return success()
    })
    assert.notEqual(changedModules.code, 0)
    assert.match(changedModules.stderr, /destination changed/)
    assert.equal(await fs.readFile(path.join(rollbackRoot, 'node_modules/concurrent.txt'), 'utf8'), 'newer work')
    await noInstallScratch(rollbackRoot)

    const metadataTamper = await installWorkspacePackages('npm install published-fixture', rollbackRoot, async ({ cwd }) => {
      await fakePreparedInstall(cwd)
      const manifest = JSON.parse(await fs.readFile(path.join(cwd, 'package.json'), 'utf8'))
      manifest.scripts = { postinstall: 'unreviewed command' }
      await writeJson(path.join(cwd, 'package.json'), manifest)
      return success()
    })
    assert.notEqual(metadataTamper.code, 0)
    assert.match(metadataTamper.stderr, /non-dependency package metadata/)
    assert.equal(await fs.readFile(path.join(rollbackRoot, 'package.json'), 'utf8'), manifestBefore)
    await noInstallScratch(rollbackRoot)

    const collision = await installWorkspacePackages('npm install published-fixture', rollbackRoot, async ({ cwd }) => {
      await fakePreparedInstall(cwd)
      return success()
    }, { beforePublish: async (relative) => {
      if (relative !== 'node_modules') return
      const manifestPath = path.join(rollbackRoot, 'package.json')
      const manifest = JSON.parse(await fs.readFile(manifestPath, 'utf8'))
      manifest.description = 'Concurrent edit after publication'
      await writeJson(manifestPath, manifest)
    } })
    assert.notEqual(collision.code, 0)
    assert.match(collision.stderr, /Manifest changed/, 'published manifests are checked against the installer-owned current state')
    assert.match(collision.stderr, /Recovery copies were retained/)
    assert.equal(JSON.parse(await fs.readFile(path.join(rollbackRoot, 'package.json'), 'utf8')).description,
      'Concurrent edit after publication')
    const backups = (await fs.readdir(rollbackRoot)).filter((name) => name.startsWith('.tensorfold-dependency-backup-'))
    assert.ok(backups.length > 0)
    assert.ok((await Promise.all(backups.map((name) => fs.readFile(path.join(rollbackRoot, name), 'utf8'))))
      .includes(manifestBefore))

    const raceRoot = path.join(base, 'parent-race')
    const outside = path.join(base, 'outside')
    await writeJson(path.join(raceRoot, 'package.json'), {
      name: 'race-fixture', version: '1.0.0', workspaces: ['packages/*'],
    })
    await writeJson(path.join(raceRoot, 'packages/local/package.json'), { name: 'local-fixture', version: '1.0.0' })
    await fs.mkdir(outside)
    await fs.writeFile(path.join(outside, 'keep.txt'), 'outside remains untouched')
    const parentRace = await installWorkspacePackages('npm install published-fixture', raceRoot, async ({ cwd }) => {
      await fakePreparedInstall(cwd)
      await fs.mkdir(path.join(cwd, 'packages/local/node_modules'), { recursive: true })
      await fs.writeFile(path.join(cwd, 'packages/local/node_modules/fixture.txt'), 'new module')
      return success()
    }, { beforePublish: async (relative) => {
      if (relative !== 'packages/local/node_modules') return
      await fs.rename(path.join(raceRoot, 'packages/local'), path.join(raceRoot, 'packages/moved'))
      await fs.symlink(outside, path.join(raceRoot, 'packages/local'))
    } })
    assert.notEqual(parentRace.code, 0)
    assert.match(parentRace.stderr, /directory changed/)
    assert.deepEqual(await fs.readdir(outside), ['keep.txt'])
    assert.equal(await fs.readFile(path.join(outside, 'keep.txt'), 'utf8'), 'outside remains untouched')

    const cleanupRoot = path.join(base, 'cleanup-retry')
    await writeJson(path.join(cleanupRoot, 'package.json'), { name: 'cleanup-fixture', version: '1.0.0' })
    const remove = fs.rm
    let cleanupAttempts = 0
    fs.rm = (async (target: any, options: any) => {
      if (path.dirname(String(target)) === path.dirname(cleanupRoot) && path.basename(String(target)).startsWith('.tensorfold-install-')) {
        cleanupAttempts++
        if (cleanupAttempts <= 2) throw Object.assign(new Error('Synthetic busy directory during cleanup'), { code: 'ENOTEMPTY' })
      }
      return remove(target, options)
    }) as typeof fs.rm
    try {
      const retried = await installWorkspacePackages('npm install', cleanupRoot, async ({ cwd }) => {
        await fakePreparedInstall(cwd); return success()
      })
      assert.equal(retried.code, 0, retried.stderr)
      assert.equal(cleanupAttempts, 3, 'Transient stage cleanup failures are retried after checking ownership')
      assert.doesNotMatch(retried.stderr, /could not be removed/)
      await noInstallScratch(cleanupRoot)
    } finally { fs.rm = remove }

    const replacedRoot = path.join(base, 'cleanup-owner-change')
    await writeJson(path.join(replacedRoot, 'package.json'), { name: 'replacement-fixture', version: '1.0.0' })
    let replacedStage = ''
    fs.rm = (async (target: any, options: any) => {
      if (!replacedStage && path.dirname(String(target)) === path.dirname(replacedRoot) && path.basename(String(target)).startsWith('.tensorfold-install-')) {
        replacedStage = String(target)
        await fs.rename(target, `${target}-previous`)
        await fs.mkdir(target)
        await fs.writeFile(path.join(target, 'new-owner.txt'), 'preserve this concurrent replacement')
        throw Object.assign(new Error('Synthetic replacement before retry'), { code: 'ENOTEMPTY' })
      }
      return remove(target, options)
    }) as typeof fs.rm
    try {
      const retained = await installWorkspacePackages('npm install', replacedRoot, async ({ cwd }) => {
        await fakePreparedInstall(cwd); return success()
      })
      assert.equal(retained.code, 0, retained.stderr)
      assert.match(retained.stderr, /stage identity changed/)
      assert.equal(await fs.readFile(path.join(replacedStage, 'new-owner.txt'), 'utf8'), 'preserve this concurrent replacement')
    } finally { fs.rm = remove }

    const queuedRoot = path.join(base, 'serial-installs')
    await writeJson(path.join(queuedRoot, 'package.json'), { name: 'serial-fixture', version: '1.0.0' })
    let active = 0
    let highestActive = 0
    const serialRunner: IsolatedPackageRunner = async ({ cwd }) => {
      highestActive = Math.max(highestActive, ++active)
      await fakePreparedInstall(cwd)
      await new Promise((resolve) => setTimeout(resolve, 20))
      active--
      return success()
    }
    const queued = await Promise.all([
      installWorkspacePackages('npm install published-fixture', queuedRoot, serialRunner),
      installWorkspacePackages('npm install published-fixture', queuedRoot, serialRunner),
    ])
    assert.ok(queued.every((result) => result.code === 0), queued.map((result) => result.stderr).join('\n'))
    assert.equal(highestActive, 1)
    await noInstallScratch(queuedRoot)

    console.log('Dependency setup: persistent npm packages, workspaces, lifecycle isolation, concurrency checks and rollback pass.')
  } finally { await fs.rm(base, { recursive: true, force: true }) }
}

main().catch((error) => { console.error(error); process.exitCode = 1 })
