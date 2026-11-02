import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { execFile, spawn } from 'node:child_process'
import { promisify } from 'node:util'
import { installWorkspacePackages, type IsolatedPackageRunner } from './package-install'
import { localPackageBins } from './package-install-bins'
import { sandboxCommandEnv } from './sandbox-process'

const execute = promisify(execFile)
const actualNpm: IsolatedPackageRunner = ({ command, cwd }) => new Promise((resolve) => {
  const child = spawn('/bin/sh', ['-c', command], { cwd, env: sandboxCommandEnv(cwd), stdio: ['ignore', 'pipe', 'pipe'] })
  let stdout = '', stderr = ''
  child.stdout.on('data', (value) => { stdout += value })
  child.stderr.on('data', (value) => { stderr += value })
  child.on('error', (error) => resolve({ code: 127, stdout, stderr: error.message, sandboxed: false, profile: 'package-install', implementation: 'unsandboxed' }))
  child.on('close', (code) => resolve({ code: code ?? 1, stdout, stderr, sandboxed: false, profile: 'package-install', implementation: 'unsandboxed' }))
})
async function json(target: string, value: unknown) {
  await fs.mkdir(path.dirname(target), { recursive: true })
  await fs.writeFile(target, JSON.stringify(value, null, 2) + '\n')
}

async function main() {
  const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorfold-local-bin-test-')))
  try {
    await json(path.join(root, 'package.json'), {
      name: 'bin-fixture', version: '1.0.0', private: true, workspaces: ['packages/*'],
      dependencies: { 'file-cli': 'file:vendor/cli' },
      scripts: { preinstall: 'node -e "require(\'fs\').writeFileSync(\'lifecycle-ran\',\'bad\')"' },
    })
    const bins = [
      { directory: 'packages/workspace', name: 'workspace-cli', bin: 'bin/cli.js' },
      { directory: 'vendor/cli', name: 'file-cli', bin: { 'file-cli': 'bin/cli.js' } },
    ]
    for (const item of bins) {
      await json(path.join(root, item.directory, 'package.json'), { name: item.name, version: '1.0.0', bin: item.bin })
      await fs.mkdir(path.join(root, item.directory, 'bin'), { recursive: true })
      await fs.writeFile(path.join(root, item.directory, 'bin/cli.js'), `#!/usr/bin/env node\nprocess.stdout.write('${item.name}: private source ran')\n`, { mode: 0o644 })
      await fs.writeFile(path.join(root, item.directory, 'private-source.js'), 'Workspace source must stay outside the networked installation stage.')
    }
    const checkedNpm: IsolatedPackageRunner = async (input) => {
      for (const item of bins) {
        assert.equal(await fs.readFile(path.join(input.cwd, item.directory, 'bin/cli.js'), 'utf8'), '#!/usr/bin/env node\n', 'the installer sees an inert placeholder, not source bytes')
        await assert.rejects(fs.access(path.join(input.cwd, item.directory, 'private-source.js')))
      }
      return actualNpm(input)
    }
    const installed = await installWorkspacePackages('npm install --offline', root, checkedNpm)
    assert.equal(installed.code, 0, installed.stderr)
    for (const item of bins) {
      const command = path.join(root, 'node_modules/.bin', item.name)
      assert.equal((await fs.lstat(command)).isSymbolicLink(), true, 'npm creates its normal relative executable link')
      const result = await execute(command, { cwd: root, env: sandboxCommandEnv(root) })
      assert.equal(result.stdout, `${item.name}: private source ran`, 'the published executable runs the actual host-owned source')
      assert.equal((await fs.stat(path.join(root, item.directory, 'bin/cli.js'))).mode & 0o777, 0o755)
      assert.ok(installed.changedPaths?.includes(`${item.directory}/bin/cli.js`), 'permission changes are included in the mutation receipt')
    }
    await assert.rejects(fs.access(path.join(root, 'lifecycle-ran')))
    const clean = await installWorkspacePackages('npm ci --offline', root, checkedNpm)
    assert.equal(clean.code, 0, clean.stderr)
    assert.equal((await execute(path.join(root, 'node_modules/.bin/workspace-cli'), { cwd: root, env: sandboxCommandEnv(root) })).stdout, 'workspace-cli: private source ran')

    const local = path.join(root, 'packages/workspace/bin/cli.js')
    await fs.chmod(local, 0o640)
    const modulesBefore = await fs.stat(path.join(root, 'node_modules'))
    const controller = new AbortController()
    const open = fs.open
    fs.open = async (...args) => {
      const handle = await open(...args)
      if (String(args[0]) === local) {
        const chmod = handle.chmod.bind(handle)
        handle.chmod = async (mode) => { await chmod(mode); controller.abort() }
      }
      return handle
    }
    let cancelled
    try { cancelled = await installWorkspacePackages('npm ci --offline', root, checkedNpm, { signal: controller.signal }) }
    finally { fs.open = open }
    assert.equal(cancelled.code, 130, cancelled.stderr)
    assert.equal((await fs.stat(local)).mode & 0o777, 0o640, 'cancellation after permission promotion restores the exact original mode')
    assert.equal((await fs.stat(path.join(root, 'node_modules'))).ino, modulesBefore.ino, 'the old dependency tree is restored with the old executable mode')
    await assert.rejects(fs.access(path.join(root, 'lifecycle-ran')))
    assert.deepEqual((await fs.readdir(root)).filter((name) => name.startsWith('.tensorfold-')), [])

    for (const bin of ['../outside.js', '/outside.js', '.git/hooks/pre-commit', 'node_modules/tool/index.js']) {
      await assert.rejects(localPackageBins(root, 'packages/workspace', { name: 'fixture', bin }), /inside|hidden|node_modules/)
    }
    await assert.rejects(localPackageBins(root, 'packages/workspace', { name: 'fixture', directories: { bin: 'bin' } }), /explicit bin/)
    await fs.symlink(local, path.join(root, 'packages/workspace/bin/link.js'))
    await assert.rejects(localPackageBins(root, 'packages/workspace', { name: 'fixture', bin: 'bin/link.js' }), /owned regular file/)
    await fs.link(local, path.join(root, 'packages/workspace/bin/hardlink.js'))
    await assert.rejects(localPackageBins(root, 'packages/workspace', { name: 'fixture', bin: 'bin/hardlink.js' }), /owned regular file/)
    console.log('Local workspace/file CLI installs execute offline, keep source private, skip lifecycle scripts and roll back permissions safely')
  } finally { await fs.rm(root, { recursive: true, force: true }) }
}

main().catch((error) => { console.error(error); process.exitCode = 1 })
