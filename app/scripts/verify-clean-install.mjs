#!/usr/bin/env node
import { execFile } from 'node:child_process'
import { access, copyFile, cp, mkdir, mkdtemp, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { promisify } from 'node:util'
import { assertRepoPackageManager } from './check-package-manager.mjs'

const run = promisify(execFile)
const root = process.cwd()
const { version } = await assertRepoPackageManager(root)
const tmp = await mkdtemp(path.join(os.tmpdir(), 'vontra-clean-install-'))

try {
  for (const file of ['package.json', 'package-lock.json', '.npmrc']) {
    await copyFile(path.join(root, file), path.join(tmp, file))
  }
  // The ONNX logger uses our reviewed precision-bounded sprintf implementation.
  // Keep the local dependency present in a clean install, rather than testing a
  // machine-specific node_modules link that disappears in a fresh checkout.
  await mkdir(path.join(tmp, 'mobile', 'vendor'), { recursive: true })
  await cp(path.join(root, 'mobile', 'vendor', 'sprintf-js'), path.join(tmp, 'mobile', 'vendor', 'sprintf-js'), { recursive: true })
  await cp(path.join(root, 'mobile', 'vendor', 'tensorfold-kokoro-phonemizer'), path.join(tmp, 'mobile', 'vendor', 'tensorfold-kokoro-phonemizer'), { recursive: true })
  // Lifecycle scripts are ignored, but keeping the directory mirrors the project shape and
  // makes future config-only lifecycle validation deterministic.
  await mkdir(path.join(tmp, 'scripts'), { recursive: true })
  const npmCli = process.env.npm_execpath
  const command = npmCli ? process.execPath : 'npm'
  const npmArgs = (...args) => npmCli ? [npmCli, ...args] : args
  const args = npmArgs('ci', '--ignore-scripts', '--no-audit', '--no-fund')
  const versionArgs = npmCli ? [npmCli, '--version'] : ['--version']
  const actualVersion = (await run(command, versionArgs, { cwd: tmp })).stdout.trim()
  if (actualVersion !== version) throw new Error(`Clean-install verification requires npm ${version}; found npm ${actualVersion}.`)
  const { stdout, stderr } = await run(command, args, { cwd: tmp, maxBuffer: 10 * 1024 * 1024 })
  if (/ERR!|npm error/i.test(`${stdout}\n${stderr}`)) throw new Error(`${stdout}\n${stderr}`.trim())
  await access(path.join(tmp, 'node_modules', '.package-lock.json'))
  // Electron 42.11.x intentionally installs lazily on require; perform its reviewed
  // binary download in the explicit trusted stage instead of the later import check.
  await run(process.execPath, [path.join(tmp, 'node_modules', 'electron', 'install.js')], { cwd: tmp, maxBuffer: 10 * 1024 * 1024 })
  const rebuilt = await run(command, npmArgs('rebuild', '--ignore-scripts=false', '@lydell/node-pty', '--no-audit', '--no-fund'), { cwd: tmp, maxBuffer: 10 * 1024 * 1024 })
  if (/ERR!|npm error/i.test(`${rebuilt.stdout}\n${rebuilt.stderr}`)) throw new Error(`${rebuilt.stdout}\n${rebuilt.stderr}`.trim())
  await run(process.execPath, ['-e', "const electron=require('electron'); const pty=require('@lydell/node-pty'); const speech=require('kokoro-js'); require('roarr'); if(typeof electron!=='string'||typeof pty.spawn!=='function'||typeof speech.KokoroTTS!=='function') process.exit(1)"], { cwd: tmp })
  console.log(`clean npm ci and package-scoped trusted rebuild passed in an isolated directory (npm ${version}); current node_modules was untouched`)
} finally {
  await rm(tmp, { recursive: true, force: true })
}
