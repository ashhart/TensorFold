import { spawn, execFile } from 'node:child_process'
import { mkdtemp, realpath, rm, stat } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import net from 'node:net'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { promisify } from 'node:util'
import { assertSafeLinuxSmokeTemp, linuxSmokeEnvironment, probePackagedService } from './smoke-packaged-linux.mjs'

const exec = promisify(execFile)

export function windowsSmokeEnvironment(base, profile, port) {
  const env = linuxSmokeEnvironment(base, profile, port)
  for (const key of ['SystemRoot', 'WINDIR', 'COMSPEC', 'PATHEXT', 'TEMP', 'TMP', 'USERPROFILE', 'APPDATA', 'LOCALAPPDATA']) {
    if (typeof base[key] === 'string') env[key] = base[key]
  }
  return env
}

export function resolvePackagedWindowsExecutable(root = fileURLToPath(new URL('..', import.meta.url)), platform = process.platform) {
  if (platform !== 'win32') throw new Error('The packaged Windows smoke requires Windows.')
  return path.join(root, 'release/windows/win-unpacked/TensorFold.exe')
}

async function freePort() {
  const server = net.createServer()
  await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve) })
  const port = server.address().port
  await new Promise(resolve => server.close(resolve))
  return port
}

export async function runPackagedWindowsSmoke() {
  const executable = resolvePackagedWindowsExecutable()
  if (!(await stat(executable)).isFile()) throw new Error('Build the Windows application before running its smoke check.')
  const parent = assertSafeLinuxSmokeTemp(os.tmpdir(), [os.homedir(), process.env.USERPROFILE].filter(Boolean))
  const profile = await realpath(await mkdtemp(path.join(parent, 'tensorfold-windows-smoke-')))
  const port = await freePort()
  let child
  let spawnFailure
  try {
    child = spawn(executable, ['--headless-server'], {
      env: windowsSmokeEnvironment(process.env, profile, port), cwd: path.dirname(executable), stdio: 'ignore',
    })
    child.once('error', error => { spawnFailure = error })
    const deadline = Date.now() + 45_000
    let failure
    while (Date.now() < deadline) {
      if (spawnFailure) throw spawnFailure
      if (child.exitCode !== null) throw new Error(`Windows application exited before readiness (${child.exitCode}).`)
      try {
        const checks = await probePackagedService(port)
        return checks
      } catch (error) { failure = error }
      await new Promise(resolve => setTimeout(resolve, 250))
    }
    throw new Error(`Windows packaged smoke timed out: ${failure?.message}`)
  } finally {
    if (child?.pid && child.exitCode === null) {
      await exec('taskkill.exe', ['/PID', String(child.pid), '/T', '/F'], { timeout: 8_000 }).catch(() => {})
      const deadline = Date.now() + 8_000
      while (child.exitCode === null && Date.now() < deadline) await new Promise(resolve => setTimeout(resolve, 100))
      if (child.exitCode === null) throw new Error('Windows smoke process did not stop; its isolated profile was preserved.')
    }
    await rm(profile, { recursive: true, force: true, maxRetries: 8, retryDelay: 250 })
  }
}

if (process.argv[1] && pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url) {
  runPackagedWindowsSmoke().then(checks => console.log('Packaged Windows smoke passed:', checks), error => {
    console.error(error.message); process.exitCode = 1
  })
}
