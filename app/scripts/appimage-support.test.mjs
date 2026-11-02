import assert from 'node:assert/strict'
import { readFile, mkdtemp, writeFile, chmod, rm, mkdir, copyFile } from 'node:fs/promises'
import path from 'node:path'
import os from 'node:os'
import { execFileSync } from 'node:child_process'
import { createRequire } from 'node:module'

const root = path.resolve(import.meta.dirname, '..'), require = createRequire(import.meta.url)
const config = require('./linux-builder-config.cjs')
assert.equal(config.toolsets?.appimage, '1.0.3', 'Both architectures need the checksum-pinned static AppImage runtime')
assert.deepEqual(config.appImage.executableArgs, [], 'The desktop entry cannot silently turn off Chromium sandboxing')
const launcher = path.join(root, 'build/linux/AppRun')
const installer = path.join(root, 'build/linux/install-sandbox-support.sh')
const source = await readFile(launcher, 'utf8'), installSource = await readFile(installer, 'utf8')
assert.doesNotMatch(source, /(?:exec|export).*--no-sandbox/)
assert.doesNotMatch(installSource, /sysctl|chmod\s+4755|\/\*\*|\/tmp\/\.mount/)
if (process.platform !== 'win32') {
  execFileSync('/bin/bash', ['-n', launcher]); execFileSync('/bin/bash', ['-n', installer])
  const hash = 'a'.repeat(64)
  const policy = execFileSync('/bin/bash', [installer, '--policy', hash], { encoding: 'utf8' })
  assert.ok(policy.includes(`/opt/tensorfold-appimage/${hash}/app/tensorfold-studio`))
  assert.match(policy, /userns,/); assert.doesNotMatch(policy, /[*?]|\/home\//)
  assert.throws(() => execFileSync('/bin/bash', [installer, '--policy', '../escape'], { stdio: 'pipe' }))
  const fixture = await mkdtemp(path.join(os.tmpdir(), 'tf-appimage-launch-'))
  try {
    const bin = path.join(fixture, 'bin'), app = path.join(fixture, 'app')
    await import('node:fs/promises').then(fs => Promise.all([fs.mkdir(bin), fs.mkdir(app)]))
    const image = path.join(fixture, 'TensorFold with spaces.AppImage')
    await writeFile(image, 'fixture image')
    const tool = async (name, text) => { const file = path.join(bin, name); await writeFile(file, '#!/bin/bash\n' + text + '\n'); await chmod(file, 0o755) }
    await tool('realpath', 'printf "%s\\n" "$2"')
    await tool('stat', 'if [[ $2 == %u ]]; then printf "0\\n"; else printf "755\\n"; fi')
    await tool('id', 'printf "1000\\n"')
    await tool('sha256sum', 'printf "'+ 'a'.repeat(64) +'  fixture\\n"')
    await tool('unshare', 'exit "${TF_PROBE_RESULT:-1}"')
    await tool('zenity', 'if [[ $1 == --question ]]; then printf "dialog\\n" >> "$TF_CALLS"; fi; exit 1')
    await tool('pkexec', 'printf "privilege-request\\n" >> "$TF_CALLS"; exit 126')
    await writeFile(path.join(app, 'tensorfold-studio'), '#!/bin/bash\nprintf "%s\\n" "$@" > "$TF_STARTED"\n')
    await chmod(path.join(app, 'tensorfold-studio'), 0o755)
    const env = { PATH: `${bin}:/usr/bin:/bin`, HOME: fixture, APPDIR: app, APPIMAGE: image, DISPLAY: ':test', TF_CALLS: path.join(fixture, 'calls'), TF_STARTED: path.join(fixture, 'started') }
    assert.throws(() => execFileSync('/bin/bash', [launcher], { env, stdio: 'pipe' }), error => { assert.equal(error.status, 78, String(error.stderr)); return true })
    assert.equal(await readFile(env.TF_CALLS, 'utf8'), 'dialog\n', 'Cancelling the first-run prompt cannot request administrator access')
    assert.throws(() => execFileSync('/bin/bash', [launcher, '--no-sandbox'], { env, stdio: 'pipe' }))
    assert.equal(await readFile(env.TF_CALLS, 'utf8'), 'dialog\n')
    execFileSync('/bin/bash', [launcher, '--headless-server', '--ozone-platform=headless'], { env: { ...env, TF_PROBE_RESULT: '0' } })
    assert.equal(await readFile(env.TF_STARTED, 'utf8'), '--headless-server\n--ozone-platform=headless\n')
    assert.equal(await readFile(env.TF_CALLS, 'utf8'), 'dialog\n', 'Supported hosts launch without privilege requests')
    await mkdir(path.join(app, 'resources/appimage-support'), { recursive: true })
    await copyFile(installer, path.join(app, 'resources/appimage-support/install-sandbox-support.sh'))
    await tool('unsquashfs', 'exit 1')
    await rm(env.TF_STARTED)
    assert.throws(() => execFileSync('/bin/bash', [launcher, '--install-sandbox-support'], { env, stdio: 'pipe' }))
    assert.equal(await readFile(env.TF_CALLS, 'utf8'), 'dialog\nprivilege-request\n')
    await assert.rejects(readFile(env.TF_STARTED), /ENOENT/, 'Cancelled administrator authorization cannot start the application')

  } finally { await rm(fixture, { recursive: true, force: true }) }
}
console.log('AppImage launcher preserves sandboxing, cancellation, ordinary arguments and narrowly scoped policy generation.')
