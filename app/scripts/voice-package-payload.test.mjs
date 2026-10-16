import assert from 'node:assert/strict'
import { cp, mkdir, mkdtemp, readFile, realpath, rm, writeFile } from 'node:fs/promises'
import { createRequire } from 'node:module'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const root = fileURLToPath(new URL('..', import.meta.url))
const require = createRequire(import.meta.url)
const { FileMatcher, getNodeModuleFileMatcher } = require('app-builder-lib/out/fileMatcher.js')
const { NodeModuleCopyHelper } = require('app-builder-lib/out/util/NodeModuleCopyHelper.js')
const { createPackage, listPackage, extractFile } = await import('@electron/asar')
const config = JSON.parse(await readFile(path.join(root, 'package.json'), 'utf8')).build
const temp = await mkdtemp(path.join(os.tmpdir(), 'tensorfold-voice-payload-'))

try {
  const source = path.join(temp, 'source')
  const destination = path.join(temp, 'packaged')
  const moduleRoot = path.join(source, 'node_modules', 'kokoro-js')
  await mkdir(path.dirname(moduleRoot), { recursive: true })
  await cp(path.join(root, 'node_modules', 'kokoro-js'), moduleRoot, { recursive: true })
  // Cover a future browser source map as well as the real installed browser payload.
  await writeFile(path.join(moduleRoot, 'dist', 'kokoro.web.js.map'), '{"version":3,"sources":[],"sourcesContent":["unused browser fixture"]}')
  const packager = { config, debugLogger: { isEnabled: false }, appInfo: { type: 'commonjs' }, getWorkspaceRoot: async () => source }
  const moduleMatcher = getNodeModuleFileMatcher(source, destination, (value) => value, {}, packager)
  const matcher = new FileMatcher(moduleRoot, path.join(destination, 'node_modules', 'kokoro-js'), (value) => value, moduleMatcher.patterns)
  const helper = new NodeModuleCopyHelper(matcher, packager)
  const files = await helper.collectNodeModules({ name: 'kokoro-js', dir: moduleRoot }, [], 'node_modules/kokoro-js')
  const relative = files.map((file) => path.relative(moduleRoot, file).split(path.sep).join('/'))
  assert.ok(!relative.includes('dist/kokoro.web.js'), 'the installed production dependency collector must omit the unused self-contained browser payload')
  assert.ok(!relative.includes('dist/kokoro.web.js.map'), 'a browser source map must not reintroduce excluded browser code')
  for (const kept of ['package.json', 'LICENSE', 'dist/kokoro.cjs', 'dist/kokoro.js']) {
    assert.ok(relative.includes(kept), `the Node worker still ships ${kept}`)
  }
  for (const file of files) {
    const output = path.join(destination, 'node_modules', 'kokoro-js', path.relative(moduleRoot, file))
    await mkdir(path.dirname(output), { recursive: true })
    await cp(file, output)
  }
  for (const fixture of [
    { name: 'pino', location: 'node_modules/pino', unused: 'pretty-demo.png' },
    { name: 'pino', location: 'node_modules/imapflow/node_modules/pino', unused: 'pretty-demo.png' },
    { name: 'thread-stream', location: 'node_modules/thread-stream', unused: '.claude/settings.local.json' },
    { name: 'thread-stream', location: 'node_modules/imapflow/node_modules/thread-stream', unused: '.claude/settings.local.json' },
  ]) {
    const moduleSource = path.join(source, fixture.location)
    await mkdir(path.dirname(path.join(moduleSource, fixture.unused)), { recursive: true })
    await writeFile(path.join(moduleSource, fixture.unused), 'synthetic development fixture')
    await writeFile(path.join(moduleSource, 'package.json'), JSON.stringify({ name: fixture.name, version: '0.0.0-fixture', main: 'index.js' }))
    await writeFile(path.join(moduleSource, 'index.js'), 'module.exports = "synthetic runtime entry"\n')
    const moduleDestination = path.join(destination, fixture.location)
    const fixtureMatcher = new FileMatcher(moduleSource, moduleDestination, (value) => value, moduleMatcher.patterns)
    const collector = new NodeModuleCopyHelper(fixtureMatcher, packager)
    const collected = await collector.collectNodeModules({ name: fixture.name, dir: moduleSource }, [], fixture.location)
    assert.ok(!collected.includes(path.join(moduleSource, fixture.unused)), `${fixture.name} development-only payload must not ship`)
    for (const kept of ['package.json', 'index.js']) assert.ok(collected.includes(path.join(moduleSource, kept)), `${fixture.name} retains its runtime ${kept}`)
    for (const file of collected) {
      const output = path.join(moduleDestination, path.relative(moduleSource, file))
      await mkdir(path.dirname(output), { recursive: true })
      await cp(file, output)
    }
    const fixtureRequire = createRequire(path.join(moduleDestination, 'package.json'))
    assert.equal(fixtureRequire.resolve('.'), await realpath(path.join(moduleDestination, 'index.js')), `${fixture.name} still resolves its unchanged runtime entry`)
  }
  const scopedRequire = createRequire(path.join(destination, 'package.json'))
  assert.equal(scopedRequire.resolve('kokoro-js'), await realpath(path.join(destination, 'node_modules', 'kokoro-js', 'dist', 'kokoro.cjs')), 'the worker resolves the retained Node require export without executing package code')
  const archive = path.join(temp, 'app.asar')
  await createPackage(destination, archive)
  const members = listPackage(archive)
  assert.ok(!members.some((member) => /\/kokoro\.web\.js(?:\.map)?$/.test(member)), 'the actual throwaway ASAR contains no unused browser entry or source map')
  assert.ok(!members.some((member) => /\/pino\/pretty-demo\.png$|\/\.claude(?:\/|$)/.test(member)), 'the actual ASAR omits the unused upstream demo image and development settings')
  const cjs = extractFile(archive, 'node_modules/kokoro-js/dist/kokoro.cjs')
  assert.deepEqual(cjs, await readFile(path.join(root, 'node_modules', 'kokoro-js', 'dist', 'kokoro.cjs')), 'packaging preserves the official Node worker bytes')
  assert.match(await readFile(path.join(root, 'electron', 'voice', 'kokoro-worker.ts'), 'utf8'), /requireLocal\('kokoro-js'\)/, 'the desktop worker uses the retained Node entry')
} finally {
  await rm(temp, { recursive: true, force: true })
}

console.log('voice package payload tests passed')
