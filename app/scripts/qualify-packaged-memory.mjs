#!/usr/bin/env node
// Lowered model budgets on the current machine qualify guard/recommendation behavior, not small-Mac performance.
import assert from 'node:assert/strict'
import { execFile, spawn } from 'node:child_process'
import { createRequire } from 'node:module'
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { promisify } from 'node:util'
import { buildPackagedSmokeEnvironment, createIsolatedSmokeProfile, isPathInside, removeExactSmokeProfile, selectUnusedLoopbackPort } from './smoke-packaged-mac.mjs'

export const MEMORY_QUALIFICATION_PROFILES_GIB = [8, 16, 24, 32, 64]
const GIB = 1024 ** 3
const root = fileURLToPath(new URL('..', import.meta.url))
const run = promisify(execFile)
const require = createRequire(import.meta.url)

export function parseMemoryQualificationArgs(argv) {
  const args = {}
  for (let index = 0; index < argv.length; index += 2) {
    const key = argv[index]
    if (!['--app', '--out', '--playwright-module'].includes(key) || !argv[index + 1] || argv[index + 1].startsWith('--')) throw new Error('Use --app <bundle.app> --out <private evidence directory> and optional --playwright-module <module path>.')
    args[key.slice(2)] = argv[index + 1]
  }
  if (!args.app || !args.out) throw new Error('--app and --out are required; qualification never uses the normal app profile.')
  const app = path.resolve(args.app), out = path.resolve(args.out)
  if (!app.endsWith('.app')) throw new Error('--app must name a packaged macOS .app bundle.')
  if (out === root || isPathInside(root, out)) throw new Error('Keep private qualification evidence outside the publication candidate.')
  return { ...args, app, out }
}

export function qualificationEnvironment(base, profile, debugPort, gib) {
  if (!MEMORY_QUALIFICATION_PROFILES_GIB.includes(gib)) throw new Error('Only the fixed 8/16/24/32/64-GiB qualification budgets are supported.')
  return { ...buildPackagedSmokeEnvironment(base, profile, debugPort), VONTRA_HEADLESS: '0', TENSORFOLD_STUDIO_MEMORY_BUDGET_BYTES: String(gib * GIB) }
}

async function stopOwnedChild(child) {
  if (child.exitCode !== null || child.signalCode !== null) return
  const exited = new Promise((resolve) => child.once('exit', resolve))
  child.kill('SIGTERM')
  const timeout = setTimeout(() => child.kill('SIGKILL'), 5000)
  await exited
  clearTimeout(timeout)
}

export async function qualifyPackagedMemory(options) {
  if (process.platform !== 'darwin') throw new Error('Packaged memory qualification requires macOS.')
  if (os.totalmem() < 64 * GIB) throw new Error('All five lowering-only profiles require a host with at least 64 GiB of physical RAM; use focused tests on smaller machines.')
  const { chromium } = require(options['playwright-module'] || 'playwright')
  const { createJiti } = require('jiti')
  const jiti = createJiti(import.meta.url)
  const { recommendEngineModel } = await jiti.import(path.join(root, 'src/lib/model-recommendation.ts'))
  const executable = path.join(options.app, 'Contents', 'MacOS', 'TensorFold')
  const signature = await run('/usr/bin/codesign', ['--display', '--verbose=4', options.app])
  const { macos } = JSON.parse(await readFile(path.join(root, 'build/release-trust.json'), 'utf8'))
  const productionSigned = signature.stderr.includes(`Authority=${macos.identity}`) && signature.stderr.includes(`TeamIdentifier=${macos.teamId}`) && !signature.stderr.includes('Signature=adhoc')
  await mkdir(options.out, { recursive: true, mode: 0o700 })
  const rows = []
  for (const gib of MEMORY_QUALIFICATION_PROFILES_GIB) {
    const profile = await createIsolatedSmokeProfile()
    const port = await selectUnusedLoopbackPort()
    const child = spawn(executable, [`--remote-debugging-address=127.0.0.1`, `--remote-debugging-port=${port}`], { env: qualificationEnvironment(process.env, profile.root, port, gib), stdio: 'ignore' })
    let browser, page
    try {
      const deadline = Date.now() + 30_000
      while (!browser) {
        try { browser = await chromium.connectOverCDP(`http://127.0.0.1:${port}`, { timeout: 1000 }) } catch {
          if (child.exitCode !== null || Date.now() > deadline) throw new Error('The packaged qualification app did not open its isolated loopback debugging endpoint.')
          await new Promise((resolve) => setTimeout(resolve, 100))
        }
      }
      const context = browser.contexts()[0]
      page = context.pages().find((candidate) => candidate.url().startsWith('file:')) || context.pages()[0] || await context.waitForEvent('page', { timeout: 30_000 })
      await page.waitForSelector('body.ready', { timeout: 30_000 })
      await page.setViewportSize({ width: 1280, height: 860 })
      const snapshot = await page.evaluate(async () => window.tensorfoldStudio.engineSnapshot())
      assert.equal(snapshot.status.totalMemoryBytes, os.totalmem(), 'qualification never fabricates physical hardware RAM')
      assert.equal(snapshot.status.memoryBudgetBytes, gib * GIB, 'the packaged app must contain the lowering-only budget seam')
      assert.equal(snapshot.status.memoryLimitBytes, Math.floor(gib * GIB * 0.7))
      assert.equal(snapshot.status.state, 'stopped')
      const diagnostics = await page.evaluate(async (suggestions) => await Promise.all(suggestions.map(async (suggestion) => {
        try { return { repo: suggestion.repo, result: await window.tensorfoldStudio.enginePreflight(suggestion.repo, suggestion.revision) } } catch (error) { return { repo: suggestion.repo, result: null, error: String(error.message) } }
      })), snapshot.suggestions)
      const checks = Object.fromEntries(diagnostics.map((entry) => [entry.repo, entry.result]))
      const recommendation = recommendEngineModel({ status: snapshot.status, installed: snapshot.installed, suggestions: snapshot.suggestions, preflights: checks })
      await writeFile(path.join(options.out, `compatibility-${gib}gib.json`), `${JSON.stringify({ physicalMemoryGiB: os.totalmem() / GIB, qualificationBudgetGiB: gib, recommendation, checks: diagnostics }, null, 2)}\n`, { mode: 0o600 })
      assert.equal(recommendation.status, 'ready', 'real model compatibility checks must finish before claiming a recommendation')
      assert.ok(recommendation.selected.memoryBytes <= snapshot.status.memoryLimitBytes)
      if (gib <= 24) assert.equal(recommendation.selected.repo, 'mlx-community/Qwen3.5-2B-MLX-4bit', 'small budgets must select the lightweight checkpoint')
      const chooser = page.getByTestId('onboarding-choose-model')
      if (await chooser.count()) await chooser.click()
      else await page.getByRole('button', { name: 'Models', exact: true }).first().click()
      await page.getByTestId('model-recommendation').getByRole('heading', { name: `Recommended: ${recommendation.selected.title}`, exact: true }).waitFor({ timeout: 60_000 })
      await page.screenshot({ path: path.join(options.out, `first-start-${gib}gib.png`), scale: 'css' })
      const model = snapshot.installed.find((candidate) => candidate.repo.includes('Nemotron'))
      assert.ok(model, 'read-only qualification needs the existing cached Nemotron model; it does not download model weights')
      assert.ok(Number.isFinite(model.memoryFloorBytes) && model.memoryFloorBytes > 0, 'cached model memory must be measured before evaluating a load')
      const readyForModel = model.memoryFloorBytes <= snapshot.status.memoryLimitBytes
      const loading = await page.evaluate(async (repo) => {
        try { return { ok: true, status: await window.tensorfoldStudio.engineLoad(repo) } } catch (error) { return { ok: false, message: String(error.message) } }
      }, model.repo)
      assert.equal(loading.ok, readyForModel, 'the real packaged supervisor must enforce each lowered model limit')
      if (!readyForModel) {
        assert.match(loading.message, /memory.*limit|limit.*memory/)
        assert.equal(await page.evaluate(() => window.tensorfoldStudio.engineConnection()), null, 'refused loads never open a working API endpoint')
      }
      const fixture = path.join(profile.root, 'worker-fixture')
      await mkdir(fixture)
      await writeFile(path.join(fixture, 'README.md'), 'Readiness qualification fixture.\n')
      const git = async (args) => run('git', ['-c', 'core.hooksPath=/dev/null', '-c', 'core.fsmonitor=false', '-c', 'commit.gpgsign=false', ...args], { cwd: fixture })
      await git(['init', '--initial-branch=main']); await git(['add', 'README.md'])
      await git(['-c', 'user.name=Example Developer', '-c', 'user.email=developer@example.com', 'commit', '-m', 'Qualification fixture'])
      await page.evaluate(async ({ cwd, repo }) => {
        const bridge = window.tensorfoldStudio, settings = await bridge.getSettings()
        await bridge.saveSettings({ ...settings, activeProviderId: 'tensorfold', activeModel: repo, autoCheckUpdates: false })
        await bridge.createProject({ name: 'Qualification fixture', cwd, workspace: { kind: 'local', path: cwd }, executionTarget: 'local' })
      }, { cwd: fixture, repo: model.repo })
      const readiness = await page.evaluate(() => window.tensorfoldStudio.getModelReadiness({ providerId: 'tensorfold' }))
      assert.equal(readiness.ready, readyForModel, 'point-of-use provider readiness must reject the same memory-refused model')
      await page.reload()
      await page.waitForSelector('body.ready')
      await page.getByRole('button', { name: 'Models', exact: true }).first().click()
      await page.getByRole('tab', { name: 'Workers', exact: true }).click()
      await page.getByTestId('worker-title').fill('Readiness qualification')
      await page.getByTestId('worker-objective').fill('Read README.md and summarize it without editing files or running commands.')
      await page.waitForFunction((expected) => document.querySelector('[data-testid="worker-start"]')?.disabled === !expected, readyForModel, { timeout: 10_000 })
      assert.equal(await page.getByTestId('worker-start').isEnabled(), readyForModel, 'workers cannot launch on a memory-refused model')
      rows.push({ qualificationBudgetGiB: gib, allowedModelGiB: snapshot.status.memoryLimitBytes / GIB, recommendation: recommendation.selected.repo, recommendationAction: recommendation.selected.action, cachedModel: model.repo, cachedMemoryFloorGiB: model.memoryFloorBytes / GIB, cachedLoadAllowed: loading.ok, providerReady: readiness.ready, readinessReason: readiness.reason, workerReady: readyForModel, physicalMemoryGiB: os.totalmem() / GIB })
      console.log(`Simulated ${gib}-GiB budget: recommendation, native load gate and worker readiness passed.`)
    } finally {
      try { await page?.evaluate(() => window.tensorfoldStudio.engineUnload()) } catch { /* shutdown below owns this disposable process */ }
      await browser?.close()
      await stopOwnedChild(child)
      await removeExactSmokeProfile(profile)
    }
  }
  const report = { evidenceKind: 'lowered-budget packaged profile qualification, not physical-small-Mac performance', productionSigned, physicalMemoryGiB: os.totalmem() / GIB, profiles: rows, modelWeightsDownloaded: false, workerTasksExecuted: false }
  await writeFile(path.join(options.out, 'memory-profiles.json'), `${JSON.stringify(report, null, 2)}\n`, { mode: 0o600 })
  return report
}

if (process.argv[1] && pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url) {
  qualifyPackagedMemory(parseMemoryQualificationArgs(process.argv.slice(2))).catch((error) => { console.error(error.message); process.exitCode = 1 })
}
