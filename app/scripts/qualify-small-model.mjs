#!/usr/bin/env node
// Actual download/chat/document worker on this host with a lowered 8-GiB model budget.
import assert from 'node:assert/strict'
import { execFile, spawn } from 'node:child_process'
import { createRequire } from 'node:module'
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { pathToFileURL } from 'node:url'
import { promisify } from 'node:util'
import { createIsolatedSmokeProfile, removeExactSmokeProfile, selectUnusedLoopbackPort } from './smoke-packaged-mac.mjs'
import { parseMemoryQualificationArgs, qualificationEnvironment } from './qualify-packaged-memory.mjs'

const run = promisify(execFile)
const require = createRequire(import.meta.url)
const MODEL = 'mlx-community/Qwen3.5-2B-MLX-4bit'

export async function qualifySmallModel(options) {
  const { chromium } = require(options['playwright-module'] || 'playwright')
  const profile = await createIsolatedSmokeProfile()
  const port = await selectUnusedLoopbackPort()
  const child = spawn(path.join(options.app, 'Contents', 'MacOS', 'TensorFold'), [`--remote-debugging-address=127.0.0.1`, `--remote-debugging-port=${port}`], { env: qualificationEnvironment(process.env, profile.root, port, 8), stdio: 'ignore' })
  let browser, page
  try {
    const deadline = Date.now() + 30_000
    while (!browser) {
      try { browser = await chromium.connectOverCDP(`http://127.0.0.1:${port}`, { timeout: 1000 }) } catch {
        if (Date.now() > deadline) throw new Error('The isolated small-model app did not start.')
        await new Promise((resolve) => setTimeout(resolve, 100))
      }
    }
    const context = browser.contexts()[0]
    page = context.pages()[0] || await context.waitForEvent('page')
    await page.waitForSelector('body.ready')
    const initial = await page.evaluate(() => window.tensorfoldStudio.engineSnapshot())
    assert.equal(initial.status.memoryBudgetBytes, 8 * 1024 ** 3)
    const suggestion = initial.suggestions.find((entry) => entry.repo === MODEL)
    assert.ok(suggestion?.revision, 'qualification uses the shipped pinned lightweight checkpoint')
    const preflight = await page.evaluate(({ repo, revision }) => window.tensorfoldStudio.enginePreflight(repo, revision), suggestion)
    assert.equal(preflight.formatOk, true)
    assert.equal(preflight.fitsMemory, true)
    const downloadRequested = !initial.installed.some((entry) => entry.repo === MODEL)
    if (downloadRequested) {
      await page.evaluate(({ repo, revision }) => window.tensorfoldStudio.enginePull(repo, revision), suggestion)
      let done = false, lastBytes = -1
      for (let attempt = 0; attempt < 900; attempt++) {
        const pull = await page.evaluate(async (repo) => (await window.tensorfoldStudio.engineSnapshot()).pulls.find((entry) => entry.repo === repo), MODEL)
        if (pull?.state === 'failed' || pull?.state === 'cancelled') throw new Error('The real small-model pull did not finish successfully.')
        if (pull && (attempt % 10 === 0 || pull.state === 'done') && pull.receivedBytes !== lastBytes) { console.log(`Small-model pull: ${pull.receivedBytes} of ${pull.totalBytes ?? 'unknown'} model bytes, ${pull.state}.`); lastBytes = pull.receivedBytes }
        if (pull?.state === 'done') { done = true; break }
        await page.waitForTimeout(1000)
      }
      assert.ok(done, 'real model pull must finish within its qualification timeout')
    }
    const status = await page.evaluate((repo) => window.tensorfoldStudio.engineLoad(repo), MODEL)
    assert.equal(status.state, 'ready')
    console.log('Pinned Qwen3.5-2B downloaded and loaded through the packaged IPC.')
    const chatId = await page.evaluate(async (repo) => {
      const bridge = window.tensorfoldStudio, settings = await bridge.getSettings()
      await bridge.saveSettings({ ...settings, activeProviderId: 'tensorfold', activeModel: repo, autoCheckUpdates: false, reasoningEffort: 'low', temperature: 0, approvalMode: 'auto' })
      const thread = await bridge.createThread('', 'chat')
      await bridge.configureThread(thread.id, 'tensorfold', repo, null)
      await bridge.startRun({ threadId: thread.id, userText: 'Reply with exactly the word ready. Do not explain or use tools.', approvalMode: 'readonly' })
      return thread.id
    }, MODEL)
    let chat
    for (let attempt = 0; attempt < 120; attempt++) {
      chat = await page.evaluate(async (id) => {
        const bridge = window.tensorfoldStudio, thread = (await bridge.getThreads()).find((entry) => entry.id === id)
        return { status: thread?.status, text: thread?.timeline.filter((item) => item.kind === 'assistant').map((item) => item.text).join('\n'), performance: (await bridge.engineSnapshot()).performance }
      }, chatId)
      if (chat?.status === 'error') throw new Error('The real small-model chat failed.')
      if (chat?.status !== 'running' && chat?.text && chat.performance?.model === MODEL) break
      await page.waitForTimeout(500)
    }
    assert.match(chat?.text ?? '', /\bready\b/i, 'the real lightweight model must answer the qualification prompt')
    assert.ok(chat.performance?.completionTokens > 0)
    await mkdir(options.out, { recursive: true, mode: 0o700 })
    await writeFile(path.join(options.out, 'small-model-chat.json'), `${JSON.stringify({ evidenceKind: 'real chat under lowered budget on the current physical host', physicalMemoryGiB: os.totalmem() / 1024 ** 3, qualificationBudgetGiB: 8, model: MODEL, downloadRequested, modelPayloadBytes: preflight.neededDiskBytes, ...chat }, null, 2)}\n`, { mode: 0o600 })
    console.log('Real small-model chat and engine-reported usage passed.')
    const fixture = path.join(profile.root, 'worker-fixture')
    await mkdir(fixture)
    await writeFile(path.join(fixture, 'README.md'), 'Before worker edit.\n')
    const git = (args) => run('git', ['-c', 'core.hooksPath=/dev/null', '-c', 'core.fsmonitor=false', '-c', 'commit.gpgsign=false', ...args], { cwd: fixture })
    await git(['init', '--initial-branch=main']); await git(['add', 'README.md'])
    await git(['-c', 'user.name=Example Developer', '-c', 'user.email=developer@example.com', 'commit', '-m', 'Qualification fixture'])
    await page.evaluate((cwd) => window.tensorfoldStudio.createProject({ name: 'Small-model fixture', cwd, workspace: { kind: 'local', path: cwd }, executionTarget: 'local' }), fixture)
    await page.reload(); await page.waitForSelector('body.ready')
    await page.getByRole('button', { name: 'Models', exact: true }).first().click()
    await page.getByRole('tab', { name: 'Workers', exact: true }).click()
    await page.getByTestId('worker-title').fill('Small-model document edit')
    await page.getByTestId('worker-objective').fill('Replace README.md by calling write_file with exactly these JSON arguments:\n\n```json\n{"path":"README.md","content":"Worker edit verified.\\n"}\n```\n\nThe content must include the final newline shown by \\n. Change no other files. Use read_file to verify README.md, then reply Done.')
    await page.waitForFunction(() => document.querySelector('[data-testid="worker-start"]')?.disabled === false)
    await page.getByTestId('worker-start').click()
    let worker, edited = false
    for (let attempt = 0; attempt < 180; attempt++) {
      worker = await page.evaluate(async () => (await window.tensorfoldStudio.getThreads()).find((entry) => entry.kind === 'task' && entry.title === 'Small-model document edit'))
      if (worker?.status === 'error') throw new Error('The small-model document worker reported an error.')
      if (worker?.cwd && worker.cwd !== fixture) {
        try { edited = (await readFile(path.join(worker.cwd, 'README.md'), 'utf8')) === 'Worker edit verified.\n' } catch { /* worker setup is still creating its worktree */ }
      }
      if (edited && worker.status !== 'running') break
      if (worker?.lastRunReceipt && worker.status !== 'running') break
      if (attempt % 20 === 0) console.log(`Small-model worker: ${worker?.status ?? 'setting up'}, document edit ${edited ? 'present' : 'pending'}.`)
      await page.waitForTimeout(500)
    }
    await writeFile(path.join(options.out, 'small-model-worker-trace.json'), `${JSON.stringify({ edited, worker }, null, 2)}\n`, { mode: 0o600 })
    assert.ok(edited, 'the actual model tool call must write the requested document')
    assert.notEqual(worker.cwd, fixture)
    assert.equal(await readFile(path.join(fixture, 'README.md'), 'utf8'), 'Before worker edit.\n', 'the source repository must stay unchanged')
    await page.evaluate((id) => window.tensorfoldStudio.cancelRun(id), worker.id)
    await mkdir(options.out, { recursive: true, mode: 0o700 })
    await page.screenshot({ path: path.join(options.out, 'small-model-worker.png'), scale: 'css' })
    const final = await page.evaluate((repo) => window.tensorfoldStudio.enginePreflight(repo), MODEL)
    const diff = await git(['-C', worker.cwd, 'diff', '--', 'README.md'])
    assert.match(diff.stdout, /^\+Worker edit verified\.$/m)
    const report = { evidenceKind: 'real packaged download/chat/document worker on a lowered 8-GiB budget, not physical-small-Mac performance', physicalMemoryGiB: os.totalmem() / 1024 ** 3, qualificationBudgetGiB: 8, model: MODEL, revision: suggestion.revision, downloadRequested, modelPayloadBytes: preflight.neededDiskBytes, modelCacheBytesAdded: final.cachedBytes - preflight.cachedBytes, modelCacheRetained: true, chatAnswered: true, performance: chat.performance, workerDocumentEdited: true, workerContentByteExact: true, workerIsolated: true, sourceRepositoryPreserved: true, priorTrials: 'A colon-delimited payload included following instructions; a plain fenced payload wrote the intended text but omitted its final newline and ran a harmless cat check. This final trial uses explicit JSON arguments and verifies actual bytes and git diff.' }
    await writeFile(path.join(options.out, 'small-model.json'), `${JSON.stringify(report, null, 2)}\n`, { mode: 0o600 })
    return report
  } finally {
    try { await page?.evaluate(async () => { const bridge = window.tensorfoldStudio; for (const thread of await bridge.getThreads()) if (thread.status === 'running') await bridge.cancelRun(thread.id); await bridge.engineUnload() }) } catch { /* private child shutdown below */ }
    await browser?.close()
    if (child.exitCode === null && child.signalCode === null) {
      const exited = new Promise((resolve) => child.once('exit', resolve))
      child.kill('SIGTERM')
      const timeout = setTimeout(() => child.kill('SIGKILL'), 5000)
      await exited; clearTimeout(timeout)
    }
    await removeExactSmokeProfile(profile)
  }
}

if (process.argv[1] && pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url) {
  qualifySmallModel(parseMemoryQualificationArgs(process.argv.slice(2))).then((report) => console.log(JSON.stringify(report))).catch((error) => { console.error(error.message); process.exitCode = 1 })
}
