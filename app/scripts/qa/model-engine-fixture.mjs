// Only the QA browser page receives these controllable engine replies; production APIs remain unchanged.
export async function installModelEngineFixture(page) {
  await page.evaluate(async () => {
    const loadedModule = path => performance.getEntriesByType('resource').map(item => item.name)
      .findLast(name => new URL(name).pathname === path) || path
    const { api } = await import(loadedModule('/src/lib/ipc.ts'))
    const { refreshEngine } = await import(loadedModule('/src/lib/engine.ts'))
    const snapshot = structuredClone(await api.engineSnapshot())
    const model = snapshot.installed[0]
    if (!model) throw new Error('The built-in renderer fixture must contain one installed model.')
    const suggestion = snapshot.suggestions.find(item => item.repo === model.repo)
    snapshot.suggestions = suggestion ? [suggestion] : []
    snapshot.installed = []
    snapshot.pulls = []
    snapshot.performance = null
    snapshot.status = { ...snapshot.status, state: 'stopped', model: null, port: null, message: null, contextTokens: null }
    const log = []
    let load = null
    let readError = false
    const publish = () => refreshEngine(true)
    api.engineSnapshot = async () => structuredClone(snapshot)
    api.engineRefresh = async () => {
      if (readError) throw new Error('Fixture: engine status could not be refreshed. Try Refresh models.')
      return structuredClone(snapshot)
    }
    api.enginePull = async (repo, revision) => {
      log.push({ action: 'download', repo, revision })
      const previous = snapshot.pulls.find(item => item.repo === repo)
      const pull = {
        repo, revision, state: 'running', phase: `Downloading ${repo}`, receivedBytes: previous?.receivedBytes ?? 512 * 1024 ** 2,
        totalBytes: 2 * 1024 ** 3, lastLine: null, message: null, startedAt: Date.now(), finishedAt: null,
        updatedAt: Date.now(), bytesPerSecond: 4 * 1024 ** 2, etaSeconds: 384,
      }
      snapshot.pulls = [pull]
      await publish()
      return structuredClone(pull)
    }
    api.engineCancelPull = async repo => {
      log.push({ action: 'cancel-download', repo })
      const pull = snapshot.pulls.find(item => item.repo === repo)
      if (!pull) return { cancelled: false }
      Object.assign(pull, { state: 'cancelled', phase: 'Cancelled', finishedAt: Date.now(), message: null })
      await publish()
      return { cancelled: true }
    }
    api.engineLoad = async repo => {
      log.push({ action: 'load', repo })
      snapshot.status = { ...snapshot.status, state: 'starting', model: repo, startedAt: Date.now(), readyAt: null,
        port: null, contextTokens: null, message: `Loading ${repo}` }
      const pending = new Promise((resolve, reject) => { load = { resolve, reject } })
      await publish()
      return pending
    }
    api.engineUnload = async () => {
      log.push({ action: 'unload' })
      snapshot.status = { ...snapshot.status, state: 'stopped', model: null, startedAt: null, readyAt: null,
        port: null, contextTokens: null, message: null }
      load?.reject(new Error('Fixture: model load cancelled.'))
      load = null
      await publish()
      return structuredClone(snapshot.status)
    }
    globalThis.__modelQa = {
      log, repo: model.repo, snapshot: () => structuredClone(snapshot),
      async failDownload() {
        Object.assign(snapshot.pulls[0], { state: 'failed', phase: 'Failed', finishedAt: Date.now(),
          message: 'Fixture: connection interrupted. Resume to keep the files already downloaded.' })
        await publish()
      },
      async completeDownload() {
        const pull = snapshot.pulls[0]
        Object.assign(pull, { state: 'done', phase: 'Downloaded', receivedBytes: pull.totalBytes,
          finishedAt: Date.now(), message: null })
        snapshot.installed = [model]
        await publish()
      },
      async failLoad() {
        snapshot.status = { ...snapshot.status, state: 'failed', port: null,
          message: 'Fixture: the model could not start. Retry loading or choose another model.' }
        await publish()
        load?.reject(new Error(snapshot.status.message))
        load = null
      },
      async completeLoad() {
        snapshot.status = { ...snapshot.status, state: 'ready', port: 52814, readyAt: Date.now(),
          contextTokens: 32768, message: null }
        await publish()
        load?.resolve(structuredClone(snapshot.status))
        load = null
      },
      async setReadError(value) { readError = value; await publish() },
    }
    await publish()
  })
}
