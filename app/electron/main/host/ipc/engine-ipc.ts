import { assertCortheonReleaseAllowed } from '../../../../shared/release-feature-policy'
import { normalizeEngineEnvOverrides } from '../../../../shared/engine-advanced-env'
import { BrowserWindow, dialog } from 'electron'
import type { EngineService } from '../../../agent/engine/service'
import { comparisonId, comparisonInput } from '../../../agent/engine/model-comparison'
import { isRepoId, isRevision } from '../../../agent/engine/paths'
import { auditEvent, requireSecurityRole } from '../audit'
import { handle } from '../ipc'

// Built-in engine channels for the Models page, the composer picker and onboarding. The renderer
// supplies only repo/library IDs, revisions and search text. A native chooser admits model folders.

export function engineRepo(value: unknown): string {
  if (!isRepoId(value)) throw new Error('Expected a Hugging Face repo id like owner/name.')
  return value
}

export function engineRevision(value: unknown): string | null {
  if (value == null || value === '') return null
  if (!isRevision(value)) throw new Error('Expected a branch, tag or commit for the revision.')
  return value
}

export function engineSearchText(value: unknown): string {
  if (typeof value !== 'string') throw new Error('Search text must be a string.')
  const text = value.trim()
  if (text.length > 200) throw new Error('Search text is too long.')
  return text
}

export function engineLibraryFolderId(value: unknown): string {
  if (typeof value !== 'string' || !/^[a-f0-9]{64}$/.test(value)) throw new Error('Choose a saved model folder.')
  return value
}

export function engineLibraryScanOptions(value: unknown): { includeHuggingFace?: boolean } {
  if (value == null) return {}
  if (typeof value !== 'object' || Array.isArray(value) || Object.keys(value).some((key) => key !== 'includeHuggingFace')) throw new Error('Expected local model scan options.')
  const enabled = (value as { includeHuggingFace?: unknown }).includeHuggingFace
  if (enabled !== undefined && typeof enabled !== 'boolean') throw new Error('Hugging Face cache scanning must be on or off.')
  return enabled === undefined ? {} : { includeHuggingFace: enabled }
}

export function registerEngineIpc(engine: () => EngineService | null, options: { reserveOperation?: () => { release(): void }; assertCanEnableCortheon?: () => void; assertCanCompare?: () => void } = {}) {
  const operation = async <T>(run: () => Promise<T>): Promise<T> => {
    const lease = options.reserveOperation?.()
    try { return await run() } finally { lease?.release() }
  }
  const svc = () => {
    const s = engine()
    if (!s) throw new Error('The built-in engine is not running in this process.')
    return s
  }

  handle('engine:snapshot', async () => {
    requireSecurityRole('operator', 'read the built-in engine status')
    return svc().snapshot()
  })

  handle('engine:connection', async () => {
    requireSecurityRole('operator', 'copy the built-in engine connection')
    return svc().connection()
  })

  handle('engine:refresh', async () => {
    requireSecurityRole('operator', 'refresh installed models')
    return operation(async () => { await svc().installed(true); return svc().snapshot() })
  })

  handle('engine:search', async (_e, query: unknown) => {
    requireSecurityRole('operator', 'search Hugging Face')
    return svc().search(engineSearchText(query))
  })

  handle('engine:preflight', async (_e, input: { repo?: unknown; revision?: unknown }) => {
    requireSecurityRole('operator', 'check a model before download')
    return svc().preflight(engineRepo(input?.repo), engineRevision(input?.revision))
  })

  handle('engine:pull', async (_e, input: { repo?: unknown; revision?: unknown }) => {
    requireSecurityRole('operator', 'download a model')
    const repo = engineRepo(input?.repo)
    const revision = engineRevision(input?.revision)
    return operation(async () => {
      const progress = await svc().pull(repo, revision)
      await auditEvent('engine.pull', { repo, revision })
      return progress
    })
  })

  handle('engine:cancelPull', async (_e, repo: unknown) => {
    requireSecurityRole('operator', 'cancel a model download')
    return { cancelled: svc().cancelPull(engineRepo(repo)) }
  })

  handle('engine:load', async (_e, repo: unknown) => {
    requireSecurityRole('operator', 'load a model')
    const model = engineRepo(repo)
    return operation(async () => { await auditEvent('engine.load', { repo: model }); return svc().load(model) })
  })

  handle('engine:compareModels', async (_e, value: unknown) => {
    requireSecurityRole('operator', 'compare two local models')
    const input = comparisonInput(value)
    return operation(async () => {
      const service = svc()
      service.setComparisonAdmission(options.assertCanCompare ?? null)
      options.assertCanCompare?.()
      const result = service.compareModels(input) // Reserve synchronously before audit IO or cancellation can race startup.
      try { await auditEvent('engine.comparison', { models: input.models }) }
      catch (error) { service.cancelComparison(input.id); await result.catch(() => {}); throw error }
      return result
    })
  })

  handle('engine:comparisonStatus', async (_e, id: unknown) => {
    requireSecurityRole('operator', 'read a local model comparison')
    return svc().comparisonStatus(comparisonId(id))
  })

  handle('engine:cancelComparison', async (_e, id: unknown) => {
    requireSecurityRole('operator', 'cancel a local model comparison')
    return svc().cancelComparison(comparisonId(id))
  })

  handle('engine:tryModel', async (_e, repo: unknown) => {
    requireSecurityRole('operator', 'measure a local model preview')
    const model = engineRepo(repo)
    return operation(async () => { await auditEvent('engine.preview', { repo: model }); return svc().tryModel(model) })
  })

  handle('engine:cancelTryModel', async () => {
    requireSecurityRole('operator', 'cancel a local model preview')
    return svc().cancelTryModel()
  })

  handle('engine:setCortheon', async (_e, enabled: unknown) => {
    requireSecurityRole('operator', 'change Cortheon mode')
    if (typeof enabled !== 'boolean') throw new Error('Cortheon mode must be on or off.')
    assertCortheonReleaseAllowed(enabled)
    return operation(async () => { await auditEvent('engine.cortheon', { enabled }); if (enabled) options.assertCanEnableCortheon?.(); return svc().setCortheon(enabled) })
  })

  handle('engine:learning', async () => {
    requireSecurityRole('operator', 'read the Living Weights graph')
    return svc().learningSnapshot()
  })

  handle('engine:setLearning', async (_e, input: { enabled?: unknown; model?: unknown }) => {
    requireSecurityRole('operator', 'change Living Weights mode')
    if (typeof input?.enabled !== 'boolean') throw new Error('Living Weights mode must be on or off.')
    const model = input.model == null ? undefined : engineRepo(input.model)
    return operation(async () => {
      const snapshot = await svc().setLearning(input.enabled as boolean, model)
      await auditEvent('engine.learning', { enabled: input.enabled, model: snapshot.model })
      return snapshot
    })
  })

  handle('engine:envOverrides', async () => {
    requireSecurityRole('operator', 'read the advanced engine variables')
    return svc().envOverrides()
  })

  handle('engine:setEnvOverrides', async (_e, value: unknown) => {
    requireSecurityRole('operator', 'change the advanced engine variables')
    const overrides = normalizeEngineEnvOverrides(value)
    return operation(async () => {
      // Only the names are audited; values may hold paths or engine data.
      await auditEvent('engine.envOverrides', { keys: Object.keys(overrides).sort() })
      return svc().setEnvOverrides(overrides)
    })
  })

  handle('engine:teach', async (_e, text: unknown) => {
    requireSecurityRole('operator', 'teach the private Living Weights model')
    if (typeof text !== 'string' || !text.trim() || Buffer.byteLength(text, 'utf8') > 8192) throw new Error('Teach a short fact or a few sentences, up to 8 KB.')
    return operation(async () => {
      const snapshot = await svc().teach(text)
      await auditEvent('engine.teach', { model: snapshot.model, jobId: snapshot.job?.id })
      return snapshot
    })
  })

  handle('engine:cancelTeach', async () => {
    requireSecurityRole('operator', 'stop a Living Weights lesson')
    return svc().cancelTeach()
  })

  handle('engine:unload', async () => {
    requireSecurityRole('operator', 'unload the model')
    return operation(() => svc().unload())
  })

  handle('engine:delete', async (_e, repo: unknown) => {
    requireSecurityRole('owner', 'delete a downloaded model')
    const model = engineRepo(repo)
    return operation(async () => {
      const installed = await svc().remove(model)
      await auditEvent('engine.delete', { repo: model })
      return installed
    })
  })

  handle('engine:library', async () => {
    requireSecurityRole('operator', 'read the local model library')
    return svc().librarySnapshot()
  })

  handle('engine:scanLibrary', async (_e, input: unknown) => {
    requireSecurityRole('operator', 'scan configured model folders')
    const options = engineLibraryScanOptions(input)
    return operation(() => svc().scanLibrary(options))
  })

  handle('engine:addLibraryFolder', async (event) => {
    requireSecurityRole('operator', 'choose a local model folder')
    return operation(async () => {
      const parent = BrowserWindow.fromWebContents(event.sender)
      const options = { title: 'Choose a folder containing local models', properties: ['openDirectory'] as ('openDirectory')[] }
      const result = parent ? await dialog.showOpenDialog(parent, options) : await dialog.showOpenDialog(options)
      if (result.canceled || result.filePaths.length !== 1) return null
      // This path comes only from the OS picker, never from renderer arguments or audit logs.
      return svc().addLibraryFolder(result.filePaths[0])
    })
  })

  handle('engine:removeLibraryFolder', async (_e, id: unknown) => {
    requireSecurityRole('operator', 'unlink a local model search folder')
    const folder = engineLibraryFolderId(id)
    return operation(() => svc().removeLibraryFolder(folder))
  })

  handle('engine:cancelLibraryScan', async () => {
    requireSecurityRole('operator', 'stop a local model scan')
    return svc().cancelLibraryScan()
  })
}
