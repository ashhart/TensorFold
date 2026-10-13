import type { EngineService } from '../../../agent/engine/service'
import { isRepoId, isRevision } from '../../../agent/engine/paths'
import { auditEvent, requireSecurityRole } from '../audit'
import { handle } from '../ipc'

// Built-in engine channels for the Models page, the composer picker and onboarding. The renderer
// supplies only repo ids, revisions and search text; every path and URL is derived here.

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

export function registerEngineIpc(engine: () => EngineService | null, options: { reserveOperation?: () => { release(): void } } = {}) {
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

  handle('engine:setCortheon', async (_e, enabled: unknown) => {
    requireSecurityRole('operator', 'change Cortheon mode')
    if (typeof enabled !== 'boolean') throw new Error('Cortheon mode must be on or off.')
    return operation(async () => { await auditEvent('engine.cortheon', { enabled }); return svc().setCortheon(enabled) })
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
}
