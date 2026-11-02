import type { EngineRuntimeManager } from '../../../agent/engine/runtime/manager'
import { runtimeId } from '../../../agent/engine/runtime/catalog'
import { requireSecurityRole } from '../audit'
import { handle } from '../ipc'

export function registerEngineRuntimeIpc(runtimes: () => EngineRuntimeManager,
  deps: { reserve: () => { release(): void }; assertIdle: () => void }) {
  const operation = async <T>(run: () => Promise<T>) => {
    const lease = deps.reserve()
    try { return await run() } finally { lease.release() }
  }
  handle('engine:runtimeStatus', async () => {
    requireSecurityRole('operator', 'read engine runtimes')
    return runtimes().snapshot()
  })
  handle('engine:runtimeCheck', async () => {
    requireSecurityRole('owner', 'check signed engine updates')
    return operation(() => runtimes().check())
  })
  handle('engine:runtimeInstall', async (_e, id: unknown) => {
    requireSecurityRole('owner', 'download a signed engine runtime')
    const selected = runtimeId(id)
    return operation(() => runtimes().install(selected))
  })
  handle('engine:runtimeUse', async (_e, id: unknown) => {
    requireSecurityRole('owner', 'switch engine runtimes')
    const selected = id === null ? null : runtimeId(id)
    return operation(async () => { deps.assertIdle(); return runtimes().use(selected) })
  })
  handle('engine:runtimeCancel', async () => {
    requireSecurityRole('owner', 'cancel an engine runtime download')
    return runtimes().cancel()
  })
}
