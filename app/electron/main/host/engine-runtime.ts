import { app } from 'electron'
import os from 'node:os'
import { EngineService } from '../../agent/engine/service'
import { resolveEngineBinaryPath } from '../../agent/engine/paths'
import { ENGINE_PROVIDER_ID } from '../../agent/engine/provider'
import type { EngineEvent } from '../../agent/types'
import { vontraHomeDir } from '../../store'
import { store } from '../state'
import { persistSettings } from './persistence'
import { sendToRenderer } from './renderer-events'
import { loweredEngineMemoryBudget } from '../../agent/engine/memory-budget'

// The one EngineService for this app process: binary from the bundle (or build/engine in dev), engine
// events to the renderer, and the built-in provider's model list kept equal to what `models` reports.

let service: EngineService | null = null

export function engineService(): EngineService | null {
  return service
}

function syncProviderModels(repos: string[]): void {
  const index = store.settings.providers.findIndex((p) => p.id === ENGINE_PROVIDER_ID)
  if (index < 0) return
  const provider = store.settings.providers[index]
  if (JSON.stringify(provider.models) === JSON.stringify(repos)) return
  const providers = store.settings.providers.map((p, i) => (i === index ? { ...p, models: repos } : p))
  const activeModel = store.settings.activeProviderId === ENGINE_PROVIDER_ID && !repos.includes(store.settings.activeModel ?? '')
    ? repos[0] ?? null
    : store.settings.activeModel
  store.settings = { ...store.settings, providers, activeModel }
  persistSettings('engineModels')
}

export function createEngineRuntime(): EngineService {
  if (service) return service
  const binaryPath = resolveEngineBinaryPath({ packaged: app.isPackaged, resourcesPath: process.resourcesPath, appRoot: app.getAppPath(), env: process.env })
  service = new EngineService({
    binaryPath,
    appHome: vontraHomeDir(),
    memoryBudgetBytes: loweredEngineMemoryBudget(os.totalmem(), process.env.TENSORFOLD_STUDIO_MEMORY_BUDGET_BYTES),
    emit: (event: EngineEvent) => { sendToRenderer('engine:event', event) },
    onInstalled: (repos) => { try { syncProviderModels(repos) } catch { /* settings not loaded yet */ } },
  })
  service.install()
  // Last resort if the app exits without before-quit: no engine may outlive it.
  process.once('exit', () => service?.shutdown(true))
  return service
}
