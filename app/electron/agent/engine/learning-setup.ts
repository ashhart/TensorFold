import { promises as fs } from 'node:fs'
import path from 'node:path'
import type { EngineEvent, EngineLearningSnapshot } from '../types'
import type { InstalledModels } from './installed-models'
import type { EngineLearning } from './learning'
import { LEARNING_MODEL, type LearningCopies } from './learning-copy'
import type { ModelRevisions } from './model-revisions'
import { assertRepoId, repoCacheDir } from './paths'
import type { EnginePreferenceStore } from './preferences'
import type { EngineSupervisor } from './supervisor'

// Turning Living Weights on or off: the private model copy is prepared, and the server
// restarts with the new setting, all under the supervisor's reconfigure transaction.

export interface LearningSetupDeps {
  learning: EngineLearning
  supervisor: EngineSupervisor
  preferences: EnginePreferenceStore
  learningCopies: LearningCopies
  revisions: ModelRevisions
  installed: InstalledModels
  hub: () => string
  supported: () => boolean
  emit: (event: EngineEvent) => void
  emitStatus: () => void
}

export async function configureLearning(deps: LearningSetupDeps, enabled: boolean, model?: string): Promise<EngineLearningSnapshot> {
  const { learning, supervisor, preferences, learningCopies, revisions } = deps
  if (enabled && !deps.supported()) throw new Error('Living Weights is unavailable in this build or on this hardware.')
  if (learning.busy) throw new Error('Wait for the current lesson or model preparation to finish.')
  const prior = preferences.get()
  const repo = enabled ? assertRepoId(model ?? supervisor.status().model ?? prior.learningModel ?? '') : prior.learningModel
  if (enabled && repo !== LEARNING_MODEL) throw new Error('Living Weights currently supports only Nemotron 3.5 Lightning MLX 4-bit.')
  if (prior.learningEnabled === enabled && prior.learningModel === repo) return learning.snapshot()
  const running = supervisor.status()
  if (enabled && running.model && running.model !== repo && ['ready', 'starting'].includes(running.state)) throw new Error('Load the Nemotron model before enabling Living Weights; the current model will not be replaced.')
  try {
    await supervisor.reconfigure(async () => {
      if (enabled) {
        const current = supervisor.status()
        if (current.model && current.model !== repo && ['ready', 'starting'].includes(current.state)) throw new Error('Load the Nemotron model before enabling Living Weights; the current model will not be replaced.')
        const learningInfo = await deps.installed.info(repo!)
        if (learningInfo.memoryFloorBytes > supervisor.memoryLimitFor(learningInfo.modelType)) throw new Error('This Mac does not have enough model memory for Living Weights on Nemotron. Choose a smaller model for ordinary chat.')
        if (!await learningCopies.existing(repo!)) {
          let source = await revisions.selected(repo!)
          if (!source) {
            const cache = repoCacheDir(await fs.realpath(deps.hub()), repo!)
            const revision = (await fs.readFile(path.join(cache, 'refs', 'main'), 'utf8')).trim()
            source = await revisions.snapshot(repo!, revision)
          }
          learning.preparation(0, 0, 'Preparing an independent private model copy; the original download stays untouched.')
          await learningCopies.prepare(repo!, source, repoCacheDir(deps.hub(), repo!), (copied, total) => learning.preparation(copied, total))
        }
      }
      await preferences.set({ learningEnabled: enabled, learningModel: repo })
    }, async () => { await preferences.set(prior) })
  } catch (error) {
    learning.preparation(null, 0, (error as Error).message)
    throw error
  }
  learning.preparation(null)
  deps.installed.invalidate()
  deps.emitStatus()
  const snapshot = await learning.snapshot()
  deps.emit({ kind: 'learning', snapshot })
  return snapshot
}
