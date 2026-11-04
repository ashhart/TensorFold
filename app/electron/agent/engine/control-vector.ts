import { existsSync } from 'node:fs'
import { mergeControlVector, type EngineControlVector } from '../../../shared/control-vector'
import type { EngineControlVectorSnapshot } from '../types'
import type { EnginePreferenceStore } from './preferences'
import type { EngineSupervisor } from './supervisor'

// The control vector the operator chose for the local engine: the file it reads, the
// strength it is applied with, and whether it is in force. The engine takes the two as
// --glp and --glp-alpha at its next start, and the file only ever arrives from the OS
// picker, never from the renderer.

export interface ControlVectorDeps {
  preferences: EnginePreferenceStore
  supervisor: EngineSupervisor
  // Whether the engine in use lists --glp among its serve flags.
  supported: () => boolean
}

export class ControlVectorSetting {
  constructor(private readonly deps: ControlVectorDeps) {}

  // What the operator chose, and what the engine in use can do with it.
  snapshot(): EngineControlVectorSnapshot {
    const prefs = this.deps.preferences.get().controlVector
    return { supported: this.deps.supported(), enabled: prefs.enabled, file: prefs.file, alpha: prefs.alpha, active: this.deps.supervisor.status().controlVectorActive }
  }

  // The file the operator picked: it is remembered, not applied.
  async choose(file: string): Promise<EngineControlVectorSnapshot> {
    const prior = this.deps.preferences.get().controlVector
    await this.deps.preferences.set({ controlVector: { ...prior, file } })
    return this.snapshot()
  }

  // Forget the file: the engine stops taking a vector at its next start.
  clear(): Promise<EngineControlVectorSnapshot> {
    return this.setNow({ clear: true })
  }

  // Whether the vector is in force, and at what strength.
  set(input: { enabled?: unknown; alpha?: unknown }): Promise<EngineControlVectorSnapshot> {
    return this.setNow(input)
  }

  private async setNow(input: { enabled?: unknown; alpha?: unknown; clear?: boolean }): Promise<EngineControlVectorSnapshot> {
    const prior = this.deps.preferences.get().controlVector
    if (input.clear) {
      if (!prior.file && !prior.enabled) return this.snapshot()
      await this.apply({ ...prior, enabled: false, file: null }, prior)
      return this.snapshot()
    }
    // A field the caller did not name keeps the value it had.
    const { enabled, alpha } = mergeControlVector(prior, input)
    if (enabled) {
      if (!this.deps.supported()) throw new Error("This build's TensorFold engine does not offer control vectors yet.")
      if (!prior.file) throw new Error('Choose a control vector file first.')
      if (!existsSync(prior.file)) throw new Error('That control vector file is no longer on this Mac.')
    }
    if (prior.enabled === enabled && prior.alpha === alpha) return this.snapshot()
    await this.apply({ enabled, file: prior.file, alpha }, prior)
    return this.snapshot()
  }

  // The change and the model restart happen together, so a refused start leaves the old choice in place.
  private async apply(next: EngineControlVector, prior: EngineControlVector): Promise<void> {
    await this.deps.supervisor.reconfigure(async () => { await this.deps.preferences.set({ controlVector: next }) }, async () => { await this.deps.preferences.set({ controlVector: prior }) })
  }
}
