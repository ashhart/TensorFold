import os from 'node:os'
import type { EngineCapabilities } from '../types'
import { parseCapabilities, parseVersion, runEngineCli } from './cli'
import type { EngineBinary } from './paths'

// What the engine binary in use is and offers: its version and the serve flags that gate
// Cortheon, the control vector and Living Weights. Each answer is asked once and kept.

export interface EngineFeaturesDeps {
  binary: () => EngineBinary | null
  requireBinary: () => EngineBinary
  env: () => NodeJS.ProcessEnv
}

export class EngineFeatures {
  private versionValue: string | null = null
  private versionChecked = false
  private doc: EngineCapabilities | null = null
  private cortheonValue: boolean | null = null
  private glpValue: boolean | null = null
  private learningValue = false

  constructor(private readonly deps: EngineFeaturesDeps) {}

  get version(): string | null { return this.versionValue }

  // Whether the bundled engine lists --cortheon among its serve flags; null until asked.
  get cortheon(): boolean | null { return this.cortheonValue }

  // Whether the engine in use lists --glp among its serve flags; null until asked.
  get glp(): boolean | null { return this.glpValue }

  get learning(): boolean { return this.learningValue }

  // Asks the binary for its version and flags the first time one is present.
  async probe(): Promise<void> {
    if (!this.versionChecked && this.deps.binary()) {
      this.versionChecked = true
      try { this.versionValue = parseVersion((await runEngineCli(this.deps.requireBinary(), ['--version'], this.deps.env(), 10_000)).stdout) } catch { this.versionValue = null }
    }
    if (this.cortheonValue == null && this.deps.binary()) {
      try {
        const flags = (await this.capabilities()).flags
        this.cortheonValue = flags.includes('--cortheon')
        this.glpValue = flags.includes('--glp')
        this.learningValue = flags.includes('--slide') && flags.includes('--slide-graph') && os.platform() === 'darwin' && os.arch() === 'arm64'
      } catch { this.cortheonValue = false; this.glpValue = false }
    }
  }

  async capabilities(): Promise<EngineCapabilities> {
    if (this.doc) return this.doc
    const result = await runEngineCli(this.deps.requireBinary(), ['capabilities', '--json'], this.deps.env())
    if (result.code !== 0) throw new Error(result.stderr.trim() || 'tensorfold-native capabilities --json failed.')
    this.doc = parseCapabilities(result.stdout)
    return this.doc
  }

  /** The environment this engine build declares it honours, or null when it could not be asked. */
  async declaredEnv(): Promise<string[] | null> {
    try {
      const caps = await this.capabilities()
      return Array.isArray(caps.env) ? caps.env : null
    } catch {
      return null
    }
  }
}
