import { randomUUID } from 'node:crypto'
import { promises as fs, readFileSync } from 'node:fs'
import path from 'node:path'

// Built-in engine choices the user makes once, kept beside the model registry.
export interface EnginePreferences {
  // Serve with --cortheon when the bundled engine offers it.
  cortheon: boolean
  learningEnabled: boolean
  learningModel: string | null
}

const DEFAULTS: EnginePreferences = { cortheon: false, learningEnabled: false, learningModel: null }

function readPreferences(file: string): EnginePreferences {
  try {
    const doc = JSON.parse(readFileSync(file, 'utf8'))
    const learningModel = doc?.learningModel === 'TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit' ? doc.learningModel : null
    return { cortheon: doc?.cortheon === true, learningEnabled: doc?.learningEnabled === true && learningModel != null, learningModel }
  } catch {
    return { ...DEFAULTS }
  }
}

export class EnginePreferenceStore {
  private readonly file: string
  private value: EnginePreferences
  private writes: Promise<unknown> = Promise.resolve()

  constructor(appHome: string) {
    this.file = path.join(appHome, 'engine', 'preferences.json')
    this.value = readPreferences(this.file)
  }

  get(): EnginePreferences {
    return { ...this.value }
  }

  set(patch: Partial<EnginePreferences>): Promise<EnginePreferences> {
    const next = this.writes.then(async () => {
      const value = { ...this.value, ...patch }
      const directory = path.dirname(this.file)
      await fs.mkdir(directory, { recursive: true, mode: 0o700 })
      const temporary = path.join(directory, `.preferences-${randomUUID()}.tmp`)
      try {
        const file = await fs.open(temporary, 'wx', 0o600)
        try { await file.writeFile(JSON.stringify({ version: 1, ...value })); await file.sync() } finally { await file.close() }
        await fs.rename(temporary, this.file)
      } finally { await fs.rm(temporary, { force: true }) }
      this.value = value
      return this.get()
    })
    this.writes = next.catch(() => {})
    return next
  }
}
