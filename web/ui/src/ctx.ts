import type { EngineInfo, Machine, Settings } from './api'

export type Ctx = {
  machine: Machine | null
  engine: EngineInfo | null
  settings: Settings | null
  refreshEngine: () => Promise<unknown>
  refreshSettings: () => Promise<unknown>
  refreshBrands: () => Promise<unknown>   // the makers kept in Settings, read again after a change
  saveSettings: (patch: Partial<Settings>) => Promise<Settings>
  toast: (text: string, kind?: 'ok' | 'bad') => void
  openPalette: () => void
  stale: boolean                 // TensorFold Web stopped answering: figures are old, engine actions are paused
  decodeHistory: number[]        // decode tok/s of the last polls, this session
}
