export interface EngineRuntimeRelease {
  id: string
  version: string
  platform: 'darwin' | 'linux' | 'win32'
  arch: 'arm64' | 'x64'
  backend: string
  notes: string
}

export interface EngineRuntimeSnapshot {
  bundledVersion: string
  active: EngineRuntimeRelease | null
  installed: EngineRuntimeRelease[]
  available: EngineRuntimeRelease[]
  state: 'idle' | 'checking' | 'downloading' | 'installing' | 'switching'
  progress: { receivedBytes: number; totalBytes: number | null } | null
  checkedAt: number | null
  error: string | null
}
