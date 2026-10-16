export interface BotVmStatus {
  botId: string
  state: 'unsupported' | 'setup-required' | 'runtime-stopped' | 'not-created' | 'stopped' | 'running' | 'blocked'
  detail: string
  runtimeVersion: string | null
  workspace: '/workspace'
  network: 'none'
  capabilities: Array<'files' | 'commands'>
  busy: boolean
  setupUrl: string
}

export interface BotVmResult {
  ok: boolean
  output: string
  code?: number
  changedPaths?: string[]
  truncated?: boolean
  interrupted?: boolean
}

export interface BotVmSetupInput { startRuntime?: boolean }

export interface BotVmSetupStatus {
  active: boolean
  botId?: string | null
  phase: 'idle' | 'downloading' | 'verifying' | 'extracting' | 'activating' | 'starting' | 'provisioning' | 'ready' | 'cancelled' | 'error'
  bytes: number
  total: number
  detail: string
}
