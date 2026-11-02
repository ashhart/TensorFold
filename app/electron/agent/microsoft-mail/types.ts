import type { ConnectorHttpDeps } from '../connector-http'
import type { EmailSourceConfig } from '../types'

export interface MicrosoftMailAccount {
  id: string
  clientId: string
  email: string
  revision: number
  accessToken: string
  refreshToken: string
  expiresAt: number
  scopes: string[]
}

export interface MicrosoftMailState { version: 1; accounts: MicrosoftMailAccount[] }

export interface MicrosoftMailDeps extends ConnectorHttpDeps {
  store: { read(): Promise<MicrosoftMailState | null>; write(state: MicrosoftMailState, validate: () => void): Promise<void> }
  now?: () => number
  assertSourceCurrent?: (source: EmailSourceConfig) => void
}
