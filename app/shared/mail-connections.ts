export type MailProvider = 'gmail' | 'microsoft'
export type MailHealthState = 'untested' | 'healthy' | 'reconnect' | 'permission' | 'unreachable' | 'rate-limited' | 'error' | 'disconnected'

export interface MailConnectionAccount {
  provider: MailProvider
  id: string
  email: string
  revision: string
  authorization: 'connected' | 'reconnect' | 'disconnected'
  read: boolean
  send: boolean
  inbox: boolean
  sourceId?: string
}

export interface MailConnectionHealth extends MailConnectionAccount {
  state: MailHealthState
  checkedAt: number | null
  lastSuccessAt: number | null
  message: string
}

export interface MailConnectionsSnapshot {
  googleConfigured: boolean
  accounts: MailConnectionHealth[]
}

export interface MailConnectionTarget { provider: MailProvider; id: string; revision: string }
export interface MailConnectionsApi {
  mailConnections(): Promise<MailConnectionsSnapshot>
  testMailConnection(target: MailConnectionTarget): Promise<MailConnectionHealth>
}

export const MAIL_CONNECTIONS_API_SCHEMA = {
  mailConnections: 'method', testMailConnection: 'method',
} as const
