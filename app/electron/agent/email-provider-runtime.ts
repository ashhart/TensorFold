import type { EmailReadResult, EmailSourceConfig, ConnectorEvent } from './types'

export interface EmailProviderRuntime {
  list(source: EmailSourceConfig, limit: number): Promise<EmailReadResult[]>
  read(source: EmailSourceConfig, id: string): Promise<EmailReadResult | null>
  send(source: EmailSourceConfig, event: ConnectorEvent): Promise<NonNullable<ConnectorEvent['delivery']>>
}

let microsoft: EmailProviderRuntime | undefined
export function setMicrosoftMailRuntime(runtime: EmailProviderRuntime): void { microsoft = runtime }
export function microsoftMailRuntime(): EmailProviderRuntime {
  if (!microsoft) throw new Error('Microsoft mail is unavailable on this host. Restart TensorFold before reconnecting.')
  return microsoft
}
