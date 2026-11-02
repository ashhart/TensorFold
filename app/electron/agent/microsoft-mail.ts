import type { ConnectorEvent, EmailSourceConfig } from './types'
import { MicrosoftMailAccounts } from './microsoft-mail/accounts'
import { MicrosoftMailbox } from './microsoft-mail/graph'
import { MicrosoftMailTransport } from './microsoft-mail/transport'
import type { MicrosoftMailDeps } from './microsoft-mail/types'

export { microsoftMailSource } from './microsoft-mail/accounts'
export type { MicrosoftMailState, MicrosoftMailDeps } from './microsoft-mail/types'

export class MicrosoftMailService {
  private readonly accounts: MicrosoftMailAccounts
  private readonly mailbox: MicrosoftMailbox

  constructor(deps: MicrosoftMailDeps) {
    const transport = new MicrosoftMailTransport(deps)
    this.accounts = new MicrosoftMailAccounts(deps, transport)
    this.mailbox = new MicrosoftMailbox(this.accounts, transport)
  }

  begin(clientId: string, allowSend: boolean) { return this.accounts.begin(clientId, allowSend) }
  poll(id: string) { return this.accounts.poll(id) }
  cancel(id: string): void { this.accounts.cancel(id) }
  connections() { return this.accounts.connections() }
  remove(id: string) { return this.accounts.remove(id) }
  list(source: EmailSourceConfig, limit: number) { return this.mailbox.list(source, limit) }
  read(source: EmailSourceConfig, id: string) { return this.mailbox.read(source, id) }
  send(source: EmailSourceConfig, event: ConnectorEvent) { return this.mailbox.send(source, event) }
}
