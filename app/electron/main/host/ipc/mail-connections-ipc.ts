import type { MailConnectionTarget } from '../../../../shared/mail-connections'
import { mailConnectionsService } from '../mail-connections-runtime'
import { requireSecurityRole } from '../audit'
import { handle as defaultHandle } from '../ipc'

export function registerMailConnectionsIpc(deps: { handle?: typeof defaultHandle; service?: typeof mailConnectionsService } = {}): void {
  const handle = deps.handle ?? defaultHandle, service = deps.service ?? mailConnectionsService
  handle('mailConnections:status', async () => { requireSecurityRole('owner', 'read mail connection health'); return service().snapshot() })
  handle('mailConnections:test', async (_event, target: MailConnectionTarget) => { requireSecurityRole('owner', 'test mail connection access'); return service().test(target) })
}
