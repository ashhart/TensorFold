import { setMicrosoftMailRuntime } from '../../../agent/email-provider-runtime'
import { microsoftMailService } from '../microsoft-mail-runtime'
import { requireSecurityRole } from '../audit'
import { saveSettings } from '../../../store'
import { store } from '../../state'
import { handle as defaultHandle } from '../ipc'
import type { ConnectorIpcDeps } from '../connector-ipc'

export function registerMicrosoftMailIpc(deps: ConnectorIpcDeps): void {
  const handle = deps.handle ?? defaultHandle
  setMicrosoftMailRuntime({
    list: (source, limit) => microsoftMailService().list(source, limit),
    read: (source, id) => microsoftMailService().read(source, id),
    send: (source, event) => microsoftMailService().send(source, event),
  })
  handle('microsoftMail:begin', async (_e, input: { clientId: string; allowSend: boolean }) => {
    requireSecurityRole('owner', 'connect Microsoft mail')
    return microsoftMailService().begin(input?.clientId, input?.allowSend)
  })
  handle('microsoftMail:poll', async (_e, id: string) => {
    requireSecurityRole('owner', 'complete Microsoft mail sign-in')
    const result = await microsoftMailService().poll(id)
    if (result.status === 'connected' && result.source) {
      const source = result.source
      store.settings = { ...store.settings, email: { ...store.settings.email, sources: [...store.settings.email.sources.filter(item => item.id !== source.id), source] } }
      await saveSettings(store.settings)
    }
    return result.status === 'pending' ? result : { status: 'connected' as const }
  })
  handle('microsoftMail:cancel', async (_e, id: string) => {
    requireSecurityRole('owner', 'cancel Microsoft mail sign-in')
    microsoftMailService().cancel(id)
  })
}
