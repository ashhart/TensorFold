import { BrowserWindow, dialog } from 'electron'
import type { PhoneHost } from '../phone-host'
import type { PhoneActionInput, PhoneDeveloperConnectInput, PhonePairInput } from '../../../agent/types'
import { handle } from '../ipc'
import { requireSecurityRole } from '../audit'

export function registerPhoneIpc(host: PhoneHost, options: { reserveOperation?: () => { release(): void } } = {}) {
  const owner = () => requireSecurityRole('owner', 'manage a connected phone')
  const guarded: typeof handle = (channel, listener) => handle(channel, async (event, ...args) => {
    const lease = options.reserveOperation?.()
    try { return await listener(event, ...args) } finally { lease?.release() }
  })
  handle('phone:snapshot', async () => { owner(); return host.service.snapshot() })
  guarded('phone:refresh', async () => { owner(); return host.service.refresh() })
  guarded('phone:connect', async (_e, input: { deviceId?: unknown }) => { owner(); if (typeof input?.deviceId !== 'string' || input.deviceId.length > 120) throw new Error('Choose a discovered phone.'); return host.service.connect({ deviceId: input.deviceId }) })
  guarded('phone:connectDeveloper', async (_e, input: PhoneDeveloperConnectInput) => {
    owner()
    if (!input || typeof input.endpoint !== 'string' || input.endpoint.length > 256 || !['usb', 'wifi'].includes(input.transport)) throw new Error('Choose a USB or Wi-Fi iPhone developer bridge endpoint.')
    return host.service.connectDeveloper(input)
  })
  guarded('phone:grantChat', async (_e, id: unknown) => { owner(); if (typeof id !== 'string') throw new Error('Open a local chat first.'); return host.grantChat(id) })
  guarded('phone:observe', async () => { owner(); return host.service.observe() })
  guarded('phone:action', async (_e, input: PhoneActionInput) => { owner(); return host.service.action(input) })
  handle('phone:stop', async () => { owner(); return host.service.stop() })
  handle('phone:disconnect', async () => { owner(); return host.service.disconnect() })
  guarded('phone:pairWifi', async (_e, input: PhonePairInput) => {
    owner()
    if (!input || typeof input.endpoint !== 'string' || typeof input.code !== 'string') throw new Error('Enter the pairing address and code shown on your Android phone.')
    return host.pairWifi(input)
  })
  guarded('phone:connectWifi', async (_e, endpoint: unknown) => { owner(); if (typeof endpoint !== 'string') throw new Error('Enter the phone’s Wireless debugging connection address.'); return host.connectWifi(endpoint) })
  guarded('phone:mirroringPermissions', async () => { owner(); return host.permissions() })
  guarded('phone:pickAdb', async (event) => {
    owner()
    if (host.service.snapshot().session || host.service.snapshot().busy) throw new Error('Disconnect the phone before changing Android tools.')
    const parent = BrowserWindow.fromWebContents(event.sender)
    const options = { title: 'Choose Android Platform Tools adb', properties: ['openFile'] as ['openFile'] }
    const result = parent ? await dialog.showOpenDialog(parent, options) : await dialog.showOpenDialog(options)
    if (result.canceled || result.filePaths.length !== 1) return null
    return host.setAdbPath(result.filePaths[0])
  })
}
