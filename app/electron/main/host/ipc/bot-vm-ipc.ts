import { BrowserWindow, dialog } from 'electron'
import { handle } from '../ipc'
import { requireSecurityRole } from '../audit'
import type { BotVmHost } from '../bot-vm-host'

export function registerBotVmIpc(host: BotVmHost, options: { authorizeBot(botId: string): void; authorizeBotStart?(botId: string): void; cancelBotRuns?(botId: string): void; reserveOperation?: () => { release(): void } }) {
  const owner = () => requireSecurityRole('owner', 'manage isolated bot VMs')
  const bot = (id: unknown): string => { if (typeof id !== 'string' || !/^[a-zA-Z0-9][a-zA-Z0-9_-]{0,99}$/.test(id)) throw new Error('Choose a saved bot.'); options.authorizeBot(id); return id }
  const startBot = (id: unknown): string => { const botId = bot(id); options.authorizeBotStart?.(botId); return botId }
  const guarded: typeof handle = (channel, listener) => handle(channel, async (event, ...args) => {
    owner(); const lease = options.reserveOperation?.()
    try { return await listener(event, ...args) } finally { lease?.release() }
  })
  handle('botVm:status', async (_event, id) => { owner(); return host.status(bot(id)) })
  handle('botVm:setupStatus', async () => { owner(); return host.setupStatus() })
  handle('botVm:cancelSetup', async () => { owner(); return host.cancelSetup() })
  guarded('botVm:managedSetup', async (_event, id) => { const botId = startBot(id); return host.managedSetup(botId, () => { startBot(botId) }) })
  guarded('botVm:provision', async (_event, id) => host.provision(startBot(id)))
  guarded('botVm:start', async (_event, id) => host.start(startBot(id)))
  handle('botVm:stop', async (_event, id) => { owner(); const botId = bot(id); options.cancelBotRuns?.(botId); return host.stop(botId) })
  guarded('botVm:rebuild', async (_event, id) => { const botId = bot(id); options.cancelBotRuns?.(botId); return host.rebuild(botId) })
  guarded('botVm:startRuntime', async () => host.startRuntime())
  guarded('botVm:pickRuntime', async (event) => {
    const parent = BrowserWindow.fromWebContents(event.sender)
    const config = { title: 'Choose the signed Apple container executable', defaultPath: '/usr/local/bin/container', properties: ['openFile'] as ['openFile'] }
    const result = parent ? await dialog.showOpenDialog(parent, config) : await dialog.showOpenDialog(config)
    if (result.canceled || result.filePaths.length !== 1) return null
    await host.setRuntimePath(result.filePaths[0]); return true
  })
}
