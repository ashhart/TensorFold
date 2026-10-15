import type { BotVmStatus, BotVmSetupStatus } from '../types'
export function createMockBotVmApi() {
  const unavailable = async (botId: string): Promise<BotVmStatus> => ({ botId, state: 'unsupported', detail: 'Isolated bot VMs are available in the local Mac app on Apple silicon with macOS 26 or later.', runtimeVersion: null, workspace: '/workspace', network: 'none', capabilities: [], busy: false, setupUrl: 'https://github.com/apple/container/releases/tag/1.5.0' })
  const reject = async (): Promise<never> => { throw new Error('VM setup is available only in the local desktop app.') }
  const setup = async (): Promise<BotVmSetupStatus> => ({ active: false, botId: null, phase: 'idle', bytes: 0, total: 118045087, detail: 'Managed VM setup is available only in the local Mac desktop app.' })
  return { botVmStatus: unavailable, botVmManagedSetup: reject, botVmSetupStatus: setup, botVmCancelSetup: setup, botVmProvision: unavailable, botVmStart: unavailable, botVmStop: unavailable, botVmRebuild: unavailable, botVmPickRuntime: reject, botVmStartRuntime: reject }
}
