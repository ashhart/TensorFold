import type { Settings, SecurityPurgeResult } from '../types'
import { mockState } from './state'
import { mockLocalMemoryCounts, mockSecurityControls, redactedMockSettings } from './runtime'
import type { TensorFoldStudioApi } from '../ipc'

function isMaskedSecret(value: unknown): boolean { return value === '••••••••' || value === '[set]' }

function preserveMaskedSourceSecrets<T extends { id: string }>(incoming: T[], current: T[], keys: string[]): T[] {
  const currentById = new Map(current.map((source) => [source.id, source as Record<string, unknown>]))
  return incoming.map((source) => {
    const existing = currentById.get(source.id)
    if (!existing) return source
    const next = { ...(source as Record<string, unknown>) }
    for (const key of keys) {
      if (isMaskedSecret(next[key])) next[key] = existing[key] ?? ''
    }
    return next as T
  })
}

export function createMockSettingsApi(): Pick<TensorFoldStudioApi, 'getSecurityControls' | 'purgeLocalMemory' | 'getSettings' | 'desktopModeSet' | 'desktopModeHide' | 'onDesktopModeChanged' | 'onDesktopNavigate' | 'saveSettings' | 'revokeApprovalRule'> {
  const desktopModeSubs = new Set<(settings: Settings) => void>()
  return {
    getSecurityControls: async () => mockSecurityControls(),
    purgeLocalMemory: async (): Promise<SecurityPurgeResult> => {
      const counts = mockLocalMemoryCounts()
      mockState.threads = []
      mockState.projects = []
      mockState.mockRooms = []
      mockState.mockTickets = []
      mockState.mockJobs = []
      mockState.mockAutomations = []
      mockState.mockConnectorEvents = []
      mockState.mockMissions = []
      mockState.mockProjectMemory = []
      mockState.mockProjectMemoryCandidates = []
      mockState.mockIncidents = []
      mockState.settings = {
        ...mockState.settings,
        whatsapp: {
          ...mockState.settings.whatsapp,
          sources: mockState.settings.whatsapp.sources.map((source) => ({
            ...source,
            paired: false,
            phoneNumber: null,
            credentialRotatedAt: null,
            lastError: 'Linked-device credentials were removed by the local-state purge; pair again.',
          })),
        },
      }
      return {
        ok: true,
        purgedAt: Date.now(),
        counts,
        removedPaths: ['/Users/example/.vontra/archives', '/Users/example/.vontra/rooms', '/Users/example/.vontra/updates', '/Users/example/.vontra/whatsapp-auth'],
        preservedCredentialsAndSettings: ['App settings and configured provider/cloud credentials (linked WhatsApp device credentials are removed)'],
        preservedResiduals: [],
      }
    },

    getSettings: async () => redactedMockSettings(),
    desktopModeSet: async (mode, onboardingComplete) => {
      if (mode !== 'minimal' && mode !== 'full') throw new Error('Choose Minimal or Full mode.')
      mockState.settings = { ...mockState.settings, desktopMode: mode, ...(onboardingComplete === undefined ? {} : { desktopOnboardingComplete: onboardingComplete === true }) }
      const settings = redactedMockSettings()
      for (const cb of desktopModeSubs) cb(settings)
      return settings
    },
    desktopModeHide: async () => ({ hidden: false }),
    // Browser previews have no system tray.
    onDesktopModeChanged: (cb) => { desktopModeSubs.add(cb); return () => desktopModeSubs.delete(cb) },
    onDesktopNavigate: () => () => { },
    saveSettings: async (s) => {
      const providers = s.providers.map((p) => {
        const cur = mockState.settings.providers.find((x) => x.id === p.id)
        const masked = p.apiKey === '••••••••' || p.apiKey === '[set]'
        const accounts = p.accounts?.map((account) => {
          const curAccount = cur?.accounts?.find((x) => x.id === account.id)
          const accountMasked = account.apiKey === '••••••••' || account.apiKey === '[set]'
          return {
            ...account,
            apiKey: accountMasked ? (curAccount?.apiKey ?? '') : account.apiKey,
            env: Object.fromEntries(Object.entries(account.env ?? {}).map(([key, value]) => [key, value === '••••••••' || value === '[set]' ? (curAccount?.env?.[key] ?? '') : value])),
          }
        })
        return {
          ...p,
          apiKey: masked ? (cur?.apiKey ?? '') : p.apiKey,
          cliEnv: Object.fromEntries(Object.entries(p.cliEnv ?? {}).map(([key, value]) => [key, value === '••••••••' || value === '[set]' ? (cur?.cliEnv?.[key] ?? '') : value])),
          accounts,
        }
      })
      const botControl = {
        ...s.botControl,
        telegram: {
          ...s.botControl.telegram,
          botToken: s.botControl.telegram.botToken === '••••••••' || s.botControl.telegram.botToken === '[set]'
            ? mockState.settings.botControl.telegram.botToken
            : s.botControl.telegram.botToken,
        },
        whatsapp: {
          ...s.botControl.whatsapp,
          accessToken: s.botControl.whatsapp.accessToken === '••••••••' || s.botControl.whatsapp.accessToken === '[set]'
            ? mockState.settings.botControl.whatsapp.accessToken
            : s.botControl.whatsapp.accessToken,
          verifyToken: s.botControl.whatsapp.verifyToken === '••••••••' || s.botControl.whatsapp.verifyToken === '[set]'
            ? mockState.settings.botControl.whatsapp.verifyToken
            : s.botControl.whatsapp.verifyToken,
          appSecret: s.botControl.whatsapp.appSecret === '••••••••' || s.botControl.whatsapp.appSecret === '[set]'
            ? mockState.settings.botControl.whatsapp.appSecret
            : s.botControl.whatsapp.appSecret,
        },
      }
      const voice = {
        ...s.voice,
        apiKey: s.voice.apiKey === '••••••••' || s.voice.apiKey === '[set]'
          ? mockState.settings.voice.apiKey
          : s.voice.apiKey,
        elevenLabsApiKey: s.voice.elevenLabsApiKey === '••••••••' || s.voice.elevenLabsApiKey === '[set]'
          ? mockState.settings.voice.elevenLabsApiKey
          : s.voice.elevenLabsApiKey,
      }
      const remoteControl = {
        ...s.remoteControl,
        authToken: s.remoteControl.authToken === '••••••••' || s.remoteControl.authToken === '[set]'
          ? mockState.settings.remoteControl.authToken
          : s.remoteControl.authToken,
        lastStartedAt: s.remoteControl.enabled ? (s.remoteControl.lastStartedAt ?? Date.now()) : s.remoteControl.lastStartedAt,
        lastError: null,
      }
      const remoteClient = {
        ...s.remoteClient,
        authToken: s.remoteClient.authToken === '••••••••' || s.remoteClient.authToken === '[set]'
          ? mockState.settings.remoteClient.authToken
          : s.remoteClient.authToken,
      }
      const remoteOrchestrators = s.remoteOrchestrators.map((host) => {
        const current = mockState.settings.remoteOrchestrators.find((item) => item.id === host.id)
        return {
          ...host,
          authToken: host.authToken === '••••••••' || host.authToken === '[set]' ? (current?.authToken ?? mockState.settings.remoteClient.authToken ?? '') : host.authToken,
        }
      })
      const email = { ...s.email, sources: preserveMaskedSourceSecrets(s.email.sources, mockState.settings.email.sources, ['password']) }
      const slack = { ...s.slack, sources: preserveMaskedSourceSecrets(s.slack.sources, mockState.settings.slack.sources, ['token']) }
      const discord = { ...s.discord, sources: preserveMaskedSourceSecrets(s.discord.sources, mockState.settings.discord.sources, ['token']) }
      const mattermost = { ...s.mattermost, sources: preserveMaskedSourceSecrets(s.mattermost.sources, mockState.settings.mattermost.sources, ['token']) }
      const matrix = { ...s.matrix, sources: preserveMaskedSourceSecrets(s.matrix.sources, mockState.settings.matrix.sources, ['token']) }
      const googlechat = { ...s.googlechat, sources: preserveMaskedSourceSecrets(s.googlechat.sources, mockState.settings.googlechat.sources, ['webhookUrl', 'accessToken']) }
      Object.assign(mockState.settings, { ...s, providers, botControl, voice, remoteControl, remoteClient, remoteOrchestrators, email, slack, discord, mattermost, matrix, googlechat })
      return redactedMockSettings()
    },
    revokeApprovalRule: async (id) => {
      mockState.settings.approvalRules = mockState.settings.approvalRules.filter((rule) => rule.id !== id)
      return redactedMockSettings()
    },
  }
}
