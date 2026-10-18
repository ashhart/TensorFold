import type { Settings } from './types'
import { PRESETS } from '../../shared/theme-presets'

/** Inert browser appearance defaults, never copied from a demo or local account. */
export function pairedSettingsDefaults(): Settings {
  return {
    providers: [], activeProviderId: null, activeProviderAccountId: null, activeModel: null,
    agentPresets: [], theme: 'system', approvalMode: 'auto', approvalRules: [], defaultCwd: null,
    temperature: 0.4, reasoningEffort: 'xhigh', autoCompact: true, contextLimit: 128_000,
    localContextLimit: 24_000, autoCheckUpdates: false, updateFeedUrl: '',
    power: { keepComputerAlive: false, lastStartedAt: null, lastError: null },
    email: { sources: [], maxMessages: 300, lastSyncAt: null },
    slack: { sources: [], maxMessages: 300, lastSyncAt: null },
    calendar: { sources: [], maxEvents: 500, lastSyncAt: null },
    documents: { sources: [], maxDocuments: 500, maxDocumentBytes: 1_000_000, lastSyncAt: null },
    browser: { sources: [], maxItems: 300, lastSyncAt: null },
    imessage: { enabled: false, allowedChatIds: '', maxMessages: 200, lastSyncAt: null, lastError: null },
    whatsapp: { sources: [], lastSyncAt: null }, discord: { sources: [], lastSyncAt: null },
    mattermost: { sources: [], lastSyncAt: null }, matrix: { sources: [], lastSyncAt: null },
    googlechat: { sources: [], lastSyncAt: null },
    botControl: {
      telegram: { enabled: false, botToken: '', allowedChatIds: '', allowedUserIds: '', polling: false, lastUpdateId: null, lastPollAt: null, lastError: null },
      whatsapp: { enabled: false, accessToken: '', phoneNumberId: '', verifyToken: '', appSecret: '', allowedChatIds: '', lastEventAt: null, lastError: null },
    },
    capabilities: { disabledCapabilityIds: [], health: {}, localScripts: [], packagedPlugins: [] },
    controller: { enabled: false, includeAccessibilityText: false, maxTextChars: 4_000, lastCapturedAt: null, lastError: null },
    voice: {
      enabled: true, ttsEnabled: true, sttEnabled: true, ttsProvider: 'browser', sttProvider: 'browser',
      voiceName: '', rate: 1.6, pitch: 1, volume: 1, autoSummarizeBeforeRead: true,
      summaryMaxChars: 900, autoReadRoomUpdates: false, apiBaseUrl: 'https://api.openai.com/v1', apiKey: '',
      ttsModel: 'gpt-4o-mini-tts', sttModel: 'gpt-4o-mini-transcribe', elevenLabsApiKey: '',
      elevenLabsVoiceId: '', elevenLabsModelId: 'eleven_multilingual_v2', elevenLabsStability: 0.4,
      elevenLabsStyle: 0.3, localTtsBaseUrl: 'http://127.0.0.1:8080', localTtsVoice: '', localTtsEmotion: '',
    },
    remoteControl: { enabled: false, serverName: 'TensorFold host', bindHost: '127.0.0.1', port: 8765, authToken: '', allowLan: false, lastStartedAt: null, lastClientAt: null, lastError: null },
    remoteClient: { enabled: false, serverUrl: '', authToken: '', lastConnectedAt: null, lastError: null },
    remoteOrchestrators: [], defaultExecutionTarget: 'local', defaultOrchestratorId: null,
    preset: 'tensorfold', light: { ...PRESETS.tensorfold.light }, dark: { ...PRESETS.tensorfold.dark },
    pointerCursors: true, doomTodos: true, notifyRuns: true, sshHosts: [], trustedSshHostKeys: [],
  }
}
