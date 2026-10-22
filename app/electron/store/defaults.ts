import type { Settings, Provider } from '../agent/types'
import { PRESETS } from '../agent/theme-presets'
import { CODEX_CLI_MODELS, CLAUDE_CLI_MODELS } from '../agent/cli-models'
import { makeRemoteControlToken } from './paths'
import { builtinEngineProvider } from '../agent/engine/provider'

// Built-in defaults + provider seeding + the small settings-migration gate.
// The giant Settings literal lives here so the persistence module (load/save) and
// importers that just need defaults can read it without pulling in the whole
// write/read/protected-json stack.

export function defaultSettings(): Settings {
  const now = Date.now()
  return {
    desktopMode: 'full',
    desktopOnboardingComplete: false,
    providers: [
      { id: 'openai', name: 'OpenAI', baseUrl: 'https://api.openai.com/v1', apiKey: '', models: ['gpt-4o', 'gpt-4o-mini', 'o4-mini'], kind: 'cloud', source: 'openai' },
      { id: 'codex-cli', name: 'Codex CLI OAuth', transport: 'codex-cli', auth: 'codex-oauth', baseUrl: '', apiKey: '', models: CODEX_CLI_MODELS, kind: 'cloud', cliPath: 'codex', cliSandbox: 'workspace-write', cliApprovalPolicy: 'on-request', accounts: [] },
      { id: 'claude-cli', name: 'Claude CLI OAuth', transport: 'claude-cli', auth: 'claude-oauth', baseUrl: '', apiKey: '', models: CLAUDE_CLI_MODELS, kind: 'cloud', cliPath: 'claude', cliPermissionMode: 'default', accounts: [] },
      { id: 'omlx', name: 'oMLX (local)', transport: 'chat-completions', auth: 'api-key', baseUrl: 'http://127.0.0.1:9000/v1', apiKey: '', models: ['local-model'], kind: 'local', source: 'omlx' },
      { id: 'ollama', name: 'Ollama (local)', baseUrl: 'http://127.0.0.1:11434/v1', apiKey: 'ollama', models: ['qwen2.5-coder:14b', 'llama3.1:8b'], kind: 'local', source: 'ollama' },
      { id: 'lmstudio', name: 'LM Studio (local)', baseUrl: 'http://127.0.0.1:1234/v1', apiKey: 'lm-studio', models: ['local-model'], kind: 'local', source: 'lmstudio' },
      { id: 'openrouter', name: 'OpenRouter', baseUrl: 'https://openrouter.ai/api/v1', apiKey: '', models: ['openrouter/free', 'openrouter/auto', 'openrouter/fusion', 'z-ai/glm-5.2', 'nvidia/nemotron-3-ultra-550b-a55b:free', 'qwen/qwen3-coder:free', 'openai/gpt-oss-120b:free'], kind: 'cloud', source: 'openrouter' },
      { id: 'zai', name: 'z.ai GLM', baseUrl: 'https://api.z.ai/api/paas/v4', apiKey: '', models: ['glm-5.2', 'glm-5.1', 'glm-4.6', 'glm-4-flash'], kind: 'cloud' },
      builtinEngineProvider(),
    ],
    cliToolOverrides: {},
    // A fresh profile starts on the built-in engine; the Models page picks its first model.
    activeProviderId: 'tensorfold',
    activeProviderAccountId: null,
    activeModel: null,
    assistantProviderId: null,
    assistantProviderAccountId: null,
    assistantModel: null,
    codeProviderId: null,
    codeProviderAccountId: null,
    codeModel: null,
    agentPresets: [],
    theme: 'system',
    approvalMode: 'suggest',
    approvalRules: [],
    defaultCwd: null,
    temperature: 0.1,
    reasoningEffort: 'xhigh',
    serviceTier: 'fast',
    autoCompact: true,
    allowTaskNetwork: false,
    contextLimit: 128_000,
    localContextLimit: 24_000,
    contextLimitExplicit: false,
    localContextLimitExplicit: false,
    autoCheckUpdates: true,
    updateFeedUrl: '',
    power: {
      keepComputerAlive: false,
      lastStartedAt: null,
      lastError: null,
    },
    email: {
      sources: [],
      maxMessages: 300,
      lastSyncAt: null,
    },
    slack: {
      sources: [],
      maxMessages: 300,
      lastSyncAt: null,
    },
    calendar: {
      sources: [],
      maxEvents: 500,
      lastSyncAt: null,
    },
    documents: {
      sources: [],
      maxDocuments: 500,
      maxDocumentBytes: 1_000_000,
      lastSyncAt: null,
    },
    browser: {
      sources: [],
      maxItems: 300,
      lastSyncAt: null,
    },
    imessage: {
      enabled: false, // opt-in; reading chat.db needs Full Disk Access
      allowedChatIds: '',
      maxMessages: 200,
      lastSyncAt: null,
      lastError: null,
    },
    whatsapp: {
      sources: [], // multi-account: each paired number is one source
      lastSyncAt: null,
    },
    discord: {
      sources: [], // each source = one bot token across its servers
      lastSyncAt: null,
    },
    mattermost: {
      sources: [], // each source = one server (base URL) + token
      lastSyncAt: null,
    },
    matrix: {
      sources: [], // each source = one homeserver (base URL) + access token
      lastSyncAt: null,
    },
    googlechat: {
      sources: [], // each source = one space webhook URL (and optional service-account token)
      lastSyncAt: null,
    },
    botControl: {
      telegram: {
        enabled: false,
        botToken: '',
        allowedChatIds: '',
        allowedUserIds: '',
        polling: false,
        lastUpdateId: null,
        lastPollAt: null,
        lastError: null,
        secretRotations: {},
      },
      whatsapp: {
        enabled: false,
        accessToken: '',
        phoneNumberId: '',
        verifyToken: '',
        appSecret: '',
        allowedChatIds: '',
        lastEventAt: null,
        lastError: null,
        secretRotations: {},
      },
    },
    capabilities: {
      disabledCapabilityIds: [],
      health: {},
      localScripts: [],
      packagedPlugins: [],
    },
    controller: {
      enabled: false,
      includeAccessibilityText: false,
      maxTextChars: 4_000,
      lastCapturedAt: null,
      lastError: null,
    },
    voice: {
      enabled: true,
      ttsEnabled: true,
      sttEnabled: true,
      ttsProvider: 'browser',
      sttProvider: 'browser',
      voiceName: '',
      rate: 1.6,
      pitch: 1,
      volume: 1,
      autoSummarizeBeforeRead: true,
      summaryMaxChars: 900,
      autoReadRoomUpdates: false,
      apiBaseUrl: 'https://api.openai.com/v1',
      apiKey: '',
      ttsModel: 'gpt-4o-mini-tts',
      sttModel: 'gpt-4o-mini-transcribe',
      elevenLabsApiKey: '',
      elevenLabsVoiceId: '',
      elevenLabsModelId: 'eleven_multilingual_v2',
      elevenLabsStability: 0.4,
      elevenLabsStyle: 0.3,
      localTtsBaseUrl: 'http://127.0.0.1:8080',
      localTtsVoice: '',
      localTtsEmotion: '',
    },
    remoteControl: {
      enabled: false,
      serverName: 'TensorFold Orchestrator',
      bindHost: '127.0.0.1',
      port: 8765,
      authToken: makeRemoteControlToken(),
      secretRotations: { authToken: Date.now() },
      allowLan: false,
      headlessBootstrappedAt: null,
      lastStartedAt: null,
      lastClientAt: null,
      lastError: null,
    },
    remoteClient: {
      enabled: false,
      serverUrl: '',
      authToken: '',
      lastConnectedAt: null,
      lastError: null,
    },
    remoteOrchestrators: [],
    preset: 'tensorfold',
    light: PRESETS.tensorfold.light,
    dark: PRESETS.tensorfold.dark,
    pointerCursors: true,
    doomTodos: true,
    notifyRuns: true,
    sshHosts: [],
    trustedSshHostKeys: [],
    showGitInitPrompt: true,
    personas: [],
    userProfile: '',
  }
}

// Reconcile the saved provider list with the built-in defaults on load.
// First run (no saved list at all) → seed the defaults. Once a list has been saved it is
// AUTHORITATIVE: the user's deletions stick (the old code seeded a Map with every default then
// overlaid the saved list — a union by id — so a deleted default like z.ai/openrouter silently
// came back on the next launch). For providers the user KEPT we still backfill any fields a
// default defines, so a field added to a default in a later build appears without clobbering the
// user's own values (their saved fields win via the `...p` spread).
export function seedProviders(defaults: Provider[], loaded: Provider[] | undefined | null): Provider[] {
  if (!loaded) return defaults
  const byId = new Map(defaults.map((p) => [p.id, p]))
  return loaded.map((p) => {
    const shipped = byId.get(p.id)
    if (!shipped) return p
    // Saved configuration remains authoritative, but `models` is a shipped catalogue,
    // not a user-deletion ledger. Union the latest defaults with saved/custom entries so
    // an upgrade can add a model without stale settings hiding it forever.
    const models = [...new Set([...(shipped.models ?? []), ...(p.models ?? [])])]
    return { ...shipped, ...p, models }
  })
}

/** Security migrations are required writes: callers must observe a persistence failure. */
export async function persistSettingsMigration(required: boolean, persist: () => Promise<void>): Promise<void> {
  if (required) await persist()
}
