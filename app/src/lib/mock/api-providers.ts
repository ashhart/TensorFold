import { modelForReadiness } from '../../../shared/model-readiness'
import { mockState } from './state'
import { CODEX_CLI_MODELS, CLAUDE_CLI_MODELS } from '../cli-models'
import { createMockWorktreeApi } from './worktrees'
import type { TensorFoldStudioApi } from '../ipc'
import { createMockEngineApi } from './engine'

export function createMockProvidersApi(mockEngineReadiness: ReturnType<typeof createMockEngineApi>['mockEngineReadiness']): Pick<TensorFoldStudioApi, 'getProviderStatus' | 'onProviderStatus' | 'onUsageCleared' | 'testProviderResponse' | 'fetchModels' | 'getModelReadiness' | 'loginCliProvider' | 'cancelCliLogin' | 'checkCliTransport' | 'checkoutBranch' | 'startLoop' | 'listLoops' | 'stopLoop' | 'onLoopUpdate' | 'onOrchestrationEvent' | 'discoverLocalProvider' | 'discoverOmlxProvider' | 'synthesizeVoice' | 'synthesizeLocalVoice' | 'getKokoroStatus' | 'installKokoro' | 'cancelKokoroInstall' | 'removeKokoro' | 'synthesizeKokoroVoice' | 'cancelKokoroVoice' | 'transcribeVoice' | 'discoverLocalTts'> & ReturnType<typeof createMockWorktreeApi> {

  return {

    getProviderStatus: async () => {
      const now = Date.now()
      const codex = mockState.settings.providers.find((p) => p.transport === 'codex-cli')
      const accountId = codex ? (mockState.settings.activeProviderAccountId ?? codex.defaultAccountId ?? codex.accounts?.[0]?.id ?? null) : null
      const key = `${codex?.id ?? 'codex-cli'}:${accountId ?? 'default'}`
      return {
        version: 1 as const,
        statuses: {
          [key]: {
            providerId: codex?.id ?? 'codex-cli',
            accountId,
            raw: {
              primary: { usedPercent: 62, resetsAt: now + 90 * 60_000, windowDurationMins: 300 },
              secondary: { usedPercent: 34, resetsAt: now + 4 * 24 * 60 * 60_000, windowDurationMins: 7 * 24 * 60 },
              planType: 'plus',
              limitName: 'ChatGPT Plus',
            },
            updatedAt: now,
          },
        },
      }
    },
    onProviderStatus: () => () => { },
    onUsageCleared: () => () => { },
    testProviderResponse: async () => { throw new Error('Response testing runs in the desktop app; this browser preview does not contact model endpoints or measure latency.') },
    fetchModels: async (provider) => {
      await new Promise((r) => setTimeout(r, 700))
      if (provider.transport === 'codex-cli') return provider.models?.length ? provider.models : CODEX_CLI_MODELS
      if (provider.transport === 'claude-cli') return provider.models?.length ? provider.models : CLAUDE_CLI_MODELS
      const baseUrl = provider.baseUrl
      if (provider.source === 'openrouter' || baseUrl.includes('openrouter.ai')) return ['openrouter/free', 'openrouter/auto', 'openrouter/fusion', 'z-ai/glm-5.2', 'nvidia/nemotron-3-ultra-550b-a55b:free', 'nvidia/nemotron-3-super-120b-a12b:free', 'qwen/qwen3-coder:free', 'openai/gpt-oss-120b:free']
      if (provider.source === 'ai-gateway' || baseUrl.includes('ai-gateway.vercel.sh')) return ['openai/gpt-5.4', 'openai/gpt-5.5', 'openai/gpt-4.1', 'openai/o4-mini', 'anthropic/claude-sonnet-4.6', 'anthropic/claude-opus-4.1', 'xai/grok-4.1-fast-reasoning', 'google/gemini-2.5-pro']
      if (baseUrl.includes('9000')) return ['Qwen3-Coder-Next-MLX-8bit', 'Qwen2.5-Coder-14B-Instruct-MLX-8bit']
      if (baseUrl.includes('11434')) return ['qwen2.5-coder:14b', 'llama3.1:8b', 'deepseek-r1:7b', 'mistral-nemo']
      if (baseUrl.includes('1234')) return ['local-model', 'qwen2.5-coder-7b-instruct']
      if (baseUrl.includes('openai')) return ['gpt-4o', 'gpt-4o-mini', 'o3', 'o4-mini']
      if (baseUrl.includes('z.ai')) return ['glm-5.2', 'glm-5.1', 'glm-4.6', 'glm-4-flash', 'glm-4-air']
      return ['model-a', 'model-b']
    },
    getModelReadiness: async (input) => {
      await new Promise((r) => setTimeout(r, 120))
      const providerId = input?.providerId ?? mockState.settings.activeProviderId
      const provider = mockState.settings.providers.find((p) => p.id === providerId)
      if (!provider) return { ready: false, reason: 'no-provider', message: 'Please go to Settings and connect a cloud or local model.' }
      if (provider.source === 'tensorfold') return mockEngineReadiness(modelForReadiness(input, mockState.settings.activeProviderId, mockState.settings.activeModel))
      const name = provider.name || 'This provider'

      if (provider.transport === 'codex-cli' || provider.transport === 'claude-cli') {
        return { ready: true, reason: 'ok', message: `${name} demo-ready.` }
      }
      const isLocal = provider.kind === 'local' || ['ollama', 'lmstudio', 'omlx'].includes(provider.source || '')
      if (!provider.baseUrl?.trim()) return { ready: false, reason: 'needs-endpoint', message: 'Please go to Settings and connect a cloud or local model.' }
      if (!isLocal && !provider.apiKey?.trim()) return { ready: true, reason: 'ok', message: `${name} demo-ready.` }
      return { ready: true, reason: 'ok', message: '' }
    },
    loginCliProvider: async () => {
      await new Promise((r) => setTimeout(r, 300))
      return { ok: false, loggedIn: false, message: 'CLI sign-in runs in the TensorFold desktop app.' }
    },
    cancelCliLogin: async () => ({ cancelled: true }),
    checkCliTransport: async () => ({ loggedIn: false, binaryPresent: false, message: 'CLI auth check unavailable in browser demo mode.' }),
    ...createMockWorktreeApi(),
    checkoutBranch: async () => ({ ok: true }),
    startLoop: async (input) => ({ id: 'mock-loop', source: input.source, surfaceId: input.surfaceId, goal: input.goal, mode: input.mode, status: 'running', cwd: '/Users/example/project', branch: 'vontra/loop-mock', phase: 'dev', cycle: 1, maxCycles: input.mode === 'ship' ? 7 : 5, checkpoints: 0, lastVerdict: null, blockers: [], log: ['Loop started (mock)'], roleThreadIds: [], createdAt: Date.now(), updatedAt: Date.now() }),
    listLoops: async () => [],
    stopLoop: async () => ({ stopped: true }),
    onLoopUpdate: () => () => { },

    onOrchestrationEvent: () => () => { },
    discoverLocalProvider: async (kind) => {
      await new Promise((r) => setTimeout(r, 500))
      if (kind === 'ollama') return {
        kind,
        found: true,
        name: 'Ollama (local)',
        source: 'ollama',
        baseUrl: 'http://127.0.0.1:11434/v1',
        apiKey: 'ollama',
        models: ['qwen2.5-coder:14b', 'llama3.1:8b', 'deepseek-r1:7b'],
        defaultModel: 'qwen2.5-coder:14b',
        configPath: null,
        message: 'Demo Ollama server discovered.',
      }
      if (kind === 'lmstudio') return {
        kind,
        found: true,
        name: 'LM Studio (local)',
        source: 'lmstudio',
        baseUrl: 'http://127.0.0.1:1234/v1',
        apiKey: 'lm-studio',
        models: ['local-model', 'qwen2.5-coder-7b-instruct'],
        defaultModel: 'local-model',
        configPath: null,
        message: 'Demo LM Studio server discovered.',
      }
      return {
        kind,
        found: true,
        name: 'oMLX (local)',
        source: 'omlx',
        baseUrl: 'http://127.0.0.1:9000/v1',
        apiKey: 'demo-key',
        models: ['Qwen3-Coder-Next-MLX-8bit', 'Qwen2.5-Coder-14B-Instruct-MLX-8bit'],
        defaultModel: 'Qwen3-Coder-Next-MLX-8bit',
        configPath: '~/.omlx/settings.json',
        message: 'Demo oMLX config discovered.',
      }
    },
    discoverOmlxProvider: async () => ({
      found: true,
      baseUrl: 'http://127.0.0.1:9000/v1',
      apiKey: 'demo-key',
      models: ['Qwen3-Coder-Next-MLX-8bit', 'Qwen2.5-Coder-14B-Instruct-MLX-8bit'],
      defaultModel: 'Qwen3-Coder-Next-MLX-8bit',
      configPath: '~/.omlx/settings.json',
      message: 'Demo oMLX config discovered.',
    }),

    synthesizeVoice: async () => ({ ok: false, error: 'Provider speech synthesis runs in the desktop app. Falling back to the system voice.', fallback: true }),
    synthesizeLocalVoice: async () => ({ ok: false, error: 'Local TTS runs in the desktop app. Falling back to the system voice.', fallback: true }),
    getKokoroStatus: async () => ({ state: 'missing', installed: false, downloadedBytes: 0, totalBytes: 92364770, message: 'Open the desktop app to download Kokoro and read replies offline.', voices: [{ id: 'af_heart', name: 'Heart', language: 'en-US' }, { id: 'af_bella', name: 'Bella', language: 'en-US' }], modelRevision: '1939ad2a8e416c0acfeecc08a694d14ef25f2231' }),
    installKokoro: async () => ({ state: 'error', installed: false, downloadedBytes: 0, totalBytes: 92364770, message: 'Kokoro downloads run in the desktop app.', error: 'Open the TensorFold desktop app to install Kokoro.', voices: [{ id: 'af_heart', name: 'Heart', language: 'en-US' }], modelRevision: '1939ad2a8e416c0acfeecc08a694d14ef25f2231' }),
    cancelKokoroInstall: async () => ({ state: 'missing', installed: false, downloadedBytes: 0, totalBytes: 92364770, message: 'No Kokoro download is running in this preview.', voices: [{ id: 'af_heart', name: 'Heart', language: 'en-US' }], modelRevision: '1939ad2a8e416c0acfeecc08a694d14ef25f2231' }),
    removeKokoro: async () => ({ state: 'missing', installed: false, downloadedBytes: 0, totalBytes: 92364770, message: 'Kokoro is not installed in this preview.', voices: [{ id: 'af_heart', name: 'Heart', language: 'en-US' }], modelRevision: '1939ad2a8e416c0acfeecc08a694d14ef25f2231' }),
    synthesizeKokoroVoice: async () => ({ ok: false, error: 'Kokoro playback runs in the TensorFold desktop app.' }),
    cancelKokoroVoice: async () => ({ cancelled: false }),
    transcribeVoice: async () => ({ ok: false, error: 'API speech transcription runs in the desktop app.' }),
    discoverLocalTts: async (baseUrl) => ({ found: false, baseUrl: baseUrl || 'http://127.0.0.1:8080', voices: [], message: 'Local TTS discovery runs in the desktop app.' }),
  }
}
