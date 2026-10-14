import type { AgentPreset, ApprovalMode, BotThreadConfig, Provider, Thread } from '../electron/agent/types'
import { isExternalProvider } from './provider-trust'

const lanes = new Set(['planning', 'development', 'review', 'research', 'operations'])
const apiTransports = new Set<Provider['transport']>([undefined, 'chat-completions'])

export function botProviderSupported(provider: Provider): boolean {
  // Third-party CLIs own their tools and cannot enforce a bot's execution boundary.
  return apiTransports.has(provider.transport)
}

export function normalizeAgentPresets(value: unknown, strict = false): AgentPreset[] {
  if (!Array.isArray(value)) {
    if (strict) throw new Error('Bots must be a list.')
    return []
  }
  if (value.length > 200 && strict) throw new Error('Keep at most 200 bots.')
  const seen = new Set<string>()
  const result: AgentPreset[] = []
  for (const item of value.slice(0, 200)) {
    if (!item || typeof item !== 'object' || Array.isArray(item)) { if (strict) throw new Error('Invalid bot.'); continue }
    const raw = item as Record<string, unknown>
    const valid = typeof raw.id === 'string' && /^[a-zA-Z0-9_-]{1,100}$/.test(raw.id) && !seen.has(raw.id)
      && typeof raw.name === 'string' && typeof raw.role === 'string' && typeof raw.providerId === 'string'
      && typeof raw.model === 'string' && lanes.has(String(raw.lane))
      && (raw.providerAccountId == null || typeof raw.providerAccountId === 'string')
      && (raw.instructions === undefined || typeof raw.instructions === 'string')
      && (raw.approvalScope === undefined || raw.approvalScope === 'readonly' || raw.approvalScope === 'review')
      && (raw.executionTarget === undefined || raw.executionTarget === 'workspace' || raw.executionTarget === 'vm')
      && (raw.paused === undefined || typeof raw.paused === 'boolean')
      && (raw.archivedAt == null || typeof raw.archivedAt === 'number' && Number.isFinite(raw.archivedAt) && raw.archivedAt > 0)
    if (!valid) { if (strict) throw new Error('Invalid bot configuration.'); continue }
    const preset = raw as unknown as AgentPreset
    seen.add(preset.id)
    result.push({
      id: preset.id, name: preset.name.slice(0, 80), role: preset.role.slice(0, 240), lane: preset.lane,
      providerId: preset.providerId.slice(0, 200), providerAccountId: preset.providerAccountId?.slice(0, 200) ?? null,
      model: preset.model.slice(0, 300), instructions: preset.instructions?.slice(0, 16_000) ?? '',
      approvalScope: preset.approvalScope ?? 'readonly', executionTarget: preset.executionTarget ?? 'workspace',
      paused: preset.paused === true, archivedAt: preset.archivedAt ?? null,
      createdAt: Number.isFinite(preset.createdAt) ? preset.createdAt : 0,
      updatedAt: Number.isFinite(preset.updatedAt) ? preset.updatedAt : 0,
    })
  }
  return result
}

export function captureBotConfig(preset: AgentPreset, providers: Provider[], now = Date.now()): BotThreadConfig {
  const normalized = normalizeAgentPresets([preset], true)[0]
  if (normalized.archivedAt) throw new Error('Restore this bot before starting a chat.')
  if (normalized.paused) throw new Error('Resume this bot before starting a chat.')
  if (!normalized.name.trim()) throw new Error('Give this bot a name.')
  const snapshot: BotThreadConfig = {
    presetId: normalized.id, name: normalized.name.trim(), role: normalized.role, instructions: normalized.instructions ?? '',
    providerId: normalized.providerId, providerAccountId: normalized.providerAccountId ?? null, model: normalized.model,
    approvalScope: normalized.approvalScope ?? 'readonly', executionTarget: normalized.executionTarget ?? 'workspace', capturedAt: now,
  }
  assertBotModel(snapshot, providers)
  return snapshot
}

export function assertBotModel(bot: BotThreadConfig, providers: Provider[]): Provider {
  const provider = providers.find((candidate) => candidate.id === bot.providerId)
  if (!provider) throw new Error('This bot\'s model source is missing; edit the bot and start a new chat.')
  if (!botProviderSupported(provider)) throw new Error('Bots require an API model source so TensorFold can enforce their permissions.')
  if (bot.executionTarget === 'vm' && isExternalProvider(provider)) throw new Error('Bot VMs require a model running on this Mac; choose the TensorFold engine or a local API model.')
  const account = bot.providerAccountId ? provider.accounts?.find((candidate) => candidate.id === bot.providerAccountId && candidate.enabled !== false) : null
  if (bot.providerAccountId && !account) throw new Error('This bot\'s model account is unavailable; edit the bot and start a new chat.')
  const models = account?.models?.length ? account.models : provider.models
  if (!bot.model || !models.includes(bot.model)) throw new Error('This bot\'s saved model is unavailable; edit the bot and start a new chat.')
  return provider
}

export function botRunApprovalMode(thread: Pick<Thread, 'bot'>, presets: AgentPreset[], providers: Provider[], fallback: ApprovalMode): ApprovalMode {
  const bot = thread.bot
  if (!bot) return fallback
  if (typeof bot.presetId !== 'string' || !/^[a-zA-Z0-9_-]{1,100}$/.test(bot.presetId) || typeof bot.name !== 'string' || !bot.name.trim() || bot.name.length > 80 || typeof bot.instructions !== 'string' || bot.instructions.length > 16_000 || typeof bot.role !== 'string' || bot.role.length > 240 || typeof bot.providerId !== 'string' || typeof bot.model !== 'string' || (bot.providerAccountId !== null && typeof bot.providerAccountId !== 'string')) throw new Error('This chat has an invalid bot configuration.')
  const current = presets.find((preset) => preset.id === bot.presetId)
  if (!current || current.archivedAt) throw new Error('This bot is archived; restore it before continuing its chat.')
  if (current.paused) throw new Error('This bot is paused; resume it before continuing its chat.')
  if (!['workspace', 'vm'].includes(bot.executionTarget) || !['readonly', 'review'].includes(bot.approvalScope)) throw new Error('This chat has an invalid bot configuration.')
  assertBotModel(bot, providers)
  return bot.approvalScope === 'readonly' ? 'readonly' : 'suggest'
}

export function botInstructions(bot?: BotThreadConfig): string {
  return bot ? `BOT IDENTITY\nYou are ${bot.name}.\nResponsibility: ${bot.role}\n${bot.instructions}\n\nThis chat uses the bot configuration captured when it was created. Ask the user before changing scope; instructions do not grant tool permissions.` : ''
}
