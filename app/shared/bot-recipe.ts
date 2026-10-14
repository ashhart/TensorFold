import type { AgentPreset, Provider } from '../electron/agent/types'
import { captureBotConfig } from './bot-policy'
import { isExternalProvider } from './provider-trust'

export const BOT_RECIPE_MAX_BYTES = 64 * 1024
export interface BotRecipe {
  format: 'tensorfold.bot-recipe'
  version: 1
  bot: { name: string; role: string; instructions: string; lane: AgentPreset['lane'] }
}
export interface BotRecipePreview { recipe: BotRecipe; text: string }
export interface BotRecipeTarget {
  providerId: string
  providerAccountId: string | null
  model: string
  workspace: 'unassigned'
}
export interface BotRecipeImportInput { text: string; target: BotRecipeTarget; reviewed: true }
export interface BotRecipeExportInput { presetId: string; reviewedText: string; reviewed: true }

const lanes = new Set(['planning', 'development', 'review', 'research', 'operations'])
export function recipeObject(value: unknown, keys: readonly string[], message = 'Invalid recipe fields.'): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value) || ![Object.prototype, null].includes(Object.getPrototypeOf(value))) throw new Error(message)
  const raw = value as Record<string, unknown>
  if (Object.keys(raw).length !== keys.length || keys.some((key) => !Object.prototype.hasOwnProperty.call(raw, key)) || Object.keys(raw).some((key) => !keys.includes(key))) throw new Error(message)
  return raw
}
function textField(value: unknown, max: number, label: string, required = false): string {
  if (typeof value !== 'string' || value.length > max || /[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/.test(value) || (required && !value.trim())) throw new Error(`Check the recipe ${label}.`)
  return value
}
function assertNoKnownSensitiveText(text: string): void {
  // A small, disclosed guard for common mistakes, not a general PII/secret detector.
  if (/\/(?:Users|home)\/[^\s/\\]+/i.test(text) || /[A-Za-z]:[\\/]Users[\\/]/i.test(text)
    || /-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----/.test(text)
    || /\b(?:sk-(?:proj-)?[a-zA-Z0-9_-]{24,}|gh[pousr]_[a-zA-Z0-9]{30,}|AKIA[A-Z0-9]{16})\b/.test(text)
    || /\b(?:api[_-]?key|access[_-]?token|password)\s*[:=]\s*["']?[^\s"']{12,}/i.test(text)
    || /\bAuthorization\s*:\s*Bearer\s+\S{12,}/i.test(text)) throw new Error('The recipe contains a common credential or private home-folder path pattern. Remove it before sharing or importing.')
}
export function parseBotRecipe(text: unknown): BotRecipe {
  if (typeof text !== 'string' || new TextEncoder().encode(text).byteLength > BOT_RECIPE_MAX_BYTES) throw new Error('Recipes must be JSON text at most 64 KiB.')
  let value: unknown
  try { value = JSON.parse(text) } catch { throw new Error('The recipe is not valid JSON.') }
  const raw = recipeObject(value, ['format', 'version', 'bot'])
  if (raw.format !== 'tensorfold.bot-recipe' || raw.version !== 1) throw new Error('This recipe format or version is not supported. TensorFold accepts version 1.')
  const bot = recipeObject(raw.bot, ['name', 'role', 'instructions', 'lane'])
  const name = textField(bot.name, 80, 'name', true).trim()
  if (/[/\\\r\n]/.test(name) || name.includes('..')) throw new Error('Use a bot name without path separators or traversal segments.')
  const role = textField(bot.role, 240, 'responsibility')
  const instructions = textField(bot.instructions, 16_000, 'instructions')
  if (typeof bot.lane !== 'string' || !lanes.has(bot.lane)) throw new Error('Choose a supported recipe responsibility type.')
  assertNoKnownSensitiveText([name, role, instructions].join('\n'))
  return { format: 'tensorfold.bot-recipe', version: 1, bot: { name, role, instructions, lane: bot.lane as AgentPreset['lane'] } }
}
export function serializeBotRecipe(recipe: BotRecipe): string {
  const text = JSON.stringify(recipe, null, 2) + '\n'
  return JSON.stringify(parseBotRecipe(text), null, 2) + '\n'
}
export function previewBotRecipe(text: unknown): BotRecipePreview {
  const recipe = parseBotRecipe(text)
  return { recipe, text: serializeBotRecipe(recipe) }
}
export function recipeForBot(bot: AgentPreset): BotRecipePreview {
  // Construct the whitelist; never serialize the preset/settings object itself.
  return previewBotRecipe(JSON.stringify({ format: 'tensorfold.bot-recipe', version: 1, bot: { name: bot.name, role: bot.role, instructions: bot.instructions ?? '', lane: bot.lane } }))
}
export function validateRecipeTarget(value: unknown): BotRecipeTarget {
  const raw = recipeObject(value, ['providerId', 'providerAccountId', 'model', 'workspace'], 'Choose a local model, account and unassigned workspace for this import.')
  const providerId = textField(raw.providerId, 200, 'model source', true)
  const model = textField(raw.model, 300, 'model', true)
  const providerAccountId = raw.providerAccountId === null ? null : textField(raw.providerAccountId, 200, 'account', true)
  if (raw.workspace !== 'unassigned') throw new Error('Imported recipes start with no project workspace or folder grants.')
  return { providerId, providerAccountId, model, workspace: 'unassigned' }
}
export function reviewedRecipeImport(value: unknown): { recipe: BotRecipe; target: BotRecipeTarget } {
  const raw = recipeObject(value, ['text', 'target', 'reviewed'], 'Review this recipe and choose its local setup before importing.')
  if (raw.reviewed !== true) throw new Error('Review the recipe instructions before importing.')
  return { recipe: parseBotRecipe(raw.text), target: validateRecipeTarget(raw.target) }
}
export function importedRecipeBot(recipe: BotRecipe, target: BotRecipeTarget, providers: Provider[], id: string, now: number): AgentPreset & { toolScope: { mode: 'selected'; names: string[] } } {
  const checked = parseBotRecipe(serializeBotRecipe(recipe));const local = validateRecipeTarget(target)
  if (!/^[a-zA-Z0-9_-]{1,100}$/.test(id) || !Number.isFinite(now)) throw new Error('Could not create a new bot identity.')
  const bot = { id, ...checked.bot, providerId: local.providerId, providerAccountId: local.providerAccountId, model: local.model, executionTarget: 'workspace' as const, approvalScope: 'readonly' as const, paused: true, archivedAt: null, createdAt: now, updatedAt: now, toolScope: { mode: 'selected' as const, names: [] as string[] } }
  const provider = captureBotConfig({ ...bot, paused: false }, providers)
  if (isExternalProvider(providers.find((item) => item.id === provider.providerId))) throw new Error('Choose a model running on this Mac for the imported bot.')
  return bot
}
