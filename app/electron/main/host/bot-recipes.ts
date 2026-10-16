import { randomUUID } from 'node:crypto'
import type { Settings } from '../../agent/types'
import { importedRecipeBot, recipeForBot, recipeObject, reviewedRecipeImport, type BotRecipePreview } from '../../../shared/bot-recipe'

export function exportRecipePreview(settings: Settings, presetId: unknown): BotRecipePreview {
  if (typeof presetId !== 'string' || !/^[a-zA-Z0-9_-]{1,100}$/.test(presetId)) throw new Error('Choose a saved bot to export.')
  const bot = settings.agentPresets.find((item) => item.id === presetId)
  if (!bot) throw new Error('This bot is no longer available.')
  return recipeForBot(bot)
}
export function reviewedRecipeExport(settings: Settings, input: unknown): BotRecipePreview {
  const raw = recipeObject(input, ['presetId', 'reviewedText', 'reviewed'], 'Review this exact recipe before saving it.')
  if (raw.reviewed !== true) throw new Error('Review the recipe instructions before saving it.')
  const preview = exportRecipePreview(settings, raw.presetId)
  if (raw.reviewedText !== preview.text) throw new Error('The bot changed after your review. Preview it again before saving.')
  return preview
}
export async function importReviewedRecipe(
  input: unknown,
  commit: (prepare: (current: Settings) => Promise<Settings>) => Promise<Settings>,
  identity = () => `bot_${randomUUID()}`,
  clock = Date.now,
): Promise<Settings> {
  const { recipe, target } = reviewedRecipeImport(input)
  // Parse before queueing, then bind to the authoritative current model catalogue.
  return commit(async (current) => {
    if (current.agentPresets.length >= 200) throw new Error('Keep at most 200 bots. Archive/export and remove an unused bot first.')
    const bot = importedRecipeBot(recipe, target, current.providers, identity(), clock())
    if (current.agentPresets.some((item) => item.id === bot.id)) throw new Error('Could not create a unique bot identity. Retry the import.')
    return { ...current, agentPresets: [bot, ...current.agentPresets] }
  })
}
