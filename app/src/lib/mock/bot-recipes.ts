import { importedRecipeBot, previewBotRecipe, recipeForBot, recipeObject, reviewedRecipeImport } from '../../../shared/bot-recipe'
import type { TensorFoldStudioApi } from '../ipc'
import { mockState } from './state'
import { redactedMockSettings } from './runtime'

export function createMockBotRecipeApi(): Pick<TensorFoldStudioApi, 'botRecipePreview' | 'botRecipeImport' | 'botRecipeExport'> {
  return {
    botRecipePreview: async ({ kind, value }) => {
      if (kind === 'import') return previewBotRecipe(value)
      const preset = mockState.settings.agentPresets.find((bot) => bot.id === value)
      if (!preset) throw new Error('Choose a saved bot to export.')
      return recipeForBot(preset)
    },
    botRecipeImport: async (input) => {
      const { recipe, target } = reviewedRecipeImport(input)
      if (mockState.settings.agentPresets.length >= 200) throw new Error('Keep at most 200 bots.')
      const bot = importedRecipeBot(recipe, target, mockState.settings.providers, `bot_${crypto.randomUUID()}`, Date.now())
      mockState.settings = { ...mockState.settings, agentPresets: [bot, ...mockState.settings.agentPresets] }
      return redactedMockSettings()
    },
    botRecipeExport: async (input) => {
      const raw = recipeObject(input, ['presetId', 'reviewedText', 'reviewed'])
      const preset = mockState.settings.agentPresets.find((bot) => bot.id === raw.presetId)
      if (!preset || raw.reviewed !== true || recipeForBot(preset).text !== raw.reviewedText) throw new Error('Review the current recipe before saving it.')
      throw new Error('Saving recipe files is available in the desktop app. This preview has not been saved.')
    },
  }
}
