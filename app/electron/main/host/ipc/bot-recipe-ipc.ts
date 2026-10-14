import { BrowserWindow, dialog } from 'electron'
import { promises as fs } from 'node:fs'
import type { Settings } from '../../../agent/types'
import { previewBotRecipe, recipeObject } from '../../../../shared/bot-recipe'
import { exportRecipePreview, importReviewedRecipe, reviewedRecipeExport } from '../bot-recipes'
import { handle } from '../ipc'
import { requireSecurityRole } from '../audit'

export interface BotRecipeIpcDeps {
  currentSettings(): Settings
  commitImport(prepare: (current: Settings) => Promise<Settings>): Promise<Settings>
}
export function registerBotRecipeIpc(deps: BotRecipeIpcDeps) {
  handle('botRecipe:preview', async (_event, input: unknown) => {
    requireSecurityRole('owner', 'review portable bot recipes')
    const raw = recipeObject(input, ['kind', 'value'])
    if (raw.kind === 'import') return previewBotRecipe(raw.value)
    if (raw.kind === 'export') return exportRecipePreview(deps.currentSettings(), raw.value)
    throw new Error('Choose import or export.')
  })
  handle('botRecipe:import', async (_event, input: unknown) => {
    requireSecurityRole('owner', 'import a portable bot recipe')
    return importReviewedRecipe(input, deps.commitImport)
  })
  handle('botRecipe:export', async (event, input: unknown) => {
    requireSecurityRole('owner', 'save a portable bot recipe')
    reviewedRecipeExport(deps.currentSettings(), input)
    const parent = BrowserWindow.fromWebContents(event.sender)
    const options = { title: 'Save the reviewed bot recipe', defaultPath: 'TensorFold-bot-recipe.json', filters: [{ name: 'Bot recipe JSON', extensions: ['json'] }] }
    const selected = parent ? await dialog.showSaveDialog(parent, options) : await dialog.showSaveDialog(options)
    if (selected.canceled || !selected.filePath) return { saved: false }
    // The dialog can remain open while settings change. Never export unseen edits.
    const preview = reviewedRecipeExport(deps.currentSettings(), input)
    try { await fs.writeFile(selected.filePath, preview.text, { encoding: 'utf8', flag: 'wx', mode: 0o600 }) }
    catch (cause) {
      if ((cause as NodeJS.ErrnoException)?.code === 'EEXIST') throw new Error('Choose a new filename. The existing file was not changed.')
      throw new Error('Could not save this recipe. Choose a writable location and try again.')
    }
    return { saved: true }
  })
}
