import type { Command } from '../../agent/types'
import { store } from '../state'
import { requireSecurityRole } from './audit'
import { handle } from './ipc'
import { persistCommands, persistSkills, persistSuggestions } from './persistence'
import { acceptSuggestion, dismissSuggestion } from '../../agent/suggestions'
import { assertCommandNameNotReserved, assertUniqueConfigName, canonicalizeSuggestedCommand, canonicalizeSuggestedSkill, sanitizeCommandSave } from './public-config-policy'

export function registerCommandsIpc() {
  handle('commands:get', async () => store.commands)

  // Consent-first suggestion spine: list / accept (-> real Skill or Command) / dismiss.
  handle('suggestions:get', async () => store.suggestions)
  handle('suggestions:accept', async (_e, id: string) => {
    requireSecurityRole('operator', 'accept a suggestion')
    const now = Date.now()
    const pending = store.suggestions.find((suggestion) => suggestion.id === id && suggestion.status === 'pending')
    const currentSkill = pending?.kind === 'skill' && pending.targetSkillId ? store.skills.find((skill) => skill.id === pending.targetSkillId) ?? null : null
    const acceptedId = currentSkill?.id ?? `${id}_accepted`
    const res = acceptSuggestion(store.suggestions, id, now, acceptedId, new Set(store.skills.map((skill) => skill.id)))
    // Revalidate at consent time against the CURRENT corpus. The agent-authored suggestion may be
    // stale, over-bound, reserved, or collide with an artifact created after it was filed.
    const command = res.command ? canonicalizeSuggestedCommand(res.command, store.commands, now) : null
    const skill = res.skill ? canonicalizeSuggestedSkill(res.skill, currentSkill, store.skills, now) : null
    // Only now is it safe to mark the suggestion accepted.
    store.suggestions = res.list
    if (command) { store.commands.unshift(command); persistCommands() }
    if (skill) {
      // An accepted improvement carries an existing skill id — overwrite it in place
      // (keeping the original createdAt) instead of appending a duplicate.
      const idx = store.skills.findIndex((s) => s.id === skill.id)
      if (idx >= 0) store.skills[idx] = skill
      else store.skills.unshift(skill)
      persistSkills()
    }
    persistSuggestions()
    store.win?.webContents.send('suggestions:update', store.suggestions)
    return { ok: !!(command || skill), skill, command }
  })
  handle('suggestions:dismiss', async (_e, id: string) => {
    store.suggestions = dismissSuggestion(store.suggestions, id)
    persistSuggestions()
    store.win?.webContents.send('suggestions:update', store.suggestions)
  })
  handle('commands:save', async (_e, requested: Partial<Command>) => {
    requireSecurityRole('operator', 'save commands')
    const c = sanitizeCommandSave(requested)
    if (c.id) {
      const ex = store.commands.find((x) => x.id === c.id)
      if (!ex) throw new Error('Command not found.')
      assertCommandNameNotReserved(c.name ?? ex.name)
      if (c.name !== undefined) assertUniqueConfigName(store.commands, c.name, ex.id, 'Command')
      const { id: _id, ...patch } = c
      Object.assign(ex, patch, { updatedAt: Date.now() })
      persistCommands()
      return ex
    }
    const name = c.name ?? 'command'
    assertUniqueConfigName(store.commands, name, undefined, 'Command')
    const now = Date.now()
    const cmd: Command = { id: 'c_' + Math.random().toString(36).slice(2, 9), name, description: c.description ?? '', prompt: c.prompt ?? '', argHint: c.argHint, createdAt: now, updatedAt: now }
    store.commands.unshift(cmd)
    persistCommands()
    return cmd
  })
  handle('commands:delete', async (_e, id: string) => {
    requireSecurityRole('operator', 'delete commands')
    store.commands = store.commands.filter((c) => c.id !== id)
    persistCommands()
  })
}
