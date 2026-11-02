export interface VerificationDraft {
  commands: string[]
  error: string | null
}

export function parseVerificationDraft(value: string): VerificationDraft {
  const commands = value.split(/\r?\n/).map(command => command.trim()).filter(Boolean)
  if (!commands.length) return { commands, error: 'Add at least one verification command.' }
  if (commands.length > 32) return { commands, error: 'Use at most 32 verification commands.' }
  if (commands.some(command => command.length > 2000 || command.includes('\0'))) {
    return { commands, error: 'Each verification command must be valid text of at most 2,000 characters.' }
  }
  return { commands, error: null }
}
