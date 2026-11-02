export const FIRST_ANSWER_STARTERS = [
  { id: 'plan', label: 'Plan something small', description: 'Turn an idea into manageable next steps.', prompt: 'Help me plan a small project. Ask me what I want to achieve and how much time I have before suggesting a few practical next steps.' },
  { id: 'write', label: 'Improve a piece of writing', description: 'Start with the purpose and reader.', prompt: 'Help me improve a short piece of writing. Ask me to paste the text and tell you who will read it before suggesting changes.' },
  { id: 'learn', label: 'Understand a topic', description: 'Get an explanation at the right level.', prompt: 'Help me understand a topic. Ask me what I want to learn and what I already know, then explain it with a small example.' },
  { id: 'blank', label: 'Start with an empty draft', description: 'Write your own first question.', prompt: '' },
] as const
export type FirstAnswerStarterId = typeof FIRST_ANSWER_STARTERS[number]['id']
const STARTER_KEY = 'tensorfold.onboarding-task-starter'

export function readFirstAnswerStarter(replaying = false): FirstAnswerStarterId {
  if (replaying) return 'blank'
  try {
    const value = localStorage.getItem(STARTER_KEY)
    return FIRST_ANSWER_STARTERS.find((starter) => starter.id === value)?.id ?? 'plan'
  } catch { return 'plan' }
}
export function saveFirstAnswerStarter(id: FirstAnswerStarterId): void {
  if (!FIRST_ANSWER_STARTERS.some((starter) => starter.id === id)) return
  try { localStorage.setItem(STARTER_KEY, id) } catch {}
}

export interface FirstAnswerReadiness {
  providerKind: 'local' | 'cloud' | null
  builtIn: boolean
  available?: boolean
  selected: boolean
  cached: boolean
  fitsMemory: boolean | null
  configured: boolean
  loading: boolean
  loaded: boolean
}
export function firstAnswerNextAction(state: FirstAnswerReadiness): { action: 'models' | 'provider' | 'draft'; label: string; message: string } {
  if (state.builtIn && state.available === false) return { action: 'models', label: 'Check local engine', message: 'The local engine is unavailable. Open Models to review its status before sending a message.' }
  if (!state.selected) return { action: state.builtIn || state.providerKind === null ? 'models' : 'provider', label: 'Choose a model', message: 'Choose a model before sending your first question.' }
  if (state.builtIn && state.fitsMemory === false) return { action: 'models', label: 'Choose a model that fits', message: 'The selected model exceeds the current memory budget.' }
  if (state.builtIn && !state.cached) return { action: 'models', label: 'Finish model download', message: 'Return to Models to finish or resume the selected download.' }
  if (!state.builtIn && !state.configured) return { action: 'provider', label: 'Finish model connection', message: 'Check the provider connection before sending this draft.' }
  return { action: 'draft', label: 'Open my first draft', message: state.loading ? 'The local model is loading; your editable draft can wait in the chat.' : 'Open a new chat, review the draft and choose Send when you are ready.' }
}
