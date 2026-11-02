import type { AskSavedState } from './session'
import { boundedText, validateAskDraft, validateAskAttachments, ASK_HISTORY_LIMIT, ASK_OUTPUT_LIMIT } from './validation'
import { validateAskGeneration } from './generate'

export function validateAskSavedState(state: AskSavedState): AskSavedState {
  if (!state || !state.session || JSON.stringify(state).length > 16_000_000) throw new Error('The saved ask session is invalid or too large.')
  const session = state.session
  if (!/^[a-f0-9-]{36}$/.test(session.id)) throw new Error('The saved ask session ID is invalid.')
  validateAskDraft(session.draft)
  if (!Array.isArray(session.turns) || session.turns.length > ASK_HISTORY_LIMIT) throw new Error('The saved ask history is invalid.')
  if (session.prompt !== null) boundedText(session.prompt, 8000, 'Saved instructions')
  if (session.route) {
    const route = session.route
    boundedText(route.id, 2000, 'Saved model route'); boundedText(route.providerId, 200, 'Saved provider'); boundedText(route.model, 500, 'Saved model')
    if (route.accountId !== null) boundedText(route.accountId, 200, 'Saved account')
    if (!['device', 'remote', 'cloud'].includes(route.destination) || !/^[a-f0-9]{64}$/.test(route.binding)) throw new Error('The saved destination is invalid.')
  }
  for (const turn of session.turns) {
    if (!turn || !/^[a-f0-9-]{36}$/.test(turn.id) || !['running', 'complete', 'cancelled', 'error'].includes(turn.status) || !Number.isFinite(turn.at)) throw new Error('The saved answer is invalid.')
    boundedText(turn.text, 32_000, 'Saved question'); boundedText(turn.answer, ASK_OUTPUT_LIMIT, 'Saved answer'); boundedText(turn.reasoning, ASK_OUTPUT_LIMIT, 'Saved reasoning')
    validateAskAttachments(turn.attachments)
    if (turn.error) boundedText(turn.error, 500, 'Saved error')
    if (!Array.isArray(turn.tools) || turn.tools.length > 6) throw new Error('The saved web history is invalid.')
    for (const tool of turn.tools) if (!['web_search', 'web_fetch'].includes(tool.name) || typeof tool.detail !== 'string' || tool.detail.length > 2000 || !['running', 'complete', 'error'].includes(tool.status)) throw new Error('The saved web activity is invalid.')
  }
  validateAskGeneration({ routeId: session.route?.id ?? 'unselected', binding: session.route?.binding ?? '0'.repeat(64), prompt: session.prompt ?? '', messages: state.messages, web: false })
  if (session.handoffThreadId && session.handoffThreadId !== `ask-${session.id}`) throw new Error('The saved conversation handoff is invalid.')
  return state
}
