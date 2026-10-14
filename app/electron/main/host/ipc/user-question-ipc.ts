import { handle } from '../ipc'
import { requireSecurityRole } from '../audit'
import type { UserQuestionHost } from '../user-question-host'
import type { UserQuestionAnswer } from '../../../agent/types'
export function registerUserQuestionIpc(host: UserQuestionHost, options: { reserveOperation?: () => { release(): void } } = {}) {
  handle('userQuestion:answer', async (_event, input: UserQuestionAnswer) => {
    requireSecurityRole('operator', 'answer an active chat question')
    const lease = options.reserveOperation?.()
    try { return await host.answer(input) } finally { lease?.release() }
  })
  handle('userQuestion:state', async (_event, input: { threadId?: unknown; questionId?: unknown }) => {
    requireSecurityRole('operator', 'read an active chat question')
    if (typeof input?.threadId !== 'string' || input.threadId.length > 120 || typeof input.questionId !== 'string' || input.questionId.length > 120) throw new Error('Choose a question in this chat.')
    const lease = options.reserveOperation?.()
    try { return await host.state(input.threadId, input.questionId) } finally { lease?.release() }
  })
}
