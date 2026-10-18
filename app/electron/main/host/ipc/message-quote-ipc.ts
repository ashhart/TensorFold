import type { PrepareMessageQuote } from '../../../../shared/message-quotes'
import { prepareMessageQuote } from '../../../agent/conversation-quotation'
import { vontraHomeDir } from '../../../store'
import { store } from '../../state'
import { requireSecurityRole } from '../audit'
import { handle } from '../ipc'

export function registerMessageQuoteIpc(options: { reserveOperation(): { release(): void } }): void {
  handle('threads:prepareMessageQuote', async (_event, input: PrepareMessageQuote) => {
    requireSecurityRole('operator', 'quote a visible conversation message')
    const profile = vontraHomeDir(), lease = options.reserveOperation()
    try {
      const thread = store.threads.find(row => row.id === input?.threadId)
      if (!thread || thread.archivedAt) throw new Error('Open an active conversation before quoting a message.')
      const provider = store.settings.providers.find(row => row.id === thread.providerId)
      if (thread.engine === 'omp' || provider?.transport && provider.transport !== 'chat-completions') throw new Error('Quoted replies need a model using the TensorFold chat API.')
      const quote = prepareMessageQuote(thread, input)
      if (vontraHomeDir() !== profile || !store.threads.includes(thread)) throw new Error('The conversation profile changed; try again.')
      return quote
    } finally { lease.release() }
  })
}
