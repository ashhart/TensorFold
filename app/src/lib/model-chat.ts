import { api } from './ipc'
import { useStore } from './store'
import { resolveProviderModel } from './provider-accounts'

interface Selection { providerId: string; model: string; providerAccountId?: string | null }

// Explicit model choices configure the new empty chat before opening it, independent of surface defaults.
export async function openModelChat(selection: Selection, canOpen: () => boolean = () => true): Promise<void> {
  if (!canOpen()) return
  const provider = useStore.getState().settings?.providers.find(item => item.id === selection.providerId)
  if (!provider) throw new Error('The selected model provider is no longer configured.')
  const accountId = selection.providerAccountId ?? null
  const model = resolveProviderModel(provider, accountId, selection.model) ?? selection.model
  const thread = await api.createThread('', 'chat', null, crypto.randomUUID())
  if (!canOpen()) return
  if (thread.kind !== 'chat' || thread.orchestratorHostId || thread.incognito || thread.timeline.length || thread.apiMessages.length) {
    throw new Error('The new local chat changed before its model could be selected.')
  }
  await api.configureThread(thread.id, provider.id, model, accountId)
  if (!canOpen()) return
  const configured = { ...thread, providerId: provider.id, providerAccountId: accountId, model }
  useStore.setState(state => ({
    threads: [configured, ...state.threads.filter(item => item.id !== thread.id)],
    view: { name: 'thread', id: thread.id },
  }))
}
