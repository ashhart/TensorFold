import type { TelegramRemoteConfigInput, TelegramRemotePublicStatus } from '../../../shared/telegram-remote'
import { defaultTelegramRemoteSettings } from '../../../shared/telegram-remote'
import type { Settings, Thread, Room } from '../types'

export function createMockTelegramRemoteApi(settings: () => Settings, threads: () => Thread[], rooms: () => Room[]) {
  const status = (): TelegramRemotePublicStatus => {
    const remote = settings().botControl.telegram.remote ?? defaultTelegramRemoteSettings()
    return {
      enabled: remote.enabled, scope: remote.scope, threadIds: [...remote.threadIds], roomIds: [...remote.roomIds], revision: remote.revision,
      eligibleThreads: threads().filter((thread) => !thread.incognito && !thread.phoneLocalOnly && !thread.googleLocalOnly && !thread.bot && !thread.archivedAt && !thread.orchestratorHostId && thread.memoryPolicy?.useExisting !== false && thread.memoryPolicy?.contribute !== false && (!thread.workspace || thread.workspace.kind === 'local')).map((thread) => ({ id: thread.id, title: thread.title })),
      eligibleRooms: rooms().filter((room) => room.state !== 'archived').map((room) => ({ id: room.id, name: room.name, state: room.state })),
    }
  }
  return {
    verifyTelegram: async () => ({ ok: false, botId: null, username: null, error: 'Live Telegram verification requires the desktop app; browser preview sends no requests.' }),
    listTelegramRemoteTargets: async () => status(),
    configureTelegramRemote: async (input: TelegramRemoteConfigInput) => {
      const current = status()
      if (input.threadIds.some((id) => !current.eligibleThreads.some((thread) => thread.id === id)) || input.roomIds.some((id) => !current.eligibleRooms.some((room) => room.id === id))) throw new Error('A selected chat or room is unavailable.')
      settings().botControl.telegram.remote = { ...structuredClone(input), revision: current.revision + 1, sessions: [] }
      return status()
    },
  }
}
