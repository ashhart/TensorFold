import type { CliSession } from '../../agent/types'

const sessions = new Map<string, CliSession>()

export function registerLiveTerminalSession(session: CliSession): void {
  sessions.set(session.id, session)
}

export function removeLiveTerminalSession(sessionId: string): void {
  sessions.delete(sessionId)
}

export function clearLiveTerminalSessions(): void {
  sessions.clear()
}

export function liveTerminalActivity(): { total: number; running: number; attention: number } {
  const live = [...sessions.values()].filter((session) => session.status !== 'done')
  return {
    total: live.length,
    running: live.filter((session) => session.status === 'busy').length,
    attention: live.filter((session) => session.status === 'attention').length,
  }
}
