import assert from 'node:assert/strict'
import { store } from '../state'
import { remoteWorkSnapshot } from './remote-snapshot'
import { clearLiveTerminalSessions, registerLiveTerminalSession } from './terminal-live-state'

const previous = {
  jobs: store.jobs,
  connectorEvents: store.connectorEvents,
  missions: store.missions,
  cliSessions: store.cliSessions,
}
try {
  store.jobs = []
  store.connectorEvents = []
  store.missions = []
  store.cliSessions = [{ id: 'persisted-only', projectId: 'p', cliId: 'shell', title: 'Old snapshot', status: 'busy', createdAt: 1 }]
  clearLiveTerminalSessions()
  registerLiveTerminalSession({ id: 'live-idle', projectId: 'p', cliId: 'shell', title: 'Live PTY', status: 'idle', createdAt: 2 })

  assert.deepEqual(remoteWorkSnapshot(100).terminals, { total: 1, running: 0, attention: 0 }, 'host snapshot ignores persisted-only terminal rows')
} finally {
  clearLiveTerminalSessions()
  store.jobs = previous.jobs
  store.connectorEvents = previous.connectorEvents
  store.missions = previous.missions
  store.cliSessions = previous.cliSessions
}

console.log('remote snapshot live-terminal regression passed')
