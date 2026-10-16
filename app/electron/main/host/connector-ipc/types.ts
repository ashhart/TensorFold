import type { SerialTaskQueue, ConnectorRuntimeCallbacks } from '../connector-runtime'
import type { ConnectorEventSpine } from '../connector-event-spine'
import type { ConnectorEmailTransport } from '../connector-email-transport'
import type { ConnectorChannels } from '../connector-channels'
import type { handle as registerHandler } from '../ipc'

export interface ConnectorIpcDeps {
  spine: ConnectorEventSpine
  emailTransport: ConnectorEmailTransport
  channels: ConnectorChannels
  syncQueue: SerialTaskQueue
  callbacks: ConnectorRuntimeCallbacks
  handle?: typeof registerHandler
}
