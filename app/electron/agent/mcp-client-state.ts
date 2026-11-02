import type { ChildProcessWithoutNullStreams } from 'node:child_process'
import type { StringDecoder } from 'node:string_decoder'
import type { McpServerConfig, McpServerRuntime } from './types'
import type { McpIsolation } from './mcp-isolation'

export type JsonRpcId = string | number
interface PendingRequest {
  resolve: (value: any) => void
  reject: (error: Error) => void
  timer: ReturnType<typeof setTimeout>
}
export interface McpClientState {
  config: McpServerConfig
  runtime: McpServerRuntime
  proc: ChildProcessWithoutNullStreams | null
  buffer: string
  decoder: StringDecoder
  pending: Map<JsonRpcId, PendingRequest>
  nextId: number
  generation: number
  connectPromise: Promise<McpServerRuntime> | null
  isolation?: McpIsolation
}
