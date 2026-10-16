import { streamClaudeCli } from './claude-cli'
import { streamCodexCli } from './codex-cli'
import { streamCursorAgent } from './cursor-agent-cli'
import { streamGeminiCli } from './gemini-cli'
import { streamPiCli } from './pi-cli'
import { planningGuard } from './planning-guard'
import type { ApiMessage, Provider } from './types'

// CLI providers as room seats: a seat pinned to a CLI-backed provider runs its turn as a
// one-shot CLI invocation (this module) instead of streamChat's HTTP streaming call.
// OpenCode/Kilo are server protocols with no read-only prompt bridge (streamChat refuses
// them in readonly mode), so they stay on the HTTP path where the refusal surfaces.

const CLI_SEAT_TRANSPORTS: ReadonlySet<Provider['transport']> = new Set([
  'codex-cli',
  'claude-cli',
  'pi-cli',
  'cursor-cli',
  'gemini-cli',
])

export function isCliSeatProvider(provider: Provider | null | undefined): boolean {
  return !!provider && CLI_SEAT_TRANSPORTS.has(provider.transport)
}

export interface CliSeatTurnParams {
  provider: Provider
  model: string
  // The seat's room context: system primer (with the ACTION protocol), rendered
  // transcript, and the moderator's turn instruction. Built by rooms.ts.
  messages: ApiMessage[]
  cwd?: string | null
  reasoningEffort?: string
  signal?: AbortSignal
}

// The transports yield content/reasoning text chunks plus transport-specific cursor chunks.
export type CliSeatChunk = { type: string; text?: string }
export type CliSeatSource = (params: CliSeatTurnParams) => AsyncGenerator<CliSeatChunk>

// One-shot, non-agentic dispatch: no resume cursor and no steering channel, so every turn
// is a fresh single prompt (rooms re-flatten the full transcript into messages each turn).
// approvalMode 'readonly' pins each CLI to its most confined non-interactive mode:
// claude --permission-mode plan, codex -s read-only, pi --tools read, cursor --mode=ask,
// gemini --approval-mode plan. claude plan mode still allows read tools; the room tool
// policy (ACTION: TOOL ...) remains the only mutating surface, and it hard-blocks writes.
const rawCliSeatSource: CliSeatSource = (params) => {
  const base = {
    provider: params.provider,
    model: params.model,
    messages: params.messages,
    cwd: params.cwd,
    approvalMode: 'readonly' as const,
    signal: params.signal,
  }
  switch (params.provider.transport) {
    case 'claude-cli': return streamClaudeCli({ ...base, reasoningEffort: params.reasoningEffort })
    case 'codex-cli': return streamCodexCli({ ...base, reasoningEffort: params.reasoningEffort })
    case 'pi-cli': return streamPiCli({ ...base, reasoningEffort: params.reasoningEffort })
    case 'cursor-cli': return streamCursorAgent(base)
    case 'gemini-cli': return streamGeminiCli(base)
    default: throw new Error(`Provider "${params.provider.name}" is not a CLI seat transport.`)
  }
}

// Seats run on external CLIs, so every seat turn goes through the planning guard: redacted
// transcript in, empty planning folder and seatbelt for the CLI, placeholders rehydrated out.
export const defaultCliSeatSource: CliSeatSource = (params) => planningGuard(params, rawCliSeatSource) ?? rawCliSeatSource(params)

// Run one CLI seat turn: accumulate content text (what gateAction parses) and forward
// deltas to onDelta for the live room stream. Reasoning/status lines and resume cursors
// are dropped; abort/timeout signaling kills the child process group inside the transport.
export async function runCliSeatTurn(
  params: CliSeatTurnParams,
  onDelta: (text: string) => void,
  source: CliSeatSource = defaultCliSeatSource,
): Promise<string> {
  let reply = ''
  for await (const chunk of source(params)) {
    if (chunk.type === 'content' && chunk.text) {
      reply += chunk.text
      onDelta(chunk.text)
    }
  }
  return reply
}
