import type { CliSessionStatus } from './types/cli'

const BELL = 0x07

// Pure status transition on a chunk of PTY output. The debounce back to 'idle' after silence and
// the 'done' transition on PTY exit are driven by the runtime (timer / exit event), not here.
export function nextStatus(prev: CliSessionStatus, chunk: Uint8Array): CliSessionStatus {
  if (prev === 'done') return 'done'
  if (chunk.includes(BELL)) return 'attention'
  if (prev === 'attention') return 'attention'
  return 'busy'
}
