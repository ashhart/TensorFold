// Terminal stream sanitizer: strips OSC 8 hyperlink sequences (and nothing else) from
// PTY chunks before they reach xterm. xterm's built-in link provider turns
// `\x1b]8;;<url>\x1b\\` into clickable links confirmed via window.confirm with the
// attacker-supplied URL — a compromised repo toolchain or remote orchestrator can print
// convincing in-terminal phishing UI otherwise. Colors, cursor moves, titles, and every
// other escape sequence pass through untouched. Stateful: a sequence split across two
// chunks is held back and stripped once complete.

const OSC_8 = /\x1b\]8;[^\x07\x1b]*(?:\x07|\x1b\\)/g

export function sanitizeTerminalChunk(chunk: string): string {
  return chunk.replace(OSC_8, '')
}

export function createTerminalSanitizer(): (chunk: string) => string {
  let carry = ''
  return (chunk: string): string => {
    const input = carry + chunk
    carry = ''
    let out = input.replace(OSC_8, '')
    // A trailing partial OSC opener (no BEL/ST terminator yet) may continue in the next
    // chunk — hold it back instead of rendering half a sequence.
    const lastOpen = out.lastIndexOf('\x1b]')
    if (lastOpen >= 0) {
      const tail = out.slice(lastOpen)
      if (!/[\x07]|\x1b\\/.test(tail)) {
        carry = tail
        out = out.slice(0, lastOpen)
      }
    }
    return out
  }
}
