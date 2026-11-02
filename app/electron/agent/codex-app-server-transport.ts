import { spawnCliProcess as spawn } from './cli-process'
import readline from 'node:readline'
import { detachedProcessGroup, terminateChildProcess } from './child-process-lifecycle'
import { CodexStdioTransport } from './codex-app-server-session'

// --- production transport (lazy spawn keeps the binary out of pure tests) ---

export function spawnCodexAppServerTransport(input: {
  binary: string
  args?: string[]
  // Full argv, replacing `app-server ...args` (a planning-mode seatbelt wrapper).
  argv?: string[]
  cwd?: string
  env?: NodeJS.ProcessEnv
}): CodexStdioTransport {
  const child = spawn(input.binary, input.argv ?? ['app-server', ...(input.args ?? [])], {
    cwd: input.cwd,
    detached: detachedProcessGroup(),
    env: input.env ?? process.env,
    stdio: ['pipe', 'pipe', 'pipe'],
  })
  const lines = readline.createInterface({ input: child.stdout! })
  let lineCb: ((line: string) => void) | null = null
  lines.on('line', (line) => lineCb?.(line))

  let stderrTail = ''
  child.stderr?.on('data', (data: Buffer) => {
    stderrTail = (stderrTail + data.toString()).slice(-400)
  })
  let closeCb: ((reason: string) => void) | null = null
  let closeReported = false
  const reportClose = (reason: string) => {
    if (closeReported) return
    closeReported = true
    closeCb?.(reason)
  }
  child.on('error', (error) => reportClose(`codex app-server failed to start: ${error.message}`))
  child.on('exit', (code, signalName) => {
    const suffix = stderrTail.trim() ? ` — ${stderrTail.trim().split('\n').slice(-2).join(' / ')}` : ''
    reportClose(`codex app-server exited (${signalName ?? code ?? 'unknown'})${suffix}`)
  })
  return {
    write(line: string) {
      child.stdin!.write(line + '\n')
    },
    onLine(cb) {
      lineCb = cb
    },
    kill() {
      terminateChildProcess(child, { processGroup: true })
    },
    onClose(cb) {
      closeCb = cb
    },
  }
}
