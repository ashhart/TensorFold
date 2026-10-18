// Drive a CLI agent (Claude Code, Codex) inside a hidden, detached tmux session:
// launch it, send prompts, watch its pane output. This lets TensorFold orchestrate the
// CLIs' real, long-lived agent sessions — keeping their native tools/approvals/context
// across turns — instead of one-shot spawns. The arg builders are pure (unit-tested);
// spawnTmuxRunner() executes the real tmux binary.

import { spawn } from 'node:child_process'

export interface TmuxRunResult { code: number | null; stdout: string; stderr: string }
export interface TmuxRunner { run(args: string[]): Promise<TmuxRunResult> }

export function startArgs(name: string, command: string[], cwd?: string): string[] {
  return ['new-session', '-d', '-s', name, ...(cwd ? ['-c', cwd] : []), ...command]
}
export function sendTextArgs(name: string, text: string): string[] {
  return ['send-keys', '-t', name, '-l', text] // -l: literal text, not interpreted as key names
}
export function sendEnterArgs(name: string): string[] {
  return ['send-keys', '-t', name, 'Enter']
}
export function captureArgs(name: string): string[] {
  return ['capture-pane', '-t', name, '-p', '-S', '-'] // -p print, -S - from start of history
}
export function hasArgs(name: string): string[] {
  return ['has-session', '-t', name]
}
export function killArgs(name: string): string[] {
  return ['kill-session', '-t', name]
}

export async function startSession(runner: TmuxRunner, opts: { name: string; command: string[]; cwd?: string }): Promise<void> {
  const res = await runner.run(startArgs(opts.name, opts.command, opts.cwd))
  if (res.code !== 0) throw new Error(`tmux new-session failed: ${res.stderr || res.code}`)
}
export async function sendLine(runner: TmuxRunner, name: string, text: string): Promise<void> {
  await runner.run(sendTextArgs(name, text))
  await runner.run(sendEnterArgs(name))
}
export async function capturePane(runner: TmuxRunner, name: string): Promise<string> {
  return (await runner.run(captureArgs(name))).stdout
}
export async function hasSession(runner: TmuxRunner, name: string): Promise<boolean> {
  return (await runner.run(hasArgs(name))).code === 0
}
export async function killSession(runner: TmuxRunner, name: string): Promise<void> {
  await runner.run(killArgs(name))
}

/** tmux session names can't contain '.' or ':'; produce a safe id. */
export function tmuxSessionName(prefix: string, id: string): string {
  return `${prefix}_${id}`.replace(/[^a-zA-Z0-9_-]/g, '_')
}

/** Real runner: execute the tmux binary. */
export function spawnTmuxRunner(bin = 'tmux'): TmuxRunner {
  return {
    run(args) {
      return new Promise((resolve) => {
        const proc = spawn(bin, args, { stdio: ['ignore', 'pipe', 'pipe'] })
        let stdout = ''
        let stderr = ''
        proc.stdout.on('data', (d) => { stdout += String(d) })
        proc.stderr.on('data', (d) => { stderr += String(d) })
        proc.on('error', (e) => resolve({ code: null, stdout, stderr: stderr || e.message }))
        proc.on('close', (code) => resolve({ code, stdout, stderr }))
      })
    },
  }
}
