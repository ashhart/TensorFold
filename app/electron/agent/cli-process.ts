import { spawn as nodeSpawn, type SpawnOptions } from 'node:child_process'
import { readFileSync, existsSync } from 'node:fs'
import path from 'node:path'
import { resolveCliBinary } from './cli-path'

export function windowsCliInvocation(command: string, args: readonly string[], options: {
  read?: (file: string) => string; exists?: (file: string) => boolean; env?: NodeJS.ProcessEnv
} = {}): { command: string; args: string[] } {
  if (!/\.(cmd|bat)$/i.test(command)) return { command, args: [...args] }
  const read = options.read ?? (file => readFileSync(file, 'utf8'))
  const exists = options.exists ?? existsSync
  const text = read(command)
  if (text.length > 64 * 1024) throw new Error('CLI launcher is too large.')
  // npm shims delegate to a Node script; launch that script with argv, without a command shell.
  const targets = [...text.matchAll(/"%(?:dp0%|~dp0)([^"\r\n]+\.(?:cjs|mjs|js))"\s+%\*/gi)]
  if (targets.length !== 1) throw new Error('This Windows batch launcher is unsupported. Choose the CLI executable or its npm launcher.')
  const base = path.win32.dirname(command)
  const relative = targets[0][1].replace(/^[\\/]+/, '')
  if (relative.split(/[\\/]/).some(part => !part || part === '..' || part.includes('%') || part.includes(':'))) throw new Error('Unsafe npm CLI launcher target.')
  const script = path.win32.resolve(base, relative)
  if (!script.toLowerCase().startsWith(base.toLowerCase().replace(/[\\/]+$/, '') + '\\') || !exists(script)) throw new Error('The npm CLI script is missing.')
  const adjacentNode = path.win32.join(base, 'node.exe')
  const node = exists(adjacentNode) ? adjacentNode : resolveCliBinary(null, ['node'], { platform: 'win32', env: options.env, exists })
  return { command: node, args: [script, ...args] }
}

export const spawnCliProcess: typeof nodeSpawn = ((command: string, argsOrOptions: string[] | SpawnOptions = [], spawnOptions: SpawnOptions = {}) => {
  const args = Array.isArray(argsOrOptions) ? argsOrOptions : []
  const options = Array.isArray(argsOrOptions) ? spawnOptions : argsOrOptions
  const launch = process.platform === 'win32' ? windowsCliInvocation(command, args, { env: options.env }) : { command, args }
  return nodeSpawn(launch.command, launch.args, { ...options, ...(process.platform === 'win32' ? { windowsHide: true } : {}) })
}) as typeof nodeSpawn
