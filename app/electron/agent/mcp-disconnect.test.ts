import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { McpHost } from './mcp-client'
import { StringDecoder } from 'node:string_decoder'

const child = spawn(process.execPath, ['-e', 'setInterval(()=>{},1000)'], { stdio: 'pipe', detached: process.platform !== 'win32' })
child.stdout.on('data', () => {}); child.stderr.on('data', () => {})
await new Promise<void>((resolve, reject) => { child.once('spawn', () => resolve()); child.once('error', reject) })
const host = new McpHost()
const state = { config: { name: 'owned lifecycle fixture' }, proc: child, pending: new Map(), generation: 0,
  connectPromise: null, buffer: '', decoder: new StringDecoder('utf8') }
;(host as any).stopState(state)
const closed = new Promise<void>(resolve => child.once('close', () => resolve()))
let deadline: ReturnType<typeof setTimeout>
try {
  await Promise.race([closed, new Promise<void>((_, reject) => {
    deadline = setTimeout(() => reject(new Error('MCP shutdown lost the native child-process close event.')), 4000)
  })])
  assert.ok(child.exitCode !== null || child.signalCode !== null)
  await host.shutdown()
  console.log('MCP disconnect preserves Node stdio close listeners and completes owned-process shutdown.')
} finally { clearTimeout(deadline!); if (child.exitCode === null && child.signalCode === null) child.kill('SIGKILL') }
