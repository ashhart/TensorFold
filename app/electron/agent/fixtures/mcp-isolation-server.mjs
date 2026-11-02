import readline from 'node:readline'
import fs from 'node:fs'
import net from 'node:net'
import { spawn } from 'node:child_process'

const reply = (id, result) => process.stdout.write(JSON.stringify({ jsonrpc: '2.0', id, result }) + '\n')
for await (const line of readline.createInterface({ input: process.stdin })) {
  const message = JSON.parse(line)
  if (message.id == null) continue
  try {
    if (message.method === 'initialize') {
      reply(message.id, { protocolVersion: '2025-06-18', capabilities: { tools: {} }, serverInfo: { name: 'isolated fixture', version: '1' } })
    } else if (message.method === 'tools/list') {
      reply(message.id, { tools: [{ name: 'probe', description: 'Owned isolation test fixture', inputSchema: { type: 'object', properties: {} } }] })
    } else if (message.method === 'tools/call') {
      const args = message.params.arguments
      let result = { ok: true }
      if (args.operation === 'write') fs.writeFileSync('result.txt', args.value)
      if (args.operation === 'read') result = { value: fs.readFileSync('result.txt', 'utf8') }
      if (args.operation === 'private') fs.readFileSync(args.file)
      if (args.operation === 'git') fs.writeFileSync('.git/config', 'unsafe fixture')
      if (args.operation === 'new-git') {
        fs.writeFileSync('result.txt', 'must not publish')
        fs.mkdirSync('nested/.git/hooks', { recursive: true })
        fs.writeFileSync('nested/.git/hooks/pre-commit', 'unsafe fixture')
      }
      if (args.operation === 'environment') result = { inherited: !!process.env.TENSORFOLD_ISOLATION_TEST_SECRET, explicit: process.env.TF_MCP_FIXTURE_TOKEN === 'explicit fixture' }
      if (args.operation === 'network') result = await new Promise(resolve => {
        const socket = net.connect({ host: '127.0.0.1', port: args.port })
        socket.setTimeout(1000)
        socket.once('connect', () => { socket.destroy(); resolve({ connected: true }) })
        socket.once('error', () => resolve({ connected: false }))
        socket.once('timeout', () => { socket.destroy(); resolve({ connected: false }) })
      })
      if (args.operation === 'child') {
        const child = spawn(process.execPath, ['-e', 'setTimeout(()=>{},30000)', '--', args.token], { detached: true, stdio: 'ignore' })
        child.unref()
        result = { pid: child.pid }
      }
      reply(message.id, { content: [{ type: 'text', text: JSON.stringify(result) }] })
    } else process.stdout.write(JSON.stringify({ jsonrpc: '2.0', id: message.id, error: { code: -32601, message: 'Unsupported fixture method' } }) + '\n')
  } catch (error) {
    process.stdout.write(JSON.stringify({ jsonrpc: '2.0', id: message.id, error: { code: -32000, message: error.code || error.message } }) + '\n')
  }
}
