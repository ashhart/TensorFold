import assert from 'node:assert/strict'
import { runVmProcess } from './service'

const result = await runVmProcess(process.execPath, ['-e', 'process.stdin.pipe(process.stdout)'], { stdin: 'fixture input', timeoutMs: 5000 })
assert.deepEqual(result, { code: 0, stdout: 'fixture input', stderr: '' })
await assert.rejects(runVmProcess(process.execPath, ['-e', 'setInterval(()=>{},1000)'], { timeoutMs: 50 }), /time limit/)
await assert.rejects(runVmProcess(process.execPath, ['-e', 'process.stdout.write("x".repeat(3*1024*1024));setInterval(()=>{},1000)'], { timeoutMs: 5000 }), /output exceeded/)
const controller = new AbortController()
const cancelled = runVmProcess(process.execPath, ['-e', 'setInterval(()=>{},1000)'], { signal: controller.signal, timeoutMs: 5000 })
controller.abort()
await assert.rejects(cancelled, /cancelled/)
await assert.rejects(runVmProcess('/nonexistent-fictional-runtime', [], { timeoutMs: 1000 }), /could not be started/)
console.log('bot-vm CLI process input, cancellation, time and output bounds passed')
