import assert from 'node:assert/strict'
import { runSandboxProcess } from './sandbox-process'

const code = 'process.on("SIGTERM",()=>process.exit(0));console.log("ready");setInterval(()=>{},1000)'
const timed = await runSandboxProcess([process.execPath, '-e', code], {
  cwd: process.cwd(), timeoutMs: 700, sandboxed: false, profile: 'build-test', implementation: 'unsandboxed',
})
assert.match(timed.stdout, /ready/)
assert.equal(timed.code, 124, 'A timeout remains a timeout when the child handles termination with exit zero')
assert.match(timed.stderr, /Timed out/)
const overflow = await runSandboxProcess([process.execPath, '-e',
  'process.on("SIGTERM",()=>process.exit(0));process.stdout.write("x".repeat(65536));setInterval(()=>{},1000)'], {
  cwd: process.cwd(), timeoutMs: 10000, maxBuffer: 1024, sandboxed: false, profile: 'build-test', implementation: 'unsandboxed',
})
assert.equal(overflow.code, 126, 'Output overflow must never authorize workspace publication')
console.log('Sandbox timeout and overflow retain failure status even when a child exits successfully after termination.')
