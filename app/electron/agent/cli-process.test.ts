import assert from 'node:assert/strict'
import { windowsCliInvocation } from './cli-process'
const command = 'C:\\Example\\npm\\claude.cmd'
const script = 'C:\\Example\\npm\\node_modules\\fixture-cli\\bin.js'
const text = '@ECHO off\r\n"%_prog%" "%dp0%\\node_modules\\fixture-cli\\bin.js" %*\r\n'
const args = ['--model', 'model&echo unsafe', 'literal%PATH%', '$(touch nope)', 'quote"here']
const result = windowsCliInvocation(command, args, { read: () => text, exists: p => [script, 'C:\\Example\\npm\\node.exe'].includes(p) })
assert.deepEqual(result, { command: 'C:\\Example\\npm\\node.exe', args: [script, ...args] })
assert.deepEqual(windowsCliInvocation('D:\\Tools\\codex.exe', args), { command: 'D:\\Tools\\codex.exe', args })
for (const source of ['echo arbitrary batch', '"node" "%dp0%\\..\\escape.js" %*', text + text]) {
  assert.throws(() => windowsCliInvocation(command, args, { read: () => source, exists: () => true }), /unsupported|Unsafe/)
}
assert.throws(() => windowsCliInvocation(command, args, { read: () => text, exists: () => false }), /missing/)
console.log('Windows npm CLI shims preserve literal argv and reject arbitrary batch or escaping targets.')
