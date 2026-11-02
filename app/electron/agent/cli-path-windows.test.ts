import assert from 'node:assert/strict'
import { knownBinDirs, mergeCliPaths, resolveCliBinary } from './cli-path'
const env = { APPDATA: 'C:\\Example\\Roaming', SystemRoot: 'C:\\Windows', PATH: 'D:\\Tools;C:\\Windows\\System32' }
assert.equal(mergeCliPaths('win32', 'C:\\Windows;D:\\Tools', 'c:\\windows\\;"E:\\Node Tools"'), 'C:\\Windows;D:\\Tools;E:\\Node Tools')
assert.ok(knownBinDirs('C:\\Example', 'win32', env).includes('C:\\Example\\Roaming\\npm'))
assert.equal(resolveCliBinary(null, ['codex'], { platform: 'win32', env, exists: p => p === 'D:\\Tools\\codex.exe' }), 'D:\\Tools\\codex.exe')
assert.equal(resolveCliBinary(null, ['claude'], { platform: 'win32', env, exists: p => p === 'C:\\Example\\Roaming\\npm\\claude.cmd' }), 'C:\\Example\\Roaming\\npm\\claude.cmd')
assert.equal(resolveCliBinary('F:\\Explicit\\tool.exe', ['codex'], { platform: 'win32', env }), 'F:\\Explicit\\tool.exe')
assert.equal(resolveCliBinary(null, ['codex'], { platform: 'win32', env, exists: () => false }), 'codex')
console.log('Windows drive paths, case-insensitive merging, npm/WinGet roots and executable extension lookup passed.')
