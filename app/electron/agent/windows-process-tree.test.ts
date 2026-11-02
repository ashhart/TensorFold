import assert from 'node:assert/strict'
import { windowsTaskkill } from './windows-process-tree'
assert.deepEqual(windowsTaskkill(123, 'D:\\Windows'), { command: 'D:\\Windows\\System32\\taskkill.exe', args: ['/PID', '123', '/T', '/F'] })
for (const pid of [0, -1, 1.5, NaN, 0x1_0000_0000]) assert.throws(() => windowsTaskkill(pid), /owned process/)
assert.throws(() => windowsTaskkill(123, 'relative'), /absolute/)
console.log('Windows descendant termination uses a fixed system executable and validated numeric PID argv.')
