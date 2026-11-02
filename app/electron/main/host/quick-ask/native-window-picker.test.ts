import assert from 'node:assert/strict'
import { EventEmitter } from 'node:events'
import { PassThrough } from 'node:stream'
import { createNativeAskPicker, supportsNativeAskPicker } from './native-window-picker'
import { qualificationImage } from '../../../agent/model-qualification/fixture-image'

class Helper extends EventEmitter {
  stdout = new PassThrough(); stderr = new PassThrough(); stdin = new PassThrough()
  killed = false
  kill() { this.killed = true; queueMicrotask(() => this.emit('close', null)); return true }
  result(value: unknown) { this.stdout.write(JSON.stringify(value)); this.emit('close', 0) }
}
let child: Helper, config: any, args: string[] = [], launches = 0
const picker = createNativeAskPicker(() => '/synthetic/helper', {
  stat: (async () => ({ isFile: () => true, isSymbolicLink: () => false, mode: 0o755 })) as any,
  launch: ((_file: string, argv: string[], options: unknown) => { launches++; args = argv; config = options; child = new Helper(); return child }) as any,
})
const turn = () => new Promise<void>(resolve => setImmediate(resolve))
assert.equal(supportsNativeAskPicker('darwin', '24.2.0'), true)
assert.equal(supportsNativeAskPicker('darwin', '24.1.0'), false)
assert.equal(supportsNativeAskPicker('linux', '27.0.0'), false)
let result = picker.pick(); await turn()
await assert.rejects(picker.pick(), /Finish or cancel/)
assert.equal(launches, 1)
assert.deepEqual(args, ['--ask-window-picker', '--own-bundle-id', 'dev.tensorfold.studio'])
assert.deepEqual(Object.keys(config.env).sort(), ['HOME', 'LANG', 'PATH'])
child!.result({ kind: 'document', app: 'TextEdit', title: 'sample.txt', text: 'Local sample', png: qualificationImage().split(',')[1] })
assert.equal((await result)?.length, 2)
result = picker.pick(); await turn(); child!.stdout.write('{"state":"picking"}\n{"state":"capturing"}\n'); picker.cancel()
assert.equal(await result, null); assert.equal(child!.killed, true)
result = picker.pick(); await turn(); child!.result({ cancelled: true }); assert.equal(await result, null)
result = picker.pick(); await turn(); child!.stdout.write('{"state":"picking"}\n')
assert.equal(picker.cancel(), false, 'an open native picker must be cancelled through its own UI before the helper can exit')
assert.equal(child!.killed, false)
child!.result({ cancelled: true }); assert.equal(await result, null)
for (const value of [null, [], { error: 'Synthetic denied window' }, { kind: 'document', png: 'bad' }]) {
  result = picker.pick(); await turn(); child!.result(value); await assert.rejects(result)
}
result = picker.pick(); await turn(); child!.stdout.write(Buffer.alloc(2_700_001))
await assert.rejects(result, /context limit/); assert.equal(child!.killed, true)
let finishStat!: (value: any) => void
const delayed = createNativeAskPicker(() => '/synthetic/helper', {
  stat: (() => new Promise(resolve => { finishStat = resolve })) as any,
  launch: (() => { throw new Error('A cancelled request must not launch a picker') }) as any,
})
result = delayed.pick(); delayed.cancel()
finishStat({ isFile: () => true, isSymbolicLink: () => false, mode: 0o755 })
assert.equal(await result, null)
const timed = createNativeAskPicker(() => '/synthetic/helper', {
  timeoutMs: 15,
  stat: (async () => ({ isFile: () => true, isSymbolicLink: () => false, mode: 0o755 })) as any,
  launch: (() => { child = new Helper(); return child }) as any,
})
result = timed.pick(); await turn()
child!.stdout.write('{"state":"picking"}\n')
await new Promise(resolve => setTimeout(resolve, 35))
assert.equal(timed.busy, true, 'human selection has no artificial timeout')
assert.equal(timed.cancel(), false); assert.equal(child!.killed, false)
child!.result({ cancelled: true }); assert.equal(await result, null)
result = timed.pick(); await turn()
child!.stdout.write('{"state":"picking"}\n{"state":"capturing"}\n')
const deadline = assert.rejects(result, /stopped responding/)
await new Promise(resolve => setTimeout(resolve, 35)); await deadline
assert.equal(child!.killed, true, 'capture after selection remains bounded')
console.log('Native window picker enforces one helper, narrow arguments and environment, bounded output, cancellation and safe result validation.')
