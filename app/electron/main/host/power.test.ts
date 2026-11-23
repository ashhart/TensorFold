import assert from 'node:assert/strict'
import { EventEmitter } from 'node:events'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import { mock } from 'node:test'
import ts from 'typescript'

const source = readFileSync(path.join(__dirname, 'power.ts'), 'utf8')
const output = ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS },
}).outputText
const loadPower = new Function('require', 'exports', 'process', output)

class FakeCaffeinate extends EventEmitter {
  signals: NodeJS.Signals[] = []

  get killed() { return this.signals.length > 0 }

  kill(signal: NodeJS.Signals) {
    this.signals.push(signal)
    return true
  }
}

function makeFixture(platform: NodeJS.Platform) {
  const store = { settings: { power: { keepComputerAlive: true, lastError: null as string | null } } }
  const children: FakeCaffeinate[] = []
  const writes: Array<{ key: string; lastError: string | null }> = []
  const imports = new Map<string, unknown>([
    ['node:child_process', {
      spawn(command: string, args: string[], options: object) {
        assert.equal(command, '/usr/bin/caffeinate')
        assert.deepEqual(args, ['-dimsu'])
        assert.deepEqual(options, { stdio: 'ignore' })
        const child = new FakeCaffeinate()
        children.push(child)
        return child
      },
    }],
    ['../state', { store }],
    ['./persistence', {
      persistSettings(key: string) {
        writes.push({ key, lastError: store.settings.power.lastError })
      },
    }],
  ])
  const power = {} as typeof import('./power')
  loadPower((id: string) => {
    assert.ok(imports.has(id), `unexpected power dependency ${id}`)
    return imports.get(id)
  }, power, { platform })
  return { power, children, writes }
}

const unsupported = makeFixture('linux')
unsupported.power.applyPowerSettings()
assert.equal(unsupported.children.length, 0)
assert.equal(unsupported.writes.at(-1)?.key, 'power.lastError')
assert.match(unsupported.writes.at(-1)?.lastError ?? '', /only available on macOS/)

const { power, children, writes } = makeFixture('darwin')
mock.timers.enable({ apis: ['setTimeout'] })
try {
  power.applyPowerSettings()
  power.applyPowerSettings()
  assert.equal(children.length, 1, 'reapplying settings keeps the same live helper')
  assert.deepEqual(writes, [{ key: 'power.lastStartedAt', lastError: null }])

  children[0].emit('exit', null, 'SIGKILL')
  assert.deepEqual(writes.at(-1), { key: 'power.lastError', lastError: 'caffeinate exited: SIGKILL' })
  mock.timers.tick(1999)
  assert.equal(children.length, 1, 'respawn waits for the backoff')
  mock.timers.tick(1)
  assert.equal(children.length, 2, 'the helper respawns once at the deadline')
  assert.deepEqual(writes.at(-1), { key: 'power.lastStartedAt', lastError: null })

  children[1].emit('exit', 1, null)
  power.stopCaffeinate()
  mock.timers.tick(2000)
  assert.equal(children.length, 2, 'stopping cancels a pending respawn')

  power.applyPowerSettings()
  assert.equal(children.length, 3)
  const savedBeforeStop = writes.length
  power.stopCaffeinate()
  assert.deepEqual(children.map((child) => child.signals), [[], [], ['SIGTERM']], 'stop signals only the current child')
  children[2].emit('exit', null, 'SIGTERM')
  mock.timers.tick(2000)
  assert.equal(children.length, 3, 'a deliberate stop does not respawn')
  assert.equal(writes.length, savedBeforeStop, 'a deliberate stop does not record an error')
} finally {
  power.stopCaffeinate()
  mock.timers.reset()
}

console.log('Desktop power: platform guard, owned shutdown, error persistence and restart backoff passed')
