import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { existsSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const native = fileURLToPath(new URL('../native/computer-use/', import.meta.url))
const shared = path.join(native, 'ComputerUseChecks', 'Sources', 'ComputerUseChecks')
const packaged = existsSync(path.join(shared, 'Protocol.swift'))
const compiler = spawnSync('swiftc', ['--version'], { encoding: 'utf8', timeout: 10_000 })
if (compiler.status !== 0 || !packaged && process.platform !== 'darwin') {
  console.log('computer parser checks skipped: Swift and a supported production source layout are required')
} else {
  const root = mkdtempSync(path.join(os.tmpdir(), 'tf-parser-bounds-'))
  try {
    const names = packaged ? readdirSync(shared).filter((name) => name.endsWith('.swift')).sort() : ['Keys.swift', 'Protocol.swift']
    for (const name of names) writeFileSync(path.join(root, name), readFileSync(path.join(packaged ? shared : native, name)))
    if (!packaged) {
      const phone = readFileSync(path.join(native, 'Phone.swift'), 'utf8')
      const start = phone.indexOf('func phoneLocalPoint('), end = phone.indexOf('func phoneContentRect(')
      assert.ok(start >= 0 && end > start, 'compile the production phone validators verbatim')
      writeFileSync(path.join(root, 'PhoneValidation.swift'), `import Foundation\n${phone.slice(start, end)}`)
      names.push('PhoneValidation.swift')
    }
    writeFileSync(path.join(root, 'main.swift'), `import Foundation
while let line = readLine() {
    let parsed = parseRequest(line)
    switch parsed.command {
    case .failure(let error):
        FileHandle.standardOutput.write(encodeLine(errorReply(parsed.id, error)))
    case .success(.scroll(_, let dx, let dy)):
        FileHandle.standardOutput.write(encodeLine(okReply(parsed.id ?? 0, ["dx": dx, "dy": dy])))
    case .success:
        fatalError("Probe accepts scroll requests only")
    }
}
`)
    const build = spawnSync('swiftc', ['-O', '-swift-version', '5', '-o', 'parser-probe', ...names, 'main.swift'], { cwd: root, encoding: 'utf8', timeout: 90_000, maxBuffer: 1024 * 1024 })
    assert.equal(build.status, 0, `production parser must compile: ${build.stderr}`)
    const parse = (delta) => {
      const result = spawnSync(path.join(root, 'parser-probe'), [], { cwd: root, input: `${JSON.stringify({ id: 7, cmd: 'scroll', ...delta })}\n`, encoding: 'utf8', timeout: 5_000, maxBuffer: 1024 * 1024 })
      assert.equal(result.status, 0, `parser must reject ${JSON.stringify(delta)} without terminating (${result.signal ?? result.error?.code ?? result.status})`)
      const lines = result.stdout.trim().split('\n')
      assert.equal(lines.length, 1, 'one request produces one JSON-line reply')
      return JSON.parse(lines[0])
    }
    for (const axis of ['dx', 'dy']) {
      for (const value of [1e100, -1e100, 1e308, -1e308, -9223372036854775808, 201, -201, 200.5, -200.5, true, '3']) {
        const reply = parse({ [axis]: value })
        assert.equal(reply.id, 7)
        assert.equal(reply.ok, false)
        assert.equal(reply.code, 'bad_request')
        assert.ok(typeof reply.error === 'string' && reply.error.length > 0)
      }
    }
    for (const [input, expected] of [
      [{ dx: 200, dy: -200 }, { dx: 200, dy: -200 }],
      [{ dx: 200.49, dy: -200.49 }, { dx: 200, dy: -200 }],
      [{ dx: -0.5, dy: 0.5 }, { dx: -1, dy: 1 }],
      [{ dy: 3 }, { dx: 0, dy: 3 }],
    ]) assert.deepEqual(parse(input), { id: 7, ok: true, ...expected })
    for (const input of [{}, { dx: 0, dy: 0 }, { dx: 0.49, dy: -0.49 }]) assert.equal(parse(input).code, 'bad_request')
    console.log('Production Swift parser rejects oversized scroll values without crashing and preserves limits, rounding and JSON-line replies')
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
}
