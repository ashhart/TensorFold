import assert from 'node:assert/strict'
import { writeFileSync } from 'node:fs'
import { mkdir, mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { buildComputerHelper as buildHelper, helperPaths, parseSelfTest, privateBuildPath, sourceDigest, swiftcArgs } from './build-computer-helper.mjs'

const buildComputerHelper = (options) => buildHelper({ platform: 'darwin', ...options })

// A stand-in for xcrun and the built binary: records every call and writes the -o output itself.
function toolchain({ swiftcMissing = false, compileFails = false, selfTest = '{"ok":true,"checks":7}\n', binaryText = 'fake helper binary' } = {}) {
  const calls = []
  const run = (command, args, options = {}) => {
    calls.push({ command, args, cwd: options.cwd })
    if (command === 'xcrun' && args[0] === '--find') {
      return swiftcMissing ? { status: null, error: Object.assign(new Error('spawnSync xcrun ENOENT'), { code: 'ENOENT' }) } : { status: 0, stdout: '/usr/bin/swiftc\n' }
    }
    if (command === 'xcrun' && args[0] === 'swiftc') {
      if (compileFails) return { status: 1, stdout: '', stderr: 'main.swift:1:1: error: expected expression' }
      writeFileSync(args[args.indexOf('-o') + 1], binaryText)
      return { status: 0, stdout: '', stderr: '' }
    }
    if (args[0] === '--self-test') return { status: selfTest.includes('"ok":true') ? 0 : 1, stdout: selfTest, stderr: '' }
    throw new Error(`unexpected command ${command} ${args.join(' ')}`)
  }
  return { calls, run, compiles: () => calls.filter((call) => call.command === 'xcrun' && call.args[0] === 'swiftc').length }
}

async function main() {
  const root = await mkdtemp(path.join(os.tmpdir(), 'tf-computer-build-'))
  try {
    const paths = helperPaths(root)
    await mkdir(paths.sourceDir, { recursive: true })
    await writeFile(path.join(paths.sourceDir, 'main.swift'), 'print("hi")\n')
    await writeFile(path.join(paths.sourceDir, 'Keys.swift'), 'let a = 1\n')
    await writeFile(path.join(paths.sourceDir, 'README.txt'), 'not a source\n')

    assert.deepEqual(swiftcArgs(['a.swift'], 'out').slice(0, 9), ['swiftc', '-O', '-swift-version', '5', '-module-name', 'TensorFoldComputer', '-target', 'arm64-apple-macos13.0', '-framework'])
    assert.equal(parseSelfTest('{"ok":true,"checks":12}\n'), 12)
    assert.equal(parseSelfTest('{"ok":false,"checks":12}'), null)
    assert.equal(parseSelfTest('{"ok":true,"checks":0}'), null)
    assert.equal(parseSelfTest('garbage'), null)
    assert.equal(privateBuildPath(Buffer.from('x\0/Users/someone/src/main.swift\0')), '/Users/someone/')
    assert.equal(privateBuildPath(Buffer.from('main.swift /usr/lib/swift')), null)
    const digestA = sourceDigest([{ name: 'a.swift', bytes: Buffer.from('x') }])
    assert.notEqual(digestA, sourceDigest([{ name: 'b.swift', bytes: Buffer.from('x') }]), 'a rename changes the digest')
    assert.notEqual(digestA, sourceDigest([{ name: 'a.swift', bytes: Buffer.from('y') }]), 'an edit changes the digest')

    // First build compiles from the source folder by relative names, stamps, and self-tests.
    const first = toolchain()
    const logs = []
    const built = await buildComputerHelper({ root, run: first.run, log: (line) => logs.push(line) })
    assert.equal(built.compiled, true)
    assert.equal(built.checks, 7)
    assert.match(logs.join('\n'), /self-test passed 7 checks/)
    assert.equal(first.compiles(), 1)
    const compile = first.calls.find((call) => call.args[0] === 'swiftc')
    assert.equal(compile.cwd, paths.sourceDir)
    assert.deepEqual(compile.args.slice(-2), ['Keys.swift', 'main.swift'], 'sources go by sorted relative name')
    assert.ok(!compile.args.some((arg) => arg.endsWith('README.txt')))
    assert.equal(await readFile(paths.binary, 'utf8'), 'fake helper binary')
    assert.equal((await stat(paths.binary)).mode & 0o777, 0o755)
    assert.ok(JSON.parse(await readFile(paths.stamp, 'utf8')).digest)
    assert.ok(first.calls.some((call) => call.command === paths.binary && call.args[0] === '--self-test'))

    // Unchanged sources: no swiftc and no xcrun lookup, but the self-test still runs.
    const second = toolchain()
    const again = await buildComputerHelper({ root, run: second.run, log: () => {} })
    assert.equal(again.compiled, false)
    assert.equal(second.compiles(), 0)
    assert.equal(second.calls.filter((call) => call.command === 'xcrun').length, 0)
    assert.equal(second.calls.filter((call) => call.args[0] === '--self-test').length, 1)

    // An edited source, a new source, a replaced binary and --force each rebuild.
    await writeFile(path.join(paths.sourceDir, 'Keys.swift'), 'let a = 2\n')
    const edited = toolchain()
    assert.equal((await buildComputerHelper({ root, run: edited.run, log: () => {} })).compiled, true)
    await writeFile(path.join(paths.sourceDir, 'Extra.swift'), 'let b = 1\n')
    const added = toolchain()
    assert.equal((await buildComputerHelper({ root, run: added.run, log: () => {} })).compiled, true)
    assert.ok(added.calls.find((call) => call.args[0] === 'swiftc').args.includes('Extra.swift'))
    await writeFile(paths.binary, 'tampered', { mode: 0o755 })
    const tampered = toolchain()
    assert.equal((await buildComputerHelper({ root, run: tampered.run, log: () => {} })).compiled, true)
    const forced = toolchain()
    assert.equal((await buildComputerHelper({ root, run: forced.run, force: true, log: () => {} })).compiled, true)
    const settled = toolchain()
    assert.equal((await buildComputerHelper({ root, run: settled.run, log: () => {} })).compiled, false)

    // Failures: a missing toolchain, a compile error, a local path in the binary, a failing self-test, another OS.
    await writeFile(path.join(paths.sourceDir, 'Extra.swift'), 'let b = 2\n')
    await assert.rejects(() => buildComputerHelper({ root, run: toolchain({ swiftcMissing: true }).run, log: () => {} }), /xcrun could not find swiftc.*xcode-select --install/s)
    await assert.rejects(() => buildComputerHelper({ root, run: toolchain({ compileFails: true }).run, log: () => {} }), /swiftc failed[\s\S]*expected expression/)
    await assert.rejects(stat(paths.binary), /ENOENT/, 'a failed compile leaves no binary behind')
    await assert.rejects(stat(paths.stamp), /ENOENT/)
    await assert.rejects(() => buildComputerHelper({ root, run: toolchain({ binaryText: 'bin /Users/someone/Studio/main.swift' }).run, log: () => {} }), /local build path \/Users\/someone\//)
    await assert.rejects(stat(paths.binary), /ENOENT/)
    await assert.rejects(() => buildComputerHelper({ root, run: toolchain({ selfTest: '{"ok":false,"checks":7,"failures":["key f12 code"]}\n' }).run, log: () => {} }), /--self-test failed[\s\S]*key f12 code/)
    await assert.rejects(stat(paths.binary), /ENOENT/, 'a failing self-test removes the binary')
    await assert.rejects(stat(paths.stamp), /ENOENT/)
    await assert.rejects(() => buildComputerHelper({ root, run: toolchain().run, platform: 'linux', log: () => {} }), /only on macOS/)

    await rm(paths.sourceDir, { recursive: true, force: true })
    await mkdir(paths.sourceDir)
    await assert.rejects(() => buildComputerHelper({ root, run: toolchain().run, log: () => {} }), /No Swift sources/)
  } finally {
    await rm(root, { recursive: true, force: true })
  }
  console.log('build-computer-helper tests passed')
}

main().catch((error) => { console.error(error); process.exit(1) })
