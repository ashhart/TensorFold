import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
import { copyFile, mkdtemp, mkdir, readFile, rm, symlink, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import hook from './strip-packaged-debug-paths.cjs'

if (process.platform === 'darwin') {
  const root = await mkdtemp(path.join(os.tmpdir(), 'studio-native-debug-'))
  try {
    const nativeRoot = path.join(root, 'unpacked')
    await mkdir(nativeRoot)
    const source = path.join(root, 'fixture.c')
    const binary = path.join(nativeRoot, 'fixture.node')
    await writeFile(source, 'int preserved_runtime_export(void) { return 42; }\n')
    const compiled = path.join(root, 'compiled.node')
    execFileSync('/usr/bin/xcrun', ['clang', '-g', '-bundle', '-o', compiled, source])
    await copyFile(compiled, binary)
    const codeBefore = execFileSync('/usr/bin/otool', ['-s', '__TEXT', '__text', binary], { encoding: 'utf8' })
    assert.match(execFileSync('/usr/bin/nm', ['-gU', binary], { encoding: 'utf8' }), /preserved_runtime_export/)
    assert.equal(await hook.stripNativeDebugSymbols(nativeRoot), 1)
    assert.equal(execFileSync('/usr/bin/otool', ['-s', '__TEXT', '__text', binary], { encoding: 'utf8' }), codeBefore, 'stripping debug symbols preserves executable instructions')
    assert.match(execFileSync('/usr/bin/nm', ['-gU', binary], { encoding: 'utf8' }), /preserved_runtime_export/)
    assert.equal((await readFile(binary)).includes(Buffer.from(source)), false)
    await symlink(source, path.join(nativeRoot, 'escape.node'))
    await assert.rejects(() => hook.stripNativeDebugSymbols(nativeRoot), /symlinks/)
  } finally { await rm(root, { recursive: true, force: true }) }
}
console.log('packaged native debug-path regression passed')
