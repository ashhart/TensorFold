import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { createMcpServer, mcpSpawnSpec } from './mcp-process'
import { canUseMacSandbox } from './sandbox'

function main() {
  if (process.platform !== 'darwin' || !canUseMacSandbox()) {
    console.log('SKIP MCP app bundle: macOS sandbox unavailable')
    return
  }
  const compiler = spawnSync('/usr/bin/xcrun', ['--find', 'clang'], { encoding: 'utf8', timeout: 10_000 })
  const sdk = spawnSync('/usr/bin/xcrun', ['--sdk', 'macosx', '--show-sdk-path'], { encoding: 'utf8', timeout: 10_000 })
  if (compiler.status !== 0 || sdk.status !== 0) {
    console.log('SKIP MCP app bundle: Xcode C compiler or macOS SDK unavailable')
    return
  }
  // Use a private home fixture: the sandbox already permits the system temporary directories.
  const root = mkdtempSync(path.join(os.homedir(), '.tensorfold-mcp-bundle-test-'))
  try {
    const cwd = path.join(root, 'workspace')
    const contents = path.join(root, 'Fixture With Spaces.app', 'Contents')
    const binary = path.join(contents, 'MacOS', 'fixture')
    const library = path.join(contents, 'Frameworks', 'libfixture.dylib')
    const resource = path.join(contents, 'Resources', 'fixture.txt')
    const unrelated = path.join(root, 'Unrelated.app', 'Contents', 'private.txt')
    for (const dir of [cwd, path.dirname(binary), path.dirname(library), path.dirname(resource), path.dirname(unrelated)]) {
      mkdirSync(dir, { recursive: true })
    }
    writeFileSync(resource, 'bundle resource')
    writeFileSync(unrelated, 'unrelated data')
    const librarySource = path.join(root, 'library.c')
    const mainSource = path.join(root, 'main.c')
    writeFileSync(librarySource, 'int fixture_value(void) { return 42; }\n')
    writeFileSync(mainSource, [
      '#include <stdio.h>',
      '#include <string.h>',
      'int fixture_value(void);',
      'int main(int argc, char **argv) {',
      '  if (argc != 2 || strcmp(argv[1], "argument with spaces")) return 2;',
      '  printf("loaded %d\\n", fixture_value());',
      '  return 0;',
      '}',
    ].join('\n'))
    for (const args of [
      ['-dynamiclib', librarySource, '-install_name', '@rpath/libfixture.dylib', '-o', library],
      [mainSource, '-L', path.dirname(library), '-lfixture', '-Wl,-rpath,@executable_path/../Frameworks', '-o', binary],
    ]) {
      const built = spawnSync(compiler.stdout.trim(), ['-isysroot', sdk.stdout.trim(), ...args], {
        encoding: 'utf8', timeout: 30_000,
      })
      assert.equal(built.status, 0, built.stderr || built.error?.message)
    }
    const config = createMcpServer({ name: 'Bundle fixture', command: binary, args: ['argument with spaces'], cwd })
    const spec = mcpSpawnSpec(config)
    assert.equal(spec.sandboxed, true)
    const options = { cwd: spec.cwd, env: spec.env, encoding: 'utf8' as const, timeout: 10_000 }
    const launched = spawnSync(spec.command, spec.args, options)
    assert.equal(launched.status, 0, launched.stderr || launched.error?.message)
    assert.equal(launched.stdout, 'loaded 42\n', 'dyld must load the sibling framework without splitting arguments')

    const probe = (script: string, file: string) => spawnSync(spec.command, [
      ...spec.args.slice(0, 2), '/bin/sh', '-c', script, 'fixture', file,
    ], options)
    const assertDenied = (result: ReturnType<typeof probe>, message: string) => {
      assert.ifError(result.error)
      assert.equal(result.signal, null, message)
      assert.ok(result.status !== null && result.status !== 0, message)
      assert.match(result.stderr, /Operation not permitted|Permission denied/, message)
    }
    const read = probe('cat "$1"', resource)
    assert.equal(read.status, 0, read.stderr)
    assert.equal(read.stdout, 'bundle resource')
    assertDenied(probe('cat "$1"', unrelated), 'other application bundles remain unreadable')
    assertDenied(probe('printf changed > "$1"', resource), 'the selected bundle remains read-only')
    assert.equal(readFileSync(resource, 'utf8'), 'bundle resource')
    const argumentOnly = mcpSpawnSpec(createMcpServer({ command: '/bin/echo', args: [binary], cwd }))
    const argumentRead = spawnSync(argumentOnly.command, [
      ...argumentOnly.args.slice(0, 2), '/bin/cat', resource,
    ], options)
    assertDenied(argumentRead, 'passing a bundle path as an argument does not grant access to its Contents')
    console.log('MCP app bundle: framework loading, spaced arguments and sandbox boundaries passed')
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
}

main()
