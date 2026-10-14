import { execFile } from 'node:child_process'
import { lstat, open, readdir } from 'node:fs/promises'
import path from 'node:path'
import { promisify } from 'node:util'

const exec = promisify(execFile)
const MACHO_MAGICS = new Set(['cffaedfe', 'cefaedfe', 'feedfacf', 'feedface', 'cafebabe', 'bebafeca', 'cafebabf', 'bfbafeca'])

function version(value) {
  if (!/^\d+\.\d+(?:\.\d+)?$/.test(value)) throw new Error(`Invalid minimum macOS version: ${value.slice(0, 80)}.`)
  return value.split('.').map(Number).concat([0]).slice(0, 3)
}

function newer(left, right) {
  const a = version(left)
  const b = version(right)
  for (let i = 0; i < 3; i++) if (a[i] !== b[i]) return a[i] > b[i]
  return false
}

// Inspect every architecture's OS load command, including older LC_VERSION_MIN_MACOSX binaries.
export function macOSBuildMinimums(output) {
  const blocks = []
  let current = null
  for (const line of output.split(/\r?\n/).map(value => value.trim())) {
    const command = /^cmd (LC_\w+)$/.exec(line)
    if (command) {
      current = command[1] === 'LC_BUILD_VERSION' || command[1].startsWith('LC_VERSION_MIN_') ? { command: command[1] } : null
      if (current) blocks.push(current)
    } else if (current) {
      const platform = /^platform (\S+)$/.exec(line)
      const minimum = /^(?:minos|version) (\S+)$/.exec(line)
      if (platform) current.platform = platform[1]
      if (minimum && current.minimum === undefined) current.minimum = minimum[1]
    }
  }
  if (!blocks.length) throw new Error('Native binary has no recognized OS build-version command.')
  return blocks.map(block => {
    if (block.command === 'LC_BUILD_VERSION' ? block.platform !== 'MACOS' : block.command !== 'LC_VERSION_MIN_MACOSX') {
      throw new Error('Native binary has a non-macOS build target.')
    }
    version(block.minimum ?? '')
    return block.minimum
  })
}

// Static inspection only: do not load or execute native modules to discover their requirements.
export async function assertPackagedMacOSMinimum(app, { run = exec } = {}) {
  const appInfo = await lstat(app)
  if (!appInfo.isDirectory() || appInfo.isSymbolicLink()) {
    throw new Error('Minimum OS verification requires a regular app directory.')
  }
  const plist = path.join(app, 'Contents', 'Info.plist')
  const result = await run('/usr/bin/plutil', ['-extract', 'LSMinimumSystemVersion', 'raw', '-o', '-', plist])
  const declaredMinimum = String(result.stdout ?? '').trim()
  version(declaredMinimum)
  const binaries = []
  async function visit(directory) {
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      const file = path.join(directory, entry.name)
      if (entry.isSymbolicLink()) continue // Framework aliases are inspected at their actual regular-file target.
      if (entry.isDirectory()) { await visit(file); continue }
      if (!entry.isFile()) continue
      const handle = await open(file, 'r')
      const magic = Buffer.alloc(4)
      try { await handle.read(magic, 0, 4, 0) } finally { await handle.close() }
      if (!MACHO_MAGICS.has(magic.toString('hex'))) continue
      const build = await run('/usr/bin/xcrun', ['vtool', '-show-build', file])
      let minimums
      try { minimums = macOSBuildMinimums(`${build.stdout ?? ''}\n${build.stderr ?? ''}`) }
      catch (error) { throw new Error(`${path.relative(app, file)}: ${error.message}`) }
      for (const minimum of minimums) if (newer(minimum, declaredMinimum)) {
        throw new Error(`${path.relative(app, file)} requires macOS ${minimum}, but the app declares ${declaredMinimum}.`)
      }
      binaries.push({ path: path.relative(app, file), minimums })
    }
  }
  await visit(app)
  if (!binaries.length) throw new Error('App contains no native binaries to verify against its minimum macOS version.')
  return { declaredMinimum, binaries }
}
