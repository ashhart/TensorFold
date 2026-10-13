const fs = require('node:fs/promises')
const path = require('node:path')
const { execFile } = require('node:child_process')
const { promisify } = require('node:util')
const exec = promisify(execFile)

// Native dependencies may carry DWARF/N_OSO records from the machine that built them.
// Strip only debug symbols in the packaging copy, before Developer ID signing.
async function stripNativeDebugSymbols(root) {
  const canonicalRoot = await fs.realpath(root)
  let count = 0
  async function visit(directory) {
    for (const entry of await fs.readdir(directory, { withFileTypes: true })) {
      const file = path.join(directory, entry.name)
      if (entry.isSymbolicLink()) throw new Error('Packaged native dependencies must not contain symlinks.')
      if (entry.isDirectory()) await visit(file)
      else if (entry.isFile() && entry.name.endsWith('.node')) {
        const canonicalFile = await fs.realpath(file)
        const relative = path.relative(canonicalRoot, canonicalFile)
        if (!relative || relative.startsWith('..') || path.isAbsolute(relative)) throw new Error('Native dependency escaped its packaging root.')
        await exec('/usr/bin/strip', ['-S', canonicalFile])
        const bytes = await fs.readFile(canonicalFile)
        if (/\/(?:Users|home)\/[^/\x00\s]+\//.test(bytes.toString('latin1'))) throw new Error('A packaged native dependency still contains a private build path after stripping debug symbols.')
        count += 1
      }
    }
  }
  await visit(canonicalRoot)
  return count
}

async function afterPack(context) {
  if (context.electronPlatformName !== 'darwin') return
  const root = path.join(context.appOutDir, `${context.packager.appInfo.productFilename}.app`, 'Contents', 'Resources', 'app.asar.unpacked')
  await stripNativeDebugSymbols(root)
}

module.exports = afterPack
module.exports.stripNativeDebugSymbols = stripNativeDebugSymbols
