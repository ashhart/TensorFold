const fs = require('node:fs/promises')
const path = require('node:path')
const { execFile } = require('node:child_process')
const { promisify } = require('node:util')
const { createHash } = require('node:crypto')
const exec = promisify(execFile)

// ONNX Runtime's published N-API binaries retain one __FILE__ diagnostic from
// Microsoft's generic CI runner. These are the exact stripped npm 1.21.0 bytes,
// not an exemption for arbitrary vendor files or user directories.
const reviewedOnnxCiBinaries = new Map([
  ['node_modules/onnxruntime-node/bin/napi-v3/darwin/arm64/onnxruntime_binding.node', '8cb6285d228800d3f6ef3e97229286d27ff5570bdf1b0ca37f162e232c12588e'],
  ['node_modules/onnxruntime-node/bin/napi-v3/darwin/x64/onnxruntime_binding.node', '3528e827364903df36f98e607063e658f1b5e52a477f4f3b09ecd13c8947a0c8'],
])

function reviewedVendorCiPath(relative, bytes) {
  const expected = reviewedOnnxCiBinaries.get(relative.split(path.sep).join('/'))
  if (!expected || createHash('sha256').update(bytes).digest('hex') !== expected) return false
  const paths = bytes.toString('latin1').match(/\/(?:Users|home)\/[^/\x00\s]+\/[^\x00\s]*/g) ?? []
  return paths.length === 1 && paths[0] === '/Users/runner/work/1/s/js/node/src/main.cc'
}

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
        if (/\/(?:Users|home)\/[^/\x00\s]+\//.test(bytes.toString('latin1')) && !reviewedVendorCiPath(relative, bytes)) throw new Error('A packaged native dependency still contains a private build path after stripping debug symbols.')
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
module.exports.reviewedVendorCiPath = reviewedVendorCiPath
