const { readFile, stat } = require('node:fs/promises')
const path = require('node:path')
const { createHash } = require('node:crypto')
const { extractFile } = require('@electron/asar')

module.exports = async function verifyPackagedWindowsSandbox(context) {
  const resources = path.join(context.appOutDir, 'resources')
  const pin = extractFile(path.join(resources, 'app.asar'), 'sandbox-native.json')
  if (pin.length > 16384) throw new Error('Windows sandbox manifest exceeds its size limit.')
  const manifest = JSON.parse(pin.toString('utf8'))
  const binary = path.join(resources, 'sandbox/TensorFoldSandbox.exe')
  const info = await stat(binary)
  if (!info.isFile() || info.size > 8 * 1024 * 1024) throw new Error('Windows sandbox helper is not a bounded regular file.')
  const bytes = await readFile(binary)
  if (manifest.abi !== 1 || manifest.platform !== 'win32' || manifest.arch !== 'x64'
    || bytes.toString('ascii', 0, 2) !== 'MZ'
    || createHash('sha256').update(bytes).digest('hex') !== manifest.sha256) {
    throw new Error('Windows package is missing its protected sandbox helper pin or has mismatching helper bytes.')
  }
}
