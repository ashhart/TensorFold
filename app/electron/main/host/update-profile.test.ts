import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import { PRIVATE_UPDATE_PROFILE_FILE, PRIVATE_UPDATE_PROFILE_PREFIX, privateUpdateRelaunchProfile } from './update-profile'

const temp = await fs.realpath('/tmp')
const root = await fs.mkdtemp(path.join(temp, PRIVATE_UPDATE_PROFILE_PREFIX))
const appBundle = path.join(root, 'TensorFold Studio.app')
const executable = path.join(appBundle, 'Contents', 'MacOS', 'TensorFold Studio')
const marker = path.join(root, PRIVATE_UPDATE_PROFILE_FILE)
try {
  await fs.chmod(root, 0o700)
  await fs.mkdir(path.dirname(executable), { recursive: true })
  await fs.writeFile(executable, 'fixture')
  const descriptor = { schemaVersion: 1, root, appBundle, debugPort: 50001 }
  await fs.writeFile(marker, JSON.stringify(descriptor), { mode: 0o600 })
  assert.deepEqual(privateUpdateRelaunchProfile(executable), { root, appBundle, debugPort: 50001 })
  await fs.chmod(marker, 0o644)
  assert.equal(privateUpdateRelaunchProfile(executable), null, 'a readable-by-others descriptor cannot choose the restart profile')
  assert.throws(() => privateUpdateRelaunchProfile(executable, true), /will not open another profile/, 'an invalid recognized fixture refuses startup instead of falling back to real user data')
  await fs.chmod(marker, 0o600)
  await fs.writeFile(marker, JSON.stringify({ ...descriptor, root: path.dirname(root) }))
  assert.equal(privateUpdateRelaunchProfile(executable), null, 'the descriptor cannot select another profile or the real home')
  await fs.writeFile(marker, JSON.stringify({ ...descriptor, debugPort: 80 }))
  assert.equal(privateUpdateRelaunchProfile(executable), null)
  await fs.writeFile(marker, JSON.stringify({ ...descriptor, publicListener: true }))
  assert.equal(privateUpdateRelaunchProfile(executable), null, 'unknown schema fields cannot extend the restart permissions')
  await fs.writeFile(marker, JSON.stringify(descriptor))
  await fs.chmod(root, 0o755)
  assert.equal(privateUpdateRelaunchProfile(executable), null, 'the bundle must sit in an owner-only managed temporary profile')
  assert.equal(privateUpdateRelaunchProfile('/Applications/TensorFold Studio.app/Contents/MacOS/TensorFold Studio'), null, 'ordinary apps never read a test profile marker')
  assert.equal(privateUpdateRelaunchProfile('/Applications/TensorFold Studio.app/Contents/MacOS/TensorFold Studio', true), null, 'ordinary Applications installs remain unaffected')
} finally { await fs.rm(root, { recursive: true, force: true }) }
console.log('Native-update relaunch recovery is restricted to exact owner-only temporary test bundles')
