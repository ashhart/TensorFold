import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import { mkdtemp, copyFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const run = promisify(execFile)
const mobile = fileURLToPath(new URL('..', import.meta.url))
const profile = await mkdtemp(path.join(os.tmpdir(), 'tensorfold-native-pinning-'))
const device = process.argv[2]
if (!device) throw new Error('Pass an already booted iOS simulator UUID; no device settings or UI are changed.')
const sdk = (await run('xcrun', ['--sdk', 'iphonesimulator', '--show-sdk-path'])).stdout.trim()
await copyFile(path.join(mobile, 'scripts/native-pinning-main.swift'), path.join(profile, 'main.swift'))
const binary = path.join(profile, 'pinning-checks')
await run('xcrun', ['swiftc', '-sdk', sdk, '-target', 'arm64-apple-ios15.1-simulator', path.join(mobile, 'modules/tensorfold-pinned-network/ios/PinnedCertificate.swift'), path.join(profile, 'main.swift'), '-o', binary])
try {
  const result = await run('xcrun', ['simctl', 'spawn', device, binary, path.join(mobile, 'src/fixtures/pinning.json')])
  process.stdout.write(result.stdout)
} catch (error) { process.stdout.write(error.stdout ?? ''); process.stderr.write(error.stderr ?? ''); process.exitCode = 1 }
