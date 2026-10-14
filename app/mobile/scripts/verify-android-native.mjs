import { readFile, mkdir, createWriteStream } from 'node:fs'
import { promisify } from 'node:util'
import { spawn } from 'node:child_process'
import path from 'node:path'

const read = promisify(readFile)
const root = path.resolve(import.meta.dirname, '..')
const connection = JSON.parse(await read(path.join(root, '.expo/native-smoke/connection.json'), 'utf8'))
const url = new URL(connection.pairingLink)
const credential = new URLSearchParams(url.hash.slice(1)).get('pair')
const pin = new URLSearchParams(url.hash.slice(1)).get('pin')
if (url.protocol !== 'https:' || !/^(?:10\.|192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.)/.test(url.hostname) || !credential || !/^[a-f0-9]{64}$/.test(pin ?? '')) throw new Error('Start the explicit disposable private-LAN QA host first.')
await promisify(mkdir)(path.join(root, '.expo/native-smoke'), { recursive: true })
const log = createWriteStream(path.join(root, '.expo/native-smoke/android-native.log'))
const child = spawn(path.join(root, 'android/gradlew'), ['--project-dir', 'android', ':tensorfold-pinned-network:connectedDebugAndroidTest', '-PreactNativeArchitectures=arm64-v8a', `-Pandroid.testInstrumentationRunnerArguments.pairingLink=${connection.pairingLink}`, '--console=plain', '--max-workers=4'], { cwd: root, env: process.env, stdio: ['ignore', 'pipe', 'pipe'] })
for (const output of [child.stdout, child.stderr]) output.on('data', chunk => log.write(String(chunk).replaceAll(connection.pairingLink, '[disposable pairing link]').replaceAll(credential, '[disposable credential]')))
const code = await new Promise((resolve, reject) => { child.once('error', reject); child.once('exit', resolve) })
log.end()
if (code !== 0) throw new Error('Android native transport checks failed; see .expo/native-smoke/android-native.log.')
console.log('Android emulator native certificate and paired private bridge checks passed.')
