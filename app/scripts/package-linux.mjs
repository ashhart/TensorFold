import { finalizePackagedAppImage } from './finalize-linux-appimage.mjs'
import { spawn } from 'node:child_process'
import { createRequire } from 'node:module'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const require = createRequire(import.meta.url)
const root = fileURLToPath(new URL('..', import.meta.url))

export function linuxPackageArgs(args, platform = process.platform, arch = process.arch) {
  if (platform !== 'linux' || !['x64', 'arm64'].includes(arch)) throw new Error('Desktop packaging requires a Linux x64 or arm64 build host.')
  if (args.length !== 1 || !['--dir', '--appimage', '--deb'].includes(args[0])) {
    throw new Error('Choose exactly --dir or --appimage or --deb; this script never publishes artifacts.')
  }
  return [
    '--config', 'scripts/linux-builder-config.cjs', '--linux',
    ...(args[0] === '--dir' ? ['--dir'] : [args[0] === '--deb' ? 'deb' : 'AppImage']), `--${arch}`, '--publish', 'never',
  ]
}

export async function packageLinux(args) {
  const builderArgs = linuxPackageArgs(args)
  const cli = require.resolve('electron-builder/cli.js')
  await new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [cli, ...builderArgs], { cwd: root, env: process.env, stdio: 'inherit' })
    child.once('error', reject)
    child.once('exit', (code, signal) => {
      if (code === 0) resolve()
      else reject(new Error(`Linux packaging failed (${signal ?? code}).`))
    })
  })
  if (args[0] === '--appimage') await finalizePackagedAppImage(root, process.arch)
}

if (process.argv[1] && pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url) {
  packageLinux(process.argv.slice(2)).catch(error => { console.error(error.message); process.exitCode = 1 })
}
