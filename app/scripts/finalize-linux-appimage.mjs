import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import { createHash } from 'node:crypto'
import { createReadStream } from 'node:fs'
import { chmod, copyFile, lstat, mkdir, mkdtemp, open, readFile, readdir, rename, rm, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { readAppImageLayout } from './appimage-format.mjs'

const exec = promisify(execFile)
const project = fileURLToPath(new URL('..', import.meta.url))
async function regular(file) {
  const stat = await lstat(file)
  if (!stat.isFile() || stat.isSymbolicLink()) throw new Error('AppImage integration expects ordinary files.')
}
export async function finalizeLinuxAppImage(file, options = {}) {
  const root = options.root ?? project, run = options.exec ?? exec
  const layout = await readAppImageLayout(file)
  const temporary = await mkdtemp(path.join(path.dirname(file), '.tensorfold-appimage-'))
  const tree = path.join(temporary, 'tree'), output = path.join(temporary, 'output.AppImage')
  try {
    try {
      await run('unsquashfs', ['-no-progress', '-processors', '2', '-o', String(layout.offset), '-d', tree, file], { timeout: 120000 })
    } catch (error) { throw new Error('AppImage preparation needs squashfs-tools. Install it on the Linux build host.', { cause: error }) }
    await regular(path.join(tree, 'AppRun'))
    await regular(path.join(tree, 'tensorfold-studio'))
    await copyFile(path.join(root, 'build/linux/AppRun'), path.join(tree, 'AppRun'))
    await chmod(path.join(tree, 'AppRun'), 0o755)
    const resources = path.join(tree, 'resources'), support = path.join(resources, 'appimage-support')
    const stat = await lstat(resources)
    if (!stat.isDirectory() || stat.isSymbolicLink()) throw new Error('AppImage resources must be a real directory.')
    const existing = await lstat(support).catch(error => { if (error.code !== 'ENOENT') throw error; return null })
    if (existing && (!existing.isDirectory() || existing.isSymbolicLink())) throw new Error('AppImage setup resources cannot be symlinks.')
    if (!existing) await mkdir(support)
    await copyFile(path.join(root, 'build/linux/install-sandbox-support.sh'), path.join(support, 'install-sandbox-support.sh'))
    await chmod(path.join(support, 'install-sandbox-support.sh'), 0o755)
    for (const name of (await readdir(tree)).filter(name => name.endsWith('.desktop'))) {
      const desktop = path.join(tree, name)
      await regular(desktop)
      const text = (await readFile(desktop, 'utf8')).replace(/\s+--no-sandbox(?=\s|$)/g, '')
      if (/--(?:no-sandbox|disable-(?:sandbox|setuid-sandbox|seccomp-filter-sandbox|namespace-sandbox))/.test(text)) throw new Error('The AppImage desktop entry cannot disable sandboxing.')
      await writeFile(desktop, text)
    }
    await run('/bin/bash', ['-n', path.join(tree, 'AppRun')])
    await run('/bin/bash', ['-n', path.join(support, 'install-sandbox-support.sh')])
    await run('mksquashfs', [tree, output, '-offset', String(layout.offset), '-all-root', '-noappend', '-no-progress', '-quiet', '-no-xattrs', '-no-fragments', '-comp', 'gzip', '-processors', '2'], { timeout: 120000 })
    const handle = await open(output, 'r+')
    try { await handle.writeFile(layout.prefix.subarray(0, layout.offset)); await handle.sync() } finally { await handle.close() }
    const completed = await readAppImageLayout(output)
    if (completed.offset !== layout.offset || completed.arch !== layout.arch) throw new Error('AppImage reconstruction changed its runtime or target architecture.')
    await chmod(output, 0o755)
    await rename(output, file)
    const digest = createHash('sha256')
    for await (const bytes of createReadStream(file)) digest.update(bytes)
    return { file, arch: layout.arch, sha256: digest.digest('hex'), sandboxIntegration: 'owner-approved immutable copy and scoped AppArmor policy' }
  } finally { await rm(temporary, { recursive: true, force: true }) }
}
export async function finalizePackagedAppImage(root, arch) {
  const pkg = JSON.parse(await readFile(path.join(root, 'package.json'), 'utf8'))
  const folder = path.join(root, 'release/linux'), prefix = `${pkg.build.productName}-${pkg.version}-linux-`
  const matches = []
  for (const name of (await readdir(folder)).filter(name => name.startsWith(prefix) && name.endsWith('.AppImage'))) {
    if ((await readAppImageLayout(path.join(folder, name))).arch === arch) matches.push(path.join(folder, name))
  }
  if (matches.length !== 1) throw new Error('Expected exactly one newly packaged AppImage for this version and architecture.')
  try { return await finalizeLinuxAppImage(matches[0], { root }) }
  catch (error) { await rm(matches[0], { force: true }); throw error }
}
if (process.argv[1] && pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url) {
  if (process.argv.length !== 3) throw new Error('Provide one locally built AppImage path.')
  finalizeLinuxAppImage(path.resolve(process.argv[2])).then(result => console.log(JSON.stringify(result)), error => { console.error(error.message); process.exitCode = 1 })
}
