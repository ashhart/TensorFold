import { existsSync, lstatSync, realpathSync } from 'node:fs'
import path from 'node:path'

export function safeResolve(cwd: string, relPath: string | undefined): string {
  const root = path.resolve(cwd)
  const realRoot = realpathSync.native(root)
  const target = path.resolve(root, relPath || '.')
  if (target !== root && !target.startsWith(root + path.sep)) {
    throw new Error(`Path "${relPath}" escapes the workspace.`)
  }
  let probe = target
  while (!existsSync(probe)) {
    const parent = path.dirname(probe)
    if (parent === probe) break
    probe = parent
  }
  const realProbe = realpathSync.native(probe)
  if (realProbe !== realRoot && !realProbe.startsWith(realRoot + path.sep)) {
    throw new Error(`Path "${relPath}" escapes the workspace through a symlink.`)
  }
  // The ancestor-walk only validates the nearest EXISTING ancestor. A symlink at the FINAL
  // component slips through when it's dangling: existsSync() follows the link and reports
  // "missing", so the walk climbs to the (in-workspace) parent and passes — yet a write would
  // follow the link to its out-of-workspace destination. lstat the final component directly
  // (no-follow) and reject any symlink that escapes (a dangling link can't be proven safe).
  let finalStat: ReturnType<typeof lstatSync> | null = null
  try { finalStat = lstatSync(target) } catch { finalStat = null } // truly absent => fine
  if (finalStat?.isSymbolicLink()) {
    let linkReal: string | null = null
    try { linkReal = realpathSync.native(target) } catch { linkReal = null } // dangling link
    if (linkReal === null || (linkReal !== realRoot && !linkReal.startsWith(realRoot + path.sep))) {
      throw new Error(`Path "${relPath}" escapes the workspace through a symlink.`)
    }
  }
  return target
}

export function rel(cwd: string, abs: string): string {
  return path.relative(path.resolve(cwd), abs) || '.'
}
