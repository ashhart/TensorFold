import { existsSync, lstatSync, readFileSync, realpathSync, statSync } from 'node:fs'
import path from 'node:path'
import { insideWorkspace } from './isolated-workspace-files'

function textFile(file: string): string {
  const stat = lstatSync(file)
  if (!stat.isFile() || stat.isSymbolicLink() || stat.size > 65536) throw new Error('Git metadata pointer is not a bounded regular file.')
  return readFileSync(file, 'utf8').trim()
}

export function gitPointerRoots(control: string): { git: string; common: string } {
  const text = textFile(control)
  if (!/^gitdir: [^\r\n]+$/.test(text)) throw new Error('Invalid Git metadata pointer.')
  const git = realpathSync(path.resolve(path.dirname(control), text.slice(8)))
  if (!statSync(git).isDirectory()) throw new Error('Git metadata directory is invalid.')
  if (!existsSync(path.join(git, 'commondir'))) {
    let ancestor = path.dirname(control)
    while (ancestor !== path.dirname(ancestor)) {
      const repository = path.join(ancestor, '.git')
      if (existsSync(repository) && lstatSync(repository).isDirectory() && insideWorkspace(realpathSync(repository), git)) return { git, common: git }
      ancestor = path.dirname(ancestor)
    }
    throw new Error('An external Git directory needs a validated worktree backlink.')
  }
  const common = realpathSync(path.resolve(git, textFile(path.join(git, 'commondir'))))
  const relative = path.relative(common, git).split(path.sep)
  let backlink = ''
  try { backlink = realpathSync(path.resolve(git, textFile(path.join(git, 'gitdir')))) } catch {}
  const expected = realpathSync(control)
  const matches = process.platform === 'win32' ? backlink.toLowerCase() === expected.toLowerCase() : backlink === expected
  if (path.basename(common).toLowerCase() !== '.git' || relative.length !== 2 || relative[0] !== 'worktrees'
    || !existsSync(path.join(common, 'HEAD')) || !statSync(path.join(common, 'objects')).isDirectory()
    || !matches) {
    throw new Error('Git worktree metadata does not point back to the selected workspace.')
  }
  return { git, common }
}

export function linuxGitReadRoots(cwd: string): string[] {
  let directory = cwd
  while (true) {
    const control = path.join(directory, '.git')
    if (existsSync(control)) {
      const stat = lstatSync(control)
      if (stat.isSymbolicLink()) throw new Error('Git control entries cannot be symbolic links.')
      if (stat.isFile()) { const roots = gitPointerRoots(control); return [...new Set([roots.git, roots.common])] }
      if (stat.isDirectory()) {
        if (!insideWorkspace(cwd, control) && (!existsSync(path.join(control, 'HEAD')) || !existsSync(path.join(control, 'objects')))) {
          throw new Error('An ancestor Git directory is invalid.')
        }
        return [realpathSync(control)]
      }
      throw new Error('Unsupported Git metadata entry.')
    }
    if (directory === path.dirname(directory)) return []
    directory = path.dirname(directory)
  }
}
