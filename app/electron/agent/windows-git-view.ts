import { promises as fs } from 'node:fs'
import path from 'node:path'
import { gitPointerRoots } from './sandbox-git-roots'
import { insideWorkspace, type WorkspaceInventory } from './isolated-workspace-files'
import { windowsLocalInput } from './windows-local-input'
import { snapshotWindowsWorkspace } from './windows-workspace-snapshot'

async function pointerText(file: string): Promise<string> {
  if (!await windowsLocalInput(file)) throw new Error('Git metadata needs local non-reparse paths.')
  const stat = await fs.lstat(file)
  if (!stat.isFile() || stat.size > 65536) throw new Error('Git metadata pointer is not a bounded regular file.')
  return (await fs.readFile(file, 'utf8')).trim()
}

async function validatedPointer(root: string, control: string): Promise<{ git: string; common: string }> {
  const text = await pointerText(control)
  if (!/^gitdir: [^\r\n]+$/.test(text)) throw new Error('Invalid Git metadata pointer.')
  const git = path.resolve(path.dirname(control), text.slice(8))
  if (!await windowsLocalInput(git)) throw new Error('Git metadata cannot use network or reparse paths.')
  const commonFile = path.join(git, 'commondir')
  const commonExists = await fs.lstat(commonFile).then(() => true, error => { if (error.code === 'ENOENT') return false; throw error })
  if (commonExists) {
    const common = path.resolve(git, await pointerText(commonFile))
    const back = path.resolve(git, await pointerText(path.join(git, 'gitdir')))
    for (const file of [common, path.join(common, 'HEAD'), path.join(common, 'objects'), back]) {
      if (!await windowsLocalInput(file)) throw new Error('Git worktree metadata cannot use network or reparse paths.')
    }
  } else {
    if (!insideWorkspace(root, git)) throw new Error('An external Git directory needs a validated worktree backlink.')
    return { git, common: git }
  }
  return gitPointerRoots(control)
}

export async function prepareWindowsGitView(root: string, workspace: string, container: string, original: WorkspaceInventory): Promise<void> {
  let count = 0
  for (const [relative, entry] of original) {
    if (!/(^|\/)\.git$/i.test(relative) || entry.kind !== 'file') continue
    const roots = await validatedPointer(root, path.join(root, relative))
    const target = path.join(workspace, relative)
    if (insideWorkspace(root, roots.git)) {
      await fs.writeFile(target, `gitdir: ${path.join(workspace, path.relative(root, roots.git))}\n`)
      continue
    }
    if (++count > 32) throw new Error('Too many external Git worktree metadata directories.')
    const common = path.join(container, 'git-metadata', String(count))
    await fs.mkdir(common, { recursive: true, mode: 0o700 })
    await snapshotWindowsWorkspace(roots.common, common, true)
    await fs.unlink(target)
    await fs.mkdir(target, { mode: 0o700 })
    await snapshotWindowsWorkspace(roots.git, target, true)
    await fs.writeFile(path.join(target, 'commondir'), common + '\n')
    await fs.writeFile(path.join(target, 'gitdir'), target + '\n')
  }
}
