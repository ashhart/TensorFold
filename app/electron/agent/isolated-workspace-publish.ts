import { constants, promises as fs } from 'node:fs'
import path from 'node:path'
import { randomUUID } from 'node:crypto'
import {
  protectedGitPath, sameWorkspaceContent, scanWorkspace, snapshotWorkspaceFile,
  type WorkspaceEntry, type WorkspaceInventory,
} from './isolated-workspace-files'
import type { IsolatedWorkspace } from './isolated-workspace'
import { snapshotWindowsWorkspace } from './windows-workspace-snapshot'

interface Publication {
  relative: string
  source: string | null
  next?: WorkspaceEntry
  previous?: WorkspaceEntry
  incoming: string | null
  backup: string
  backedUp: boolean
  published: boolean
  publishedIdentity?: { dev: number; ino: number }
}
const locks = new Map<string, Promise<void>>()

async function locked<T>(root: string, action: () => Promise<T>): Promise<T> {
  const previous = locks.get(root) ?? Promise.resolve()
  let release!: () => void
  const next = new Promise<void>(resolve => { release = resolve })
  const queue = previous.then(() => next)
  locks.set(root, queue)
  await previous
  try { return await action() } finally { release(); if (locks.get(root) === queue) locks.delete(root) }
}

async function assertParents(layout: IsolatedWorkspace, relative: string, created: Map<string, WorkspaceEntry>): Promise<void> {
  const root = await fs.lstat(layout.root)
  if (!root.isDirectory() || root.isSymbolicLink() || root.dev !== layout.rootIdentity.dev || root.ino !== layout.rootIdentity.ino
    || await fs.realpath(layout.root) !== layout.root) throw new Error('Workspace root changed during isolated execution.')
  let directory = path.dirname(relative)
  while (directory !== '.') {
    const expected = created.get(directory) ?? layout.original.get(directory)
    const stat = await fs.lstat(path.join(layout.root, directory))
    if (!expected || !stat.isDirectory() || stat.isSymbolicLink() || expected.dev !== stat.dev || expected.ino !== stat.ino) {
      throw new Error('A workspace parent changed during isolated execution.')
    }
    directory = path.dirname(directory)
  }
}

function assertChangesAllowed(layout: IsolatedWorkspace, proposed: WorkspaceInventory): string[] {
  const changed: string[] = []
  for (const relative of new Set([...layout.initial.keys(), ...proposed.keys()])) {
    const before = layout.initial.get(relative), after = proposed.get(relative)
    if (sameWorkspaceContent(before, after)) continue
    if (protectedGitPath(relative)) throw new Error('Git metadata cannot be published from an isolated process.')
    if (before?.kind === 'directory' && after?.kind === 'directory' && before.mode !== after.mode) {
      throw new Error('Isolation refuses changes to existing directory permissions.')
    }
    if ((process.platform === 'win32' && (layout.original.get(relative)?.kind === 'link' || before?.kind === 'link' || after?.kind === 'link'))
      || (after?.kind === 'file' && after.nlink !== 1)) {
      throw new Error('Isolation refuses changed symlinks, reparse points or hardlinks.')
    }
    if (before && after && before.kind !== after.kind && (before.kind === 'directory' || after.kind === 'directory')) throw new Error('Isolation refuses a changed directory entry type.')
    if (before?.kind === 'file' || after?.kind === 'file' || before?.kind === 'link' || after?.kind === 'link') changed.push(relative)
  }
  return changed.sort()
}

async function preparePublications(layout: IsolatedWorkspace, proposed: WorkspaceInventory, changed: string[], snapshot: string, view: string): Promise<Publication[]> {
  const current = await scanWorkspace(layout.root, undefined, { externalReadRoots: layout.externalReadRoots })
  let total = 0
  const publications: Publication[] = []
  for (const relative of changed) {
    const previous = layout.original.get(relative), actual = current.get(relative), next = proposed.get(relative)
    if (!sameWorkspaceContent(previous, actual) || previous?.dev !== actual?.dev || previous?.ino !== actual?.ino) {
      throw new Error(`Workspace changed concurrently; no isolated changes were published: ${relative}`)
    }
    const source = next?.kind === 'file' ? path.join(snapshot, String(publications.length)) : null
    if (source) {
      const copied = await snapshotWorkspaceFile(path.join(view, relative), source)
      if (copied.sha256 !== next!.sha256) throw new Error('An isolated file changed during publication.')
      total += copied.size
      if (total > 2 * 1024 * 1024 * 1024) throw new Error('Isolated changes exceed the publication size limit.')
    }
    publications.push({ relative, source, previous, next, incoming: null,
      backup: path.join(layout.root, path.dirname(relative), `.tensorfold-isolation-backup-${randomUUID()}`),
      backedUp: false, published: false })
  }
  return publications
}

async function assertDestination(layout: IsolatedWorkspace, entry: Publication): Promise<void> {
  const target = path.join(layout.root, entry.relative)
  const stat = await fs.lstat(target).catch(error => { if (error.code !== 'ENOENT') throw error; return null })
  if (!entry.previous) { if (stat) throw new Error('A concurrent file prevents isolated publication.'); return }
  if (entry.previous.kind === 'link') {
    if (!stat?.isSymbolicLink() || stat.dev !== entry.previous.dev || stat.ino !== entry.previous.ino
      || await fs.readlink(target) !== entry.previous.rawLink) throw new Error('Workspace link changed before publication.')
    return
  }
  if (!stat?.isFile() || stat.isSymbolicLink() || stat.dev !== entry.previous.dev || stat.ino !== entry.previous.ino
    || (stat.mode & 0o777) !== entry.previous.mode
    || (await snapshotWorkspaceFile(target)).sha256 !== entry.previous.sha256) throw new Error('Workspace destination changed before publication.')
}

export async function publishIsolatedWorkspace(layout: IsolatedWorkspace, signal?: AbortSignal, beforeCommit?: (relative: string) => Promise<void>): Promise<string[]> {
  return locked(layout.root, async () => {
    if (signal?.aborted) throw new Error('Isolated publication was cancelled.')
    const snapshot = await fs.mkdtemp(path.join(layout.container, 'publication-'))
    try {
    let view = layout.workspace
    if (process.platform === 'win32') {
      view = path.join(snapshot, 'captured')
      await fs.mkdir(view, { mode: 0o700 })
      await snapshotWindowsWorkspace(layout.workspace, view, false)
    }
    const proposed = await scanWorkspace(view, undefined, { externalReadRoots: layout.externalReadRoots, virtualRoot: layout.root })
    const changed = assertChangesAllowed(layout, proposed)
    const entries = await preparePublications(layout, proposed, changed, snapshot, view)
    const created = new Map<string, WorkspaceEntry>()
    try {
      const directories = [...proposed].filter(([, entry]) => entry.kind === 'directory')
        .sort(([a], [b]) => a.split('/').length - b.split('/').length)
      for (const [relative, entry] of directories) {
        if (layout.original.has(relative)) continue
        await assertParents(layout, relative, created)
        await fs.mkdir(path.join(layout.root, relative), { mode: entry.mode })
        await fs.chmod(path.join(layout.root, relative), entry.mode)
        const stat = await fs.lstat(path.join(layout.root, relative))
        created.set(relative, { ...entry, dev: stat.dev, ino: stat.ino })
      }
      for (const entry of entries) {
        await assertParents(layout, entry.relative, created)
        await assertDestination(layout, entry)
        if (entry.next) {
          entry.incoming = path.join(layout.root, path.dirname(entry.relative), `.tensorfold-isolation-next-${randomUUID()}`)
          if (entry.next.kind === 'link') await fs.symlink(entry.next.link!, entry.incoming)
          else { await fs.copyFile(entry.source!, entry.incoming, constants.COPYFILE_EXCL); await fs.chmod(entry.incoming, entry.next.mode) }
        }
      }
      for (const entry of entries) {
        if (signal?.aborted) throw new Error('Isolated publication was cancelled.')
        await beforeCommit?.(entry.relative)
        await assertParents(layout, entry.relative, created)
        await assertDestination(layout, entry)
        const target = path.join(layout.root, entry.relative)
        if (entry.previous) { await fs.rename(target, entry.backup); entry.backedUp = true }
        if (entry.incoming) {
          const identity = await fs.lstat(entry.incoming)
          entry.publishedIdentity = { dev: identity.dev, ino: identity.ino }
          await fs.rename(entry.incoming, target); entry.published = true
        }
      }
    } catch (error) {
      const retained: string[] = []
      for (const entry of [...entries].reverse()) {
        try {
          await assertParents(layout, entry.relative, created)
          const target = path.join(layout.root, entry.relative)
          if (entry.published) {
            const identity = await fs.lstat(target)
            if (identity.dev !== entry.publishedIdentity?.dev || identity.ino !== entry.publishedIdentity?.ino) {
              throw new Error('Preserving a replacement file during rollback.')
            }
            if (entry.next!.kind === 'link') {
              const stat = await fs.lstat(target)
              if (!stat.isSymbolicLink() || await fs.readlink(target) !== entry.next!.link) throw new Error('Preserving a newer link during rollback.')
            } else if ((await snapshotWorkspaceFile(target)).sha256 !== entry.next!.sha256) throw new Error('Preserving a newer edit during rollback.')
            await fs.unlink(target)
          }
          if (entry.backedUp) {
            if (await fs.lstat(target).then(() => true, e => { if (e.code === 'ENOENT') return false; throw e })) throw new Error('A newer file prevents rollback.')
            await fs.rename(entry.backup, target)
          }
          if (entry.incoming) await fs.rm(entry.incoming, { force: true })
        } catch { if (entry.backedUp) retained.push(entry.backup) }
      }
      for (const directory of [...created.keys()].reverse()) await fs.rmdir(path.join(layout.root, directory)).catch(() => {})
      if (retained.length) throw new Error(`Publication stopped; newer edits were preserved. Recovery files: ${retained.join(', ')}`)
      throw error
    }
    for (const entry of entries) if (entry.backedUp) await fs.unlink(entry.backup)
    for (const [relative, entry] of [...layout.original].reverse()) {
      if (entry.kind === 'directory' && !proposed.has(relative)) {
        await assertParents(layout, relative, created)
        await fs.rmdir(path.join(layout.root, relative)).catch(error => { if (!['ENOTEMPTY', 'ENOENT'].includes(error.code)) throw error })
      }
    }
    layout.original = await scanWorkspace(layout.root, undefined, { externalReadRoots: layout.externalReadRoots })
    layout.initial = proposed
    return changed
    } finally { await fs.rm(snapshot, { recursive: true, force: true }) }
  })
}
