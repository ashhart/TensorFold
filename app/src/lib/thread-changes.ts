import type { DiffLine, GitFileChange, GitStatus, Thread } from './types'

// The Changes tab model: files this thread changed, grouped by the turn that last touched
// them (newest turn first), joined with the working tree's git status. Pure; no IPC.

export interface ChangedFile {
  path: string
  status: GitFileChange['status'] | null
  additions: number
  deletions: number
  // The thread's latest edit of this file, for when git has no diff to show.
  editDiff: DiffLine[] | null
}

export interface ChangeTurn {
  key: string
  turn: number | null
  prompt: string | null
  files: ChangedFile[]
}

// Repo-relative form of a tool path, so it lines up with `git status` paths.
export function relativeToCwd(path: string, cwd: string): string {
  const clean = path.replace(/^\.\//, '')
  if (!cwd) return clean
  const root = cwd.replace(/\/+$/, '') + '/'
  return clean.startsWith(root) ? clean.slice(root.length) : clean
}

function countLines(diff: DiffLine[]): { additions: number; deletions: number } {
  let additions = 0
  let deletions = 0
  for (const line of diff) {
    if (line.type === 'add') additions++
    else if (line.type === 'del') deletions++
  }
  return { additions, deletions }
}

function findGitFile(byPath: Map<string, GitFileChange>, rel: string): GitFileChange | undefined {
  const exact = byPath.get(rel)
  if (exact) return exact
  // The thread may run in a subfolder of the repo: match on the path suffix.
  for (const [path, file] of byPath) if (path.endsWith('/' + rel)) return file
  return undefined
}

interface Touch { turn: number; diff: DiffLine[] | null }

export function threadChangeTurns(thread: Pick<Thread, 'timeline'>, git: GitStatus | null, cwd: string): ChangeTurn[] {
  const prompts: string[] = []
  const touched = new Map<string, Touch>()
  let turn = 0
  for (const item of thread.timeline) {
    if (item.kind === 'user') { turn++; prompts[turn] = (item.text ?? '').trim(); continue }
    if (item.kind !== 'tool' || (item.state && item.state !== 'ok')) continue
    const at = Math.max(1, turn)
    const paths = new Set<string>(item.changedPaths ?? [])
    if (item.path && (item.diff?.length || item.tool === 'edit_file' || item.tool === 'write_file')) paths.add(item.path)
    for (const path of paths) {
      const rel = relativeToCwd(path, cwd)
      const diff = path === item.path && item.diff?.length ? item.diff : touched.get(rel)?.diff ?? null
      touched.delete(rel)
      touched.set(rel, { turn: at, diff })
    }
  }

  const byPath = new Map((git?.files ?? []).map((f) => [f.path, f] as const))
  const claimed = new Set<string>()
  const byTurn = new Map<number, ChangedFile[]>()
  for (const [rel, touch] of touched) {
    const gitFile = findGitFile(byPath, rel)
    if (gitFile) claimed.add(gitFile.path)
    const counted = touch.diff ? countLines(touch.diff) : { additions: 0, deletions: 0 }
    const file: ChangedFile = {
      path: gitFile?.path ?? rel,
      status: gitFile?.status ?? null,
      additions: gitFile?.additions ?? counted.additions,
      deletions: gitFile?.deletions ?? counted.deletions,
      editDiff: touch.diff,
    }
    byTurn.set(touch.turn, [...(byTurn.get(touch.turn) ?? []), file])
  }

  const turns: ChangeTurn[] = [...byTurn.entries()]
    .sort((a, b) => b[0] - a[0])
    .map(([n, files]) => ({ key: `turn-${n}`, turn: n, prompt: prompts[n] || null, files: files.sort((a, b) => a.path.localeCompare(b.path)) }))

  const others = (git?.files ?? []).filter((f) => !claimed.has(f.path))
  if (others.length) {
    turns.push({
      key: 'other',
      turn: null,
      prompt: null,
      files: others.map((f) => ({ path: f.path, status: f.status, additions: f.additions, deletions: f.deletions, editDiff: null })),
    })
  }
  return turns
}

export function changeTotals(turns: ChangeTurn[]): { files: number; additions: number; deletions: number } {
  let files = 0
  let additions = 0
  let deletions = 0
  for (const turn of turns) for (const f of turn.files) { files++; additions += f.additions; deletions += f.deletions }
  return { files, additions, deletions }
}

// Unified diff text (from `git diff`) into the DiffLine[] shape the diff views render.
export function parseUnifiedDiff(text: string): DiffLine[] {
  const out: DiffLine[] = []
  let oldNo = 0
  let newNo = 0
  let inHunk = false
  for (const raw of text.split('\n')) {
    const hunk = /^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@/.exec(raw)
    if (hunk) {
      oldNo = Number(hunk[1])
      newNo = Number(hunk[2])
      inHunk = true
      out.push({ type: 'hunk', text: raw })
      continue
    }
    if (!inHunk || raw.startsWith('\\')) continue
    if (raw.startsWith('diff --git')) { inHunk = false; continue }
    if (raw.startsWith('+')) out.push({ type: 'add', text: raw.slice(1), newNo: newNo++ })
    else if (raw.startsWith('-')) out.push({ type: 'del', text: raw.slice(1), oldNo: oldNo++ })
    else out.push({ type: 'ctx', text: raw.startsWith(' ') ? raw.slice(1) : raw, oldNo: oldNo++, newNo: newNo++ })
  }
  // A trailing newline in the input leaves one empty context line behind.
  const last = out[out.length - 1]
  if (last?.type === 'ctx' && last.text === '' && text.endsWith('\n')) out.pop()
  return out
}
