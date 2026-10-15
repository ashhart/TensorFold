import { useState } from 'react'
import clsx from 'clsx'
import { useStore } from '../../lib/store'
import { Icon } from '../../lib/icons'
import { FileGlyph } from '../../lib/fileicon'
import { basename } from '../../lib/format'
import type { GitDiffFile } from '../../lib/types'
import { STATUS_TINT } from './primitives'

function diffLineClass(line: string) {
  if (line.startsWith('@@')) return 'bg-accent/10 text-accent'
  if (line.startsWith('+') && !line.startsWith('+++')) return 'bg-positive/10 text-positive'
  if (line.startsWith('-') && !line.startsWith('---')) return 'bg-danger/10 text-danger'
  if (line.startsWith('diff --git') || line.startsWith('index ') || line.startsWith('---') || line.startsWith('+++')) return 'text-ink-3'
  return 'text-ink-2'
}

export function ReviewChangesModal({
  files,
  activePath,
  loading,
  error,
  onPick,
  onClose,
  threadId,
}: {
  files: GitDiffFile[]
  activePath: string | null
  loading: boolean
  error: string | null
  onPick: (path: string) => void
  onClose: () => void
  threadId: string
}) {
  const setView = useStore((s) => s.setView)
  const seedComposer = useStore((s) => s.seedComposer)
  // Wrap toggle (VST-040): long diff lines either scroll horizontally (default, faithful) or
  // wrap to the panel width. Remembered so the choice sticks across files.
  const [wrap, setWrap] = useState(() => { try { return localStorage.getItem('vontra.diff.wrap') === '1' } catch { return false } })
  const toggleWrap = () => setWrap((w) => { const next = !w; try { localStorage.setItem('vontra.diff.wrap', next ? '1' : '0') } catch { /* ignore */ } return next })
  const active = files.find((f) => f.path === activePath) ?? files[0] ?? null
  const additions = files.reduce((sum, f) => sum + f.additions, 0)
  const deletions = files.reduce((sum, f) => sum + f.deletions, 0)
  const lines = active?.diff.split('\n') ?? []

  // Ask why this changed (VST-040): seed the composer with a reference naming the file + a
  // bounded hunk excerpt, then jump to the thread so the user can send the question.
  const askWhy = () => {
    if (!active) return
    const hunk = active.diff.split('\n').filter((l) => /^[+-]/.test(l) && !/^[+-]{3}/.test(l)).slice(0, 24).join('\n')
    const reference = `Why did \`${active.path}\` change this way?\n\n\`\`\`diff\n${hunk || active.diff.slice(0, 1500)}\n\`\`\`\n\n`
    seedComposer(reference)
    setView({ name: 'thread', id: threadId })
    onClose()
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/55 p-6">
      <div className="flex h-[min(760px,calc(100vh-56px))] w-[min(1040px,calc(100vw-48px))] flex-col overflow-hidden rounded-2xl border border-line bg-card shadow-pop">
        <div className="flex items-center gap-3 border-b border-line px-4 py-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-ink-5 text-ink-2">
            <Icon name="file" size={17} />
          </div>
          <div className="min-w-0 flex-1">
            <div className="text-[13px] font-semibold text-ink">Review changes</div>
            <div className="mt-0.5 flex items-center gap-2 font-mono text-[12px]">
              <span className="text-ink-3">{files.length} file{files.length === 1 ? '' : 's'}</span>
              <span className="text-positive">+{additions}</span>
              <span className="text-danger">−{deletions}</span>
            </div>
          </div>
          <button onClick={onClose} className="flex h-8 w-8 items-center justify-center rounded-lg text-ink-3 transition-colors hover:bg-ink-5 hover:text-ink" aria-label="Close review">
            <Icon name="x" size={15} />
          </button>
        </div>

        <div className="grid min-h-0 flex-1 grid-cols-[260px_1fr]">
          <div className="min-h-0 overflow-y-auto border-r border-line bg-card-2 p-2">
            {files.length === 0 && !loading ? (
              <div className="px-3 py-8 text-center text-[13px] text-ink-3">{error || 'No textual changes to review.'}</div>
            ) : files.map((f) => {
              const activeFile = active?.path === f.path
              return (
                <button
                  key={f.path}
                  onClick={() => onPick(f.path)}
                  className={clsx('flex w-full items-center gap-2 rounded-lg px-2 py-2 text-left transition-colors', activeFile ? 'bg-card text-ink shadow-card' : 'text-ink-2 hover:bg-ink-5')}
                >
                  <FileGlyph path={f.path} size={16} />
                  <span className="min-w-0 flex-1 truncate text-[13px] font-medium">{basename(f.path)}</span>
                  <span className={clsx('font-mono text-[12px]', STATUS_TINT[f.status])}>{f.status === '?' ? 'new' : f.status}</span>
                  <span className="font-mono text-[12px] text-positive">+{f.additions}</span>
                  <span className="font-mono text-[12px] text-danger">−{f.deletions}</span>
                </button>
              )
            })}
          </div>

          <div className="flex min-w-0 flex-col">
            <div className="flex items-center gap-2 border-b border-line px-3.5 py-2.5">
              {active ? (
                <>
                  <FileGlyph path={active.path} size={15} />
                  <span className="min-w-0 flex-1 truncate text-[13px] font-medium text-ink">{active.path}</span>
                  <span className="font-mono text-[12px] text-positive">+{active.additions}</span>
                  <span className="font-mono text-[12px] text-danger">−{active.deletions}</span>
                  <button onClick={toggleWrap} title={wrap ? 'Scroll long lines' : 'Wrap long lines'} className={clsx('ml-1 rounded-md px-2 py-0.5 text-[12px] font-medium transition-colors', wrap ? 'bg-ink-5 text-ink' : 'text-ink-3 hover:bg-ink-5 hover:text-ink')}>Wrap</button>
                  <button onClick={askWhy} title="Ask the agent why this file changed" className="flex items-center gap-1 rounded-md border border-line px-2 py-0.5 text-[12px] font-medium text-ink-2 transition-colors hover:bg-ink-5 hover:text-ink"><Icon name="spark" size={11} className="text-accent" />Ask why</button>
                </>
              ) : (
                <span className="text-[13px] text-ink-3">{loading ? 'Loading diff…' : 'No file selected'}</span>
              )}
            </div>
            <div className="min-h-0 flex-1 overflow-auto bg-canvas">
              {loading ? (
                <div className="p-6 text-[13px] text-ink-3">Loading diff…</div>
              ) : error ? (
                <div className="p-6 text-[13px] text-danger">{error}</div>
              ) : active ? (
                <pre className={clsx('select-text py-2 font-mono text-[13px] leading-5', wrap ? 'whitespace-pre-wrap break-words' : 'min-w-max')}>
                  {lines.map((line, i) => (
                    <div key={`${active.path}:${i}`} className={clsx('px-3', wrap && 'flex gap-4', diffLineClass(line))}>
                      <span className={clsx('inline-block w-10 select-none text-right text-ink-4', !wrap && 'mr-4', wrap && 'shrink-0')}>{i + 1}</span>
                      <span className={clsx(wrap && 'min-w-0 flex-1')}>{line || ' '}</span>
                    </div>
                  ))}
                </pre>
              ) : (
                <div className="p-6 text-[13px] text-ink-3">No textual changes to review.</div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
