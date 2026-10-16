import { useMemo } from 'react'
import clsx from 'clsx'
import { pairWordDiffs, type WordSpan } from '../../lib/diff-words'
import type { DiffLine } from '../../lib/types'

// Line diff with old/new numbers and word-level marks, the same look as the transcript's tool
// diffs. The transcript keeps its own copy in timeline.tsx, which this pane may not edit.

function Words({ spans, kind }: { spans: WordSpan[]; kind: 'add' | 'del' }) {
  return (
    <>
      {spans.map((s, i) => s.changed
        ? <mark key={i} className={clsx('rounded-[2px] px-0.5', kind === 'add' ? 'bg-[var(--add-ink)]/20 text-[var(--add-ink)]' : 'bg-[var(--del-ink)]/20 text-[var(--del-ink)]')}>{s.text || ' '}</mark>
        : <span key={i}>{s.text}</span>)}
    </>
  )
}

export function DiffView({ diff }: { diff: DiffLine[] }) {
  const words = useMemo(() => pairWordDiffs(diff), [diff])
  return (
    <div className="scroll-thin overflow-x-auto font-mono text-[13px] leading-[18px]">
      <div className="min-w-max">
        {diff.map((line, i) => {
          if (line.type === 'hunk') return <div key={i} className="select-none px-3 py-1 text-[12px] text-ink-3">{line.text}</div>
          const pair = words.get(i)
          return (
            <div key={i} className={clsx('flex', line.type === 'add' && 'bg-[var(--add-bg)]', line.type === 'del' && 'bg-[var(--del-bg)]')}>
              <span className="w-9 shrink-0 select-none pr-1 text-right text-ink-4">{line.oldNo ?? ''}</span>
              <span className="w-9 shrink-0 select-none pr-1 text-right text-ink-4">{line.newNo ?? ''}</span>
              <span className={clsx('w-4 shrink-0 select-none text-center', line.type === 'add' ? 'text-[var(--add-ink)]' : line.type === 'del' ? 'text-[var(--del-ink)]' : 'text-ink-4')}>
                {line.type === 'add' ? '+' : line.type === 'del' ? '−' : ''}
              </span>
              <span className={clsx('whitespace-pre pr-3', line.type === 'add' ? 'text-[var(--add-ink)]' : line.type === 'del' ? 'text-[var(--del-ink)]' : 'text-ink-2')}>
                {pair && line.type === 'del' ? <Words spans={pair.removed} kind="del" />
                  : pair && line.type === 'add' ? <Words spans={pair.added} kind="add" />
                    : (line.text || ' ')}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
