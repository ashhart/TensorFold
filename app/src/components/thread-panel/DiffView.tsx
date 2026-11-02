import { useMemo } from 'react'
import clsx from 'clsx'
import { pairWordDiffs, type WordSpan } from '../../lib/diff-words'
import type { DiffLine } from '../../lib/types'

// The transcript and Changes pane share readable line numbers and word-level change marks.

function Words({ spans, kind }: { spans: WordSpan[]; kind: 'add' | 'del' }) {
  return (
    <>
      {spans.map((s, i) => s.changed
        ? <mark key={i} className={clsx('rounded-[2px] px-0.5', kind === 'add' ? 'bg-[var(--add-ink)]/20 text-[var(--add-ink)]' : 'bg-[var(--del-ink)]/20 text-[var(--del-ink)]')}>{s.text || ' '}</mark>
        : <span key={i}>{s.text}</span>)}
    </>
  )
}

export function DiffView({ diff, wrap = true, label = 'File diff' }: { diff: DiffLine[]; wrap?: boolean; label?: string }) {
  const words = useMemo(() => pairWordDiffs(diff), [diff])
  return (
    <div tabIndex={0} role="region" aria-label={label} className="scroll-thin min-w-0 overflow-x-auto font-mono text-[12px] leading-5 focus-visible:outline focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-accent">
      <div className={wrap ? 'min-w-0' : 'min-w-max'}>
        {diff.map((line, i) => {
          if (line.type === 'hunk') return <div key={i} className="select-none whitespace-pre-wrap break-words border-y border-line/60 bg-surface px-3 py-1.5 text-[11px] text-ink-3">{line.text}</div>
          const pair = words.get(i)
          return (
            <div key={i} className={clsx('flex', line.type === 'add' && 'bg-[var(--add-bg)]', line.type === 'del' && 'bg-[var(--del-bg)]')}>
              <span aria-hidden="true" className="w-8 shrink-0 select-none pr-1 text-right text-ink-4">{line.oldNo ?? ''}</span>
              <span aria-hidden="true" className="w-8 shrink-0 select-none pr-1 text-right text-ink-4">{line.newNo ?? ''}</span>
              <span className={clsx('w-4 shrink-0 select-none text-center', line.type === 'add' ? 'text-[var(--add-ink)]' : line.type === 'del' ? 'text-[var(--del-ink)]' : 'text-ink-4')}>
                {line.type === 'add' ? '+' : line.type === 'del' ? '−' : ''}
              </span>
              <span className={clsx('min-w-0 flex-1 pr-3', wrap ? 'whitespace-pre-wrap [overflow-wrap:anywhere]' : 'whitespace-pre', line.type === 'add' ? 'text-[var(--add-ink)]' : line.type === 'del' ? 'text-[var(--del-ink)]' : 'text-ink-2')}>
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
