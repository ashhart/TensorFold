import { useState } from 'react'
import type { ComposerBudget } from '../../lib/composer-budget'
import { currentIntlLocale } from '../../lib/preferences/current-locale'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'

export function ContextBudget({ budget }: { budget: ComposerBudget }) {
  const tx = useAuthoredText(), [open, setOpen] = useState(false)
  const format = (value: number) => Math.ceil(value).toLocaleString(currentIntlLocale())
  const warning = budget.remaining !== null && budget.remaining < 0
  return <div className="px-4 pb-1 text-[11px] text-ink-3">
    <button type="button" aria-expanded={open} aria-label={tx('Context budget before sending')} onClick={() => setOpen(value => !value)} className={`rounded-md py-1 text-left focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent ${warning ? 'text-warn' : 'hover:text-ink'}`}>
      {budget.window === null ? tx('Estimated context: {tokens} tokens; model limit unavailable', { tokens: format(budget.total) }) : tx('Estimated context: {tokens} of {limit} tokens', { tokens: format(budget.total), limit: format(budget.window) })}{budget.partial ? ' · ' + tx('Partial') : ' · ' + tx('Approximate')}
    </button>
    {open && <section aria-label={tx('Context budget breakdown')} className="mb-2 rounded-lg border border-line bg-card-2 p-3 text-[12px]">
      <dl className="grid grid-cols-[1fr_auto] gap-x-4 gap-y-1">
        <dt>{tx('Conversation history')}</dt><dd>{format(budget.history)}</dd>
        <dt>{tx('System and instructions')}</dt><dd>{format(budget.system)}</dd>
        <dt>{tx('Current draft')}</dt><dd>{format(budget.draft)}</dd>
        <dt>{tx('Selected files')}</dt><dd>{format(budget.files)}</dd>
        {!!budget.quoted && <><dt>{tx('Quoted context')}</dt><dd>{format(budget.quoted)}</dd></>}
        <dt>{tx('Reserved output')}</dt><dd>{budget.outputReserve === null ? tx('Not reported') : format(budget.outputReserve)}</dd>
      </dl>
      <p className="mt-2 leading-relaxed">{tx('Approximate text tokens only. Images, model tokenization, skills, dynamic instructions and tools can change the actual request; this does not guarantee admission.')}</p>
      {budget.outputReserve === null && <p className="mt-1">{tx('The provider has not reported a reply limit; reserved output is not included in the total.')}</p>}
      {budget.stale && <p role="status" className="mt-1">{tx('History is changing or its snapshot is stale; this is a partial preview.')}</p>}
      {budget.unknownImages > 0 && <p className="mt-1">{tx('Image token costs are unknown and are not included.')}</p>}
      {budget.unreadyFiles > 0 && <p role="status" className="mt-1 text-warn">{tx('Some files still need review, retry or reselection before sending.')}</p>}
      {warning && <p role="status" className="mt-1 text-warn">{tx('The estimated request exceeds the selected context window. Shorten the draft or files, compact history, or choose a larger model profile.')}</p>}
    </section>}
  </div>
}
