import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { QuickAskApi, QuickAskTurn } from '../../../shared/quick-ask'
import { Icon } from '../../lib/icons'
import type { UiTextParameters } from '../../lib/preferences/language-packs'

export function AskAnswer({ turn, api, tx, speak, error }: { turn: QuickAskTurn; api: QuickAskApi; tx(source: string, params?: UiTextParameters): string; speak(text: string): void; error(error: unknown): void }) {
  const [copied, setCopied] = useState(false)
  const copy = async (text: string) => { try { await api.copy(text); setCopied(true); setTimeout(() => setCopied(false), 1800) } catch (failure) { error(failure) } }
  return <article className="ask-turn" aria-label={tx('Ask and answer')}>
    <div className="mb-4 text-sm leading-relaxed">
      <div className="mb-1 text-[11px] font-semibold uppercase tracking-wider text-ink-3">{tx('You')}</div>
      <p className="whitespace-pre-wrap break-words text-ink">{turn.text}</p>
      {turn.attachments.length > 0 && <div className="mt-2 flex flex-wrap gap-2">{turn.attachments.map(item => <span key={item.id} className="rounded-md border border-line bg-card-2 px-2 py-1 text-xs text-ink-2">{item.name}</span>)}</div>}
    </div>
    {turn.tools.length > 0 && <div className="mb-3 space-y-1" aria-live="polite">{turn.tools.map((tool, index) => <p key={index} className="flex items-center gap-2 text-xs text-ink-3"><Icon name="globe" size={12} /><span>{tx(tool.name === 'web_search' ? 'Web search' : 'Read link')} · {tool.detail} · {tx(tool.status === 'running' ? 'Working…' : tool.status === 'error' ? 'Failed' : 'Done')}</span></p>)}</div>}
    {turn.reasoning && <details className="mb-3 rounded-lg border border-line px-3 py-2 text-xs text-ink-3"><summary className="cursor-pointer">{tx('Reasoning')}</summary><p className="mt-2 whitespace-pre-wrap">{turn.reasoning}</p></details>}
    <div className="ask-answer prose-tf text-[14px] leading-[1.7] text-ink" aria-busy={turn.status === 'running'}>
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={{
        a: ({ href, children }) => <a href={href} onClick={event => { event.preventDefault(); if (href) void api.openLink(href).catch(error) }}>{children}</a>,
        img: ({ alt }) => <span className="text-ink-3">{alt ?? tx('Image omitted')}</span>,
        pre: ({ children }) => <div className="group relative"><button className="ask-text-button absolute right-2 top-2 bg-card opacity-0 focus:opacity-100 group-hover:opacity-100" onClick={event => { const text = event.currentTarget.parentElement?.querySelector('pre')?.textContent; if (text) void copy(text) }}>{tx('Copy code')}</button><pre>{children}</pre></div>,
      }}>{turn.answer}</ReactMarkdown>
      {!turn.answer && turn.status === 'running' && <p className="flex items-center gap-2 text-ink-3"><span className="ask-pulse h-1.5 w-1.5 rounded-full bg-accent" />{tx('Thinking…')}</p>}
    </div>
    {turn.error && <p className="mt-3 rounded-lg border border-line bg-card-2 p-3 text-xs leading-relaxed text-ink-2" role="status">{tx(turn.error)}</p>}
    {turn.answer && <div className="mt-3 flex items-center gap-3"><button className="ask-text-button" onClick={() => void copy(turn.answer)}><Icon name={copied ? 'check' : 'copy'} size={12} />{tx(copied ? 'Copied' : 'Copy answer')}</button><button className="ask-text-button" onClick={() => speak(turn.answer)}><Icon name="speaker" size={13} />{tx('Read aloud')}</button></div>}
  </article>
}
