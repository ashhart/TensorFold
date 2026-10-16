import { useEffect, useId, useState } from 'react'
import { api } from '../../lib/ipc'
import type { TimelineItem, UserQuestion } from '../../lib/types'
import { Button } from '../ui'

export function QuestionCard({ item }: { item: TimelineItem }) {
  const recorded = item.question
  const [question, setQuestion] = useState<UserQuestion | undefined>(recorded)
  const [selected, setSelected] = useState<number | null>(null)
  const [written, setWritten] = useState('')
  const [checked, setChecked] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [checkTick, setCheckTick] = useState(0)
  const fieldId = useId(), groupId = useId()
  useEffect(() => { setQuestion(recorded); setSelected(null); setWritten(''); setError(null); setChecked(recorded?.state !== 'pending') }, [recorded?.id])
  useEffect(() => { if (recorded && recorded.state !== 'pending') { setQuestion(recorded); setChecked(true) } }, [recorded?.state, recorded?.answer])
  useEffect(() => {
    if (!recorded || recorded.state !== 'pending') return
    let alive = true
    void api.userQuestionState({ threadId: recorded.threadId, questionId: recorded.id }).then((latest) => {
      if (!alive) return
      setQuestion(latest ?? { ...recorded, state: 'interrupted', answer: null, resolvedAt: Date.now() }); setChecked(true)
    }).catch((cause) => { if (alive) setError(cause instanceof Error ? cause.message : 'Could not check this question.') })
    return () => { alive = false }
  }, [recorded?.id, checkTick])
  if (!question) return <div role="status" className="rounded-xl border border-line bg-card p-4 text-[13px] text-ink-2">{['pending', 'running', 'awaiting'].includes(item.state ?? '') ? 'Preparing the question…' : item.output || 'This question ended before it could be saved.'}</div>
  const waiting = question.state === 'pending', remote = api.platform === 'web-remote'
  const answer = written.trim() || (selected == null ? '' : question.choices[selected])
  const send = async (action: 'answer' | 'skip') => {
    if (!checked || busy || !waiting || remote || action === 'answer' && !answer) return
    setBusy(true); setError(null)
    try {
      setQuestion(await api.userQuestionAnswer({ questionId: question.id, threadId: question.threadId, runId: question.runId, ...(action === 'answer' ? { action, answer } : { action }) }))
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not save your answer; try again.')
      const latest = await api.userQuestionState({ threadId: question.threadId, questionId: question.id }).catch(() => null)
      if (latest) setQuestion(latest)
    } finally { setBusy(false) }
  }
  return <section aria-label="Question from TensorFold" aria-busy={busy} className="rounded-xl border border-accent/30 bg-card p-4 shadow-sm">
    <p className="text-[12px] font-medium text-accent">{waiting ? 'A quick question' : question.state === 'answered' ? 'Answered' : question.state === 'skipped' ? 'Skipped' : 'Interrupted'}</p>
    <h3 className="mt-1 break-words text-[15px] font-medium leading-relaxed text-ink">{question.question}</h3>
    {waiting ? <form className="mt-3 space-y-3" onSubmit={(event) => { event.preventDefault(); void send('answer') }}>
      <fieldset disabled={!checked || busy || remote} className="space-y-2">
        <legend className="sr-only">Choose one option or write your answer</legend>
        {question.choices.map((choice, index) => <label key={index} className="flex cursor-pointer items-start gap-2 rounded-lg border border-line px-3 py-2 text-[13px] text-ink has-[:checked]:border-accent has-[:checked]:bg-accent/5"><input type="radio" name={groupId} checked={selected === index && !written.trim()} onChange={() => { setSelected(index); setWritten('') }} className="mt-0.5 accent-[var(--color-accent)]" /><span className="min-w-0 break-words">{choice}</span></label>)}
        <label htmlFor={fieldId} className="block text-[12px] text-ink-2">{question.choices.length ? 'Or write your answer' : 'Your answer'}</label>
        <textarea id={fieldId} value={written} onChange={(event) => { setWritten(event.target.value); setSelected(null) }} maxLength={2000} rows={2} placeholder="Type your answer" className="w-full resize-y rounded-lg border border-line bg-input px-3 py-2 text-[13px] leading-relaxed text-ink outline-none focus:border-accent disabled:opacity-60" />
      </fieldset>
      <p className="text-[12px] leading-relaxed text-ink-3">Your answer guides this chat; tool permissions stay the same.</p>
      {remote && <p className="text-[12px] text-ink-2">Answer this question in the desktop app.</p>}
      {!checked && !error && <p role="status" className="text-[12px] text-ink-2">Checking the active run…</p>}
      <div className="flex flex-wrap gap-2"><Button type="submit" variant="primary" size="sm" disabled={!checked || busy || !answer || remote}>{busy ? 'Saving…' : 'Send answer'}</Button><Button variant="ghost" size="sm" disabled={!checked || busy || remote} onClick={() => void send('skip')}>Skip question</Button></div>
    </form> : <p className="mt-3 whitespace-pre-wrap break-words rounded-lg bg-card-2 px-3 py-2 text-[13px] leading-relaxed text-ink-2">{question.state === 'answered' ? question.answer : question.state === 'skipped' ? 'You skipped this question; no option was selected.' : 'The run ended before an answer; start a new turn to continue.'}</p>}
    {error && <p role="alert" className="mt-3 break-words text-[12px] text-danger">{error}</p>}
    {error && waiting && !checked && <Button size="xs" className="mt-2" onClick={() => { setError(null); setCheckTick((value) => value + 1) }}>Check question again</Button>}
  </section>
}
