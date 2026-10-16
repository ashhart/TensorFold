import { randomUUID } from 'node:crypto'
import { InterruptRegistry, type InterruptOutcome } from '../interrupt-registry'
import { normalizeQuestionAnswer, normalizeUserQuestion } from './validation'
import type { Thread, UserQuestion, UserQuestionAnswer, UserQuestionInput, UserQuestionOutcome, UserQuestionRun } from '../types'

export interface UserQuestionServiceOptions {
  getThread(threadId: string): Thread | null
  isRunActive(threadId: string, runId: string): boolean
  /** Must await the encrypted store write before announcing the state to the UI. */
  commitQuestion(run: Pick<UserQuestionRun, 'threadId' | 'runId' | 'itemId'>, question: UserQuestion): Promise<void>
  now?: () => number
  nextId?: () => string
}
interface Pending {
  question: UserQuestion
  run: UserQuestionRun
  ready: boolean
  settling: boolean
  chain: Promise<void>
  promise: Promise<InterruptOutcome>
  abort: () => void
}

export class UserQuestionService {
  private registry: InterruptRegistry
  private pending = new Map<string, Pending>()
  private writes = new Set<Promise<unknown>>()
  private closing = false
  private readonly now: () => number
  constructor(private readonly options: UserQuestionServiceOptions) { this.now = options.now ?? Date.now; this.registry = new InterruptRegistry({ now: this.now, nextId: options.nextId ?? randomUUID }) }
  isWaiting(questionId: string): boolean {
    const pending = this.pending.get(questionId)
    return !!pending && this.registry.has(questionId) && pending.question.state === 'pending' && !pending.run.signal.aborted && !this.closing && this.options.isRunActive(pending.run.threadId, pending.run.runId)
  }
  private live(run: Pick<UserQuestionRun, 'threadId' | 'runId' | 'itemId'>) {
    const thread = this.options.getThread(run.threadId)
    if (!thread || !this.options.isRunActive(run.threadId, run.runId)) throw new Error('This question’s run is no longer active; start a new turn to continue.')
    const item = thread.timeline.find((candidate) => candidate.id === run.itemId)
    if (!item || item.tool !== 'ask_user') throw new Error('This question does not belong to the active chat turn.')
    return thread
  }
  private commit(pending: Pending, snapshot: UserQuestion): Promise<void> {
    const write = pending.chain.catch(() => {}).then(() => this.options.commitQuestion(pending.run, { ...snapshot, choices: [...snapshot.choices] }))
    pending.chain = write
    this.writes.add(write); void write.finally(() => this.writes.delete(write)).catch(() => {})
    return write
  }
  private commitRecovered(run: Pick<UserQuestionRun, 'threadId' | 'runId' | 'itemId'>, question: UserQuestion): Promise<void> {
    const write = this.options.commitQuestion(run, question)
    this.writes.add(write); void write.finally(() => this.writes.delete(write)).catch(() => {})
    return write
  }
  private interrupt(pending: Pending): Promise<void> {
    if (!this.registry.has(pending.question.id)) return pending.chain.catch(() => {})
    pending.question = { ...pending.question, state: 'interrupted', answer: null, resolvedAt: this.now() }
    pending.settling = true
    // Release the parked run immediately; a slow disk cannot keep Stop waiting.
    this.registry.resolve(pending.question.id, { type: 'deny', reason: 'Question interrupted; no answer or default choice was selected.' })
    return this.commit(pending, pending.question)
  }
  async ask(input: UserQuestionInput, run: UserQuestionRun): Promise<UserQuestionOutcome> {
    if (this.closing || run.signal.aborted) throw new Error('This question was interrupted before it could be asked.')
    this.live(run)
    const normalized = normalizeUserQuestion(input)
    for (const existing of this.pending.values()) if (existing.run.threadId === run.threadId) {
      if (existing.question.state !== 'pending' || !this.registry.has(existing.question.id)) continue
      if (!this.options.isRunActive(existing.run.threadId, existing.run.runId)) await this.interrupt(existing)
      else throw new Error('This chat already has a question waiting for its user.')
    }
    const parked = this.registry.park({ kind: 'question', tool: 'ask_user', label: normalized.question, prompt: normalized.question, args: normalized, meta: { threadId: run.threadId, runId: run.runId, itemId: run.itemId } })
    // Observe denial now, including cancellation while the initial write is in flight.
    void parked.promise.catch(() => {})
    const pending: Pending = { question: { id: parked.id, ...normalized, threadId: run.threadId, runId: run.runId, itemId: run.itemId, state: 'pending', answer: null, createdAt: this.now(), resolvedAt: null }, run, ready: false, settling: false, chain: Promise.resolve(), promise: parked.promise, abort: () => {} }
    pending.abort = () => { void this.interrupt(pending).catch(() => {}) }
    this.pending.set(parked.id, pending); run.signal.addEventListener('abort', pending.abort, { once: true })
    try {
      // An abort must release the run even while the initial encrypted save is
      // slow; its late pending/interrupted writes remain tracked for purge/Quit.
      await Promise.race([this.commit(pending, pending.question), pending.promise.then(() => { throw new Error('Question settled before it was ready.') })])
      if (run.signal.aborted || this.closing || !this.options.isRunActive(run.threadId, run.runId)) { await this.interrupt(pending); throw new Error('This question was interrupted before an answer.') }
      pending.ready = true
      const result = await pending.promise
      if (result.type !== 'answer') throw new Error('Question was not answered.')
      return result.value as UserQuestionOutcome
    } catch (error) {
      if (this.registry.has(parked.id)) { this.registry.resolve(parked.id, { type: 'deny', reason: 'Question could not be saved or was interrupted.' }) }
      throw error
    } finally {
      run.signal.removeEventListener('abort', pending.abort)
      // Keep a terminal snapshot available while a late store publication drains.
      // It is never executable: the registry entry has already been removed.
      void pending.chain.catch(() => {}).then(() => { if (this.pending.get(parked.id) === pending) this.pending.delete(parked.id) })
    }
  }
  async answer(raw: UserQuestionAnswer): Promise<UserQuestion> {
    const input = normalizeQuestionAnswer(raw), pending = this.pending.get(input.questionId)
    if (!pending || pending.question.threadId !== input.threadId || pending.question.runId !== input.runId || !this.registry.has(input.questionId)) throw new Error('This question is stale or belongs to another chat turn.')
    this.live(pending.run)
    if (!pending.ready || pending.settling || pending.run.signal.aborted || this.closing) throw new Error('This question is already settling or is no longer waiting for an answer.')
    pending.settling = true
    const snapshot: UserQuestion = { ...pending.question, state: input.action === 'skip' ? 'skipped' : 'answered', answer: input.action === 'skip' ? null : input.answer, resolvedAt: this.now() }
    try { await this.commit(pending, snapshot) }
    catch (error) { if (this.registry.has(input.questionId) && !pending.run.signal.aborted) pending.settling = false; throw error }
    if (pending.run.signal.aborted || this.closing || !this.options.isRunActive(input.threadId, input.runId) || !this.registry.has(input.questionId)) { await this.interrupt(pending); throw new Error('The run stopped before this answer could resume it.') }
    pending.question = snapshot
    this.registry.resolve(input.questionId, { type: 'answer', value: snapshot.state === 'answered' ? { state: 'answered', answer: snapshot.answer! } : { state: 'skipped', answer: null } })
    return snapshot
  }
  async state(threadId: string, questionId: string): Promise<UserQuestion | null> {
    const thread = this.options.getThread(threadId)
    if (!thread) return null
    // The registry is authoritative until its resolution is durably accepted.
    // A store writer may be staging a candidate answer while a direct state read
    // races it; that candidate must never make a second card show Answered early.
    const pending = this.pending.get(questionId)
    if (pending?.question.threadId === threadId && thread.timeline.some((item) => item.id === pending.question.itemId && item.tool === 'ask_user')) {
      const current = pending.question
      if (current.state === 'pending' && (this.closing || pending.run.signal.aborted)) return { ...current, state: 'interrupted', answer: null, resolvedAt: this.now() }
      if (current.state === 'pending' && !this.options.isRunActive(threadId, current.runId)) { await this.interrupt(pending); return { ...pending.question, choices: [...pending.question.choices] } }
      if (current.state === 'pending' && !this.registry.has(questionId)) return { ...current, state: 'interrupted', answer: null, resolvedAt: this.now() }
      return { ...current, choices: [...current.choices] }
    }
    const item = thread.timeline.find((candidate) => candidate.question?.id === questionId), question = item?.question
    if (!question) return null
    if (this.closing && question.state === 'pending') return { ...question, state: 'interrupted', answer: null, resolvedAt: this.now() }
    if (question.state === 'pending' && (!this.pending.has(questionId) || !this.options.isRunActive(threadId, question.runId))) {
      const interrupted: UserQuestion = { ...question, state: 'interrupted', answer: null, resolvedAt: this.now() }
      await this.commitRecovered({ threadId, runId: question.runId, itemId: item!.id }, interrupted)
      return interrupted
    }
    return { ...question, choices: [...question.choices] }
  }
  async recoverInterrupted(threads: Thread[]): Promise<number> {
    let count = 0
    for (const thread of threads) for (const item of thread.timeline) if (item.question?.state === 'pending' && !this.pending.has(item.question.id)) {
      if (this.closing) return count
      await this.commitRecovered({ threadId: thread.id, runId: item.question.runId, itemId: item.id }, { ...item.question, state: 'interrupted', answer: null, resolvedAt: this.now() }); count++
    }
    return count
  }
  async quiesce(): Promise<void> { this.closing = true; for (const pending of this.pending.values()) void this.interrupt(pending).catch(() => {}); await Promise.allSettled([...this.writes]) }
  resume(): void { this.closing = false }
  async shutdown(): Promise<void> { await this.quiesce() }
}
