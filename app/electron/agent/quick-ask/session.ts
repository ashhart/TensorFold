import { randomUUID } from 'node:crypto'
import type { ApiMessage, Attachment } from '../types'
import type { QuickAskDraft, QuickAskEvent, QuickAskPreferences, QuickAskRoute, QuickAskSession, QuickAskTurn } from '../../../shared/quick-ask'
import { quickAskDefaults } from '../../../shared/quick-ask'
import { ASK_HISTORY_LIMIT, boundedText, emptyAskSession, resolveAskCommand, validateAskDraft, validateAskPreferences } from './validation'
import type { AskGeneration, AskStep } from './generate'
import { validateAskSavedState } from './state-validation'

export interface AskSavedState { session: QuickAskSession; messages: ApiMessage[] }
export interface AskSessionDeps {
  save(state: AskSavedState): Promise<void>
  generate(input: AskGeneration, signal: AbortSignal): AsyncIterable<AskStep>
  routes(): Promise<QuickAskRoute[]>
  emit(event: QuickAskEvent): void
  authorize(): void
  reserve(): { release(): void }
  handoff(state: AskSavedState): Promise<string>
}
export class QuickAskSessionHost {
  private session = emptyAskSession()
  private messages: ApiMessage[] = []
  private active: { controller: AbortController; promise: Promise<void> } | null = null
  private accepting = true
  private preferences = quickAskDefaults()
  private writes: Promise<void> = Promise.resolve()
  private handingOff: Promise<string> | null = null
  constructor(private deps: AskSessionDeps) {}
  snapshot(): QuickAskSession { return structuredClone(this.session) }
  saved(): AskSavedState { return { session: this.snapshot(), messages: structuredClone(this.messages) } }
  setPreferences(value: QuickAskPreferences): void { this.preferences = validateAskPreferences(value) }
  restore(state: AskSavedState | null): void {
    if (!state) return
    validateAskSavedState(state)
    const session = state.session
    if (!session || !Array.isArray(session.turns) || session.turns.length > ASK_HISTORY_LIMIT || !Array.isArray(state.messages)) throw new Error('The saved ask session needs recovery.')
    boundedText(session.id, 120, 'Saved session ID'); validateAskDraft(session.draft)
    if (JSON.stringify(state).length > 16_000_000) throw new Error('The saved ask session is too large.')
    this.session = structuredClone(session); this.messages = structuredClone(state.messages)
    for (const turn of this.session.turns) if (turn.status === 'running') {
      turn.status = 'cancelled'; turn.error = 'TensorFold restarted before this answer finished.'
      this.messages.push(askUserMessage(turn.text, turn.attachments), ...(turn.answer ? [{ role: 'assistant' as const, content: `[Interrupted answer]\n${turn.answer}` }] : []))
    }
  }
  private persist(): Promise<void> {
    const state = this.saved()
    const write = this.writes.catch(() => {}).then(() => this.deps.save(state))
    this.writes = write
    return write
  }
  async saveDraft(value: QuickAskDraft): Promise<QuickAskDraft> {
    this.deps.authorize()
    if (!this.accepting) throw new Error('The workspace is stopping. Your draft is preserved.')
    if (this.active) throw new Error('Wait for this answer to finish or stop it before editing.')
    const draft = validateAskDraft(value)
    if (draft.revision !== this.session.draft.revision) throw new Error('This draft changed. Reopen Ask TensorFold to recover the latest text.')
    const next = { ...draft, revision: draft.revision + 1 }
    this.session.draft = next
    try { await this.persist() } catch (error) { if (this.session.draft === next) next.revision = draft.revision; throw error }
    return structuredClone(this.session.draft)
  }
  async send(input: { revision: number; approvedRouteId: string | null }): Promise<{ requestId: string }> {
    this.deps.authorize()
    if (!this.accepting || this.active) throw new Error('An answer is already running or the workspace is stopping.')
    if (input?.revision !== this.session.draft.revision) throw new Error('Save the current draft before sending.')
    if (this.session.handoffThreadId) throw new Error('This ask was moved to a conversation. Start a new ask to continue here.')
    if (this.session.turns.length >= ASK_HISTORY_LIMIT) throw new Error('Continue in TensorFold or start a new ask.')
    const { preset, text } = resolveAskCommand(this.session.draft.text, this.preferences, this.session.draft.presetId)
    if (this.session.presetId && this.session.presetId !== preset.id) throw new Error('Start a new ask to change presets. This conversation keeps its original instructions.')
    const draft = this.session.draft
    if (!text.trim() && !draft.attachments.length) throw new Error('Write a question or attach some context.')
    const routeId = this.session.route?.id ?? preset.routeId ?? draft.routeId
    const route = (await this.deps.routes()).find(route => route.id === routeId)
    if (!route) throw new Error('Choose an available model. No alternative model was substituted.')
    if (this.session.route && route.binding !== this.session.route.binding) throw new Error('The model destination changed. Start a new ask before sharing conversation history with it.')
    if (route.state !== 'ready') throw new Error('Load the selected model first. Your draft is preserved.')
    if (route.destination !== 'device' && input.approvedRouteId !== route.id) throw new Error('Review and approve the destination before sending this conversation.')
    if (route.vision === 'unsupported' && draft.attachments.some(item => item.kind === 'image')) throw new Error('The selected model cannot read images. Choose a vision model in a new ask.')
    this.deps.authorize()
    if (!this.accepting || this.active || draft !== this.session.draft) throw new Error('The ask changed before sending. Retry with the current draft.')
    const lease = this.deps.reserve(), controller = new AbortController()
    const prior = this.saved()
    const turn: QuickAskTurn = { id: randomUUID(), text, attachments: structuredClone(draft.attachments), answer: '', reasoning: '', status: 'running', tools: [], at: Date.now() }
    const user = askUserMessage(text, turn.attachments)
    if (JSON.stringify({ ...this.saved(), user, turn }).length > 14_000_000) { lease.release(); throw new Error('This conversation is too large. Continue in TensorFold or start a new ask.') }
    this.session.route ??= structuredClone(route)
    this.session.prompt ??= preset.prompt
    this.session.presetId ??= preset.id
    this.session.turns.push(turn)
    const generation = { routeId: route.id, binding: route.binding, upstreamBinding: route.upstreamBinding, prompt: this.session.prompt, messages: [...this.messages, user], web: draft.web }
    const stored = this.persist()
    const promise = Promise.resolve().then(async () => {
      let persisted = false
      try {
        await stored; persisted = true
        controller.signal.throwIfAborted()
        let lastEmit = 0, lastSave = Date.now(), complete = false
        const generated: ApiMessage[] = []
        for await (const step of this.deps.generate(generation, controller.signal)) {
          controller.signal.throwIfAborted(); this.deps.authorize()
          if (step.type === 'content') turn.answer += step.text
          if (step.type === 'reasoning') turn.reasoning += step.text
          if (step.type === 'message') generated.push(step.message)
          if (step.type === 'finish') complete = true
          if (step.type === 'tool') {
            const current = [...turn.tools].reverse().find(tool => tool.status === 'running' && tool.name === step.name && tool.detail === step.detail)
            if (current) current.status = step.status
            else turn.tools.push({ name: step.name, detail: step.detail, status: step.status })
          }
          if (Date.now() - lastEmit > 50 || step.type === 'tool') { this.emitTurn(turn); lastEmit = Date.now() }
          if (Date.now() - lastSave > 1500) { await this.persist(); lastSave = Date.now() }
        }
        if (!complete || !turn.answer.trim() || generated.at(-1)?.role !== 'assistant') throw new Error('The response ended without a complete answer and history receipt.')
        this.messages.push(user, ...generated)
        turn.status = 'complete'
        this.session.draft = { ...draft, revision: draft.revision + 1, text: '', attachments: [] }
      } catch (error) {
        if (!persisted) { this.session = prior.session; this.messages = prior.messages; throw error }
        turn.status = controller.signal.aborted ? 'cancelled' : 'error'
        turn.error = controller.signal.aborted ? 'Stopped. Your question and partial answer are preserved.' : safeAskError(error)
        this.messages.push(user, ...(turn.answer ? [{ role: 'assistant' as const, content: `[Interrupted answer]\n${turn.answer}` }] : []))
      } finally {
        try { await this.persist() } catch { turn.status = 'error'; turn.error = 'This ask could not be saved. Keep this window open and copy your work before closing.' }
        if (this.session.turns.includes(turn)) this.emitTurn(turn)
        lease.release(); this.active = null
        this.deps.emit({ kind: 'refresh' })
      }
    })
    this.active = { controller, promise }
    this.emitTurn(turn)
    void promise.catch(() => {})
    await stored
    return { requestId: turn.id }
  }
  private emitTurn(turn: QuickAskTurn): void { this.deps.emit({ kind: 'turn', sessionId: this.session.id, turn: structuredClone(turn) }) }
  async cancel(): Promise<void> { this.deps.authorize(); this.active?.controller.abort(); await this.active?.promise.catch(() => {}) }
  async reset(): Promise<void> {
    this.deps.authorize()
    if (this.active) throw new Error('Stop the current answer before starting a new ask.')
    const draft = this.session.draft
    this.session = emptyAskSession(); this.session.draft = { ...draft, revision: draft.revision + 1 }
    this.messages = []; await this.persist(); this.deps.emit({ kind: 'refresh' })
  }
  async handoff(): Promise<string> {
    if (this.handingOff) return this.handingOff
    const operation = this.performHandoff()
    this.handingOff = operation
    try { return await operation } finally { this.handingOff = null }
  }
  private async performHandoff(): Promise<string> {
    this.deps.authorize()
    if (this.active) throw new Error('Finish or stop the answer before continuing in TensorFold.')
    if (!this.session.turns.length) throw new Error('Ask a question before continuing in TensorFold.')
    if (this.session.handoffThreadId) return this.session.handoffThreadId
    const lease = this.deps.reserve()
    try {
      const id = await this.deps.handoff(this.saved())
      this.session.handoffThreadId = id; await this.persist(); return id
    } finally { lease.release() }
  }
  async quiesce(): Promise<void> { this.accepting = false; this.active?.controller.abort(); await this.active?.promise.catch(() => {}); await this.persist() }
  resume(): void { this.accepting = true }
  async purge(): Promise<void> { await this.quiesce(); this.session = emptyAskSession(); this.messages = []; await this.persist(); this.deps.emit({ kind: 'refresh' }) }
}
export function safeAskError(error: unknown): string {
  const raw = error instanceof Error ? error.message : 'The model request failed.'
  return raw.replace(/Bearer\s+\S+|(?:sk|key|token)[-_][A-Za-z0-9_-]{12,}/gi, '[redacted]').replace(/https?:\/\/\S+/g, '[endpoint]').slice(0, 350)
}
export function askUserMessage(text: string, attachments: Attachment[]): ApiMessage {
  const content = [text, ...attachments.filter(item => item.kind === 'document').map(item => `\n<attached-context name=${JSON.stringify(item.name)}>\n${item.text}\n</attached-context>`)].join('\n')
  const images = attachments.filter(item => item.kind === 'image')
  return { role: 'user', content: images.length ? [{ type: 'text', text: content }, ...images.map(item => ({ type: 'image_url' as const, image_url: { url: item.dataUrl! } }))] : content }
}
