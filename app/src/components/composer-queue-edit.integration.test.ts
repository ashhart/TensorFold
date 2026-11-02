import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import ts from 'typescript'
import { ConversationDrafts } from '../lib/conversation-drafts'
import { composerDraftKey } from '../../shared/composer-drafts'
import { capturedDraftMatches, editQueuedFollowUp } from '../lib/queued-followups'
import { fixtureText } from './preferences/language-test-fixture'
import { harness } from './reading/reading-test-harness'
const source = ts.createSourceFile('Composer.tsx', readFileSync('src/components/Composer.tsx', 'utf8'), ts.ScriptTarget.ES2022, true, ts.ScriptKind.TSX)
const component = source.statements.find((node): node is ts.FunctionDeclaration => ts.isFunctionDeclaration(node) && node.name?.text === 'Composer')!
const printer = ts.createPrinter()
const declarations = (names: string[]) => names.map(name => {
  const statement = component.body!.statements.find(row => ts.isVariableStatement(row) && row.declarationList.declarations.some(item => ts.isIdentifier(item.name) && item.name.text === name))
  assert.ok(statement, name); return printer.printNode(ts.EmitHint.Unspecified, statement, source)
}).join('\n')
function production(names: string[], environment: Record<string, unknown>) {
  const code = ts.transpileModule(declarations(names) + '\nreturn { ' + names.join(',') + ' };', { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None } }).outputText
  return new Function(...Object.keys(environment), code)(...Object.values(environment))
}
const scope = (id: string) => ({ surface: 'chat' as const, conversationId: id, hostId: null, incognito: false })
const a = scope('queue-owner-a'), b = scope('queue-owner-b')
const drafts = new ConversationDrafts(null)
const settle = async () => { for (let index = 0; index < 20; index++) await Promise.resolve() }
function fixture(options: { idle?: boolean; switched?: boolean; newer?: boolean } = {}) {
  drafts.update(a, { text: 'original draft' }); drafts.update(b, { text: 'other draft' })
  const owner = options.switched ? b : a
  const original = drafts.capture(a), sends: any[] = [], cues: any[] = [], notices: any[] = [], timers: (() => void)[] = []
  const queuedRef = { current: [{ id: 'q', text: 'original draft', status: 'queued' as const }] as any[] }
  const queuedDraftTokens = { current: new Map([['q', original]]) }, queueEditingRef = { current: 'q' as string | null }, activeDraftKey = { current: composerDraftKey(owner) }
  if (options.newer) drafts.update(a, { text: 'newer unsent typing', attachments: [{ id: 'new-file', name: 'new.txt', kind: 'document', mime: 'text/plain', status: 'ready', text: 'new file' }] })
  let resolveCue!: (value: { queued: boolean }) => void
  const cue = async (...args: any[]) => { cues.push(args); return await new Promise<{ queued: boolean }>(resolve => { resolveCue = resolve }) }
  const environment: Record<string, any> = {
    tx: fixtureText, thread: { id: owner.conversationId }, activeDraftKey,
    draft: { scopeKey: composerDraftKey(a), capture: () => drafts.capture(a), clearAccepted: (token: any) => drafts.clearAccepted(token) },
    queueEditingRef, queuedRef, queuedDraftTokens, queueScopeRef: { current: composerDraftKey(a) }, editQueuedFollowUp, capturedDraftMatches,
    setText: (text: string) => drafts.update(a, { text }), setQueued: (change: any) => { queuedRef.current = typeof change === 'function' ? change(queuedRef.current) : change },
    api: { cueRun: cue, guideRemoteOrchestratorThread: cue }, busy: !options.idle, botBlocked: false, recoveryBlocked: false,
    send: async (...args: any[]) => { sends.push(args); return true }, useStore: { setState: (value: any) => notices.push(value) }, setTimeout: (callback: () => void) => timers.push(callback),
  }
  Object.assign(environment, production(['updateQueue'], environment))
  const handlers = production(['editQueued', 'guideNow'], environment)
  return { ...handlers, environment, queuedRef, queuedDraftTokens, queueEditingRef, activeDraftKey, sends, cues, notices, timers, resolve: (value: { queued: boolean }) => resolveCue(value) }
}
try {
  let f = fixture()
  assert.equal(f.editQueued('q', 'original draft', 'saved replacement'), true)
  assert.equal(f.queuedRef.current[0].text, 'saved replacement'); assert.equal(drafts.get(a).text, 'saved replacement')
  assert.equal(f.queuedDraftTokens.current.get('q')?.text, 'saved replacement'); assert.equal(f.sends.length, 0); assert.equal(f.cues.length, 0)
  assert.equal(f.editQueued('q', 'original draft', 'stale save'), false)
  f = fixture({ newer: true })
  assert.equal(f.editQueued('q', 'original draft', 'edited queued follow-up'), true)
  assert.equal(drafts.get(a).text, 'newer unsent typing'); assert.equal(drafts.get(a).attachments[0].name, 'new.txt')
  assert.equal(f.queuedDraftTokens.current.has('q'), false, 'Editing cannot retarget clearing to a later typed draft')
  f = fixture({ switched: true }); assert.equal(f.editQueued('q', 'original draft', 'wrong owner'), false)
  await f.guideNow('q', 'original draft'); assert.equal(f.cues.length, 0); assert.equal(f.sends.length, 0)
  f = fixture(); f.queueEditingRef.current = null; f.queuedRef.current = []; f.queuedDraftTokens.current.clear()
  assert.equal(f.editQueued('q', 'original draft', 'stale after stop'), false); await f.guideNow('q', 'original draft'); assert.equal(f.cues.length, 0)
  f = fixture({ idle: true }); f.queueEditingRef.current = null
  await f.guideNow('q', 'original draft'); assert.equal(f.sends.length, 1); assert.equal(f.cues.length, 0)
  assert.equal(drafts.get(a).text, '', 'Only accepted explicit idle send clears its exact old draft')
  f = fixture(); f.queueEditingRef.current = null
  const first = f.guideNow('q', 'original draft'); await f.guideNow('q', 'original draft')
  assert.equal(f.cues.length, 1, 'Synchronous guiding state blocks duplicate Guide clicks before rendering')
  f.resolve({ queued: true }); await first
  assert.equal(f.queuedRef.current[0].status, 'guided')

  for (const accepted of [true, false]) {
    f = fixture(); f.queueEditingRef.current = null
    const oldGuide = f.guideNow('q', 'original draft')
    f.environment.queueScopeRef.current = composerDraftKey(b); f.activeDraftKey.current = composerDraftKey(b)
    const replacement = [{ id: 'q', text: 'other-chat collision row', status: 'queued' }]
    f.queuedRef.current = replacement
    f.resolve({ queued: accepted }); await oldGuide
    assert.equal(f.queuedRef.current, replacement, 'Late guide success or refusal cannot mark or restore a collision row from a new owner scope')
    for (const timer of f.timers) timer()
    assert.equal(f.queuedRef.current, replacement, 'Late guided-confirmation cleanup cannot remove a new owner row with a colliding identity')
    assert.equal(drafts.get(b).text, 'other draft')
  }

  // Execute the actual completion effect, including its captured previous-busy transition.
  const effectStatement = component.body!.statements.find(row => ts.isExpressionStatement(row) && ts.isCallExpression(row.expression) && row.expression.expression.getText(source) === 'useEffect' && row.expression.arguments[0]?.getText(source).includes('prevBusy.current &&')) as ts.ExpressionStatement
  assert.ok(effectStatement)
  const arrow = (effectStatement.expression as ts.CallExpression).arguments[0]
  const code = ts.transpileModule('const effect = ' + printer.printNode(ts.EmitHint.Expression, arrow, source) + '; return effect;', { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None } }).outputText
  const drain = (environment: Record<string, any>) => new Function(...Object.keys(environment), code)(...Object.values(environment))()
  f = fixture(); const previousBusy = { current: true }
  drain({ ...f.environment, prevBusy: previousBusy, busy: false, st: 'idle', mode: 'thread' }); await settle()
  assert.equal(f.sends.length, 0, 'Completion while an edit is open cannot send an unreviewed follow-up')
  f.queueEditingRef.current = null
  drain({ ...f.environment, prevBusy: previousBusy, busy: false, st: 'idle', mode: 'thread' }); await settle()
  assert.equal(f.sends.length, 0, 'Closing or saving an editor after completion cannot revive the missed automatic transition')

  f = fixture(); f.queueEditingRef.current = null
  drain({ ...f.environment, prevBusy: { current: true }, busy: false, st: 'idle', mode: 'thread' })
  f.environment.queueScopeRef.current = composerDraftKey(b); f.activeDraftKey.current = composerDraftKey(b)
  const collision = [{ id: 'q', text: 'new owner queued row', status: 'queued' }]
  f.queuedRef.current = collision
  await settle()
  assert.equal(f.queuedRef.current, collision, 'Late auto-drain completion cannot remove a new owner row with a colliding identity')
  assert.equal(drafts.get(b).text, 'other draft')

  f = fixture({ switched: true }); f.queueEditingRef.current = null
  drain({ ...f.environment, draft: { ...f.environment.draft, scopeKey: composerDraftKey(b), capture: () => drafts.capture(b) }, prevBusy: { current: true }, busy: false, st: 'idle', mode: 'thread' }); await settle()
  assert.equal(f.sends.length, 0, 'Changing from running chat A to idle chat B cannot dispatch A\'s queued input into B before the queue-reset effect executes')

  // Render both production effects in source order so navigation reset cannot hide a real busy-to-idle transition.
  const queueEffects = component.body!.statements.filter(row => ts.isExpressionStatement(row) && ts.isCallExpression(row.expression) && row.expression.expression.getText(source) === 'useEffect' && row.expression.arguments[0]?.getText(source).includes('prevBusy.current'))
  assert.equal(queueEffects.length, 2)
  const lifecycleCode = ts.transpileModule(queueEffects.map(row => printer.printNode(ts.EmitHint.Unspecified, row, source)).join('\n'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None } }).outputText
  for (const scenario of ['mount-running', 'return-running', 'return-error', 'return-review', 'return-editing']) {
    f = fixture()
    const hooks = harness(), previous = { current: false }, quoteEditingRef = { current: false }
    const render = (target: typeof a, busy: boolean, terminal = false) => {
      f.activeDraftKey.current = composerDraftKey(target)
      const environment = {
        ...f.environment, quoteEditingRef, setQuoteEditing() {}, prevBusy: previous, busy,
        st: busy ? 'streaming' : terminal && scenario === 'return-error' ? 'error' : 'idle', mode: 'thread',
        thread: { id: target.conversationId, ...(terminal && scenario === 'return-review' ? { runRecovery: { state: 'needs_review', retry: 'requires_review' } } : {}) },
        draft: { scopeKey: composerDraftKey(target), capture: () => drafts.capture(target), clearAccepted: (token: any) => drafts.clearAccepted(token) },
      }
      hooks.render(() => new Function('useEffect', ...Object.keys(environment), lifecycleCode)(hooks.hooks.useEffect, ...Object.values(environment)))
      hooks.flush()
      return environment
    }
    let environment = render(a, true)
    if (scenario !== 'mount-running') { render(b, false); environment = render(a, true) }
    const followUp = `Follow up only for ${scenario}`
    drafts.update(a, { text: followUp })
    production(['queueMessage'], { ...environment, text: followUp, quotes: [], attachments: [], readiness: { ready: true }, expandSkills: (text: string) => text, crypto: { randomUUID: () => 'after-return' } }).queueMessage()
    assert.equal(f.sends.length, 0, 'Queueing while busy does not start a second run')
    render(a, true)
    if (scenario === 'return-editing') f.queueEditingRef.current = 'after-return'
    render(a, false, true)
    await settle()
    if (scenario === 'mount-running' || scenario === 'return-running') {
      assert.deepEqual(f.sends, [[followUp, undefined, a.conversationId]], `${scenario}: the first completed run drains exactly one queued message to its owner`)
      assert.equal(drafts.get(a).text, '')
      render(a, false, true); await settle()
      assert.equal(f.sends.length, 1, 'Rerendering the settled scope cannot send it again')
    } else {
      assert.equal(f.sends.length, 0, `${scenario}: restoring the busy transition must preserve error, review and edit guards`)
      assert.equal(drafts.get(a).text, followUp)
    }
    assert.equal(drafts.get(b).text, 'other draft', 'Navigation and completion preserve the other conversation draft')
    hooks.cleanup()
  }
  console.log('Production queued-edit/save/guide/completion wiring preserves owner and newer drafts, pauses completion edits, requires explicit idle sends and prevents stale/duplicate/Stop/conversation-switch dispatch')
} finally { drafts.dispose() }
