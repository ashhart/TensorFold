import assert from 'node:assert/strict'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { readFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import path from 'node:path'
import ts from 'typescript'
import type { ComposerDraftPort, ComposerDraftScope, ComposerDraftSnapshot } from '../../../../shared/composer-drafts'
import { composerDraftKey } from '../../../../shared/composer-drafts'
import { ConversationDrafts } from '../../../lib/conversation-drafts'
import type { Thread } from '../../../lib/types'
import * as firstAnswerModel from './model'
import { FIRST_ANSWER_STARTERS, firstAnswerNextAction, readFirstAnswerStarter, saveFirstAnswerStarter, type FirstAnswerReadiness } from './model'
import { openFirstAnswerDraft } from './transition'
import { flowTextFor } from '../../../lib/preferences/flow-catalogue'

// Compile the production component with the app's actual JSX mode, without a DOM or app boot.
const nodeRequire = createRequire(path.join(process.cwd(), 'package.json'))
const productionSource = readFileSync(path.join(process.cwd(), 'src/components/onboarding/first-answer/FirstAnswerStep.tsx'), 'utf8')
const compiled = ts.transpileModule(productionSource, { compilerOptions: { jsx: ts.JsxEmit.ReactJSX, module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText
const productionModule = { exports: {} as { FirstAnswerStep: (props: any) => ReturnType<typeof createElement> } }
new Function('require', 'module', 'exports', compiled)((id: string) => id === './model' ? firstAnswerModel : id === '../../../lib/preferences/useFlowText' ? { useFlowText: () => flowTextFor('en') } : nodeRequire(id), productionModule, productionModule.exports)
const FirstAnswerStep = productionModule.exports.FirstAnswerStep

const scope = (id: string): ComposerDraftScope => ({ surface: 'chat', conversationId: id, hostId: null, incognito: false })
const empty = (): ComposerDraftSnapshot => ({ revision: 0, text: '', attachments: [], updatedAt: 0 })
const thread = (id: string): Thread => ({ id, title: 'Synthetic chat', kind: 'chat', cwd: '', providerId: 'synthetic', model: 'synthetic', createdAt: 1, updatedAt: 1, status: 'idle', apiMessages: [], timeline: [] })
function fixture() {
  const saved = new Map<string, ComposerDraftSnapshot>()
  const port: ComposerDraftPort = {
    async getComposerDraft(scope) { return structuredClone(saved.get(composerDraftKey(scope)) ?? empty()) },
    async saveComposerDraft(input) {
      const key = composerDraftKey(input.scope), prior = saved.get(key) ?? empty()
      if (prior.revision !== input.expectedRevision) return { saved: false, draft: structuredClone(prior) }
      const draft = { revision: prior.revision + 1, text: input.text, attachments: input.attachments, updatedAt: 1 }
      saved.set(key, draft); return { saved: true, draft: structuredClone(draft) }
    },
    async clearComposerDraft(input) { return port.saveComposerDraft({ ...input, text: '', attachments: [] }) },
  }
  const drafts = new ConversationDrafts(port, 60_000)
  let current = thread('existing')
  const threads = [current]
  const input = { starter: 'plan' as const, drafts, persistent: true, threadIds: () => threads.map(item => item.id), currentThread: () => current,
    newChat: async () => { current = thread('new-chat'); threads.push(current) } }
  return { saved, port, drafts, input, threads, setCurrent: (next: Thread) => { current = next } }
}

const state: FirstAnswerReadiness = { providerKind: 'local', builtIn: true, selected: false, cached: false, fitsMemory: null, configured: false, loading: false, loaded: false }
assert.equal(firstAnswerNextAction(state).action, 'models')
assert.equal(firstAnswerNextAction({ ...state, selected: true }).label, 'Finish model download')
assert.equal(firstAnswerNextAction({ ...state, selected: true, cached: true, fitsMemory: false }).label, 'Choose a model that fits')
assert.equal(firstAnswerNextAction({ ...state, selected: true, cached: true }).action, 'draft')
assert.equal(firstAnswerNextAction({ ...state, providerKind: 'cloud', builtIn: false, selected: true }).action, 'provider')
assert.equal(firstAnswerNextAction({ ...state, providerKind: 'cloud', builtIn: false, selected: true, configured: true }).action, 'draft')
assert.ok(FIRST_ANSWER_STARTERS.every(starter => starter.prompt.length <= 512))
const markup = renderToStaticMarkup(createElement(FirstAnswerStep, { starter: 'plan', onStarter() {}, model: 'Synthetic model', readiness: { ...state, selected: true, cached: true } }))
assert.match(markup, /First answer checklist/)
assert.equal((markup.match(/type="radio"/gu) ?? []).length, 4)
assert.match(markup, /Validation and the first local load happen when you choose Send/)
assert.match(markup, /Existing conversations and unfinished messages stay separate/)
assert.ok(!markup.includes('tok/s') && !markup.includes('measured speed'))
const cloudMarkup = renderToStaticMarkup(createElement(FirstAnswerStep, { starter: 'write', onStarter() {}, model: 'Synthetic cloud', readiness: { ...state, providerKind: 'cloud', builtIn: false, selected: true, configured: true } }))
assert.match(cloudMarkup, /cloud provider receives your message after you choose Send/)

const first = fixture()
first.drafts.update(scope('existing'), { text: 'Existing unfinished synthetic question' })
first.drafts.update({ ...scope('home'), surface: 'home', conversationId: null }, { text: 'Unfinished synthetic home draft' })
const opened = await openFirstAnswerDraft(first.input)
assert.equal(opened.seeded, true)
assert.equal(first.saved.get(composerDraftKey(scope('new-chat')))?.text, FIRST_ANSWER_STARTERS[0].prompt)
assert.equal(first.saved.get(composerDraftKey(scope('existing')))?.text, 'Existing unfinished synthetic question')
assert.equal(first.saved.get(composerDraftKey({ ...scope('home'), surface: 'home', conversationId: null }))?.text, 'Unfinished synthetic home draft')
assert.equal(first.threads[1].apiMessages.length, 0, 'the actual transition creates no sent message or model run')
assert.equal(first.threads[1].timeline.length, 0)

const collision = fixture()
collision.saved.set(composerDraftKey(scope('new-chat')), { ...empty(), revision: 1, text: 'Saved draft that must stay' })
assert.equal((await openFirstAnswerDraft(collision.input)).preservedExistingDraft, true)
assert.equal(collision.drafts.get(scope('new-chat')).text, 'Saved draft that must stay')

const typed = fixture()
let release: (draft: ComposerDraftSnapshot) => void = () => undefined
typed.port.getComposerDraft = async (requested) => requested.conversationId === 'new-chat' ? new Promise(resolve => { release = resolve }) : empty()
const typingTransition = openFirstAnswerDraft(typed.input)
await new Promise<void>(resolve => setImmediate(resolve))
typed.drafts.update(scope('new-chat'), { text: 'New typing while the draft loads' })
release(empty())
assert.equal((await typingTransition).preservedExistingDraft, true)
assert.equal(typed.drafts.get(scope('new-chat')).text, 'New typing while the draft loads')

const refused = fixture()
refused.port.saveComposerDraft = async () => { throw new Error('Synthetic draft write refusal') }
refused.drafts.update(scope('existing'), { text: 'Unpersisted old input' })
await assert.rejects(openFirstAnswerDraft(refused.input), /could not be saved/)
assert.equal(refused.threads.length, 1, 'failed prior-draft flush cannot create a new conversation')

const remote = fixture()
remote.input.newChat = async () => { const created = { ...thread('new-chat'), orchestratorHostId: 'synthetic-remote' }; remote.threads.push(created); remote.setCurrent(created) }
await assert.rejects(openFirstAnswerDraft(remote.input), /local conversation changed/)
assert.equal(remote.saved.size, 0, 'a remote identity cannot seed a desktop draft')
const blank = fixture()
assert.equal((await openFirstAnswerDraft({ ...blank.input, starter: 'blank' })).seeded, false)
assert.equal(blank.saved.size, 0)
const navigated = fixture()
let finishHydration: (draft: ComposerDraftSnapshot) => void = () => undefined
navigated.port.getComposerDraft = async () => new Promise(resolve => { finishHydration = resolve })
const navigation = openFirstAnswerDraft(navigated.input)
await new Promise<void>(resolve => setImmediate(resolve))
navigated.setCurrent(navigated.threads[0])
finishHydration(empty())
await assert.rejects(navigation, /selected conversation changed/)
assert.equal(navigated.saved.size, 0, 'navigation while loading a draft cannot seed the newly selected conversation')

const storage = Object.getOwnPropertyDescriptor(globalThis, 'localStorage')
const values = new Map<string, string>()
Object.defineProperty(globalThis, 'localStorage', { configurable: true, value: { getItem: (key: string) => values.get(key) ?? null, setItem: (key: string, value: string) => values.set(key, value) } })
try {
  saveFirstAnswerStarter('learn')
  assert.equal(readFirstAnswerStarter(), 'learn', 'only the bounded task choice resumes after reopening setup')
  assert.equal(readFirstAnswerStarter(true), 'blank', 'existing-user setup replay does not preselect a task')
} finally { if (storage) Object.defineProperty(globalThis, 'localStorage', storage); else delete (globalThis as any).localStorage }
await Promise.all([first.drafts.flushAll(), collision.drafts.flushAll(), typed.drafts.flushAll(), blank.drafts.flushAll()])
for (const fixture of [first, collision, typed, refused, remote, blank, navigated]) fixture.drafts.forget()
console.log('Production first-answer rendering, checklist, explicit idle draft, prior/new typing preservation and refusal tests passed')
