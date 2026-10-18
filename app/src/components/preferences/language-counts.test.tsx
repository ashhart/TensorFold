import assert from 'node:assert/strict'
import { readFileSync, readdirSync } from 'node:fs'
import path from 'node:path'
import ts from 'typescript'
import * as languages from '../../../shared/languages'
import { jsx, load, text } from '../reading/reading-test-harness'

const chinese: Record<string, string> = {}
for (const name of readdirSync('src/lib/preferences/packs/zh-Hans')) if (name.endsWith('.json')) Object.assign(chinese, JSON.parse(readFileSync(path.join('src/lib/preferences/packs/zh-Hans', name), 'utf8')))
const { authoredText } = load('src/lib/preferences/language-packs.ts', {
  '../../../shared/languages': languages,
  './packs': { languagePacks: { 'zh-Hans': chinese }, languagePackCoverage: { 'zh-Hans': { complete: true } } },
})
let locale = 'zh-Hans'
const tx = (source: string, params?: Record<string, unknown>) => authoredText(locale, source, params)
const state = { settings: { uiPreferences: { get locale() { return locale } } }, projects: [] }
const store = Object.assign((select: (input: typeof state) => unknown) => select(state), { getState: () => state })
const hook = { useAuthoredText: () => tx }
const failures: string[] = []
function check(name: string, run: () => void) { try { run() } catch (error) { failures.push(name); console.error(name, String(error)) } }

const { EditedFilesCard } = load('src/components/turn/EditedFilesCard.tsx', {
  'react/jsx-runtime': jsx, react: { useState: (initial: unknown) => [initial, () => {}] },
  '../../lib/preferences/useAuthoredText': hook, '../../lib/store': { useStore: store },
  '../../lib/ipc': { api: {} }, '../../lib/icons': { Icon: 'icon' }, '../../lib/fileicon': { FileGlyph: 'file' },
  '../../lib/format': { basename: (value: string) => value.split('/').at(-1) },
  '../../lib/turn-blocks': { editTotals: (edits: unknown[]) => ({ files: edits.length, additions: 3, deletions: 1 }) },
  '../../lib/thread-workspace': { effectiveThreadCwd: () => '/synthetic', effectiveThreadWorkspace: () => null },
  '../thread-panel/ReviewChangesModal': { ReviewChangesModal: 'review' },
})
check('Actual edited-file counts keep numeric meaning without an English suffix', () => {
  for (const count of [4, 1]) {
    const edits = Array.from({ length: count }, (_, index) => ({ path: `Cancel-${index}.tsx` }))
    const view = EditedFilesCard({ thread: { id: 'synthetic-count-chat' }, turn: { index: 0, edits }, canUndo: false, onChanged() {} })
    assert.ok(text(view).includes(tx('Files changed: {count}', { count })), text(view))
    assert.ok(text(view).includes('Cancel-0.tsx'), 'Original filenames must remain unchanged')
  }
})

const helpers = load('src/components/rooms/helpers.tsx', {
  'react/jsx-runtime': jsx, '../../lib/preferences/useAuthoredText': hook,
  '../../lib/preferences/language-packs': { authoredText }, '../../lib/store': { useStore: store },
  '../../lib/preferences/display': { currentAuthoredText: tx },
  '../../lib/preferences/current-locale': { currentUiLanguage: () => locale, localDateText: () => '09:30' },
  '../../lib/format': { relativeTime: () => 'recent' },
  '../../lib/policy': { allowedProviders: () => [], projectDefaultModel: () => null },
  '../../lib/provider-accounts': { providerAccountChoices: () => [] },
  '../../lib/room-modes': { ROOM_MODE_OPTIONS: [], roomModeDefaultEffort: () => 'normal' },
})
check('Actual avatars retain Unicode letters and complete supplementary code points', () => {
  for (const [name, expected] of [['Aria Milo', 'AM'], ['Mary-Jane', 'MJ'], ['山田 太郎', '山太'], ['Анна Петрова', 'АП'], ['𠮷野 Akira', '𠮷A'], ['กมล ดี', 'กด'], ['!!!', '?']]) assert.equal(text(helpers.Avatar({ name })), expected)
})
check('Room presentation responds to language changes and preserves data boundaries', () => {
  const ticket = { assignee: 'Cancel', runState: 'running', status: 'in_progress', startedAt: 1 }
  assert.ok(helpers.ticketWorkerLabel(ticket).startsWith(tx('Worker running')))
  assert.ok(helpers.ticketWorkerLabel(ticket).includes('@Cancel'))
  assert.equal(helpers.ticketStatusLabel('in_progress'), tx('In progress'))
  assert.equal(helpers.ticketStatusLabel('custom_status'), 'custom status')
  assert.equal(helpers.ticketPhaseLabel({ status: 'review' }), tx('Work review'))
  assert.equal(helpers.ROOM_STATE_LABEL.running, tx('Running'))
  assert.equal(helpers.roomJobTitle({ kind: 'ticket', status: 'completed', metadata: { name: 'Cancel' } }), `Cancel · ${tx('Completed')}`)
  assert.deepEqual(helpers.parseModelChoice(helpers.modelChoiceValue('Cancel', 'Settings', 'Copy')), { providerId: 'Cancel', providerAccountId: 'Settings', model: 'Copy' })
  assert.ok(Object.keys(helpers.ROOM_STATE_LABEL).includes('running'))
  const reason = 'Settings <raw diagnostic>'
  assert.equal(helpers.roomJobDetail({ blockedReason: reason }), reason)
  assert.ok(helpers.roomJobDetail({ blockedReason: reason, nextWakeAt: Date.now() - 1000 }).startsWith(reason + ' · '))
  assert.match(helpers.collaborationOption().detail, /\p{Script=Han}/u)
  assert.match(helpers.deliberationBudgetLabel('quick'), /\p{Script=Han}/u)
  assert.match(helpers.roomWorkerBudgetLabel({ workerBudget: { maxActive: 2, maxStarts: 4 } }, 1), /\p{Script=Han}/u)
  locale = 'en'
  assert.equal(helpers.ROOM_STATE_LABEL.running, 'Running')
  assert.equal(helpers.ticketWorkerLabel(ticket), 'Worker running @Cancel · recent')
  assert.equal(helpers.ticketStatusLabel('in_progress'), 'in progress')
  assert.equal(helpers.deliberationBudgetLabel('quick'), '4 turns · 2m · 60%')
  assert.equal(helpers.deliberationBudgetLabel('max'), '48 turns · open · 90%')
  assert.equal(helpers.roomWorkerBudgetLabel({ workerBudget: { maxActive: 2, maxStarts: 4 } }, 1), '1/2 active · 4 starts')
  locale = 'zh-Hans'
})

check('No component asks a language pack to translate English plural suffixes', () => {
  const violations: string[] = []
  function scan(directory: string) {
    for (const entry of readdirSync(directory, { withFileTypes: true })) {
      const file = path.join(directory, entry.name)
      if (entry.isDirectory()) { scan(file); continue }
      if (!file.endsWith('.tsx') || file.includes('.test.')) continue
      const ast = ts.createSourceFile(file, readFileSync(file, 'utf8'), ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX)
      const visit = (node: ts.Node) => {
        if (ts.isCallExpression(node) && node.expression.getText(ast) === 'tx' && node.arguments[0] && ts.isStringLiteral(node.arguments[0]) && ['s', 'es'].includes(node.arguments[0].text)) violations.push(`${file}:${ast.getLineAndCharacterOfPosition(node.getStart(ast)).line + 1}`)
        ts.forEachChild(node, visit)
      }
      visit(ast)
    }
  }
  scan('src/components')
  assert.deepEqual(violations, [])
})
assert.deepEqual(failures, [])
console.log('Actual file-count and room helpers retain counts, raw names, English behavior and Unicode initials without direct English suffix translation.')
