import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import ts from 'typescript'
import { isAttachmentReady } from '../lib/attachments'
import { isComposerSendKey } from '../lib/composer-keyboard'
import { fixtureText } from './preferences/language-test-fixture'

const source = ts.createSourceFile('Composer.tsx', readFileSync('src/components/Composer.tsx', 'utf8'), ts.ScriptTarget.ES2022, true, ts.ScriptKind.TSX)
const composer = source.statements.find((node): node is ts.FunctionDeclaration => ts.isFunctionDeclaration(node) && node.name?.text === 'Composer')!
const printer = ts.createPrinter()
type Notice = { notice?: { kind?: string; message?: string } }
type ProductionHandlers = {
  dispatch: (body: string) => Promise<void>
  expandSkills: (text: string) => string
  submit: () => void
  onKeyDown: (event: { key: string; keyCode?: number; repeat?: boolean; nativeEvent: { isComposing?: boolean; keyCode?: number }; preventDefault: () => void }) => void
}
function production(names: string[], environment: Record<string, unknown>): ProductionHandlers {
  const declarations = names.map((name) => {
    const statement = composer.body!.statements.find((node) => ts.isVariableStatement(node) && node.declarationList.declarations.some((declaration) => ts.isIdentifier(declaration.name) && declaration.name.text === name))
    assert.ok(statement, `Production handler ${name} exists`)
    return printer.printNode(ts.EmitHint.Unspecified, statement, source)
  }).join('\n')
  const code = ts.transpileModule(`${declarations}\nreturn { ${names.join(',')} };`, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None } }).outputText
  return new Function('tx', ...Object.keys(environment), code)(fixtureText, ...Object.values(environment)) as ProductionHandlers
}

let button: ts.JsxElement | undefined
const visit = (node: ts.Node) => {
  if (ts.isJsxElement(node) && node.openingElement.tagName.getText(source) === 'button') {
    const click = node.openingElement.attributes.properties.find((attribute) => ts.isJsxAttribute(attribute) && attribute.name.getText(source) === 'onClick')
    if (click && ts.isJsxAttribute(click) && click.initializer && ts.isJsxExpression(click.initializer) && click.initializer.expression && ts.isIdentifier(click.initializer.expression) && click.initializer.expression.text === 'submit') button = node
  }
  ts.forEachChild(node, visit)
}
visit(composer)
assert.ok(button, 'The actual Send button invokes the production submit handler')
const disabledAttribute = button.openingElement.attributes.properties.find((attribute) => ts.isJsxAttribute(attribute) && attribute.name.getText(source) === 'disabled')!
assert.ok(ts.isJsxAttribute(disabledAttribute) && disabledAttribute.initializer && ts.isJsxExpression(disabledAttribute.initializer) && disabledAttribute.initializer.expression)
const disabledCode = ts.transpileModule(`return ${printer.printNode(ts.EmitHint.Expression, disabledAttribute.initializer.expression!, source)};`, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None } }).outputText

const readyDocument = { id: 'document', kind: 'document' as const, name: 'notes.txt', mime: 'text/plain', status: 'ready' as const, text: 'Attached document contents' }
const submittedDraft = { text: '', attachments: [readyDocument], quotes: [] }
async function settle() { for (let i = 0; i < 12; i++) await Promise.resolve() }
function scenario(overrides: Record<string, unknown> = {}) {
  const sent: unknown[][] = [], notices: Notice[] = []
  let cleared = 0, directoryChoices = 0, prevented = 0
  const environment: Record<string, unknown> = {
    text: '', attachments: [readyDocument], quoteEditingRef: { current: false }, quoteEditing: false,
    draft: { ready: true, capture: () => submittedDraft, clearAccepted: async () => { cleared++ } }, isAttachmentReady,
    useStore: { setState: (value: Notice) => notices.push(value), getState: () => ({ setNotice: (value: { kind: string; message: string }) => notices.push({ notice: value }) }) },
    currentModelFit: { disabledReason: null }, modelFitReason: () => 'Model cannot accept this draft', recoveryBlocked: false, botBlocked: false,
    setView() {}, readiness: undefined, quotes: [], commands: [], skills: [],
    submittingRef: { current: false }, submitting: false, setSubmitting() {},
    provider: { id: 'local', models: ['fixture'] }, providerAccountId: null, model: 'fixture',
    mode: 'thread', thread: { id: 'chat' }, send: async (...args: unknown[]) => { sent.push(args); return true },
    homeWorkspace: null, cwd: '', projectRoot: '', isOrchestratorProject: false,
    chooseDir: async () => { directoryChoices++ }, loadRemoteProjectDir: async () => {},
    launch: async () => ({ ok: true }), settings: {}, homeProject: null, incognito: false, newMemoryPolicy: null,
    busy: false, pick: null, pickItems: [], mention: null, capabilityItems: [], items: [], pIdx: 0, mIdx: 0,
    selectPick() {}, selectCapability() {}, selectMention() {}, queueMessage() {}, isComposerSendKey,
    ...overrides,
  }
  const handlers = production(['dispatch', 'expandSkills', 'submit', 'onKeyDown'], environment)
  const buttonDisabled = new Function(...Object.keys(environment), disabledCode)(...Object.values(environment)) === true
  return {
    handlers, sent, notices, directoryChoices: () => directoryChoices, cleared: () => cleared,
    buttonDisabled,
    keyDown: () => handlers.onKeyDown({ key: 'Enter', nativeEvent: {}, preventDefault: () => { prevented++ } }),
    prevented: () => prevented,
  }
}

const clicked = scenario()
clicked.handlers.submit()
await settle()
const keyboard = scenario()
keyboard.keyDown()
await settle()
assert.deepEqual({ buttonDisabled: clicked.buttonDisabled, buttonSends: clicked.sent.length, keyboardSends: keyboard.sent.length, keyboardPrevented: keyboard.prevented() }, { buttonDisabled: false, buttonSends: 1, keyboardSends: 1, keyboardPrevented: 1 }, 'A ready document is sendable with no typed text through either Send or Enter')
assert.deepEqual(clicked.sent[0], ['', [readyDocument], 'chat', []])
assert.equal(clicked.cleared(), 1)

const blockedCases: Array<[string, Record<string, unknown>]> = [
  ['empty draft', { attachments: [] }],
  ['attachment review', { attachments: [{ ...readyDocument, status: 'review' }] }],
  ['draft error', { draft: { ready: false, capture: () => submittedDraft, clearAccepted: async () => {} } }],
  ['model capability', { currentModelFit: { disabledReason: 'vision' } }],
  ['run recovery', { recoveryBlocked: true }],
]
for (const [label, overrides] of blockedCases) {
  const blocked = scenario(overrides)
  blocked.handlers.submit()
  await settle()
  assert.equal(blocked.sent.length, 0, `${label} still blocks attachment-only submission`)
}

const home = scenario({ mode: 'home', thread: undefined })
home.handlers.submit()
await settle()
assert.equal(home.sent.length, 0)
assert.equal(home.directoryChoices(), 1)
assert.ok(home.notices.some((value) => value.notice?.kind === 'info' && /Choose a folder/.test(value.notice.message ?? '') && /draft and attachments/.test(value.notice?.message ?? '')), 'The automatic folder picker explains why it opened and that cancellation preserves the draft')
console.log('Production Composer sends ready attachment-only drafts by button and keyboard while preserving review, model and recovery gates')
