import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import ts from 'typescript'
import { isAttachmentReady } from '../lib/attachments'
import { isComposerSendKey } from '../lib/composer-keyboard'
import { fixtureText } from './preferences/language-test-fixture'
import { isPickerGuidance, PICKER_GUIDANCE_MESSAGE } from '../lib/store/slices/uiSlice'

const source = ts.createSourceFile('useComposerSubmit.tsx', readFileSync('src/components/composer/useComposerSubmit.tsx', 'utf8'), ts.ScriptTarget.ES2022, true, ts.ScriptKind.TSX)
const composer = source.statements.find((node): node is ts.FunctionDeclaration => ts.isFunctionDeclaration(node) && node.name?.text === 'useComposerSubmit')!
const viewSource = ts.createSourceFile('ComposerView.tsx', readFileSync('src/components/composer/ComposerView.tsx', 'utf8'), ts.ScriptTarget.ES2022, true, ts.ScriptKind.TSX)
const view = viewSource.statements.find((node): node is ts.FunctionDeclaration => ts.isFunctionDeclaration(node) && node.name?.text === 'ComposerView')!
const printer = ts.createPrinter()
const workspaceSource = ts.createSourceFile('useComposerWorkspace.ts', readFileSync('src/components/composer/useComposerWorkspace.ts', 'utf8'), ts.ScriptTarget.ES2022, true, ts.ScriptKind.TSX)
const workspace = workspaceSource.statements.find((node): node is ts.FunctionDeclaration => ts.isFunctionDeclaration(node) && node.name?.text === 'useComposerWorkspace')
assert.ok(workspace?.body, 'The real useComposerWorkspace function has a body')
const chooseDirStatement = workspace.body.statements.find((node) => ts.isVariableStatement(node) && node.declarationList.declarations.some((declaration) => ts.isIdentifier(declaration.name) && declaration.name.text === 'chooseDir'))
assert.ok(chooseDirStatement, 'The workspace function declares chooseDir')
const chooseDirCode = ts.transpileModule(`${printer.printNode(ts.EmitHint.Unspecified, chooseDirStatement, workspaceSource)}\nreturn chooseDir;`, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None } }).outputText
const setFolderStatement = workspace.body.statements.find((node) =>
  ts.isVariableStatement(node) && node.declarationList.declarations.some((declaration) =>
    ts.isIdentifier(declaration.name) && declaration.name.text === 'setFolder'))
const chooseRemoteProjectDirStatement = workspace.body.statements.find((node) =>
  ts.isVariableStatement(node) && node.declarationList.declarations.some((declaration) =>
    ts.isIdentifier(declaration.name) && declaration.name.text === 'chooseRemoteProjectDir'))
assert.ok(setFolderStatement && chooseRemoteProjectDirStatement, 'The workspace function declares both folder-selection routes')
const folderSource = [
  printer.printNode(ts.EmitHint.Unspecified, setFolderStatement, workspaceSource),
  printer.printNode(ts.EmitHint.Unspecified, chooseRemoteProjectDirStatement, workspaceSource),
  'return { setFolder, chooseRemoteProjectDir };',
].join('\n')
const folderCode = ts.transpileModule(folderSource, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None } }).outputText
type Notice = { notice?: { kind?: string; persistent?: boolean; message?: string } | null }
type ProductionHandlers = {
  dispatch: (body: string) => Promise<void>
  expandSkills: (text: string) => string
  submit: () => void
  onKeyDown: (event: { key: string; keyCode?: number; repeat?: boolean; nativeEvent: { isComposing?: boolean; keyCode?: number }; preventDefault: () => void }) => void
  pickFolder: () => Promise<void>
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
  if (ts.isJsxElement(node) && node.openingElement.tagName.getText(viewSource) === 'button') {
    const click = node.openingElement.attributes.properties.find((attribute) => ts.isJsxAttribute(attribute) && attribute.name.getText(viewSource) === 'onClick')
    if (click && ts.isJsxAttribute(click) && click.initializer && ts.isJsxExpression(click.initializer) && click.initializer.expression && ts.isIdentifier(click.initializer.expression) && click.initializer.expression.text === 'submit') button = node
  }
  ts.forEachChild(node, visit)
}
visit(view)
assert.ok(button, 'The actual Send button invokes the production submit handler')
const disabledAttribute = button.openingElement.attributes.properties.find((attribute) => ts.isJsxAttribute(attribute) && attribute.name.getText(viewSource) === 'disabled')!
assert.ok(ts.isJsxAttribute(disabledAttribute) && disabledAttribute.initializer && ts.isJsxExpression(disabledAttribute.initializer) && disabledAttribute.initializer.expression)
const disabledCode = ts.transpileModule(`return ${printer.printNode(ts.EmitHint.Expression, disabledAttribute.initializer.expression!, viewSource)};`, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None } }).outputText

const readyDocument = { id: 'document', kind: 'document' as const, name: 'notes.txt', mime: 'text/plain', status: 'ready' as const, text: 'Attached document contents' }
const submittedDraft = { text: '', attachments: [readyDocument], quotes: [] }
const pickerMessage = fixtureText('Choose a folder for this conversation. Your draft and attachments stay here if you cancel.')
async function settle() { for (let i = 0; i < 12; i++) await Promise.resolve() }
function scenario(overrides: Record<string, unknown> = {}) {
  const sent: unknown[][] = [], notices: Notice[] = []
  let cleared = 0, directoryChoices = 0, prevented = 0
  let currentNotice = (overrides.initialNotice ?? null) as Notice['notice']
  let selectedCwd = typeof overrides.cwd === 'string' ? overrides.cwd : ''
  const environment: Record<string, unknown> = {
    isPickerGuidance, PICKER_GUIDANCE_MESSAGE,
    text: '', attachments: [readyDocument], quoteEditingRef: { current: false }, quoteEditing: false,
    draft: { ready: true, capture: () => submittedDraft, clearAccepted: async () => { cleared++ } }, isAttachmentReady,
    useStore: {
      setState: (value: Notice) => { notices.push(value); if ('notice' in value) currentNotice = value.notice },
      getState: () => ({ notice: currentNotice, setNotice: (value: { kind: string; message: string }) => { notices.push({ notice: value }); currentNotice = value } }),
    },
    api: { pickDirectory: async () => { directoryChoices++; return overrides.pickerDirectory ?? null } },
    setCwd: (value: string) => { selectedCwd = value; environment.cwd = value },
    setHomeWorkspace: (value: null) => { environment.homeWorkspace = value },
    currentModelFit: { disabledReason: null }, modelFitReason: () => 'Model cannot accept this draft', recoveryBlocked: false, botBlocked: false,
    setView() {}, readiness: undefined, quotes: [], commands: [], skills: [],
    submittingRef: { current: false }, submitting: false, setSubmitting() {},
    provider: { id: 'local', models: ['fixture'] }, providerAccountId: null, model: 'fixture',
    mode: 'thread', thread: { id: 'chat' }, send: async (...args: unknown[]) => { sent.push(args); return true },
    homeWorkspace: null, cwd: '', projectRoot: '', isOrchestratorProject: false,
    loadRemoteProjectDir: async () => {},
    setRemoteBrowseOpen() {}, setRemoteBrowse() {}, setRemoteBrowseError() {},
    launch: async () => ({ ok: true }), settings: {}, homeProject: null, incognito: false, newMemoryPolicy: null,
    busy: false, pick: null, pickItems: [], mention: null, capabilityItems: [], items: [], pIdx: 0, mIdx: 0,
    selectPick() {}, selectCapability() {}, selectMention() {}, queueMessage() {}, isComposerSendKey,
    ...overrides,
  }
  const folders = new Function(
    'mode', 'thread', 'setThreadWorkspace', 'setCwd', 'setHomeWorkspace', 'useStore',
    'isPickerGuidance', 'PICKER_GUIDANCE_MESSAGE', 'setRemoteBrowseOpen', 'setRemoteBrowse', 'setRemoteBrowseError', 'tx', folderCode,
  )(
    environment.mode, environment.thread, environment.setThreadWorkspace, environment.setCwd, environment.setHomeWorkspace,
    environment.useStore, isPickerGuidance, PICKER_GUIDANCE_MESSAGE, environment.setRemoteBrowseOpen,
    environment.setRemoteBrowse, environment.setRemoteBrowseError, fixtureText,
  )
  if (!Object.hasOwn(overrides, 'setFolder')) environment.setFolder = folders.setFolder
  if (!Object.hasOwn(overrides, 'chooseRemoteProjectDir')) environment.chooseRemoteProjectDir = folders.chooseRemoteProjectDir
  if (!Object.hasOwn(overrides, 'chooseDir')) environment.chooseDir = new Function('api', 'setCwd', 'setHomeWorkspace', 'useStore', 'isPickerGuidance', 'PICKER_GUIDANCE_MESSAGE', 'tx', chooseDirCode)(environment.api, environment.setCwd, environment.setHomeWorkspace, environment.useStore, isPickerGuidance, PICKER_GUIDANCE_MESSAGE, fixtureText)
  const handlers = production(['dispatch', 'expandSkills', 'submit', 'onKeyDown', 'pickFolder'], environment)
  const buttonDisabled = new Function(...Object.keys(environment), disabledCode)(...Object.values(environment)) === true
  return {
    handlers, sent, notices, directoryChoices: () => directoryChoices, cleared: () => cleared,
    buttonDisabled, currentNotice: () => currentNotice, selectedCwd: () => selectedCwd,
    keyDown: () => handlers.onKeyDown({ key: 'Enter', nativeEvent: {}, preventDefault: () => { prevented++ } }),
    prevented: () => prevented,
    chooseRemoteProjectDir: (dir: string) => environment.chooseRemoteProjectDir(dir), setFolder: (dir: string) => environment.setFolder(dir),
    replaceNotice: (notice: Notice['notice']) => environment.useStore.setState({ notice }),
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

for (const emptyDocument of [{ ...readyDocument, name: 'empty.txt', text: '' }, { ...readyDocument, name: 'blank.txt', text: ' \n\t ' }]) {
  const empty = scenario({ attachments: [emptyDocument] })
  empty.handlers.submit()
  await settle()
  assert.equal(empty.sent.length, 1, 'A ready empty or whitespace-only document remains sendable')
  assert.deepEqual(empty.sent[0], ['', [emptyDocument], 'chat', []])
}
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
assert.ok(home.notices.some((value) => value.notice?.kind === 'info' && value.notice.persistent === true && /Choose a folder/.test(value.notice.message ?? '') && /draft and attachments/.test(value.notice?.message ?? '')), 'The automatic folder picker has a persistent explanation of why it opened and that cancellation preserves the draft')
assert.equal(home.currentNotice()?.persistent, true, 'Picker cancellation retains the persistent guidance')

const selected = scenario({ mode: 'home', thread: undefined, pickerDirectory: '/chosen' })
selected.handlers.submit()
await settle()
assert.equal(selected.directoryChoices(), 1)
assert.equal(selected.selectedCwd(), '/chosen')
assert.equal(selected.currentNotice(), null, 'Successful directory selection dismisses the matching picker guidance')

const launched = scenario({ mode: 'home', thread: undefined, cwd: '/chosen', initialNotice: { kind: 'info', persistent: true, message: pickerMessage } })
launched.handlers.dispatch('')
await settle()
assert.equal(launched.cleared(), 1)
assert.equal(launched.currentNotice(), null, 'A completed launch dismisses the matching picker guidance')

const replacement = { kind: 'info', persistent: true, message: 'Another notice replaced the picker guidance' }
const manual = scenario({ mode: 'home', thread: undefined, pickerDirectory: '/manual', initialNotice: { kind: 'info', persistent: true, message: pickerMessage } })
await manual.handlers.pickFolder()
assert.equal(manual.selectedCwd(), '/manual')
assert.equal(manual.currentNotice(), null, 'The visible pickFolder route clears matching guidance after a successful Home selection')

const manualReplacement = scenario({ mode: 'home', thread: undefined, pickerDirectory: '/manual', initialNotice: replacement })
await manualReplacement.handlers.pickFolder()
assert.deepEqual(manualReplacement.currentNotice(), replacement, 'Manual selection preserves an unrelated replacement notice')

const manualCancelled = scenario({ mode: 'home', thread: undefined, initialNotice: { kind: 'info', persistent: true, message: pickerMessage } })
await manualCancelled.handlers.pickFolder()
assert.equal(manualCancelled.selectedCwd(), '')
assert.equal(manualCancelled.currentNotice()?.message, pickerMessage, 'Manual picker cancellation retains matching guidance and the draft workspace')
let remoteBrowses = 0
const remote = scenario({ mode: 'home', thread: undefined, isOrchestratorProject: true, loadRemoteProjectDir: async () => { remoteBrowses++ } })
remote.handlers.submit()
await settle()
assert.equal(remoteBrowses, 1)
assert.equal(remote.directoryChoices(), 0, 'A remote owner browses the host rather than opening the native picker')
assert.equal(remote.currentNotice()?.message, pickerMessage)
remote.chooseRemoteProjectDir('/owner/project')
assert.equal(remote.selectedCwd(), '/owner/project')
assert.equal(remote.currentNotice(), null, 'Successful remote-owner selection clears matching guidance through the shared setFolder route')
assert.equal(remote.cleared(), 0, 'Folder selection does not consume the saved draft')

const noFolder = scenario({ mode: 'home', thread: undefined, initialNotice: { kind: 'info', persistent: true, message: pickerMessage } })
noFolder.setFolder('')
assert.equal(noFolder.currentNotice()?.message, pickerMessage, 'No folder leaves picker guidance in place')

const { promise: selectedDirectory, resolve: resolveDirectory } = Promise.withResolvers<string>()
const inFlight = scenario({
  mode: 'home', thread: undefined, pickerDirectory: selectedDirectory,
  initialNotice: { kind: 'info', persistent: true, message: pickerMessage },
})
const pendingSelection = inFlight.handlers.pickFolder()
inFlight.replaceNotice(replacement)
resolveDirectory('/later')
await pendingSelection
assert.equal(inFlight.selectedCwd(), '/later')
assert.deepEqual(inFlight.currentNotice(), replacement, 'Folder completion preserves an unrelated notice installed while the picker was open')
const replaced = scenario({ mode: 'home', thread: undefined, cwd: '/chosen', initialNotice: replacement })
replaced.handlers.dispatch('')
await settle()
assert.deepEqual(replaced.currentNotice(), replacement, 'Successful completion preserves an unrelated replacement notice')
console.log('Production Composer sends ready attachments, preserves picker cancellation guidance, and clears only matching guidance after selection or launch')
