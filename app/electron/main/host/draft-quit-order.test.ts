import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import ts from 'typescript'

// Execute the production callbacks with inert lifecycle collaborators. Importing
// main-app would launch Electron; AST extraction keeps this seam isolated.
const file = path.join(process.cwd(), 'electron/main-app.ts')
const tree = ts.createSourceFile(file, readFileSync(file, 'utf8'), ts.ScriptTarget.Latest, true)
function find(predicate: (node: ts.Node) => boolean): ts.Node {
  let result: ts.Node | undefined
  const visit = (node: ts.Node) => { if (predicate(node)) result = node; ts.forEachChild(node, visit) }
  visit(tree); assert.ok(result); return result
}
function compile(node: ts.Node): string {
  return ts.transpileModule(ts.createPrinter().printNode(ts.EmitHint.Unspecified, node, tree), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None } }).outputText
}
const registration = compile(find((node) => ts.isCallExpression(node) && node.expression.getText(tree) === 'app.on' && node.arguments[0]?.getText(tree) === "'before-quit'"))
const resume = compile(find((node) => ts.isFunctionDeclaration(node) && node.name?.text === 'resumeAfterFailedQuit'))
let handler!: (event: { preventDefault(): void }) => void
let log: string[] = [], hold: string | null = null, enabled = true
const gate = (name: string) => (event: { preventDefault(): void }) => { log.push(name); if (hold === name) { event.preventDefault(); return true }; return false }
const teardown = (name: string) => () => { log.push(name) }
const store = { win: { isDestroyed: () => false, setEnabled(value: boolean) { enabled = value; log.push(`enabled:${value}`) } } }
const collaborators: Record<string, unknown> = {
  app: { on(_name: string, callback: typeof handler) { handler = callback } }, store,
  quitAdmissionPause: null, workspaceMutationGuard: { pauseAdmission() { log.push('pause-admission'); return { release() {} } } },
  aborters: new Map([['thread', { abort: teardown('abort-thread') }]]), roomAborters: new Map([['room', { abort: teardown('abort-room') }]]), missionAborters: new Map([['mission', { abort: teardown('abort-mission') }]]),
  automationRuntime: { abortAll: teardown('abort-automation') },
  telegramRemoteRuntime: { cancelPending: teardown('abort-telegram-remote') },
  deferLearningQuit: gate('learning'), deferQuestionQuit: gate('questions'), deferPhoneQuit: gate('phone'), deferBotVmQuit: gate('vm'), deferDocumentQuit: gate('documents'), deferDraftQuit: gate('drafts'),
  desktopModeRuntime: { dispose: teardown('dispose-desktop') }, botPollingApi: { stopBotControlPolling: teardown('stop-polling') },
  stopAllWhatsAppBridges: teardown('stop-whatsapp'), closeBrowserAutomationSessions: teardown('stop-browser'), stopCaffeinate: teardown('stop-power'), shutdownComputerUse: teardown('stop-computer'),
  attachmentTextService: { shutdown: teardown('stop-ocr') }, killAllTerminals: teardown('stop-terminal'), closeAllCodexAppServerSessions: teardown('stop-codex'),
  engineService: () => ({ shutdown: teardown('stop-engine') }), mcpHost: { shutdown: async () => { log.push('stop-mcp') } }, webhookServer: { stop: async () => { log.push('stop-webhooks') } }, mobileBridgeRuntime: { stop: async () => { log.push('stop-mobile') } }, remoteServer: { stop: async () => { log.push('stop-remote') } },
}
new Function(...Object.keys(collaborators), registration)(...Object.values(collaborators))
const gates = ['learning', 'questions', 'phone', 'vm', 'documents', 'drafts']
for (const name of gates) {
  hold = name; log = []; enabled = true; let prevented = 0
  handler({ preventDefault() { prevented++ } })
  assert.equal(enabled, false, 'native input is frozen before an asynchronous Quit fence')
  assert.equal(log[0], 'enabled:false')
  assert.ok(log.indexOf('abort-telegram-remote') > log.indexOf('enabled:false') && log.indexOf('abort-telegram-remote') < log.indexOf('learning'), 'remote inference is cancelled after input freezes and before any asynchronous quit fence or draft flush')
  assert.deepEqual(log.filter((entry) => gates.includes(entry)), gates.slice(0, gates.indexOf(name) + 1), 'draft flush is last, after every device/model fence')
  assert.equal(prevented, 1)
  assert.equal(log.includes('dispose-desktop'), false, 'no final teardown while any fence is pending')
}
hold = null; log = []
handler({ preventDefault() { throw new Error('Settled Quit should proceed') } })
assert.deepEqual(log.filter((entry) => gates.includes(entry)), gates)
assert.ok(log.indexOf('drafts') < log.indexOf('dispose-desktop'))
assert.ok(log.includes('stop-ocr') && log.includes('stop-engine'))
let released = 0
const resumeFixture = new Function('store', 'userQuestions', 'phones', 'botVms', 'resumeDocumentLibrary', 'resumeModelQualifications', 'resumeModelHelpers', 'release', `
let documentQuitSettled = true, draftQuitSettled = true, questionQuitSettled = true, phoneQuitSettled = true, botVmQuitSettled = true;
let quitAdmissionPause = { release };
${resume}
resumeAfterFailedQuit();
return { documentQuitSettled, draftQuitSettled, questionQuitSettled, phoneQuitSettled, botVmQuitSettled, quitAdmissionPause };
`)
log = []; enabled = false
const flags = resumeFixture(store, { resume: teardown('resume-questions') }, { service: { resume: teardown('resume-phone') } }, { resume: teardown('resume-vm') }, teardown('resume-documents'), teardown('resume-qualifications'), teardown('resume-helpers'), () => { released++ })
assert.equal(enabled, true, 'a failed Quit restores native input')
assert.deepEqual(flags, { documentQuitSettled: false, draftQuitSettled: false, questionQuitSettled: false, phoneQuitSettled: false, botVmQuitSettled: false, quitAdmissionPause: null })
assert.equal(released, 1)
console.log('Production Quit callbacks: native input frozen, model/device fences precede final draft flush, pending teardown blocked, failed-Quit draft fence and input reset passed')
