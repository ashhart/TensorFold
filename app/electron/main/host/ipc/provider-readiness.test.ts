// Issue 3539: hermetic test of live verification; checkEndpoint:true opts into one bounded request, others stay pure.
import assert from 'node:assert/strict'
import http from 'node:http'
import { once } from 'node:events'
import nodePath from 'node:path'
import { readFileSync } from 'node:fs'
import ts from 'typescript'
import * as cliAuth from '../../../agent/cli-auth'
import * as cliAuthRuntime from '../../../agent/cli-auth-runtime'
import * as providerAccounts from '../../../agent/provider-accounts'
import * as engineProvider from '../../../agent/engine/provider'
import * as localProviders from '../../../agent/local-providers'
import * as security from '../../../agent/security'
import * as generationProbe from '../../../agent/provider-generation-probe'
import * as sharedReadiness from '../../../../shared/model-readiness'
import * as providerRuntime from '../provider-runtime'
import * as engineRuntime from '../engine-runtime'
import { store } from '../../state'
import type { Provider } from '../../../agent/types'

function load(file: string, overrides: Record<string, any>) {
  const source = readFileSync(file, 'utf8'), imports: Record<string, any> = {}
  Object.assign(imports, overrides)
  const module = { exports: {} as any }, output = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, esModuleInterop: true } }).outputText
  new Function('require', 'module', 'exports', output)((id: string) => { assert.ok(id in imports); return imports[id] }, module, module.exports)
  return module.exports
}

// Literal property values of every api.getModelReadiness(...) call in a renderer source file.
function readinessCallsites(file: string): Array<Record<string, unknown>> {
  const ast = ts.createSourceFile(file, readFileSync(file, 'utf8'), ts.ScriptTarget.ES2022, true)
  const calls: Array<Record<string, unknown>> = []
  const visit = (node: ts.Node) => {
    if (ts.isCallExpression(node) && ts.isPropertyAccessExpression(node.expression) && node.expression.name.text === 'getModelReadiness') {
      const flags: Record<string, unknown> = {}
      const argument = node.arguments[0]
      if (argument && ts.isObjectLiteralExpression(argument)) {
        for (const part of argument.properties) {
          if (!ts.isPropertyAssignment(part) || !ts.isIdentifier(part.name)) continue
          if (part.initializer.kind === ts.SyntaxKind.TrueKeyword) flags[part.name.text] = true
          else if (part.initializer.kind === ts.SyntaxKind.FalseKeyword) flags[part.name.text] = false
          else flags[part.name.text] = 'non-literal'
        }
      }
      calls.push(flags)
    }
    ts.forEachChild(node, visit)
  }
  visit(ast)
  return calls
}

const handlers = new Map<string, any>()
let role = 'operator'
const denyRole = (required: string) => {
  if (role !== required) throw new Error(`Current TensorFold role "${role}" cannot check model readiness; requires ${required}.`)
}
let cliChecks = 0
const cliAuthRuntimeStub = {
  ...cliAuthRuntime,
  checkCliAuth: async () => { cliChecks += 1; return { loggedIn: true, binaryPresent: true, detail: 'fixture status' } },
}
const providerIpc = load('electron/main/host/ipc/provider-ipc.ts', {
  '../../../../shared/model-readiness': sharedReadiness,
  '../../../agent/local-providers': localProviders,
  '../../../agent/provider-accounts': providerAccounts,
  '../../../agent/security': security,
  '../../../agent/cli-auth': cliAuth,
  '../../../agent/cli-auth-runtime': cliAuthRuntimeStub,
  '../../state': { store },
  '../audit': { requireSecurityRole: denyRole, auditEvent: async () => {} },
  '../ipc': { handle: (name: string, callback: any) => handlers.set(name, callback) },
  '../../../agent/engine/provider': engineProvider,
  '../engine-runtime': engineRuntime,
  '../provider-runtime': providerRuntime,
  '../../../agent/provider-generation-probe': generationProbe,
  '../voice-runtime': { discoverLocalTts: async () => ({ found: false, models: [] }), synthesizeLocalTts: async () => ({ ok: false }) },
  electron: { app: { getPath: () => '/synthetic' } },
  'node:path': nodePath,
  '../../../voice/kokoro-service': { KokoroService: class {} },
})
providerIpc.registerProviderIpc()
const readiness = (input: Record<string, unknown>) => handlers.get('provider:readiness')({}, input)

// Loopback fixture: one real HTTP server, per-case payload, request log for the bounded checks.
const MODEL_BODY = { data: [{ id: 'fixture-model' }] }
const seen: Array<{ path: string; authorization: string }> = []
let status = 200
let body = JSON.stringify(MODEL_BODY)
let routes: Record<string, { status: number; body: string }> | null = null
const server = http.createServer((req, res) => {
  const requestPath = String(req.url ?? '')
  const route = routes?.[requestPath]
  seen.push({ path: requestPath, authorization: String(req.headers.authorization ?? '') })
  res.statusCode = route?.status ?? status
  res.setHeader('content-type', 'application/json')
  res.end(route?.body ?? body)
})
server.listen(0, '127.0.0.1')
await once(server, 'listening')
const port = (server.address() as import('node:net').AddressInfo).port
const baseUrl = `http://127.0.0.1:${port}/v1`

const serve = (json: unknown, nextStatus = 200) => { status = nextStatus; body = JSON.stringify(json); routes = null; seen.length = 0 }
const serveRaw = (raw: string, nextStatus = 200) => { status = nextStatus; body = raw; routes = null; seen.length = 0 }

const local = (id: string, source: Provider['source'], extra: Partial<Provider> = {}): Provider => ({
  id, name: `Local ${source ?? 'custom'}`, kind: 'local', transport: 'chat-completions', source, baseUrl, apiKey: '', models: [], ...extra,
})
const account = (id: string, apiKey: string) => ({ id, name: id, kind: 'api-key' as const, enabled: true, apiKey, createdAt: 1, updatedAt: 1 })
const accountsProvider = local('local-accounts', 'custom', { defaultAccountId: 'acct-one', accounts: [account('acct-one', 'fictional-account-one-key'), account('acct-two', 'fictional-account-two-key')] })
const cloudProvider: Provider = { id: 'cloud-example', name: 'Cloud Example', kind: 'cloud', transport: 'chat-completions', source: 'custom', baseUrl, apiKey: 'fictional-cloud-key', models: [] }
const engineProviderFixture: Provider = { id: 'engine-example', name: 'Built-in engine', kind: 'local', transport: 'chat-completions', source: 'tensorfold', baseUrl: '', apiKey: '', models: [] }
const cliProvider: Provider = { id: 'cli-example', name: 'Codex CLI', kind: 'cloud', transport: 'codex-cli', source: 'custom', baseUrl, apiKey: '', models: [] }
// OpenCode-protocol servers answer through their own transport, never the /v1 model list.
const opencodeProvider: Provider = { id: 'opencode-example', name: 'Local OpenCode', kind: 'local', transport: 'opencode', source: 'custom', baseUrl, apiKey: '', models: [] }

const oldSettings = store.settings
store.settings = {
  ...oldSettings,
  providers: [
    local('local-ollama', 'ollama'), local('local-lmstudio', 'lmstudio'), local('local-omlx', 'omlx'), local('local-custom', undefined),
    local('local-ollama-key', 'ollama', { apiKey: 'fictional-ollama-key' }),
    accountsProvider, cloudProvider, engineProviderFixture, cliProvider, opencodeProvider,
  ],
  activeProviderId: 'local-omlx',
  activeProviderAccountId: null,
  activeModel: null,
}
try {
  // Opt-in live verification for every local HTTP source: Ready only with a real non-empty model list.
  const expectedPath: Record<string, string> = { 'local-ollama': '/api/tags', 'local-lmstudio': '/v1/models', 'local-omlx': '/v1/models', 'local-custom': '/v1/models' }
  for (const providerId of Object.keys(expectedPath)) {
    serve(MODEL_BODY)
    const result = await readiness({ providerId, checkEndpoint: true })
    assert.equal(result.ready, true, `${providerId}: a live endpoint with a model list is ready`)
    assert.equal(result.reason, 'ok', `${providerId}: a verified model list reports the ok reason`)
    assert.ok(seen.length >= 1, `${providerId}: checkEndpoint:true must actually request the model list`)
    assert.ok(seen.length <= 2, `${providerId}: live verification stays bounded to one or two requests`)
    assert.equal(seen[0].path, expectedPath[providerId], `${providerId}: the probe hits the model list of the configured endpoint`)
  }

  // The Ollama compatibility fallback must carry the provider's own key, not a dropped one.
  status = 200; routes = { '/api/tags': { status: 404, body: '{}' }, '/v1/models': { status: 200, body: JSON.stringify(MODEL_BODY) } }; seen.length = 0
  const compat = await readiness({ providerId: 'local-ollama-key', checkEndpoint: true })
  assert.equal(compat.ready, true, 'Ollama is ready through its OpenAI-compatible route')
  assert.equal(seen.at(-1)?.authorization, 'Bearer fictional-ollama-key', 'the configured Ollama key is sent on the compatibility route')

  // The selected account's own credential must reach the live model-list request.
  serve(MODEL_BODY)
  assert.equal(accountsProvider.defaultAccountId, 'acct-one', 'the fixture does have a different default account')
  const withAccount = await readiness({ providerId: 'local-accounts', accountId: 'acct-two', checkEndpoint: true })
  assert.equal(withAccount.ready, true, 'the selected account keeps a live provider ready')
  assert.equal(seen.at(-1)?.authorization, 'Bearer fictional-account-two-key', 'the selected account key, not the default account, is sent')

  // Non-ready shapes must not report Ready while opted in.
  serve({ data: [] })
  const empty = await readiness({ providerId: 'local-omlx', checkEndpoint: true })
  assert.equal(empty.ready, false, 'an empty model list is not Ready')
  assert.equal(empty.reason, 'needs-endpoint', 'an empty model list keeps the existing not-ready reason')
  serve(MODEL_BODY, 401)
  assert.equal((await readiness({ providerId: 'local-custom', checkEndpoint: true })).ready, false, 'a 401 model list is not Ready')
  serve(MODEL_BODY, 404)
  assert.equal((await readiness({ providerId: 'local-lmstudio', checkEndpoint: true })).ready, false, 'a 404 model list is not Ready')
  serveRaw('this is not json', 200)
  assert.equal((await readiness({ providerId: 'local-omlx', checkEndpoint: true })).ready, false, 'a malformed payload is not Ready')
  status = 200; routes = { '/api/tags': { status: 404, body: '{}' }, '/v1/models': { status: 404, body: '{}' } }; seen.length = 0
  assert.equal((await readiness({ providerId: 'local-ollama', checkEndpoint: true })).ready, false, 'Ollama with no answering route is not Ready')
  assert.ok(seen.length <= 2, 'both Ollama routes stay inside the bounded request count')

  // Without opt-in the composer path stays pure, and cloud/CLI/built-in sources never probe.
  serve(MODEL_BODY)
  assert.equal((await readiness({ providerId: 'local-omlx' })).ready, true, 'a plain composer check still reads Ready')
  assert.equal(seen.length, 0, 'no opt-in means no live request')
  serve(MODEL_BODY)
  assert.equal((await readiness({ providerId: 'local-omlx', checkEndpoint: false })).ready, true, 'an explicit false keeps the pure check')
  assert.equal(seen.length, 0, 'checkEndpoint:false means no live request')
  serve(MODEL_BODY)
  assert.equal((await readiness({ providerId: 'cloud-example', checkEndpoint: true })).ready, true, 'a cloud provider keeps its key-based Ready')
  assert.equal(seen.length, 0, 'only local HTTP sources are verified live')
  serve(MODEL_BODY)
  assert.equal((await readiness({ providerId: 'engine-example', checkEndpoint: true })).ready, false, 'the built-in engine keeps its own readiness path')
  assert.equal(seen.length, 0, 'the built-in engine is never probed through the local HTTP route')
  serve(MODEL_BODY)
  const beforeCli = cliChecks
  assert.equal((await readiness({ providerId: 'cli-example', checkEndpoint: true })).ready, true, 'a CLI transport keeps its status-based Ready')
  assert.equal(seen.length, 0, 'CLI transports never send a model-list request')
  assert.equal(cliChecks, beforeCli + 1, 'the CLI branch is the one that answered')
  serve(MODEL_BODY)
  assert.equal((await readiness({ providerId: 'opencode-example', checkEndpoint: true })).ready, true, 'an OpenCode transport keeps its presence-based Ready')
  assert.equal(seen.length, 0, 'the OpenCode transport never sends a model-list request')
  assert.equal((await readiness({ providerId: 'missing-provider', checkEndpoint: true })).reason, 'no-provider', 'an unknown provider still needs a model')

  // Secrets and server bodies stay out of the readiness answers.
  serveRaw('{"data":[{"id":"fixture-model"}],"secret":"PLAINTEXT-SECRET-VALUE"}', 200)
  const withSecret = await readiness({ providerId: 'local-custom', checkEndpoint: true })
  assert.equal(withSecret.ready, true, 'a model list with extra fields is still Ready')
  assert.ok(!JSON.stringify(withSecret).includes('PLAINTEXT-SECRET-VALUE'), 'the server body is not echoed into the readiness result')

  role = 'viewer'
  await assert.rejects(() => readiness({ providerId: 'local-omlx', checkEndpoint: true }), /cannot check model readiness/)
  role = 'operator'

  // Wiring regression: both Readiness screens opt in, the composer and onboarding stay pure.
  for (const file of ['src/components/settings/ReadinessSettings.tsx', 'src/components/CommandCentre.tsx']) {
    const calls = readinessCallsites(file)
    assert.ok(calls.length >= 1, `${file}: the Readiness screen still calls getModelReadiness`)
    for (const flags of calls) assert.equal(flags.checkEndpoint, true, `${file}: its readiness call must opt into the live model-list check`)
  }
  for (const file of ['src/components/composer/useComposerState.ts', 'src/components/Onboarding.tsx']) {
    const calls = readinessCallsites(file)
    assert.ok(calls.length >= 1, `${file}: its readiness calls are still present`)
    for (const flags of calls) assert.ok(!('checkEndpoint' in flags), `${file}: non-Readiness surfaces stay presence-based`)
  }

  // A closed listener gives a real connection refusal instead of a stubbed timeout.
  await new Promise<void>((resolve) => server.close(() => resolve()))
  seen.length = 0
  const refused = await readiness({ providerId: 'local-ollama', checkEndpoint: true })
  assert.equal(refused.ready, false, 'an unreachable local server is not Ready')
  assert.equal(refused.reason, 'needs-endpoint', 'an unreachable server keeps the existing not-ready reason')
  console.log('provider:readiness opt-in: live model-list verification, empty/401/404/malformed/unreachable refusals, pure non-opt-in path, callsite wiring.')
} finally {
  store.settings = oldSettings
  if (server.listening) await new Promise<void>((resolve) => server.close(() => resolve()))
}
