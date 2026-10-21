// Run: npx tsx electron/agent/tools.test.ts
import assert from 'node:assert'
import { installDnsFixture } from './test-fixtures/dns'
import { promises as fs } from 'node:fs'
import { existsSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { CHAT_TOOL_SCHEMAS, executeTool, READ_ONLY_TOOLS, toolLabel, TOOL_SCHEMAS } from './tools.ts'
import { canUseMacSandbox } from './sandbox.ts'
import { workspaceBackend } from './tool-workspace.ts'
import { emailDraftEvent } from './email-connector.ts'
import { BROWSER_AUTOMATION_ACTIONS } from './browser-automation.ts'
import { CONTROLLER_ACTIONS } from './controller-actions.ts'
import { makeBrowserSettings, makeCalendarSettings, makeDocumentSettings, makeEmailSettings, makeSlackSettings } from './tools-test-fixtures.ts'
import type { WorkspaceBackend, WorkspaceFileEntry, WorkspaceStat } from './workspace-backend.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

async function makeWorkspace(): Promise<string> {
  const cwd = await fs.mkdtemp(path.join(os.tmpdir(), 'vontra-tools-'))
  await fs.mkdir(path.join(cwd, 'src/nested'), { recursive: true })
  await fs.mkdir(path.join(cwd, 'node_modules/pkg'), { recursive: true })
  await fs.writeFile(path.join(cwd, 'README.md'), '# Test\n', 'utf8')
  await fs.writeFile(path.join(cwd, 'src/App.tsx'), 'export const App = () => null\n', 'utf8')
  await fs.writeFile(path.join(cwd, 'src/nested/util.ts'), 'export const value = 1\n', 'utf8')
  await fs.writeFile(path.join(cwd, 'node_modules/pkg/ignored.ts'), 'ignored\n', 'utf8')
  return cwd
}

async function main() {
console.log('tool tests:')

// Non-macOS CI has no sandbox-exec. Use the existing explicit development escape hatch so
// run_command cases exercise the real exec path; the fail-closed default is covered separately.
if (!canUseMacSandbox()) process.env.VONTRA_ALLOW_UNSANDBOXED_COMMANDS = '1'

{
  assert.throws(
    () => workspaceBackend({ cwd: '/srv/project', workspace: { kind: 'ssh', hostId: 'host-1', path: '/srv/project' } }),
    /SSH workspace is disconnected.*refusing to fall back/i,
  )
  ok('an SSH workspace identity without a live backend fails closed before any local filesystem access')
}

{
  const cwd = await makeWorkspace()
  const result = await executeTool('glob', { pattern: '**/*.ts', max_results: 20 }, { cwd })
  assert.strictEqual(result.ok, true, result.output)
  assert.match(result.output, /src\/nested\/util\.ts/)
  assert.doesNotMatch(result.output, /node_modules/)
  assert.ok(READ_ONLY_TOOLS.has('glob'))
  assert.ok(TOOL_SCHEMAS.some((tool: any) => tool.function?.name === 'glob'))
  assert.deepStrictEqual(toolLabel('glob', { pattern: '**/*.ts' }), { verb: 'Glob', target: '**/*.ts' })
  ok('glob finds workspace files and stays read-only')
}

{
  const cwd = await makeWorkspace()
  await fs.writeFile(path.join(cwd, 'src/lines.txt'), Array.from({ length: 30 }, (_, i) => `line-${i + 1}`).join('\n'), 'utf8')
  const result = await executeTool('read_file', { path: 'src/lines.txt', offset: '10', limit: '3' }, { cwd })
  assert.strictEqual(result.ok, true, result.output)
  assert.match(result.output, /   10\tline-10/)
  assert.match(result.output, /   12\tline-12/)
  assert.doesNotMatch(result.output, /line-13/)
  ok('read_file normalizes string offset and limit before slicing')
}

{
  // Huge files (VST-016): the line limit alone cannot bound enormous lines.
  const cwd = await makeWorkspace()
  // 400 lines x ~200 chars ≈ 80k chars — over the ~48k read bound at full width.
  await fs.writeFile(path.join(cwd, 'src/wide.txt'), Array.from({ length: 400 }, (_, i) => `row-${i + 1} ${'w'.repeat(195)}`).join('\n'), 'utf8')
  const wide = await executeTool('read_file', { path: 'src/wide.txt' }, { cwd })
  assert.strictEqual(wide.ok, true)
  assert.ok(wide.output.length < 50_000, `bounded (got ${wide.output.length} chars)`)
  assert.match(wide.output, /row-1 /)
  assert.doesNotMatch(wide.output, /row-400 /)
  assert.match(wide.output, /\[read_file bounded by TensorFold: .*Continue with offset=\d+/)
  assert.match(wide.meta ?? '', /bounded/)
  const continueAt = Number(/offset=(\d+)/.exec(wide.output)?.[1])
  assert.ok(continueAt > 1 && continueAt <= 400, 'continuation offset points into the file')

  // A single monster line (minified bundle): bounded mid-line with a search hint.
  await fs.writeFile(path.join(cwd, 'src/minified.js'), `const x=1;${'y'.repeat(200_000)}`, 'utf8')
  const minified = await executeTool('read_file', { path: 'src/minified.js' }, { cwd })
  assert.strictEqual(minified.ok, true)
  assert.ok(minified.output.length < 50_000, `bounded (got ${minified.output.length} chars)`)
  assert.match(minified.output, /line 1 alone is .* showing its first 48,000 chars/)
  ok('read_file bounds huge ranges and single monster lines, with a named continuation')
}

{
  const cwd = await makeWorkspace()
  await fs.mkdir(path.join(cwd, '.git/hooks'), { recursive: true })
  await fs.writeFile(path.join(cwd, '.git/config'), '[core]\n\trepositoryformatversion = 0\n', 'utf8')
  await fs.writeFile(path.join(cwd, '.git/hooks/pre-commit'), '#!/bin/sh\nexit 0\n', 'utf8')

  const writeHook = await executeTool('write_file', { path: '.git/hooks/pre-commit', content: '#!/bin/sh\necho planted\n' }, { cwd })
  const writeConfig = await executeTool('write_file', { path: '.git/config', content: '[core]\nhooksPath=/tmp/evil\n' }, { cwd })
  const editHook = await executeTool('edit_file', { path: '.git/hooks/pre-commit', old_string: 'exit 0', new_string: 'echo planted' }, { cwd })
  const patchConfig = await executeTool('apply_patch', {
    patch: '*** Begin Patch\n*** Update File: .git/config\n@@\n-[core]\n+[alias]\n*** End Patch',
  }, { cwd })
  const patchHookAdd = await executeTool('apply_patch', {
    patch: '*** Begin Patch\n*** Add File: .git/hooks/post-commit\n+#!/bin/sh\n+echo planted\n*** End Patch',
  }, { cwd })
  const patchMove = await executeTool('apply_patch', {
    patch: '*** Begin Patch\n*** Update File: README.md\n*** Move to: .git/hooks/post-checkout\n*** End Patch',
  }, { cwd })

  for (const result of [writeHook, writeConfig, editHook, patchConfig, patchHookAdd, patchMove]) {
    assert.equal(result.ok, false)
    assert.match(result.output, /Blocked by TensorFold(?: Studio)? safety policy.*Git private metadata/i)
  }
  assert.equal(await fs.readFile(path.join(cwd, '.git/hooks/pre-commit'), 'utf8'), '#!/bin/sh\nexit 0\n')
  assert.equal(await fs.readFile(path.join(cwd, '.git/config'), 'utf8'), '[core]\n\trepositoryformatversion = 0\n')
  await assert.rejects(() => fs.access(path.join(cwd, '.git/hooks/post-commit')))
  assert.equal(await fs.readFile(path.join(cwd, 'README.md'), 'utf8'), '# Test\n')
  ok('write_file, edit_file, and every apply_patch shape refuse .git metadata, config, hooks, and move targets')
}

{
  const cwd = await makeWorkspace()
  await fs.mkdir(path.join(cwd, 'config'), { recursive: true })
  await fs.mkdir(path.join(cwd, 'deploy'), { recursive: true })
  await fs.writeFile(path.join(cwd, 'config/service-account.json'), '{"private_key":"SEARCH_SECRET_MARKER"}\n', 'utf8')
  await fs.writeFile(path.join(cwd, 'deploy/secrets.yaml'), 'token: SEARCH_SECRET_MARKER\n', 'utf8')
  await fs.writeFile(path.join(cwd, 'src/visible.txt'), 'SEARCH_SECRET_MARKER is documented safely here\n', 'utf8')
  const result = await executeTool('search', { query: 'SEARCH_SECRET_MARKER' }, { cwd })
  assert.equal(result.ok, true, result.output)
  assert.match(result.output, /src\/visible\.txt/)
  assert.doesNotMatch(result.output, /service-account\.json|secrets\.yaml|private_key|token:/)
  ok('local search filters service-account and secret-file contents while retaining ordinary hits')
}

{
  assert.ok(TOOL_SCHEMAS.some((tool: any) => tool.function?.name === 'launch_subagent'))
  assert.ok(TOOL_SCHEMAS.some((tool: any) => tool.function?.name === 'launch_subagents'))
  assert.ok(!READ_ONLY_TOOLS.has('launch_subagent'))
  assert.ok(!READ_ONLY_TOOLS.has('launch_subagents'))
  assert.deepStrictEqual(toolLabel('launch_subagent', { task: 'Inspect tests', name: 'Test scout' }), { verb: 'Launch sub-agent', target: 'Test scout' })
  const result = await executeTool('launch_subagent', { task: 'Inspect tests', name: 'Test scout' }, {
    cwd: await makeWorkspace(),
    launchSubAgent: async (input) => ({
      output: `launched: ${input.task}`,
      agent: {
        id: 'sub-1',
        parent: { kind: 'thread', id: 'thread-1', label: 'Parent thread', threadId: 'thread-1' },
        name: input.name || 'Thread sub-agent',
        task: input.task,
        status: 'succeeded',
        projectId: null,
        roomId: null,
        threadId: 'thread-1',
        seatId: null,
        providerId: 'openai',
        providerAccountId: null,
        model: input.model ?? 'gpt-4o',
        runId: 'run-sub-1',
        result: `launched: ${input.task}`,
        error: null,
        events: [],
        messages: [],
        createdAt: 1,
        updatedAt: 2,
        startedAt: 1,
        finishedAt: 2,
      },
    }),
  })
  assert.strictEqual(result.ok, true)
  assert.match(result.output, /launched: Inspect tests/)
  assert.strictEqual(result.title, 'Test scout')
  assert.strictEqual(result.subAgent?.id, 'sub-1')
  const launched: string[] = []
  const batch = await executeTool('launch_subagents', { tasks: [
    { task: 'Inspect tests', name: 'Test scout' },
    { task: 'Inspect docs', name: 'Docs scout' },
    { task: 'Inspect CI', name: 'CI scout' },
    { task: 'Inspect overflow', name: 'Overflow scout' },
  ] }, {
    cwd: await makeWorkspace(),
    launchSubAgent: async (input) => {
      launched.push(input.name || input.task)
      return {
        output: `launched: ${input.task}`,
        agent: {
          id: `sub-${launched.length}`,
          parent: { kind: 'thread', id: 'thread-1', label: 'Parent thread', threadId: 'thread-1' },
          name: input.name || 'Thread sub-agent',
          task: input.task,
          status: 'succeeded',
          projectId: null,
          roomId: null,
          threadId: 'thread-1',
          seatId: null,
          providerId: 'openai',
          providerAccountId: null,
          model: input.model ?? 'gpt-4o',
          runId: `run-sub-${launched.length}`,
          result: `launched: ${input.task}`,
          error: null,
          events: [],
          messages: [],
          createdAt: 1,
          updatedAt: 2,
          startedAt: 1,
          finishedAt: 2,
        },
      }
    },
  })
  assert.strictEqual(batch.ok, true)
  assert.deepStrictEqual(launched, ['Test scout', 'Docs scout', 'CI scout'])
  assert.match(batch.output, /1\. Test scout: launched: Inspect tests/)
  assert.match(batch.output, /3\. CI scout: launched: Inspect CI/)
  assert.match(batch.meta ?? '', /3 launched/)
  ok('launch_subagent and launch_subagents are approval-gated and dispatch through ToolContext')
}

{
  let readCalls = 0
  const largeSize = 9 * 1024 * 1024 * 1024
  const backend: WorkspaceBackend = {
    kind: 'ssh', rootCwd: '/srv/project',
    async readFile() { readCalls++; throw new Error('whole-file read must not happen') },
    async writeFile() {}, async readdir() { return [] },
    async stat() { return { isDirectory: false, isFile: true, size: largeSize, mtimeMs: 1 } },
    async mkdirp() {}, async rename() {}, async remove() {}, async exists() { return true },
    async assertContained() {}, async exec() { return { code: 0, stdout: '', stderr: '' } },
  }
  const result = await executeTool('save_artifact', { title: 'Large evidence', path: 'huge.log' }, {
    cwd: '/srv/project', backend, workspace: { kind: 'ssh', hostId: 'host-1', path: '/srv/project' },
    currentThread: { id: 'thread-artifact' } as any,
    registerArtifact: () => { throw new Error('remote path artifacts must not be registered') },
  })
  assert.equal(result.ok, false, result.output)
  assert.match(result.output, /SSH workspaces cannot be previewed or exported safely/i)
  assert.equal(readCalls, 0, 'registration never materializes artifact bytes')
  ok('save_artifact refuses SSH path records before stat/read so host preview cannot open a local lookalike')
}

{
  const cwd = await makeWorkspace()
  const result = await executeTool('glob', { pattern: '../*.ts' }, { cwd })
  assert.strictEqual(result.ok, false)
  assert.match(result.output, /escapes the workspace/)
  ok('glob rejects escaping patterns')
}

{
  const cwd = await makeWorkspace()
  await fs.writeFile(path.join(cwd, 'src/models.ts'), 'export const alpha = 1;\nexport const beta = 2;\n', 'utf8')
  assert.ok(READ_ONLY_TOOLS.has('ast_grep_search'))
  assert.ok(TOOL_SCHEMAS.some((tool: any) => tool.function?.name === 'ast_grep_search'))
  assert.deepStrictEqual(toolLabel('ast_grep_search', { pattern: 'export const $NAME = $VALUE' }), { verb: 'Structural search', target: 'export const $NAME = $VALUE' })
  const result = await executeTool('ast_grep_search', { pattern: 'export const $NAME = $VALUE', path: 'src' }, { cwd })
  assert.strictEqual(result.ok, true, result.output)
  assert.match(result.output, /code:src\/models\.ts:1/)
  assert.match(result.output, /NAME=alpha/)
  ok('ast_grep_search finds structural matches with source ids')
}

{
  const cwd = await makeWorkspace()
  const file = path.join(cwd, 'src/models.ts')
  await fs.writeFile(file, 'export const alpha = 1;\nexport const beta = 2;\n', 'utf8')
  assert.ok(!READ_ONLY_TOOLS.has('ast_grep_rewrite_preview'))
  assert.ok(TOOL_SCHEMAS.some((tool: any) => tool.function?.name === 'ast_grep_rewrite_preview'))
  assert.deepStrictEqual(toolLabel('ast_grep_rewrite_preview', { pattern: 'export const $NAME = $VALUE' }), { verb: 'Rewrite preview', target: 'export const $NAME = $VALUE' })
  const before = await fs.readFile(file, 'utf8')
  const result = await executeTool('ast_grep_rewrite_preview', {
    pattern: 'export const $NAME = $VALUE',
    rewrite: 'export let $NAME = $VALUE',
    path: 'src/models.ts',
  }, { cwd })
  assert.strictEqual(result.ok, true, result.output)
  assert.match(result.output, /proposed diff only/i)
  assert.match(result.meta ?? '', /not applied/)
  assert.ok(result.diff?.some((line) => line.type === 'add' && line.text.includes('export let alpha')))
  assert.strictEqual(await fs.readFile(file, 'utf8'), before)
  ok('ast_grep_rewrite_preview returns reviewable diffs without mutation')
}

{
  const cwd = await makeWorkspace()
  await fs.writeFile(path.join(cwd, 'src/dom.ts'), 'document.body.innerHTML = "<b>x</b>"\n', 'utf8')
  assert.ok(READ_ONLY_TOOLS.has('semgrep_scan'))
  assert.ok(TOOL_SCHEMAS.some((tool: any) => tool.function?.name === 'semgrep_scan'))
  assert.deepStrictEqual(toolLabel('semgrep_scan', { path: 'src' }), { verb: 'Semgrep scan', target: 'src' })
  const result = await executeTool('semgrep_scan', { path: 'src', autofix: true }, { cwd })
  assert.strictEqual(result.ok, true, result.output)
  assert.match(result.output, /js-inner-html-assignment/)
  assert.match(result.output, /proposed patch only/i)
  assert.ok(result.diff?.some((line) => line.type === 'add' && line.text.includes('textContent')))
  ok('semgrep_scan creates evidence findings and proposed autofix diffs only')
}

{
  const cwd = await makeWorkspace()
  assert.ok(READ_ONLY_TOOLS.has('codebase_map'))
  assert.ok(TOOL_SCHEMAS.some((tool: any) => tool.function?.name === 'codebase_map'))
  assert.deepStrictEqual(toolLabel('codebase_map', {}), { verb: 'Codebase map', target: 'workspace' })
  const result = await executeTool('codebase_map', {}, { cwd })
  assert.strictEqual(result.ok, true, result.output)
  assert.match(result.output, /Artifact: codebase-map/)
  assert.match(result.output, /## Dependency graph/)
  assert.match(result.meta ?? '', /files/)
  ok('codebase_map emits codebase and dependency artifacts')
}

{
  const cwd = await makeWorkspace()
  await fs.writeFile(path.join(cwd, 'src/models.ts'), [
    'export interface ModelConfig {',
    '  name: string',
    '}',
    'export function makeModel(config: ModelConfig) {',
    '  return config.name',
    '}',
    '',
  ].join('\n'), 'utf8')
  assert.ok(READ_ONLY_TOOLS.has('symbol_index'))
  assert.ok(TOOL_SCHEMAS.some((tool: any) => tool.function?.name === 'symbol_index'))
  assert.deepStrictEqual(toolLabel('symbol_index', { query: 'makeModel' }), { verb: 'Symbol index', target: 'makeModel' })
  const result = await executeTool('symbol_index', { path: 'src', query: 'makeModel' }, { cwd })
  assert.strictEqual(result.ok, true, result.output)
  assert.match(result.output, /Artifact: symbol-index/)
  assert.match(result.output, /Engine: typescript-language-service/)
  assert.match(result.output, /symbol:src\/models\.ts:4:makeModel/)
  ok('symbol_index exposes TypeScript language-service symbol exploration')
}

{
  const cwd = await makeWorkspace()
  await fs.writeFile(path.join(cwd, 'src/retrieval.ts'), 'export const approvalEvidence = "retrieval index source";\n', 'utf8')
  const email = await makeEmailSettings()
  const docs = await makeDocumentSettings()
  const slack = await makeSlackSettings()
  try {
    assert.ok(READ_ONLY_TOOLS.has('retrieval_search'))
    assert.ok(TOOL_SCHEMAS.some((tool: any) => tool.function?.name === 'retrieval_search'))
    assert.deepStrictEqual(toolLabel('retrieval_search', { query: 'approval evidence' }), { verb: 'Retrieval search', target: 'approval evidence' })
    const result = await executeTool('retrieval_search', {
      query: 'approval',
      kinds: ['code', 'doc', 'chat', 'email', 'summary', 'artifact'],
      limit: 12,
      max_code_files: 40,
      max_connector_items: 20,
    }, {
      cwd,
      emailSettings: email.settings,
      documentSettings: docs.settings,
      slackSettings: slack.settings,
      connectorEvents: [{
        id: 'evt-email-approval',
        connectorId: 'email',
        pluginName: 'Email',
        action: 'message_received',
        direction: 'inbound',
        classification: 'communication',
        priority: 'high',
        trustLevel: 'suggest',
        requiresApproval: false,
        approvalId: null,
        approvalStatus: null,
        resolvedAt: null,
        resolvedBy: null,
        title: 'Approval mail',
        summary: 'Email source says approval evidence is ready.',
        actor: 'Maya',
        threadKey: 'email:approval',
        payloadKeys: ['subject'],
        sources: [{ id: 'email:src:approval', connectorId: 'email', sourceId: 'src', itemId: 'approval', title: 'Approval mail', excerpt: 'approval evidence' }],
        createdAt: 10,
      }],
      threads: [{
        id: 'thread-approval',
        title: 'Approval chat',
        kind: 'task',
        cwd,
        providerId: null,
        model: null,
        createdAt: 1,
        updatedAt: 20,
        status: 'idle',
        apiMessages: [{ role: 'user', content: 'Find approval evidence across local sources.' }],
        timeline: [{ id: 'tl-approval', kind: 'assistant', text: 'Approval evidence is in email and docs.', createdAt: 20 }],
        compactions: [{ id: 'cmp-approval', at: 21, trigger: 'auto', messagesFolded: 3, tokensBefore: 1000, tokensAfter: 200, summaryExcerpt: 'Compacted approval evidence summary.', kindsDropped: ['assistant'] }],
      }],
      missions: [{
        id: 'mission-approval',
        title: 'Approval mission',
        cwd,
        checkpoints: [{ id: 'chk-approval', sequence: 1, label: 'Evidence', summary: 'Sandbox artifact captured approval evidence.', status: 'running', createdAt: 30, artifactRefs: ['sbox-approval'] }],
        scratchMemory: [],
      } as any],
      projectMemory: [{
        id: 'mem-approval',
        projectId: 'project-approval',
        kind: 'summary',
        text: 'Remember that approval evidence needs citations.',
        sourceType: 'connector',
        sourceRef: 'email:src:approval',
        sourceIds: ['email:src:approval'],
        confidence: 'high',
        privacyScope: 'project',
        owner: null,
        lastVerifiedAt: 30,
        createdAt: 30,
        updatedAt: 30,
      }],
      tickets: [{
        id: 'ticket-approval',
        title: 'Approval ticket',
        evidence: [{ id: 'ev-approval', kind: 'test', summary: 'Approval evidence test passed.', ref: 'artifact:test-log', createdAt: 40 }],
      } as any],
    })
    assert.strictEqual(result.ok, true, result.output)
    assert.match(result.output, /Retrieval index:/)
    assert.match(result.output, /\[code\]/)
    assert.match(result.output, /\[doc\]/)
    assert.match(result.output, /\[chat\]/)
    assert.match(result.output, /\[email\]/)
    assert.match(result.output, /\[summary\]/)
    assert.match(result.output, /\[artifact\]/)
    assert.match(result.output, /Sources:/)
    assert.match(result.output, /code:src\/retrieval\.ts/)
    assert.match(result.output, /email:src:approval/)
    assert.match(result.meta ?? '', /indexed/)
    ok('retrieval_search indexes code, docs, chats, emails, summaries, and artifacts with source ids')

    const cloudScoped = await executeTool('retrieval_search', {
      query: 'approval',
      kinds: ['code', 'doc', 'chat', 'email', 'summary', 'artifact'],
      limit: 12,
      max_code_files: 40,
      max_connector_items: 20,
    }, {
      cwd,
      emailSettings: email.settings,
      documentSettings: docs.settings,
      slackSettings: slack.settings,
      connectorEvents: [{
        id: 'evt-cloud-email',
        connectorId: 'email',
        pluginName: 'Email',
        action: 'message_received',
        direction: 'inbound',
        classification: 'communication',
        priority: 'high',
        trustLevel: 'suggest',
        requiresApproval: false,
        approvalId: null,
        approvalStatus: null,
        resolvedAt: null,
        resolvedBy: null,
        title: 'Cloud approval mail',
        summary: 'Cloud run should not receive connector approval details.',
        actor: 'Maya',
        threadKey: 'email:cloud',
        payloadKeys: ['subject'],
        sources: [{ id: 'email:src:cloud', connectorId: 'email', sourceId: 'src', itemId: 'cloud', title: 'Cloud approval mail', excerpt: 'approval evidence' }],
        createdAt: 10,
      }],
      sensitiveRetrievalAllowed: false,
      sensitiveRetrievalReason: 'cloud model route',
    })
    assert.strictEqual(cloudScoped.ok, true, cloudScoped.output)
    assert.match(cloudScoped.output, /Sensitive retrieval sources omitted/)
    assert.doesNotMatch(cloudScoped.output, /email:src:cloud/)
    assert.doesNotMatch(cloudScoped.output, /\[email\]/)
    assert.match(cloudScoped.output, /\[code\]/)
    ok('retrieval_search omits connector and memory sources when cloud context is not allowed')
  } finally {
    await fs.rm(email.root, { recursive: true, force: true })
    await fs.rm(docs.root, { recursive: true, force: true })
    await fs.rm(slack.root, { recursive: true, force: true })
  }
}

{
  const cwd = await makeWorkspace()
  const file = path.join(cwd, 'src/models.ts')
  await fs.writeFile(file, 'export const alpha = 1;\nexport const beta = 2;\n', 'utf8')
  assert.ok(!READ_ONLY_TOOLS.has('safe_refactor_preview'))
  assert.ok(TOOL_SCHEMAS.some((tool: any) => tool.function?.name === 'safe_refactor_preview'))
  assert.deepStrictEqual(toolLabel('safe_refactor_preview', { pattern: 'export const $NAME = $VALUE' }), { verb: 'Safe refactor', target: 'export const $NAME = $VALUE' })
  const before = await fs.readFile(file, 'utf8')
  const result = await executeTool('safe_refactor_preview', {
    pattern: 'export const $NAME = $VALUE',
    rewrite: 'export let $NAME = $VALUE',
    path: 'src/models.ts',
  }, { cwd })
  assert.strictEqual(result.ok, true, result.output)
  assert.match(result.output, /Workflow: safe mechanical refactor/)
  assert.match(result.output, /Apply the diff separately after approval/)
  assert.ok(result.diff?.some((line) => line.type === 'add' && line.text.includes('export let alpha')))
  assert.strictEqual(await fs.readFile(file, 'utf8'), before)
  ok('safe_refactor_preview chains search, diff preview, and scan without mutation')
}

{
  const cwd = await makeWorkspace()
  const result = await executeTool('apply_patch', {
    path: 'src/nested/util.ts',
    replacements: [{ old_string: 'export const value = 1', new_string: 'export const value = 2' }],
  }, { cwd })
  assert.strictEqual(result.ok, true, result.output)
  assert.match(await fs.readFile(path.join(cwd, 'src/nested/util.ts'), 'utf8'), /value = 2/)
  assert.ok(!READ_ONLY_TOOLS.has('apply_patch'))
  assert.ok(TOOL_SCHEMAS.some((tool: any) => tool.function?.name === 'apply_patch'))
  assert.deepStrictEqual(toolLabel('apply_patch', { path: 'src/nested/util.ts' }), { verb: 'Patch', target: 'src/nested/util.ts' })
  ok('apply_patch applies explicit search/replace hunks')
}

{
  const cwd = await makeWorkspace()
  const patch = `*** Begin Patch
*** Update File: src/nested/util.ts
@@
-export const value = 1
+export const value = 3
*** Add File: src/new.ts
+export const created = true
*** End Patch`
  const result = await executeTool('apply_patch', { patch }, { cwd })
  assert.strictEqual(result.ok, true, result.output)
  assert.match(await fs.readFile(path.join(cwd, 'src/nested/util.ts'), 'utf8'), /value = 3/)
  assert.match(await fs.readFile(path.join(cwd, 'src/new.ts'), 'utf8'), /created = true/)
  assert.match(result.meta ?? '', /2 files/)
  ok('apply_patch applies *** Begin Patch update and add operations')
}

{
  const cwd = await makeWorkspace()
  const patch = `*** Begin Patch
*** Update File: src/nested/util.ts
@@
-export const value = 1
*** End Patch`
  const result = await executeTool('apply_patch', { patch }, { cwd })
  assert.strictEqual(result.ok, true, result.output)
  assert.strictEqual(await fs.readFile(path.join(cwd, 'src/nested/util.ts'), 'utf8'), '')
  ok('apply_patch deletes whole lines without leaving blanks')
}

{
  const cwd = await makeWorkspace()
  const patch = `*** Begin Patch
*** Add File: ../outside.txt
+nope
*** End Patch`
  const result = await executeTool('apply_patch', { patch }, { cwd })
  assert.strictEqual(result.ok, false)
  assert.match(result.output, /escapes the workspace/)
  ok('apply_patch rejects escaping file paths')
}

{
  const cwd = await makeWorkspace()
  const outside = await fs.mkdtemp(path.join(os.tmpdir(), 'vontra-outside-'))
  const link = path.join(cwd, 'linked-outside')
  try {
    await fs.symlink(outside, link, 'dir')
    const result = await executeTool('read_file', { path: 'linked-outside/secret.txt' }, { cwd })
    assert.strictEqual(result.ok, false)
    assert.match(result.output, /escapes the workspace/)
    ok('read_file rejects symlink escapes')
  } catch (e: any) {
    if (existsSync(link)) throw e
    ok('symlink escape test skipped where symlinks are unavailable')
  }
}

{
  const cwd = await makeWorkspace()
  const result = await executeTool('run_command', { command: 'cat ~/.ssh/id_rsa' }, { cwd })
  assert.strictEqual(result.ok, false)
  assert.match(result.output, /Blocked by TensorFold(?: Studio)? safety policy/)
  ok('run_command blocks direct credential reads')
}

{
  const cwd = await makeWorkspace()
  const result = await executeTool('run_command', { command: 'printf ok' }, { cwd })
  assert.match(result.meta ?? '', /sandbox=(build-test|unsandboxed)/)
  assert.match(result.meta ?? '', /(macos-sandbox-exec|unsandboxed)/)
  assert.match(result.meta ?? '', /artifacts=sbox-/)
  assert.doesNotMatch(result.output, /Sandbox artifacts/)
  assert.equal(result.output, 'ok')
  assert.equal(result.artifacts?.id, result.sandboxArtifacts?.id)
  assert.match(result.artifacts?.id ?? '', /^sbox-/)
  ok('run_command keeps sandbox metadata out of the model-visible output')
}

{
  // Real build/test execution regression: a hostile repository script can print credentials even
  // though the command itself (`npm test`) looks safe enough for Full Auto. Value scanning must
  // happen after process execution and before either the model-facing output or artifact log exists.
  const cwd = await makeWorkspace()
  const vendorSecret = `ghp_${'A'.repeat(36)}`
  const opaqueSecret = 'p9Ks_4LzN7qR2vX8mC5dT1wB6hJ3sF0a'
  const ordinaryHash = '0123456789abcdef'.repeat(4)
  await fs.writeFile(path.join(cwd, 'package.json'), JSON.stringify({
    name: 'malicious-output-fixture',
    scripts: { test: 'node leak.cjs' },
  }))
  await fs.writeFile(path.join(cwd, 'leak.cjs'), [
    `console.log(${JSON.stringify(`vendor ${vendorSecret}`)})`,
    `console.error(${JSON.stringify(`opaque ${opaqueSecret}`)})`,
    `console.log(${JSON.stringify(`ordinary log checksum ${ordinaryHash}`)})`,
  ].join('\n'))
  // Keep this fixture independent of whichever version manager installed npm on the host: the
  // workspace-local package-runner shim performs a real child-process test execution with the
  // same Full Auto `npm test` command classification.
  const fixtureBin = path.join(cwd, 'fixture-bin')
  await fs.mkdir(fixtureBin)
  await fs.writeFile(path.join(fixtureBin, 'npm'), `#!/bin/sh\nexec ${JSON.stringify(process.execPath)} leak.cjs\n`)
  await fs.chmod(path.join(fixtureBin, 'npm'), 0o755)
  const previousPath = process.env.PATH
  process.env.PATH = `${fixtureBin}:/usr/bin:/bin`
  let result: Awaited<ReturnType<typeof executeTool>>
  try {
    result = await executeTool('run_command', { command: 'npm test --silent' }, { cwd })
  } finally {
    if (previousPath === undefined) delete process.env.PATH
    else process.env.PATH = previousPath
  }
  assert.strictEqual(result.ok, true, result.output)
  assert.doesNotMatch(result.output, new RegExp(vendorSecret))
  assert.doesNotMatch(result.output, new RegExp(opaqueSecret))
  assert.match(result.output, /TensorFold(?: Studio)? redacted 2 sensitive values/)
  assert.match(result.output, new RegExp(ordinaryHash), 'ordinary SHA-256-like hashes remain readable')
  assert.match(result.meta ?? '', /redacted=2/)
  assert.doesNotMatch(JSON.stringify(result.sandboxArtifacts?.logs ?? []), new RegExp(vendorSecret))
  assert.doesNotMatch(JSON.stringify(result.sandboxArtifacts?.logs ?? []), new RegExp(opaqueSecret))
  ok('malicious npm test output is value-redacted while ordinary logs and hashes survive')
}

{
  const cwd = await makeWorkspace()
  const result = await executeTool('run_command', { command: 'mkdir -p .vontra-artifacts && printf patchable > .vontra-artifacts/result.txt' }, { cwd })
  assert.strictEqual(result.ok, true)
  assert.strictEqual(result.artifacts?.patch, true)
  assert.ok(result.artifacts?.diffs.some((entry) => entry.path === '.vontra-artifacts/result.txt' && entry.status === 'added'))
  assert.match(result.sandboxArtifacts?.patch ?? '', /diff --git a\/\.vontra-artifacts\/result.txt b\/\.vontra-artifacts\/result.txt/)
  assert.match(result.sandboxArtifacts?.patch ?? '', /\+patchable/)
  ok('run_command returns sandbox patch evidence to the caller')
}

{
  const cwd = await makeWorkspace()
  await fs.writeFile(path.join(cwd, '.env'), 'SECRET=shh\n', 'utf8')
  const result = await executeTool('read_file', { path: '.env' }, { cwd })
  assert.strictEqual(result.ok, false)
  assert.match(result.output, /Blocked by TensorFold(?: Studio)? safety policy/)
  ok('read_file blocks .env secret files')
}

{
  assert.ok(READ_ONLY_TOOLS.has('web_search'))
  assert.ok(READ_ONLY_TOOLS.has('web_fetch'))
  assert.ok(TOOL_SCHEMAS.some((tool: any) => tool.function?.name === 'web_search'))
  assert.ok(TOOL_SCHEMAS.some((tool: any) => tool.function?.name === 'web_fetch'))
  assert.ok(CHAT_TOOL_SCHEMAS.some((tool: any) => tool.function?.name === 'web_search'))
  assert.ok(CHAT_TOOL_SCHEMAS.some((tool: any) => tool.function?.name === 'web_fetch'))
  assert.deepStrictEqual(toolLabel('web_search', { query: 'TensorFold' }), { verb: 'Web search', target: 'TensorFold' })
  assert.deepStrictEqual(toolLabel('web_fetch', { url: 'https://example.com' }), { verb: 'Web fetch', target: 'https://example.com' })
  ok('chat exposes public web tools as read-only tools')
}

{
  const originalFetch = globalThis.fetch
  const restoreDns = installDnsFixture({ 'news.google.com': '8.8.8.8' })
  try {
    globalThis.fetch = async () => new Response(`<?xml version="1.0" encoding="UTF-8"?>
      <rss version="2.0"><channel>
        <item>
          <title>Example &amp; Result</title>
          <link>https://example.com/one</link>
          <source url="https://example.com">Example News</source>
          <pubDate>Mon, 22 Jun 2026 13:00:00 GMT</pubDate>
        </item>
      </channel></rss>
    `, { status: 200, headers: { 'content-type': 'application/xml' } })
    const cwd = await makeWorkspace()
    const result = await executeTool('web_search', { query: 'example', limit: 3 }, { cwd })
    assert.strictEqual(result.ok, true, result.output)
    assert.match(result.output, /Example & Result/)
    assert.match(result.output, /https:\/\/example.com\/one/)
    assert.match(result.output, /Example News/)
    ok('web_search parses news results')
  } finally {
    globalThis.fetch = originalFetch
    restoreDns()
  }
}

{
  const cwd = await makeWorkspace()
  const result = await executeTool('web_fetch', { url: 'http://127.0.0.1:1234/secret' }, { cwd })
  assert.strictEqual(result.ok, false)
  assert.match(result.output, /private|metadata|local/i)
  ok('web_fetch rejects local/private hosts')
}

{
  const cwd = await makeWorkspace()
  const { root, settings } = await makeEmailSettings()
  try {
    const summary = await executeTool('email_summary', { limit: 5 }, { cwd, emailSettings: settings })
    assert.strictEqual(summary.ok, true, summary.output)
    assert.match(summary.output, /Sources:/)
    assert.match(summary.output, /email:src:/)
    assert.equal(summary.sources?.length, 1)
    assert.match(summary.meta ?? '', /1 sources/)

    const list = await executeTool('email_list', { limit: 5 }, { cwd, emailSettings: settings })
    assert.strictEqual(list.ok, true, list.output)
    assert.match(list.output, /\[email:src:/)
    assert.equal(list.sources?.[0]?.connectorId, 'email')

    const itemId = summary.sources?.[0]?.itemId
    assert.ok(itemId)
    const read = await executeTool('email_read', { id: itemId }, { cwd, emailSettings: settings })
    assert.strictEqual(read.ok, true, read.output)
    assert.match(read.output, /Source ID: email:src:/)
    assert.equal(read.sources?.[0]?.id, summary.sources?.[0]?.id)

    let draftSources = 0
    const compose = await executeTool('email_compose', {
      mode: 'send',
      to: 'maya@example.test',
      subject: 'Re: Renewal question',
      body: 'Thanks, I will confirm pricing today.',
      inReplyTo: itemId,
    }, {
      cwd,
      emailSettings: settings,
      recordEmailDraft: async (input, sources = []) => {
        draftSources = sources.length
        return emailDraftEvent(input, 123, sources)
      },
    })
    assert.strictEqual(compose.ok, true, compose.output)
    assert.match(compose.output, /reply_requested/)
    assert.match(compose.output, /waiting for approval/)
    assert.equal(compose.sources?.[0]?.connectorId, 'email')
    assert.equal(draftSources, 1)

    let resolvedAction = ''
    const resolved = await executeTool('email_resolve', { id: itemId, note: 'Reply drafted for review.' }, {
      cwd,
      emailSettings: settings,
      recordEmailEvent: async (event) => {
        resolvedAction = event.action
        return event
      },
    })
    assert.strictEqual(resolved.ok, true, resolved.output)
    assert.equal(resolvedAction, 'thread_resolved')
    assert.match(resolved.output, /Sources:/)
    assert.equal(resolved.sources?.[0]?.connectorId, 'email')
    assert.deepStrictEqual(toolLabel('email_resolve', { id: itemId }), { verb: 'Email resolve', target: itemId })
    assert.ok(TOOL_SCHEMAS.some((tool: any) => tool.function?.name === 'email_resolve'))
    ok('email tools expose source ids and source-backed thread actions')
  } finally {
    await fs.rm(root, { recursive: true, force: true })
  }
}

{
  const cwd = await makeWorkspace()
  const { root, settings } = await makeSlackSettings()
  try {
    const summary = await executeTool('message_summary', { limit: 5 }, { cwd, slackSettings: settings })
    assert.strictEqual(summary.ok, true, summary.output)
    assert.match(summary.output, /Sources:/)
    assert.match(summary.output, /slack:slack:/)
    assert.match(summary.output, /#launch-room/)
    assert.equal(summary.sources?.length, 2)

    const list = await executeTool('message_list', { query: 'drift', limit: 5 }, { cwd, slackSettings: settings })
    assert.strictEqual(list.ok, true, list.output)
    assert.match(list.output, /\[slack:slack:/)
    assert.match(list.output, /Staging data drift/)
    assert.equal(list.sources?.[0]?.connectorId, 'slack')

    const itemId = list.sources?.[0]?.itemId
    assert.ok(itemId)
    const read = await executeTool('message_read', { id: itemId }, { cwd, slackSettings: settings })
    assert.strictEqual(read.ok, true, read.output)
    assert.match(read.output, /Source ID: slack:slack:/)
    assert.match(read.output, /Staging data drift is blocking QA/)
    assert.equal(read.sources?.[0]?.id, list.sources?.[0]?.id)

    let messageEvent: any = null
    const compose = await executeTool('message_compose', { connectorId: 'telegram', target: '42', body: 'Approval-gated status update.' }, {
      cwd,
      recordMessageEvent: async (event) => {
        messageEvent = event
        return event
      },
    })
    assert.strictEqual(compose.ok, true, compose.output)
    assert.match(compose.output, /waiting for approval/)
    assert.equal(messageEvent?.connectorId, 'telegram')
    assert.equal(messageEvent?.action, 'message_send_requested')
    assert.equal(messageEvent?.requiresApproval, true)
    assert.equal(messageEvent?.approvalStatus, 'pending')
    assert.equal(messageEvent?.delivery, undefined)

    assert.ok(READ_ONLY_TOOLS.has('message_list'))
    assert.ok(READ_ONLY_TOOLS.has('message_read'))
    assert.ok(READ_ONLY_TOOLS.has('message_summary'))
    assert.ok(!READ_ONLY_TOOLS.has('message_compose'))
    assert.ok(TOOL_SCHEMAS.some((tool: any) => tool.function?.name === 'message_list'))
    assert.ok(TOOL_SCHEMAS.some((tool: any) => tool.function?.name === 'message_read'))
    assert.ok(TOOL_SCHEMAS.some((tool: any) => tool.function?.name === 'message_summary'))
    assert.ok(TOOL_SCHEMAS.some((tool: any) => tool.function?.name === 'message_compose'))
    assert.deepStrictEqual(toolLabel('message_summary', { query: 'drift' }), { verb: 'Message summary', target: 'drift' })
    assert.deepStrictEqual(toolLabel('message_compose', { connectorId: 'telegram', target: '42' }), { verb: 'Message send', target: 'telegram:42' })
    ok('Slack message tools list, read, and summarize local export messages with source ids')
  } finally {
    await fs.rm(root, { recursive: true, force: true })
  }
}

{
  const cwd = await makeWorkspace()
  const { root, settings } = await makeCalendarSettings()
  try {
    const summary = await executeTool('calendar_summary', { day: '2026-06-20', limit: 5 }, { cwd, calendarSettings: settings })
    assert.strictEqual(summary.ok, true, summary.output)
    assert.match(summary.output, /Sources:/)
    assert.match(summary.output, /calendar:cal:/)
    assert.equal(summary.sources?.length, 1)
    assert.match(summary.meta ?? '', /1 sources/)

    const list = await executeTool('calendar_list', { day: '2026-06-20', limit: 5 }, { cwd, calendarSettings: settings })
    assert.strictEqual(list.ok, true, list.output)
    assert.match(list.output, /\[calendar:cal:/)
    assert.equal(list.sources?.[0]?.connectorId, 'calendar')

    const itemId = summary.sources?.[0]?.itemId
    assert.ok(itemId)
    const read = await executeTool('calendar_read', { id: itemId }, { cwd, calendarSettings: settings })
    assert.strictEqual(read.ok, true, read.output)
    assert.match(read.output, /Source ID: calendar:cal:/)
    assert.equal(read.sources?.[0]?.id, summary.sources?.[0]?.id)

    let createdAction = ''
    const create = await executeTool('calendar_create', {
      title: 'Planning follow-up',
      startAt: '2026-06-20T12:00:00Z',
      endAt: '2026-06-20T12:30:00Z',
      location: 'Project room',
      attendees: ['maya@example.test'],
    }, {
      cwd,
      calendarSettings: settings,
      recordCalendarEvent: async (event) => {
        createdAction = event.action
        return event
      },
    })
    assert.strictEqual(create.ok, true, create.output)
    assert.equal(createdAction, 'calendar_create_requested')
    assert.match(create.output, /waiting for approval/)
    assert.match(create.meta ?? '', /approval/)

    let updatedAction = ''
    const update = await executeTool('calendar_update', {
      id: itemId,
      title: 'Planning review moved',
      startAt: '2026-06-20T11:00:00Z',
    }, {
      cwd,
      calendarSettings: settings,
      recordCalendarEvent: async (event) => {
        updatedAction = event.action
        return event
      },
    })
    assert.strictEqual(update.ok, true, update.output)
    assert.equal(updatedAction, 'calendar_update_requested')
    assert.match(update.output, /waiting for approval/)
    assert.equal(update.sources?.[0]?.id, summary.sources?.[0]?.id)

    assert.ok(READ_ONLY_TOOLS.has('calendar_list'))
    assert.ok(READ_ONLY_TOOLS.has('calendar_read'))
    assert.ok(READ_ONLY_TOOLS.has('calendar_summary'))
    assert.ok(!READ_ONLY_TOOLS.has('calendar_create'))
    assert.ok(!READ_ONLY_TOOLS.has('calendar_update'))
    assert.ok(TOOL_SCHEMAS.some((tool: any) => tool.function?.name === 'calendar_list'))
    assert.ok(TOOL_SCHEMAS.some((tool: any) => tool.function?.name === 'calendar_read'))
    assert.ok(TOOL_SCHEMAS.some((tool: any) => tool.function?.name === 'calendar_summary'))
    assert.ok(TOOL_SCHEMAS.some((tool: any) => tool.function?.name === 'calendar_create'))
    assert.ok(TOOL_SCHEMAS.some((tool: any) => tool.function?.name === 'calendar_update'))
    assert.deepStrictEqual(toolLabel('calendar_summary', { day: '2026-06-20' }), { verb: 'Calendar summary', target: '2026-06-20' })
    assert.deepStrictEqual(toolLabel('calendar_create', { title: 'Planning follow-up' }), { verb: 'Calendar create', target: 'Planning follow-up' })
    assert.deepStrictEqual(toolLabel('calendar_update', { id: itemId }), { verb: 'Calendar update', target: itemId })
    ok('calendar tools list, read, summarize, and request approval-gated mutations')
  } finally {
    await fs.rm(root, { recursive: true, force: true })
  }
}

{
  const cwd = await makeWorkspace()
  const { root, settings } = await makeDocumentSettings()
  try {
    const summary = await executeTool('document_summary', { limit: 5 }, { cwd, documentSettings: settings })
    assert.strictEqual(summary.ok, true, summary.output)
    assert.match(summary.output, /Sources:/)
    assert.match(summary.output, /docs:docs:/)
    assert.equal(summary.sources?.length, 5)
    assert.match(summary.meta ?? '', /5 sources/)

    const list = await executeTool('document_list', { query: 'Reviewer', limit: 5 }, { cwd, documentSettings: settings })
    assert.strictEqual(list.ok, true, list.output)
    assert.match(list.output, /\[docs:docs:/)
    assert.equal(list.sources?.[0]?.connectorId, 'docs')

    const itemId = list.sources?.[0]?.itemId
    assert.ok(itemId)
    const read = await executeTool('document_read', { id: itemId }, { cwd, documentSettings: settings })
    assert.strictEqual(read.ok, true, read.output)
    assert.match(read.output, /Source ID: docs:docs:/)
    assert.match(read.output, /final sign-off/)
    assert.equal(read.sources?.[0]?.id, list.sources?.[0]?.id)

    const pdfList = await executeTool('document_list', { query: 'Renewal', extension: 'pdf', limit: 5 }, { cwd, documentSettings: settings })
    assert.strictEqual(pdfList.ok, true, pdfList.output)
    assert.match(pdfList.output, /review\.pdf/)
    assert.equal(pdfList.sources?.length, 1)

    const pdfRead = await executeTool('document_read', { id: pdfList.sources?.[0]?.id }, { cwd, documentSettings: settings })
    assert.strictEqual(pdfRead.ok, true, pdfRead.output)
    assert.match(pdfRead.output, /source-backed approval/)
    assert.equal(pdfRead.sources?.[0]?.id, pdfList.sources?.[0]?.id)

    const tableSummary = await executeTool('table_summary', { limit: 5 }, { cwd, documentSettings: settings })
    assert.strictEqual(tableSummary.ok, true, tableSummary.output)
    assert.match(tableSummary.output, /metrics\.csv/)
    assert.match(tableSummary.output, /budget\.xlsx/)
    assert.match(tableSummary.output, /Revenue: min 700, max 1200, avg 950/)
    assert.match(tableSummary.output, /Budget: min 800, max 1000, avg 900/)
    assert.match(tableSummary.output, /Sources:/)
    assert.equal(tableSummary.sources?.length, 2)

    const tableList = await executeTool('table_list', { query: 'Revenue', limit: 5 }, { cwd, documentSettings: settings })
    assert.strictEqual(tableList.ok, true, tableList.output)
    assert.match(tableList.output, /\[docs:docs:/)
    assert.match(tableList.output, /3 rows x 4 columns/)

    const tableRead = await executeTool('table_read', { id: tableList.sources?.[0]?.id, limit: 2 }, { cwd, documentSettings: settings })
    assert.strictEqual(tableRead.ok, true, tableRead.output)
    assert.match(tableRead.output, /Source ID: docs:docs:/)
    assert.match(tableRead.output, /\| Team \| Revenue \| Cost \| Status \|/)
    assert.match(tableRead.output, /\| Beta \| 950 \| 500 \| yellow \|/)
    assert.equal(tableRead.sources?.[0]?.id, tableList.sources?.[0]?.id)

    const xlsxList = await executeTool('table_list', { extension: 'xlsx', limit: 5 }, { cwd, documentSettings: settings })
    assert.strictEqual(xlsxList.ok, true, xlsxList.output)
    assert.match(xlsxList.output, /budget\.xlsx/)
    assert.equal(xlsxList.sources?.length, 1)

    const xlsxRead = await executeTool('table_read', { id: xlsxList.sources?.[0]?.id, limit: 5 }, { cwd, documentSettings: settings })
    assert.strictEqual(xlsxRead.ok, true, xlsxRead.output)
    assert.match(xlsxRead.output, /\| Team \| Budget \| Actual \|/)
    assert.match(xlsxRead.output, /\| Beta \| 800 \| 850 \|/)

    assert.ok(READ_ONLY_TOOLS.has('document_list'))
    assert.ok(READ_ONLY_TOOLS.has('document_read'))
    assert.ok(READ_ONLY_TOOLS.has('document_summary'))
    assert.ok(READ_ONLY_TOOLS.has('table_list'))
    assert.ok(READ_ONLY_TOOLS.has('table_read'))
    assert.ok(READ_ONLY_TOOLS.has('table_summary'))
    assert.ok(TOOL_SCHEMAS.some((tool: any) => tool.function?.name === 'document_list'))
    assert.ok(TOOL_SCHEMAS.some((tool: any) => tool.function?.name === 'document_read'))
    assert.ok(TOOL_SCHEMAS.some((tool: any) => tool.function?.name === 'document_summary'))
    assert.ok(TOOL_SCHEMAS.some((tool: any) => tool.function?.name === 'table_list'))
    assert.ok(TOOL_SCHEMAS.some((tool: any) => tool.function?.name === 'table_read'))
    assert.ok(TOOL_SCHEMAS.some((tool: any) => tool.function?.name === 'table_summary'))
    assert.match(String(TOOL_SCHEMAS.find((tool: any) => tool.function?.name === 'document_list')?.function?.description), /PDF/)
    assert.match(String(TOOL_SCHEMAS.find((tool: any) => tool.function?.name === 'document_read')?.function?.description), /PDF/)
    assert.match(String(TOOL_SCHEMAS.find((tool: any) => tool.function?.name === 'document_summary')?.function?.description), /PDF/)
    assert.deepStrictEqual(toolLabel('document_summary', { query: 'Reviewer' }), { verb: 'Document summary', target: 'Reviewer' })
    assert.deepStrictEqual(toolLabel('table_summary', { query: 'Revenue' }), { verb: 'Table summary', target: 'Revenue' })
    ok('document and table tools list, read, and summarize local documents, PDFs, and tables with source ids')
  } finally {
    await fs.rm(root, { recursive: true, force: true })
  }
}

{
  const cwd = await makeWorkspace()
  const { root, settings } = await makeBrowserSettings()
  try {
    const summary = await executeTool('browser_summary', { query: 'auth', limit: 5 }, { cwd, browserSettings: settings })
    assert.strictEqual(summary.ok, true, summary.output)
    assert.match(summary.output, /Sources:/)
    assert.match(summary.output, /browser:/)
    assert.equal(summary.sources?.length, 2)

    const list = await executeTool('browser_list', { query: 'auth', limit: 5 }, { cwd, browserSettings: settings })
    assert.strictEqual(list.ok, true, list.output)
    assert.match(list.output, /\[browser:/)
    assert.equal(list.sources?.[0]?.connectorId, 'browser')

    const itemId = list.sources?.[0]?.itemId
    assert.ok(itemId)
    const read = await executeTool('browser_read', { id: itemId }, { cwd, browserSettings: settings })
    assert.strictEqual(read.ok, true, read.output)
    assert.match(read.output, /Source ID: browser:/)
    assert.match(read.output, /Auth launch checklist|Release notes mention auth/)
    assert.equal(read.sources?.[0]?.id, list.sources?.[0]?.id)

    assert.ok(READ_ONLY_TOOLS.has('browser_list'))
    assert.ok(READ_ONLY_TOOLS.has('browser_read'))
    assert.ok(READ_ONLY_TOOLS.has('browser_summary'))
    assert.ok(!READ_ONLY_TOOLS.has('browser_automation'))
    assert.ok(TOOL_SCHEMAS.some((tool: any) => tool.function?.name === 'browser_list'))
    assert.ok(TOOL_SCHEMAS.some((tool: any) => tool.function?.name === 'browser_read'))
    assert.ok(TOOL_SCHEMAS.some((tool: any) => tool.function?.name === 'browser_summary'))
    const automationSchema = TOOL_SCHEMAS.find((tool: any) => tool.function?.name === 'browser_automation') as any
    assert.ok(automationSchema)
    assert.deepEqual(automationSchema.function.parameters.properties.action.enum, BROWSER_AUTOMATION_ACTIONS)
    assert.deepStrictEqual(toolLabel('browser_summary', { query: 'auth' }), { verb: 'Browser summary', target: 'auth' })
    assert.deepStrictEqual(toolLabel('browser_automation', { action: 'navigate', url: 'http://localhost:5173/' }), { verb: 'Browser navigate', target: 'http://localhost:5173/' })

    let automationInput: any = null
    const automation = await executeTool('browser_automation', {
      action: 'navigate',
      url: 'http://localhost:5173/',
      includeSnapshot: true,
      captureScreenshot: true,
    }, {
      cwd,
      runBrowserAutomation: async (input) => {
        automationInput = input
        return {
          ok: true,
          action: input.action,
          sessionId: input.sessionId ?? 'default',
          tabId: 'tab-1',
          url: input.url,
          title: 'TensorFold local preview',
          loadState: 'loaded',
          accessibilitySnapshot: '- heading "TensorFold" [ref=e1]\n- button "Save" [ref=e2]',
          screenshot: { path: path.join(cwd, '.vontra-browser', 'tab-1.png'), title: 'Initial render' },
          consoleMessages: [{ level: 'warning', text: 'Demo warning', at: 1 }],
          networkErrors: [{ url: 'http://localhost:5173/api', method: 'GET', status: 404, errorText: 'Not Found', at: 2 }],
        }
      },
    })
    assert.strictEqual(automation.ok, true, automation.output)
    assert.equal(automationInput?.action, 'navigate')
    assert.equal(automationInput?.includeSnapshot, true)
    assert.match(automation.output, /Accessibility snapshot:/)
    assert.match(automation.output, /Console messages:/)
    assert.match(automation.output, /Network errors:/)
    assert.match(automation.path ?? '', /\.vontra-browser\/tab-1\.png$/)
    assert.match(automation.meta ?? '', /session=default/)
    ok('browser tools list, read, and summarize explicit sources with source ids')
  } finally {
    await fs.rm(root, { recursive: true, force: true })
  }
}

{
  const cwd = await makeWorkspace()
  const controllerSchema = TOOL_SCHEMAS.find((tool: any) => tool.function?.name === 'controller_action') as any
  assert.ok(controllerSchema)
  assert.deepEqual(controllerSchema.function.parameters.properties.action.enum, CONTROLLER_ACTIONS)
  assert.ok(!READ_ONLY_TOOLS.has('controller_action'))
  assert.deepStrictEqual(toolLabel('controller_action', { action: 'menu_item', appName: 'Safari', menuPath: ['File', 'New Tab'] }), { verb: 'Controller menu_item', target: 'Safari File > New Tab' })

  let controllerInput: any = null
  const result = await executeTool('controller_action', {
    action: 'keystroke',
    appName: 'Terminal',
    keystroke: 'l',
    modifiers: ['command'],
  }, {
    cwd,
    runControllerAction: async (input) => {
      controllerInput = input
      return {
        ok: true,
        action: input.action,
        appName: input.appName,
        target: 'Command+L',
        executedAt: Date.UTC(2026, 5, 20, 12, 0, 0),
        note: 'Keystroke delivered.',
      }
    },
  })
  assert.strictEqual(result.ok, true, result.output)
  assert.equal(controllerInput?.action, 'keystroke')
  assert.match(result.output, /Approval: controller_action tool approval/)
  assert.match(result.meta ?? '', /Terminal/)
  ok('controller_action is approval-gated and dispatches through ToolContext')
}

{
  // SSH-kind backend routing (in-memory fake) — proves file/shell/search tools
  // execute against the backend (i.e. "remotely") while policy + path containment
  // still hold.
  const files = new Map<string, string>()
  const execLog: string[] = []
  const containmentLog: string[] = []
  let remoteSnapshotCount = 0
  const sshBackend: WorkspaceBackend = {
    kind: 'ssh',
    rootCwd: '/srv/app',
    async readFile(p) { const v = files.get(p); if (v == null) throw new Error('ENOENT ' + p); return v },
    async writeFile(p, c) { files.set(p, c) },
    async readdir(p): Promise<WorkspaceFileEntry[]> {
      const prefix = p.endsWith('/') ? p : p + '/'
      const names = new Set<string>()
      for (const k of files.keys()) if (k.startsWith(prefix)) names.add(k.slice(prefix.length).split('/')[0])
      return [...names].map((name) => ({ name, isDirectory: false, isFile: true, size: files.get(prefix + name)?.length ?? 0 }))
    },
    async stat(p): Promise<WorkspaceStat> {
      const v = files.get(p); if (v == null) throw new Error('ENOENT')
      return { isDirectory: false, isFile: true, size: v.length, mtimeMs: 0 }
    },
    async mkdirp() { /* no-op */ },
    async rename(from, to) { const v = files.get(from); if (v != null) { files.set(to, v); files.delete(from) } },
    async remove(p) { files.delete(p) },
    async exists(p) { return files.has(p) },
    async assertContained(p) {
      containmentLog.push(p)
      if (p.endsWith('/linked-outside')) throw new Error(`Path "${p}" escapes the workspace through a symlink.`)
    },
    async exec(command) {
      execLog.push(command)
      if (command.includes('vontra-command-index-')) {
        remoteSnapshotCount++
        const tree = (remoteSnapshotCount % 2 ? 'a' : 'b').repeat(40)
        return { code: 0, stdout: `/srv/app\0\0${tree}\0`, stderr: '' }
      }
      if (command.startsWith('git diff --name-only -z --no-renames')) {
        return { code: 0, stdout: 'src/remote-mutated.ts\0', stderr: '' }
      }
      if (command.startsWith('grep')) return {
        code: 0,
        stdout: [
          './src/index.ts:1: hello',
          './config/service-account.json:1: {"private_key":"hello"}',
          './deploy/secrets.yaml:1: token: hello',
          './keys/private.pem:1: hello',
        ].join('\n') + '\n',
        stderr: '',
      }
      if (command === 'npm test') return {
        code: 0,
        stdout: `remote-ok ghp_${'B'.repeat(36)} checksum ${'abcdef0123456789'.repeat(4)}`,
        stderr: 'token=remote-secret-value',
      }
      return { code: 0, stdout: 'remote-ok', stderr: '' }
    },
  }
  const ctx = { cwd: '/srv/app', backend: sshBackend }

  const w = await executeTool('write_file', { path: 'src/index.ts', content: 'hello' }, ctx)
  assert.strictEqual(w.ok, true, w.output)
  assert.ok(files.has('/srv/app/src/index.ts'))

  const r = await executeTool('read_file', { path: 'src/index.ts' }, ctx)
  assert.match(r.output, /hello/)

  const ls = await executeTool('list_dir', { path: 'src' }, ctx)
  assert.match(ls.output, /index\.ts/)

  const run = await executeTool('run_command', { command: 'echo hi' }, ctx)
  assert.strictEqual(run.ok, true)
  assert.match(run.meta ?? '', /ssh/)
  assert.ok(execLog.includes('echo hi'))

  const remoteLeak = await executeTool('run_command', { command: 'npm test' }, ctx)
  assert.strictEqual(remoteLeak.ok, true)
  assert.doesNotMatch(remoteLeak.output, /ghp_B{36}|remote-secret-value/)
  assert.match(remoteLeak.output, /abcdef0123456789abcdef0123456789/, 'remote hashes remain readable')
  assert.match(remoteLeak.output, /TensorFold(?: Studio)? redacted 2 sensitive values/)
  assert.match(remoteLeak.meta ?? '', /ssh.*redacted=2/)
  assert.deepEqual(remoteLeak.changedPaths, ['src/remote-mutated.ts'], 'remote shell source edits are retained as checkpoint ownership evidence')

  const blockedRun = await executeTool('run_command', { command: 'rm -rf /' }, ctx)
  assert.strictEqual(blockedRun.ok, false) // command policy still applies on remote

  const grep = await executeTool('search', { query: 'hello' }, ctx)
  assert.match(grep.output, /index\.ts/)
  assert.doesNotMatch(grep.output, /service-account\.json|secrets\.yaml|private\.pem|private_key|token:/)
  assert.ok(execLog.some((c) => c.startsWith('grep')))
  assert.ok(containmentLog.includes('/srv/app'), 'remote search proves its shell cwd is canonically contained')

  const protectedWrite = await executeTool('write_file', { path: '.git/hooks/pre-commit', content: 'echo planted' }, ctx)
  const protectedEdit = await executeTool('edit_file', { path: '.git/config', old_string: 'x', new_string: 'y' }, ctx)
  assert.equal(protectedWrite.ok, false)
  assert.equal(protectedEdit.ok, false)
  assert.match(protectedWrite.output, /Git private metadata/i)
  assert.equal(files.has('/srv/app/.git/hooks/pre-commit'), false)

  const beforeEscapeCommands = execLog.length
  const escapedSearch = await executeTool('search', { query: 'secret', path: 'linked-outside' }, ctx)
  const escapedGlob = await executeTool('glob', { pattern: '*.txt', path: 'linked-outside' }, ctx)
  assert.equal(escapedSearch.ok, false)
  assert.equal(escapedGlob.ok, false)
  assert.match(escapedSearch.output, /escapes the workspace through a symlink/)
  assert.equal(execLog.length, beforeEscapeCommands, 'remote grep/find never run beneath a rejected symlink cwd')

  const patch = await executeTool('apply_patch', { patch: '*** Begin Patch' }, ctx)
  assert.strictEqual(patch.ok, false)
  assert.match(patch.meta ?? '', /remote-unsupported/)

  const escape = await executeTool('read_file', { path: '../../etc/passwd' }, ctx)
  assert.strictEqual(escape.ok, false) // remote path containment blocks escape

  ok('file/shell/search tools route through SSH; containment, secret filtering, and .git write protection hold')
}

{
  // #42 app-control — approval gating (Finding B) + folder validation (Finding C).
  // create_project/create_room/create_kanban_ticket MUTATE app state (and the first
  // writes files into a model-chosen cwd), so they must NOT be read-only/auto-run.
  // open_view (pure navigation) and change_setting (confirm-first in the renderer)
  // stay read-only.
  assert.ok(!READ_ONLY_TOOLS.has('create_project'), 'create_project must be approval-gated, not read-only')
  assert.ok(!READ_ONLY_TOOLS.has('create_room'), 'create_room must be approval-gated, not read-only')
  assert.ok(!READ_ONLY_TOOLS.has('create_kanban_ticket'), 'create_kanban_ticket must be approval-gated, not read-only')
  assert.ok(READ_ONLY_TOOLS.has('open_view'), 'open_view is pure navigation — stays read-only')
  assert.ok(READ_ONLY_TOOLS.has('change_setting'), 'change_setting is confirm-first — stays read-only')

  // create_project with no folder is fine.
  const noFolder = await executeTool('create_project', { name: 'Plan' }, { cwd: await makeWorkspace() })
  assert.strictEqual(noFolder.ok, true)
  assert.strictEqual(noFolder.appAction?.kind, 'create_project')
  assert.strictEqual(noFolder.appAction.payload.folder, undefined)

  // A non-absolute folder is rejected.
  const relFolder = await executeTool('create_project', { name: 'X', folder: 'some/rel' }, { cwd: await makeWorkspace() })
  assert.strictEqual(relFolder.ok, false)
  assert.match(relFolder.output, /absolute path/)

  // A nonexistent folder is rejected (no silent write to a path the model invented).
  const missingFolder = await executeTool('create_project', { name: 'X', folder: '/definitely/not/here/vontra-test-xyz' }, { cwd: await makeWorkspace() })
  assert.strictEqual(missingFolder.ok, false)
  assert.match(missingFolder.output, /does not exist/)

  // The home folder is refused — a project writes AGENTS.md/CLAUDE.md into its cwd.
  const homeFolder = await executeTool('create_project', { name: 'X', folder: os.homedir() }, { cwd: await makeWorkspace() })
  assert.strictEqual(homeFolder.ok, false)
  assert.match(homeFolder.output, /home folder/i)

  // A real existing subfolder is accepted (the authoritative symlink-resolved guard
  // also runs server-side in projects:create).
  const realFolder = await fs.mkdtemp(path.join(os.tmpdir(), 'vontra-proj-'))
  const okFolder = await executeTool('create_project', { name: 'Real', folder: realFolder }, { cwd: await makeWorkspace() })
  assert.strictEqual(okFolder.ok, true)
  assert.strictEqual(okFolder.appAction?.payload?.folder, realFolder)
  await fs.rm(realFolder, { recursive: true, force: true })

  // open_view validates its enum.
  const badView = await executeTool('open_view', { view: 'dashboard' }, { cwd: await makeWorkspace() })
  assert.strictEqual(badView.ok, false)
  assert.match(badView.output, /Unknown view/)
  const goodView = await executeTool('open_view', { view: 'settings' }, { cwd: await makeWorkspace() })
  assert.strictEqual(goodView.ok, true)
  assert.strictEqual(goodView.appAction?.kind, 'open_view')

  // change_setting stays confirm-first (state: 'pending'), never auto-applied.
  const setting = await executeTool('change_setting', { key: 'theme', value: 'dark' }, { cwd: await makeWorkspace() })
  assert.strictEqual(setting.ok, true)
  assert.strictEqual(setting.appAction?.state, 'pending')
  const badSetting = await executeTool('change_setting', { key: 'theme', value: 'neon' }, { cwd: await makeWorkspace() })
  assert.strictEqual(badSetting.ok, false)

  ok('app-control tools: mutating ones are approval-gated; create_project folder is validated')
}

console.log(`\n${passed} checks passed.`)
}

main().catch((error) => {
  console.error(error)
  process.exitCode = 1
})
