// Run: npx tsx electron/agent/capability-registry.test.ts
import assert from 'node:assert'
import {
  buildCapabilityRegistry,
  capabilityAuditForTool,
  renderCapabilityRegistryPrompt,
  selectedCapabilityIdsFromMessages,
} from './capability-registry.ts'
import { buildChatToolSchemas, buildToolSchemas } from './tools.ts'
import type { ApiMessage, Command, LocalScriptCapability, McpToolDefinition, PackagedPluginCapability, Skill } from './types.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }
const schemaNames = (schemas: any[]) => schemas.map((schema) => schema.function?.name).filter(Boolean)

const mcpTools: McpToolDefinition[] = [
  {
    serverId: 'private',
    serverName: 'Private Search',
    approvalMode: 'ask',
    name: 'search',
    callName: 'mcp__private__search',
    title: 'Private search',
    description: 'Search a private external system.',
    inputSchema: { type: 'object', properties: { query: { type: 'string' } } },
    readOnlyHint: true,
    openWorldHint: true,
  },
]

console.log('capability registry tests:')

{
  const messages: Pick<ApiMessage, 'role' | 'content'>[] = [
    { role: 'user', content: '@browser\n<vontra_capability name="PDF" id="pdf">Use PDFs.</vontra_capability>' },
  ]
  const selected = selectedCapabilityIdsFromMessages(messages)
  assert.ok(selected.includes('browser'))
  assert.ok(selected.includes('pdf'))
  ok('extracts selected built-in capability ids from composer text')
}

{
  const registry = buildCapabilityRegistry({
    mcpTools,
    selectedCapabilityIds: ['browser', 'controller', 'mcp:mcp__private__search'],
    disabledCapabilityIds: ['connector:email'],
    now: 1,
  })
  const email = registry.audit.find((entry) => entry.capabilityId === 'connector:email')
  assert.strictEqual(email?.state, 'disabled')
  assert.ok(!registry.visibleToolNames.includes('email_list'))
  assert.ok(registry.selectedSets.find((set) => set.id === 'run-selected')?.capabilityIds.includes('browser'))

  const mcp = capabilityAuditForTool(registry, 'mcp__private__search')
  assert.ok(mcp)
  assert.ok(mcp.permissions.includes('mcp_call'))
  assert.ok(mcp.riskFlags.includes('external_system'))
  assert.ok(mcp.riskFlags.includes('network'))
  assert.ok(mcp.riskFlags.includes('prompt_injection'))
  assert.strictEqual(mcp.approvalRequired, true)

  const browser = registry.audit.find((entry) => entry.capabilityId === 'browser')
  assert.strictEqual(browser?.selected, true)
  assert.strictEqual(browser?.metadata?.sandboxProfile, 'browser')
  assert.ok(browser?.toolNames.includes('browser_automation'))
  assert.ok(browser?.permissions.includes('browser_control'))
  assert.ok(browser?.riskFlags.includes('approval_required'))
  assert.match(browser?.promptInjectionWarning ?? '', /Browser page contents/)
  const controller = registry.audit.find((entry) => entry.capabilityId === 'controller')
  assert.ok(controller?.toolNames.includes('controller_action'))
  assert.ok(controller?.permissions.includes('controller_capture'))
  assert.ok(controller?.approvalRequired)
  const shell = registry.audit.find((entry) => entry.capabilityId === 'tool:shell')
  assert.deepStrictEqual(shell?.metadata?.sandboxProfiles, ['read-only', 'build-test', 'package-install', 'network-fetch', 'system-control', 'untrusted-code'])
  assert.match(String(shell?.metadata?.profilePolicy ?? ''), /approval-policy command classification/)
  const web = registry.audit.find((entry) => entry.capabilityId === 'tool:web')
  assert.strictEqual(web?.metadata?.sandboxProfile, 'network-fetch')
  const codeIntel = registry.audit.find((entry) => entry.capabilityId === 'tool:code-intelligence')
  assert.ok(codeIntel?.toolNames.includes('ast_grep_search'))
  assert.ok(codeIntel?.toolNames.includes('semgrep_scan'))
  assert.ok(codeIntel?.toolNames.includes('safe_refactor_preview'))
  assert.strictEqual(codeIntel?.metadata?.proposedPatchesOnly, true)
  const workspaceRead = registry.audit.find((entry) => entry.capabilityId === 'tool:workspace-read')
  assert.ok(workspaceRead?.toolNames.includes('retrieval_search'))
  const emailConnector = registry.audit.find((entry) => entry.capabilityId === 'connector:email')
  assert.strictEqual(emailConnector?.metadata?.sandboxProfile, 'connector-send')
  assert.match(renderCapabilityRegistryPrompt(registry), /Prompt-injection warning/)
  ok('registry audit metadata includes selected state, risks, and permissions')
}

{
  const registry = buildCapabilityRegistry({
    selectedCapabilityIds: ['skill:ui-smoke-localhost'],
    now: 2,
  })
  const recipe = registry.audit.find((entry) => entry.capabilityId === 'skill:ui-smoke-localhost')
  assert.ok(recipe)
  assert.strictEqual(recipe.source, 'skill')
  assert.strictEqual(recipe.lane, 'browser')
  assert.strictEqual(recipe.selected, true)
  assert.ok(recipe.toolNames.includes('run_command'))
  assert.ok(recipe.toolNames.includes('browser_automation'))
  assert.ok(recipe.permissions.includes('browser_control'))
  assert.strictEqual(recipe.metadata?.recipeKind, 'ui_smoke')
  assert.match(renderCapabilityRegistryPrompt(registry), /\$ui-smoke-localhost/)
  ok('registry includes built-in UI smoke recipe capabilities')
}

{
  const registry = buildCapabilityRegistry({
    mcpTools: [{
      serverId: 'danger',
      serverName: 'Danger MCP',
      approvalMode: 'trusted',
      name: 'delete_all',
      callName: 'mcp__danger__delete_all',
      title: 'Delete all records',
      description: 'Delete external records.',
      inputSchema: { type: 'object', properties: {} },
      readOnlyHint: true,
      destructiveHint: true,
    }],
    now: 2,
  })
  const mcp = capabilityAuditForTool(registry, 'mcp__danger__delete_all')
  assert.ok(mcp)
  assert.strictEqual(mcp.readOnly, false)
  assert.strictEqual(mcp.approvalRequired, true)
  assert.ok(mcp.riskFlags.includes('destructive'))
  assert.ok(mcp.riskFlags.includes('approval_required'))
  ok('MCP destructiveHint overrides misleading readOnlyHint')
}

{
  const registry = buildCapabilityRegistry({
    mcpTools: [{
      serverId: 'liar',
      serverName: 'Self-described read-only MCP',
      approvalMode: 'trusted',
      name: 'quiet_mutation',
      callName: 'mcp__liar__quiet_mutation',
      description: 'Server claims this call only reads.',
      inputSchema: { type: 'object', properties: {} },
      readOnlyHint: true,
      destructiveHint: false,
      openWorldHint: false,
    }],
    now: 2,
  })
  const mcp = capabilityAuditForTool(registry, 'mcp__liar__quiet_mutation')
  assert.ok(mcp)
  assert.strictEqual(mcp.readOnly, false)
  assert.strictEqual(mcp.approvalRequired, true)
  assert.ok(mcp.riskFlags.includes('approval_required'))
  ok('MCP server-supplied readOnlyHint never grants approval or changes the advertised gate')
}

{
  const registry = buildCapabilityRegistry({
    selectedCapabilityIds: ['browser'],
    disabledCapabilityIds: ['tool:web'],
    capabilityHealth: {
      browser: { status: 'warning', checkedAt: 42, lastError: 'Worker not installed yet.' },
    },
    now: 2,
  })
  const browser = registry.audit.find((entry) => entry.capabilityId === 'browser')
  assert.strictEqual(browser?.state, 'enabled')
  assert.strictEqual(browser?.health?.status, 'warning')
  assert.strictEqual(browser?.health?.lastError, 'Worker not installed yet.')
  assert.ok(!registry.visibleToolNames.includes('web_search'))
  ok('registry carries capability health and settings-disabled tools')
}

{
  const registry = buildCapabilityRegistry({
    agentSkills: [{
      id: 'reviewer',
      name: 'Reviewer',
      description: 'Review code changes.',
      rootPath: '/repo/.agents/reviewer',
      skillPath: '/repo/.agents/reviewer/SKILL.md',
      hasScripts: true,
      hasReferences: true,
      hasAssets: false,
      updatedAt: 10,
    }],
    selectedCapabilityIds: ['skill:reviewer'],
    now: 3,
  })
  const skill = registry.audit.find((entry) => entry.capabilityId === 'skill:reviewer')
  assert.ok(skill)
  assert.strictEqual(skill.source, 'skill')
  assert.strictEqual(skill.selected, true)
  assert.ok(skill.riskFlags.includes('prompt_injection'))
  assert.match(skill.promptInjectionWarning ?? '', /SKILL\.md/)
  ok('registry includes scanned Agent Skills as capability audit entries')
}

{
  const savedSkills: Skill[] = [{
    id: 's_review',
    name: 'reviewer',
    description: 'Review changes',
    body: 'private skill body should stay out of audit prompt',
    enabled: true,
    createdAt: 1,
    updatedAt: 2,
  }]
  const savedCommands: Command[] = [{
    id: 'c_ship',
    name: 'ship-it',
    description: 'Prepare release notes',
    prompt: 'private command prompt should stay out of audit prompt',
    argHint: '<version>',
    createdAt: 3,
    updatedAt: 4,
  }]
  const registry = buildCapabilityRegistry({
    skills: savedSkills,
    commands: savedCommands,
    selectedCapabilityIds: ['skill:reviewer', 'command:ship-it'],
    now: 4,
  })
  const skill = registry.capabilities.find((entry) => entry.id === 'skill:reviewer')
  const command = registry.capabilities.find((entry) => entry.id === 'command:ship-it')
  assert.ok(skill)
  assert.ok(command)
  assert.strictEqual(skill.source, 'skill')
  assert.strictEqual(command.source, 'command')
  assert.strictEqual(skill.metadata?.provenance, 'settings.skills')
  assert.strictEqual(command.metadata?.provenance, 'settings.commands')
  assert.strictEqual(skill.metadata?.bodyChars, savedSkills[0].body.length)
  assert.strictEqual(command.metadata?.promptChars, savedCommands[0].prompt.length)
  const prompt = renderCapabilityRegistryPrompt(registry)
  assert.match(prompt, /\$reviewer/)
  assert.match(prompt, /\/ship-it/)
  assert.ok(!prompt.includes(savedSkills[0].body))
  assert.ok(!prompt.includes(savedCommands[0].prompt))
  ok('registry includes saved skills and commands without leaking their bodies')
}

{
  const localScripts: LocalScriptCapability[] = [{
    id: 'daily-news',
    name: 'Daily News',
    description: 'Fetch a daily news digest.',
    path: '/Users/example/.vontra/scripts/daily-news.sh',
    command: 'zsh',
    args: ['daily-news.sh'],
    checksumSha256: 'abc123def4567890',
    signatureVerifiedAt: null,
    sandboxProfile: 'no-network',
    enabled: true,
    createdAt: 1,
    updatedAt: 2,
  }]
  const packagedPlugins: PackagedPluginCapability[] = [{
    id: 'repo-tools',
    name: 'Repo Tools',
    description: 'Repository helper plugin.',
    manifestPath: '/Users/example/.vontra/plugins/repo-tools/plugin.json',
    packagePath: '/Users/example/.vontra/plugins/repo-tools',
    checksumSha256: 'def456abc1237890',
    signatureVerifiedAt: 55,
    sandboxProfile: 'workspace-write',
    toolNames: ['plugin__repo_tools__scan'],
    trusted: true,
    enabled: true,
    createdAt: 3,
    updatedAt: 4,
  }]
  const registry = buildCapabilityRegistry({
    localScripts,
    packagedPlugins,
    selectedCapabilityIds: ['script:daily-news', 'plugin:repo-tools'],
    now: 5,
  })
  const script = registry.audit.find((entry) => entry.capabilityId === 'script:daily-news')
  const plugin = registry.audit.find((entry) => entry.capabilityId === 'plugin:repo-tools')
  assert.ok(script)
  assert.ok(plugin)
  assert.strictEqual(script.source, 'local_script')
  assert.strictEqual(plugin.source, 'plugin')
  assert.strictEqual(script.metadata?.trustLevel, 'checksum')
  assert.strictEqual(plugin.metadata?.trustLevel, 'signed')
  assert.strictEqual(script.metadata?.sandboxProfile, 'no-network')
  assert.strictEqual(plugin.metadata?.sandboxProfile, 'workspace-write')
  assert.ok(script.riskFlags.includes('shell_exec'))
  assert.ok(plugin.riskFlags.includes('approval_required'))
  assert.ok(registry.visibleToolNames.includes('plugin__repo_tools__scan'))
  const prompt = renderCapabilityRegistryPrompt(registry)
  assert.match(prompt, /Trust: level=checksum/)
  assert.match(prompt, /Trust: level=signed/)
  assert.ok(!prompt.includes(localScripts[0].path))
  assert.ok(!prompt.includes(packagedPlugins[0].manifestPath))
  ok('registry includes executable scripts and packaged plugins with trust metadata')
}

{
  const names = schemaNames(buildToolSchemas(mcpTools, { disabledCapabilityIds: ['connector:email'] }))
  assert.ok(!names.includes('email_list'))
  assert.ok(!names.includes('email_compose'))
  assert.ok(!names.includes('email_resolve'))
  assert.ok(names.includes('mcp__private__search'))
  assert.ok(names.includes('run_command'))
  ok('disabled connector capabilities are omitted from visible tool schemas')
}

{
  const names = schemaNames(buildToolSchemas([], { disabledCapabilityIds: ['browser'] }))
  assert.ok(!names.includes('browser_automation'))
  assert.ok(names.includes('run_command'))
  ok('disabled browser automation capability hides browser control tool')
}

{
  const names = schemaNames(buildToolSchemas(mcpTools, { disabledCapabilityIds: ['mcp:mcp__private__search'] }))
  assert.ok(!names.includes('mcp__private__search'))
  assert.ok(names.includes('email_list'))
  ok('disabled MCP capabilities are omitted from visible tool schemas')
}

{
  const names = schemaNames(buildChatToolSchemas({ disabledCapabilityIds: ['tool:web'] }))
  // A local clarification is an explicit user interaction, independent of
  // public web access; it cannot expose shell/files/connectors or grant approval.
  assert.deepStrictEqual(names, ['ask_user', 'visualize'])
  assert.ok(!names.some((name) => ['web_search', 'web_fetch', 'run_command', 'write_file', 'email_compose'].includes(name)))
  const withoutQuestions = schemaNames(buildChatToolSchemas({ disabledCapabilityIds: ['tool:web'], disabledToolNames: ['ask_user'] }))
  assert.deepStrictEqual(withoutQuestions, ['visualize'], 'the explicit question tool disable still wins')
  ok('chat schemas retain local clarification while respecting web and explicit tool restrictions')
}

console.log(`capability registry tests: ${passed} passed`)
