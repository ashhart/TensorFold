// Run: jiti electron/agent/cli-registry.test.ts
import assert from 'node:assert/strict'
import { builtinClis, cliToolStatuses, effectiveCliCommand, makeCustomCli, launchString, resumeString } from './cli-registry.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('cli-registry tests:')

{
  const ids = builtinClis().map((c) => c.id)
  assert.ok(ids.includes('shell') && ids.includes('claude') && ids.includes('codex'), 'has core built-ins')
  assert.ok(ids.includes('cursor') && ids.includes('gemini') && ids.includes('grok') && ids.includes('opencode') && ids.includes('kilo') && ids.includes('pi'), 'has provider-tool built-ins')
  assert.ok(ids.includes('hermes'), 'has the hermes harness built-in')
  assert.deepEqual([...new Set(ids)].length, ids.length, 'builtin ids are unique')
  const shell = builtinClis().find((c) => c.id === 'shell')!
  assert.equal(shell.command, null, 'shell has no command (plain shell)')
  assert.ok(builtinClis().every((c) => c.builtin), 'built-ins are flagged builtin')
  ok('builtinClis: includes shell + agent/provider CLIs, all builtin')
}

{
  const c = makeCustomCli('My Tool', 'mytool --serve')
  assert.equal(c.builtin, false)
  assert.equal(c.command, 'mytool --serve')
  assert.equal(c.short, 'mytool', 'short = first word of the command')
  assert.match(c.id, /[0-9a-f-]{8,}/i, 'custom id is a uuid')
  ok('makeCustomCli: builds an editable non-builtin definition')
}

{
  const shell = builtinClis().find((c) => c.id === 'shell')!
  const codex = builtinClis().find((c) => c.id === 'codex')!
  assert.equal(launchString(shell), null, 'shell never launches a command')
  assert.equal(launchString(codex), 'codex', 'interactive launch never appends approval-bypass flags')
  const noFlags = makeCustomCli('x', 'x')
  assert.equal(launchString(noFlags), 'x')
  assert.equal((builtinClis().find((c) => c.id === 'claude') as any).approvedFlags, undefined, 'dangerous Claude bypass metadata is absent')
  assert.equal((codex as any).approvedFlags, undefined, 'terminal metadata cannot request implicit Codex full-auto')
  ok('launchString: launches the explicit command only; shell => null')
}

{
  const codex = builtinClis().find((c) => c.id === 'codex')!
  const claude = builtinClis().find((c) => c.id === 'claude')!
  const shell = builtinClis().find((c) => c.id === 'shell')!
  const gemini = builtinClis().find((c) => c.id === 'gemini')!
  // CLIs that define resumeCommand reattach to their last session instead of relaunching fresh.
  assert.equal(resumeString(codex), 'codex resume --last', 'codex reattaches to its last session')
  assert.equal(resumeString(claude), 'claude --resume', 'claude resumes its prior session')
  assert.equal(resumeString(builtinClis().find((c) => c.id === 'pi')!), 'pi --resume', 'pi resumes its prior session')
  assert.equal(resumeString(builtinClis().find((c) => c.id === 'opencode')!), 'opencode --continue', 'opencode continues its last session')
  // A plain shell has no command at all → nothing to launch or resume.
  assert.equal(resumeString(shell), null, 'shell never launches/resumes a command')
  // No resumeCommand → fall back to launchString (relaunch fresh). gemini has a command but null resume.
  assert.equal(resumeString(gemini), 'gemini', 'no resumeCommand falls back to a fresh launch')
  assert.equal(resumeString(gemini), launchString(gemini), 'fallback === launchString')
  const codexAutoLaunch = makeCustomCli('codexish', 'codex')
  assert.equal(resumeString(codexAutoLaunch), 'codex', 'custom CLI w/ null resume falls back to its explicit command')
  ok('resumeString: uses resumeCommand when set, else falls back to the explicit launch command')
}

{
  const codex = builtinClis().find((c) => c.id === 'codex')!
  assert.equal(effectiveCliCommand(codex, '/opt/vontra/codex'), '/opt/vontra/codex', 'binary override replaces the executable')
  assert.equal(effectiveCliCommand({ ...codex, command: 'codex exec' }, '/opt/vontra/codex'), '/opt/vontra/codex exec', 'binary override preserves fixed command args')
  assert.equal(effectiveCliCommand(codex, '/Applications/Codex Preview.app/Contents/MacOS/codex'), "'/Applications/Codex Preview.app/Contents/MacOS/codex'", 'binary override shell-quotes paths with spaces')
  ok('effectiveCliCommand: applies binary overrides without losing command args')
}

{
  const dirs = ['/bin']
  const exists = (path: string) => path === '/custom/grok'
  const [grok] = cliToolStatuses([builtinClis().find((c) => c.id === 'grok')!], [], { dirs, exists, overrides: { grok: { binaryPath: '/custom/grok' } } })
  assert.equal(grok.installed, true, 'generic tool override is probed')
  assert.equal(grok.configuredPath, '/custom/grok')
  assert.equal(grok.effectiveCommand, '/custom/grok')
  ok('cliToolStatuses: probes generic binary overrides')
}

{
  for (const [toolId, transport] of [
    ['codex', 'codex-cli'], ['claude', 'claude-cli'], ['cursor', 'cursor-cli'], ['gemini', 'gemini-cli'], ['pi', 'pi-cli'],
  ] as const) {
    const configuredPath = `/provider/${toolId}`
    const [tool] = cliToolStatuses([builtinClis().find((c) => c.id === toolId)!], [
      { id: `${toolId}-provider`, name: toolId, transport, baseUrl: '', apiKey: '', models: [], cliPath: configuredPath } as any,
    ], { exists: (path) => path === configuredPath })
    assert.equal(tool.providerIds[0], `${toolId}-provider`, `${toolId} provider id is retained for settings UI`)
    assert.equal(tool.configuredPath, configuredPath, `${toolId} provider path wins`)
    assert.equal(tool.installed, true)
  }
  ok('cliToolStatuses: provider cliPath wins for every delegated CLI transport')
}

console.log(`\n${passed} passed`)
