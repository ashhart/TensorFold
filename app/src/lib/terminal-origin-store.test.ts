// Run: jiti src/lib/terminal-origin-store.test.ts
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

const g = globalThis as Record<string, unknown>
g.localStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} }
g.matchMedia = () => ({ matches: false, addEventListener: () => {}, removeEventListener: () => {} })
g.addEventListener = () => {}
g.removeEventListener = () => {}
g.window = g

await (await import('./browser-demo-bootstrap')).prepareBrowserDemo()
const { api } = await import('./ipc.ts')
const { useStore } = await import('./store.ts')

const projectId = 'terminal-origin-project'
const local = { id: 'local-1', projectId, cliId: 'shell', title: 'Local', status: 'idle', createdAt: 1, hostLocation: 'local' } as const
const remoteA = { id: 'remote-a', projectId, cliId: 'shell', title: 'Remote A', status: 'idle', createdAt: 2, hostLocation: 'orchestrator', orchestratorHostId: 'host-a' } as const
const legacyRemote = { id: 'remote-legacy', projectId, cliId: 'shell', title: 'Legacy remote', status: 'idle', createdAt: 3, hostLocation: 'orchestrator' } as const
const calls: string[] = []

useStore.setState({
  settings: {
    providers: [],
    approvalMode: 'suggest',
    theme: 'system',
    defaultExecutionTarget: 'orchestrator',
    defaultOrchestratorId: 'host-b',
    remoteOrchestrators: [
      { id: 'host-a', name: 'Host A', enabled: true },
      { id: 'host-b', name: 'Host B', enabled: true },
    ],
  } as any,
  projects: [{ id: projectId, name: 'Project', cwd: '/repo', executionTarget: 'local', createdAt: 1, updatedAt: 1 } as any],
  terminals: { [projectId]: [remoteA, legacyRemote] as any },
  activeTerminalId: { [projectId]: 'remote-a' },
  notice: null,
})

api.listTerminals = (async (id: string) => {
  calls.push(`list-local:${id}`)
  return [local] as any
}) as typeof api.listTerminals
api.remoteTerminalList = (async (hostId: string | null | undefined, id: string) => {
  calls.push(`list-remote:${hostId}:${id}`)
  if (hostId === 'host-a') return [remoteA] as any
  return [{ id: 'remote-b', projectId, cliId: 'shell', title: 'Remote B', status: 'idle', createdAt: 4 }] as any
}) as typeof api.remoteTerminalList
api.restorableTerminals = (async () => []) as typeof api.restorableTerminals

await useStore.getState().loadTerminals(projectId)
assert.deepEqual(
  new Set((useStore.getState().terminals[projectId] ?? []).map((session) => session.id)),
  new Set(['local-1', 'remote-a', 'remote-b', 'remote-legacy']),
  'global remote mode retains local, prior-host, selected-host, and unattributable legacy sessions',
)
assert.ok(calls.includes(`list-local:${projectId}`), 'local sessions are refreshed even when new terminals target a remote host')
assert.ok(calls.includes(`list-remote:host-a:${projectId}`), 'a cached immutable origin is refreshed after retargeting')
assert.ok(calls.includes(`list-remote:host-b:${projectId}`), 'the configured new-work host is refreshed')

api.killTerminal = (async (sessionId: string) => { calls.push(`kill-local:${sessionId}`) }) as typeof api.killTerminal
api.renameTerminal = (async (sessionId: string, title: string) => { calls.push(`rename-local:${sessionId}:${title}`) }) as typeof api.renameTerminal
api.remoteTerminalKill = (async (hostId: string | null | undefined, sessionId: string) => { calls.push(`kill-remote:${hostId}:${sessionId}`) }) as typeof api.remoteTerminalKill
api.remoteTerminalRename = (async (hostId: string | null | undefined, sessionId: string, title: string) => { calls.push(`rename-remote:${hostId}:${sessionId}:${title}`) }) as typeof api.remoteTerminalRename

// Retarget new work from host B to local. Existing remote A must still use A, while the local PTY
// stays local; neither may follow the changed project/global default.
useStore.setState((state) => ({
  settings: { ...state.settings!, defaultExecutionTarget: 'local', defaultOrchestratorId: null } as any,
  projects: state.projects.map((project) => project.id === projectId ? { ...project, executionTarget: 'local', orchestratorId: null } : project),
}))
await useStore.getState().renameTerminal(projectId, 'remote-a', 'Still A')
await useStore.getState().killTerminal(projectId, 'local-1')
assert.ok(calls.includes('rename-remote:host-a:remote-a:Still A'), 'remote A rename stays on immutable host A after retarget')
assert.ok(calls.includes('kill-local:local-1'), 'local terminal stays local after global/project retargets')
assert.ok(!calls.includes('rename-local:remote-a:Still A'))

const beforeLegacy = calls.length
await useStore.getState().renameTerminal(projectId, 'remote-legacy', 'Unsafe')
await useStore.getState().killTerminal(projectId, 'remote-legacy')
assert.equal(calls.length, beforeLegacy, 'legacy remote session never falls through to local or a newly selected host')
assert.match(useStore.getState().notice?.message ?? '', /cannot safely determine which host owns it/i)
assert.ok(useStore.getState().terminals[projectId]?.some((session) => session.id === 'remote-legacy'), 'fail-closed legacy row remains visible')

const sidebarSource = readFileSync(fileURLToPath(new URL('../components/sidebar/ProjectsSection.tsx', import.meta.url)), 'utf8')
const quickTerminalStart = sidebarSource.match(/<MenuItem icon="terminal"[\s\S]*?<\/MenuItem>/)?.[0] ?? ''
assert.match(quickTerminalStart, /openTerminal\(project\.id, terminalSessionKey\(s\)\)/, 'sidebar quick-start opens the collision-safe immutable terminal key')
assert.doesNotMatch(quickTerminalStart, /openTerminal\(project\.id, s\.id\)/, 'sidebar never reintroduces ambiguous raw terminal ids')

console.log('terminal origin store regressions passed')
