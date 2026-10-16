import assert from 'node:assert/strict'
import { execFile } from 'node:child_process'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { promisify } from 'node:util'
import { defaultSettings } from '../../store'
import { store } from '../state'
import { assertLocalPrProjectSupported, createLocalPrRuntime } from './local-pr-runtime'
import { assertMissionWorkspaceSupported, createMissionRuntime } from './mission-runtime'
import { assertSshToolContextConnected } from './thread-runtime'

const execFileAsync = promisify(execFile)

async function git(cwd: string, ...args: string[]): Promise<string> {
  const { stdout } = await execFileAsync('git', args, { cwd })
  return stdout
}

async function run() {
  const sshWorkspace = { kind: 'ssh' as const, hostId: 'pgx', path: '/srv/project' }

  assert.throws(() => assertSshToolContextConnected(sshWorkspace, { backend: undefined }), /disconnected during startup/, 'a backend lost after provider awaits fails at the final execution seam')
  assert.doesNotThrow(() => assertSshToolContextConnected(sshWorkspace, { backend: { exec() {} } }))
  assert.doesNotThrow(() => assertSshToolContextConnected({ kind: 'local', path: '/srv/project' }, { backend: undefined }))

  // Mission ownership is authoritative and complete: a missing/mismatched source is not allowed
  // to degrade into a local cwd run, and an SSH owner is rejected before any job is claimed.
  {
    assert.throws(
      () => assertMissionWorkspaceSupported({ projectId: null, threadId: 'missing-thread' }, { projects: [], threads: [] }),
      /source thread not found/,
    )
    assert.throws(
      () => assertMissionWorkspaceSupported({ projectId: 'missing-project', threadId: null }, { projects: [], threads: [] }),
      /project not found/,
    )
    assert.throws(
      () => assertMissionWorkspaceSupported(
        { projectId: 'project-a', threadId: 'thread-b' },
        {
          projects: [{ id: 'project-a' }, { id: 'project-b' }],
          threads: [{ id: 'thread-b', projectId: 'project-b' }],
        },
      ),
      /does not match its source thread project/,
    )
    assert.throws(
      () => assertMissionWorkspaceSupported(
        { projectId: 'project-a', threadId: 'projectless-thread' },
        {
          projects: [{ id: 'project-a' }],
          threads: [{ id: 'projectless-thread', projectId: null }],
        },
      ),
      /does not match its source thread project/,
      'a project-scoped mission cannot borrow a projectless thread',
    )
    assert.throws(
      () => assertMissionWorkspaceSupported(
        { projectId: 'remote-project', threadId: null },
        { projects: [{ id: 'remote-project', workspace: sshWorkspace }], threads: [] },
      ),
      /does not support SSH workspaces/,
    )

    const savedProjects = store.projects
    const savedThreads = store.threads
    let claims = 0
    const running = new Set<string>()
    try {
      store.projects = [{ id: 'remote-project', workspace: sshWorkspace } as any]
      store.threads = []
      const runtime = createMissionRuntime({
        runningMissions: running,
        missionAborters: new Map(),
        missionApprovalResolvers: new Map(),
        claimJob: () => { claims++; return {} as any },
      } as any)
      await assert.rejects(
        () => runtime.runMission({ id: 'remote-mission', projectId: 'remote-project', threadId: null } as any),
        /does not support SSH workspaces/,
      )
      assert.equal(claims, 0, 'SSH mission rejects before a durable job is claimed')
      assert.equal(running.has('remote-mission'), false, 'SSH mission rejects before in-memory running state changes')
    } finally {
      store.projects = savedProjects
      store.threads = savedThreads
    }
  }

  // Local PR loops are explicitly local and reject archived/SSH projects before filesystem work.
  {
    assert.throws(() => assertLocalPrProjectSupported({ archivedAt: Date.now() }), /Restore the archived project/)
    assert.throws(() => assertLocalPrProjectSupported({ workspace: sshWorkspace }), /do not support SSH workspaces/)

    const savedProjects = store.projects
    const savedSettings = store.settings
    const previousRole = process.env.VONTRA_SECURITY_ROLE
    let persistedThreads = 0
    try {
      process.env.VONTRA_SECURITY_ROLE = 'owner'
      store.settings = defaultSettings()
      store.projects = [{ id: 'remote-project', cwd: '/srv/project', workspace: sshWorkspace } as any]
      const runtime = createLocalPrRuntime({ persistThreads: () => { persistedThreads++ } } as any)
      await assert.rejects(
        () => runtime.createLocalPrLoop({ title: 'Remote', objective: 'Do work', projectId: 'remote-project' }),
        /do not support SSH workspaces/,
      )
      assert.equal(persistedThreads, 0, 'SSH rejection occurs before PR-loop state is persisted')
    } finally {
      store.projects = savedProjects
      store.settings = savedSettings
      if (previousRole == null) delete process.env.VONTRA_SECURITY_ROLE
      else process.env.VONTRA_SECURITY_ROLE = previousRole
    }
  }

  // Provider/local-only policy used to run after `git worktree add`, leaving an orphan branch and
  // worktree on failure. Pin the ordering with a real repository.
  {
    const repo = await fs.mkdtemp(path.join(os.tmpdir(), 'vontra-pr-policy-'))
    const savedProjects = store.projects
    const savedThreads = store.threads
    const savedMissions = store.missions
    const savedJobs = store.jobs
    const savedSettings = store.settings
    const previousRole = process.env.VONTRA_SECURITY_ROLE
    try {
      process.env.VONTRA_SECURITY_ROLE = 'owner'
      await git(repo, 'init', '-q')
      await fs.writeFile(path.join(repo, 'README.md'), 'baseline\n')
      await git(repo, 'add', 'README.md')
      await git(repo, '-c', 'user.name=Test', '-c', 'user.email=test@example.test', 'commit', '-q', '-m', 'initial')

      const settings = defaultSettings()
      settings.providers = [{
        id: 'cloud', name: 'Cloud', kind: 'cloud', transport: 'chat-completions',
        baseUrl: 'https://provider.invalid/v1', apiKey: '', models: ['model'],
      } as any]
      settings.activeProviderId = 'cloud'
      settings.activeModel = 'model'
      store.settings = settings
      store.projects = [{
        id: 'local-only-project', name: 'Local only', cwd: repo, memory: '', team: '', localOnly: true,
        createdAt: 1, updatedAt: 1,
      } as any]
      store.threads = []
      store.missions = []
      store.jobs = []

      const runtime = createLocalPrRuntime({ effectiveContextLimit: () => 128_000 } as any)
      await assert.rejects(
        () => runtime.createLocalPrLoop({
          title: 'Must not mutate Git', objective: 'Test invalid route ordering', projectId: 'local-only-project',
          providerId: 'cloud', model: 'model', autoStart: false,
        }),
        /local-only/,
      )
      assert.equal((await git(repo, 'branch', '--list', 'vontra/pr/*')).trim(), '', 'invalid route creates no PR branch')
      assert.equal((await git(repo, 'worktree', 'list', '--porcelain')).match(/^worktree /gm)?.length, 1, 'invalid route creates no worktree')
      assert.deepEqual(store.threads, [], 'invalid route creates no threads')
      assert.deepEqual(store.missions, [], 'invalid route creates no mission')
      assert.deepEqual(store.jobs, [], 'invalid route creates no job')
    } finally {
      store.projects = savedProjects
      store.threads = savedThreads
      store.missions = savedMissions
      store.jobs = savedJobs
      store.settings = savedSettings
      if (previousRole == null) delete process.env.VONTRA_SECURITY_ROLE
      else process.env.VONTRA_SECURITY_ROLE = previousRole
      await fs.rm(repo, { recursive: true, force: true })
    }
  }

  const missionSource = await fs.readFile(path.join(process.cwd(), 'electron/main/host/mission-runtime.ts'), 'utf8')
  const verifier = missionSource.slice(missionSource.indexOf('async function verifyMissionStopCondition'), missionSource.indexOf('async function runMission'))
  assert.match(verifier, /streamChat\(\{[\s\S]*?approvalMode: 'readonly'/, 'mission stop-condition verifier is readonly even for delegated transports')

  console.log('ssh-execution-boundaries.test: mission/local-PR ownership, pre-mutation policy, and readonly verifier passed')
}

run().catch((error) => { console.error(error); process.exit(1) })
