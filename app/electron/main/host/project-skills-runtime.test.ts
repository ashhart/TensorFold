// Run: jiti electron/main/host/project-skills-runtime.test.ts
//
// Repo-authored project skills end-to-end (VST-010): real tmpdir with
// `.vontra/skills/<name>/SKILL.md`, in-memory trust store. Proves the spec tests:
// skill load, project-local trust, disabled-until-trusted, changed-content revocation.
import assert from 'node:assert/strict'
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { createProjectSkillsRuntime, PROJECT_SKILLS_DIR } from './project-skills-runtime'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('project-skills-runtime tests:')

const root = await mkdtemp(path.join(tmpdir(), 'vontra-proj-skills-'))
const project = { id: 'proj1', cwd: root, workspace: null }
const skillDir = path.join(root, PROJECT_SKILLS_DIR, 'deploy')
await mkdir(skillDir, { recursive: true })
await writeFile(path.join(skillDir, 'SKILL.md'), [
  '---',
  'name: Deploy checklist',
  'description: Use when shipping this service to production.',
  '---',
  '',
  'Run the release script and verify the health endpoint.',
  '',
  '## Tests',
  'curl the /healthz endpoint and expect 200.',
].join('\n'))

let disk: unknown = null
const runtime = createProjectSkillsRuntime({
  loadTrust: async () => disk,
  saveTrust: (registry) => { disk = JSON.parse(JSON.stringify(registry)) },
})

try {
  {
    const views = await runtime.scanProjectSkills(project)
    assert.equal(views.length, 1, 'the repo skill is discovered')
    assert.equal(views[0].skill.name, 'Deploy checklist')
    assert.equal(views[0].trust, 'untrusted', 'repo content starts untrusted')
    assert.equal(views[0].skill.enabled, false, 'untrusted skills are disabled')
    assert.equal(views[0].skill.scope, 'project')
    assert.equal(views[0].skill.projectId, 'proj1')
    assert.deepEqual(runtime.cachedProjectSkills('proj1'), [], 'nothing untrusted reaches a run context')
    ok('scan: discovered, untrusted, excluded from runs')
  }

  {
    const views = await runtime.setProjectSkillTrust(project, 'deploy', true)
    assert.equal(views[0].trust, 'trusted')
    assert.equal(views[0].skill.enabled, true)
    const cached = runtime.cachedProjectSkills('proj1')
    assert.equal(cached.length, 1, 'a trusted skill joins the run context')
    assert.match(cached[0].body, /release script/)
    assert.ok(disk, 'trust persists')
    assert.deepEqual(runtime.cachedProjectSkills('other-project'), [], 'never leaks into another project')
    ok('trust: enabled for THIS project only, persisted')
  }

  {
    // An SSH project can have a remote path that also exists on this Mac. Never scan that local
    // lookalike or retain a previously trusted body in the synchronous run cache.
    const remoteProject = { ...project, workspace: { kind: 'ssh' as const, hostId: 'pgx', path: root } }
    const views = await runtime.scanProjectSkills(remoteProject)
    assert.deepEqual(views, [], 'SSH repo skills are not read through the local filesystem scanner')
    assert.deepEqual(runtime.cachedProjectSkills('proj1'), [], 'switching to SSH clears stale trusted repo instructions')
    await assert.rejects(
      () => runtime.setProjectSkillTrust(remoteProject, 'deploy', true),
      /unavailable for SSH projects/,
      'an SSH local-lookalike skill cannot be newly trusted',
    )
    await runtime.refreshProjectSkills(remoteProject)
    assert.deepEqual(runtime.cachedProjectSkills('proj1'), [], 'run-start refresh remains fail closed for SSH')
    await runtime.scanProjectSkills(project)
    ok('ssh: local lookalikes are not scanned and stale trusted cache is cleared')
  }

  {
    // A repo edit AFTER review must drop the skill out of runs until re-reviewed.
    await writeFile(path.join(skillDir, 'SKILL.md'), 'Totally different instructions now.')
    const views = await runtime.scanProjectSkills(project)
    assert.equal(views[0].trust, 'changed', 'edited content is flagged, not silently run')
    assert.equal(views[0].skill.enabled, false)
    assert.deepEqual(runtime.cachedProjectSkills('proj1'), [], 'changed content is out of the run context')
    ok('changed: post-review edits revoke run access until re-trusted')
  }

  {
    // A fresh runtime (restart) reads the persisted registry and still flags the change.
    const restarted = createProjectSkillsRuntime({ loadTrust: async () => disk, saveTrust: () => {} })
    const views = await restarted.scanProjectSkills(project)
    assert.equal(views[0].trust, 'changed', 'trust decisions survive restarts')

    // Re-trust pins the new content; revoke withdraws it again.
    await restarted.setProjectSkillTrust(project, 'deploy', true)
    assert.equal(restarted.cachedProjectSkills('proj1').length, 1)
    await restarted.setProjectSkillTrust(project, 'deploy', false)
    assert.deepEqual(restarted.cachedProjectSkills('proj1'), [])
    ok('restart: registry restores; re-trust and revoke round-trip')
  }

  {
    // Exact ids only: a lexically-earlier suffix collision must not steal trust intended
    // for the bare "deploy" skill.
    const collisionDir = path.join(root, PROJECT_SKILLS_DIR, '0_deploy')
    await mkdir(collisionDir, { recursive: true })
    await writeFile(path.join(collisionDir, 'SKILL.md'), 'Shadow deployment instructions.')
    const views = await runtime.setProjectSkillTrust(project, 'deploy', true)
    assert.equal(views.find((view) => view.skill.id === 'ps_proj1_deploy')?.trust, 'trusted')
    assert.equal(views.find((view) => view.skill.id === 'ps_proj1_0_deploy')?.trust, 'untrusted')
    assert.match(runtime.cachedProjectSkills('proj1')[0].body, /Totally different instructions/)
    await runtime.setProjectSkillTrust(project, 'ps_proj1_deploy', false)
    assert.deepEqual(runtime.cachedProjectSkills('proj1'), [])
    ok('trust lookup uses exact bare/full ids and rejects suffix collisions')
  }

  {
    const none = await runtime.scanProjectSkills({ id: 'proj2', cwd: path.join(root, 'nope'), workspace: null })
    assert.deepEqual(none, [], 'a project without a skills dir simply has none')
    ok('missing skills dir: empty, no errors')
  }
} finally {
  await rm(root, { recursive: true, force: true })
}

console.log(`  ${passed} passed`)
