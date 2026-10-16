import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { hostGitEnvForWorkspace } from './host-git-env'

const root = await fs.mkdtemp(path.join(os.tmpdir(), 'studio-git-env-'))
try {
  const config = path.join(root, 'gitconfig')
  await fs.writeFile(config, '[filter "fixture"]\n clean = false\n smudge = false\n process = false\n required = true\n[merge "fixture"]\n driver = fixture-command\n[commit]\n gpgSign = true\n')
  const env = await hostGitEnvForWorkspace(root, { ...process.env, GIT_CONFIG_NOSYSTEM: '1', GIT_CONFIG_GLOBAL: config })
  const settings = new Map(Array.from({ length: Number(env.GIT_CONFIG_COUNT) }, (_, i) => [env[`GIT_CONFIG_KEY_${i}`], env[`GIT_CONFIG_VALUE_${i}`]]))
  assert.equal(settings.get('filter.fixture.clean'), '')
  assert.equal(settings.get('filter.fixture.smudge'), '')
  assert.equal(settings.get('filter.fixture.process'), '')
  assert.equal(settings.get('filter.fixture.required'), 'false')
  assert.equal(settings.get('merge.fixture.driver'), 'false')
  assert.equal(settings.get('commit.gpgSign'), 'false')
  assert.equal(settings.get('core.fsmonitor'), 'false')
  assert.equal(settings.get('core.hooksPath'), '/dev/null')
} finally { await fs.rm(root, { recursive: true, force: true }) }
console.log('host Git filter isolation tests passed')
