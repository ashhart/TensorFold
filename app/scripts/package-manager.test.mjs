import assert from 'node:assert/strict'
import { mkdir, mkdtemp, rm, symlink, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { assertInstallTreeNotMixed, assertNpmInvocation, assertRepoPackageManager } from './check-package-manager.mjs'

const tmp = await mkdtemp(path.join(os.tmpdir(), 'vontra-package-manager-'))
try {
  await writeFile(path.join(tmp, 'package.json'), JSON.stringify({ name: 'fixture', version: '1.0.0', packageManager: 'npm@10.9.8' }))
  await writeFile(path.join(tmp, 'package-lock.json'), JSON.stringify({ name: 'fixture', version: '1.0.0', lockfileVersion: 3 }))
  await writeFile(path.join(tmp, '.npmrc'), 'ignore-scripts=true\n')
  assert.equal((await assertRepoPackageManager(tmp)).version, '10.9.8')

  assert.doesNotThrow(() => assertNpmInvocation({ npm_config_user_agent: 'npm/10.9.8 node/v22.23.1', npm_execpath: '/opt/npm/bin/npm-cli.js' }, '10.9.8'))
  assert.doesNotThrow(() => assertNpmInvocation({ npm_config_user_agent: 'npm/10.9.8 node/v22.23.1', npm_execpath: '/opt/corepack/dist/npm.js' }, '10.9.8'))
  assert.throws(() => assertNpmInvocation({ npm_config_user_agent: 'pnpm/10.0.0', npm_execpath: '/opt/pnpm.cjs' }, '10.9.8'), /must be installed with npm/)
  assert.throws(() => assertNpmInvocation({ npm_config_user_agent: 'npm/11.0.0', npm_execpath: '/opt/npm/bin/npm-cli.js' }, '10.9.8'), /requires npm 10\.9\.8/)
  assert.throws(() => assertNpmInvocation({ npm_config_user_agent: 'npm/10.9.8', npm_execpath: '/opt/pnpm.cjs' }, '10.9.8'), /Unexpected npm executable/)

  await writeFile(path.join(tmp, 'pnpm-lock.yaml'), 'lockfileVersion: 9\n')
  await assert.rejects(() => assertRepoPackageManager(tmp), /pnpm-lock\.yaml is not supported/)
  await rm(path.join(tmp, 'pnpm-lock.yaml'))

  await writeFile(path.join(tmp, '.npmrc'), 'ignore-scripts=false\n')
  await assert.rejects(() => assertRepoPackageManager(tmp), /lifecycle scripts are denied/)
  await writeFile(path.join(tmp, '.npmrc'), 'ignore-scripts=true\n')

  const modules = path.join(tmp, 'node_modules')
  await mkdir(path.join(modules, '.pnpm'), { recursive: true })
  await assert.rejects(() => assertInstallTreeNotMixed(tmp), /hybrid tree/)
  await rm(modules, { recursive: true, force: true })
  await mkdir(modules, { recursive: true })
  await symlink(modules, path.join(modules, 'node_modules'), 'dir')
  await assert.rejects(() => assertInstallTreeNotMixed(tmp), /self-symlink artifacts/)
} finally {
  await rm(tmp, { recursive: true, force: true })
}

console.log('package manager enforcement tests passed')
