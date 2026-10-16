#!/usr/bin/env node
import { access, lstat, readFile, readdir, realpath } from 'node:fs/promises'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

export async function assertRepoPackageManager(root) {
  const pkg = JSON.parse(await readFile(path.join(root, 'package.json'), 'utf8'))
  const declared = String(pkg.packageManager || '')
  if (!/^npm@\d+\.\d+\.\d+$/.test(declared)) {
    throw new Error('package.json must pin an exact npm version in packageManager (for example npm@10.9.8).')
  }
  await access(path.join(root, 'package-lock.json')).catch(() => {
    throw new Error('package-lock.json is required; TensorFold installs only with npm ci.')
  })
  for (const other of ['pnpm-lock.yaml', 'yarn.lock', 'bun.lock', 'bun.lockb']) {
    await access(path.join(root, other)).then(() => {
      throw new Error(`${other} is not supported. Keep package-lock.json as the only package-manager lock.`)
    }).catch((error) => {
      if (error?.message?.includes('is not supported')) throw error
    })
  }
  const npmrc = await readFile(path.join(root, '.npmrc'), 'utf8').catch(() => '')
  if (!/^ignore-scripts=true$/m.test(npmrc)) {
    throw new Error('.npmrc must set ignore-scripts=true; dependency lifecycle scripts are denied by default.')
  }
  return { pkg, declared, version: declared.slice(4) }
}

export function assertNpmInvocation(env, expectedVersion) {
  const userAgent = String(env.npm_config_user_agent || '')
  const match = /^npm\/([^\s]+)/.exec(userAgent)
  if (!match) throw new Error('TensorFold dependencies must be installed with npm, not pnpm/yarn/bun. Run: npm ci')
  if (match[1] !== expectedVersion) {
    throw new Error(`TensorFold requires npm ${expectedVersion}; received npm ${match[1]}. Enable Corepack/use the packageManager pin, then run npm ci.`)
  }
  const execPath = String(env.npm_execpath || '')
  if (!/(?:^|[/\\])npm-cli\.js$/.test(execPath) && !/[/\\]corepack[/\\].*[/\\]npm\.js$/.test(execPath)) {
    throw new Error(`Unexpected npm executable (${execPath || 'missing npm_execpath'}). Run the pinned npm CLI directly or through Corepack; shims from another manager are not supported.`)
  }
}

export async function assertInstallTreeNotMixed(root) {
  const modules = path.join(root, 'node_modules')
  const entries = await readdir(modules).catch(() => null)
  if (!entries) return
  if (entries.includes('.pnpm') || entries.includes('.modules.yaml') || entries.some((entry) => entry.startsWith('.ignored_'))) {
    throw mixedTreeError()
  }
  const nested = path.join(modules, 'node_modules')
  const stat = await lstat(nested).catch(() => null)
  if (stat?.isSymbolicLink()) {
    const [target, own] = await Promise.all([realpath(nested).catch(() => ''), realpath(modules).catch(() => '')])
    if (target && target === own) throw mixedTreeError()
  }
}

function mixedTreeError() {
  return new Error('node_modules contains pnpm/mixed-manager or self-symlink artifacts. Move/remove that install and run a clean `npm ci`; TensorFold will not build against a hybrid tree.')
}

async function main() {
  const root = process.cwd()
  const repo = await assertRepoPackageManager(root)
  if (!process.argv.includes('--repo-only')) {
    assertNpmInvocation(process.env, repo.version)
    await assertInstallTreeNotMixed(root)
  }
  console.log(`package manager check passed (${repo.declared})`)
}

const invoked = process.argv[1] ? pathToFileURL(path.resolve(process.argv[1])).href : ''
if (invoked === import.meta.url) {
  main().catch((error) => {
    console.error(`[package-manager] ${error?.message || error}`)
    process.exitCode = 1
  })
}
