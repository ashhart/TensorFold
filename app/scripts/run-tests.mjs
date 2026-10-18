// Run all *.test.ts, *.test.tsx, *.test.mjs files through jiti in isolated
// child processes. Isolation matters: several tests intentionally modify globals,
// process.env, module caches, or process.exitCode.
import { readdirSync, statSync } from 'node:fs'
import { createRequire } from 'node:module'
import { spawnSync } from 'node:child_process'
import path from 'node:path'

const root = path.resolve(process.cwd())
const ignoredDirs = new Set(['.git', '.claude', '.vite', 'build', 'dist', 'dist-electron', 'node_modules', 'release', '__pycache__'])

function walk(dir) {
  const out = []
  for (const name of readdirSync(dir)) {
    if (ignoredDirs.has(name)) continue
    const full = path.join(dir, name)
    if (statSync(full).isDirectory()) {
      out.push(...walk(full))
    } else if (/\.(test\.ts|test\.tsx|test\.mjs)$/.test(name)) {
      out.push(full)
    }
  }
  return out
}

const require = createRequire(import.meta.url)
const jitiPackage = require.resolve('jiti/package.json')
const jitiCli = path.join(path.dirname(jitiPackage), 'lib', 'jiti-cli.mjs')

const testRoots = ['electron', 'src', 'scripts', 'experiments']
const tests = testRoots.flatMap((dir) => walk(path.join(root, dir))).sort()
const total = tests.length
let passed = 0
let failed = 0
const failures = []

for (const file of tests) {
  const rel = path.relative(root, file)
  const result = spawnSync(process.execPath, [jitiCli, file], {
    cwd: root,
    env: process.env,
    encoding: 'utf8',
    maxBuffer: 10 * 1024 * 1024,
    timeout: 120_000,
  })
  if (result.status === 0 && !result.error) {
    passed++
    process.stdout.write('.')
  } else {
    failed++
    const output = [result.stdout, result.stderr].filter(Boolean).join('\n').trim()
    const reason = result.error?.message
      ?? (result.signal ? `terminated by ${result.signal}` : '')
      ?? `exit ${result.status ?? 'unknown'}`
    failures.push({ file: rel, error: output || reason })
    process.stdout.write('F')
  }
}

console.log()
console.log(`${passed}/${total} passed, ${failed} failed`)

if (failed > 0) {
  console.log()
  for (const f of failures) {
    console.log(`  FAIL ${f.file}`)
    console.log(f.error.split('\n').map((line) => `    ${line}`).join('\n'))
  }
  process.exit(1)
}
