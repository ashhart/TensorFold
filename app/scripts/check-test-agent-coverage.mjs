// Verify that every *.test.ts/tsx/mjs file is discoverable by the test runner.
// The runner (scripts/run-tests.mjs) auto-discovers test files by walking electron/, src/, and scripts/,
// so we just need to confirm no test files exist outside those directories.
import { readdirSync, statSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { join, relative } from 'node:path'

const root = fileURLToPath(new URL('..', import.meta.url))
const ignoredDirs = new Set([
  '.git',
  '.claude',
  '.research',
  '.vite',
  'build',
  'dist',
  'dist-electron',
  'node_modules',
  'release',
  '__pycache__',
])
const scannedDirs = ['electron', 'src', 'scripts', 'experiments']

function walk(dir) {
  const out = []
  for (const name of readdirSync(dir)) {
    if (ignoredDirs.has(name)) continue
    const full = join(dir, name)
    const rel = relative(root, full).split('\\').join('/')
    if (rel === 'mobile/ios' || rel === 'mobile/android') continue // Expo-generated projects and vendored native dependencies.
    const st = statSync(full)
    if (st.isDirectory()) {
      out.push(...walk(full))
    } else if (/\.test\.(?:tsx?|mjs)$/.test(name)) {
      out.push(relative(root, full).split('\\').join('/'))
    }
  }
  return out
}

// All test files
const allTests = walk(root).sort()
// Test files inside scanned directories (electron/ or src/)
const scannedTests = allTests.filter((f) => scannedDirs.some((d) => f.startsWith(d + '/'))).sort()
const orphans = allTests.filter((f) => !scannedTests.includes(f))

if (orphans.length) {
  console.error(`Test files found outside ${scannedDirs.join(', ')} (not picked up by runner):`)
  for (const file of orphans) console.error(`  - ${file}`)
  process.exit(1)
}

console.log(`Runner covers ${scannedTests.length} test files across ${scannedDirs.join(' & ')}/.`)
