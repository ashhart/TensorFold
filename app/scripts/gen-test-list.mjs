import { execSync } from 'node:child_process'
import { readdirSync, statSync } from 'node:fs'

const ignoredDirs = new Set(['.git', '.claude', '.vite', 'build', 'dist', 'dist-electron', 'node_modules', 'release', '__pycache__'])

function walk(dir) {
  const out = []
  for (const name of readdirSync(dir)) {
    if (ignoredDirs.has(name)) continue
    const full = `${dir}/${name}`
    if (statSync(full).isDirectory()) {
      out.push(...walk(full))
    } else if (/\.(test\.ts|test\.tsx|test\.mjs)$/.test(name)) {
      out.push(full)
    }
  }
  return out
}

const tests = [...walk('electron'), ...walk('src'), ...walk('scripts')].sort()
console.log('jiti ' + tests.join(' && jiti '))
