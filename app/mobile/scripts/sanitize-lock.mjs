import { readFile, writeFile } from 'node:fs/promises'
const path = new URL('../package-lock.json', import.meta.url)
const source = await readFile(path, 'utf8')
const lock = JSON.parse(source)
let changed = false
for (const entry of Object.values(lock.packages ?? {})) {
  if (typeof entry.deprecated !== 'string') continue
  // Registry deprecation prose is dispensable metadata; preserve the warning
  // while omitting its trailing commercial-support contact sentence.
  const warning = entry.deprecated.replace(/\. Support for old versions may be purchased .*$/, '.')
  if (warning !== entry.deprecated) { entry.deprecated = warning; changed = true }
}
if (changed && process.argv.includes('--check')) { console.error('Run npm run lock:sanitize to remove registry contact metadata.'); process.exitCode = 1 }
else if (changed) await writeFile(path, JSON.stringify(lock, null, 2) + '\n')
