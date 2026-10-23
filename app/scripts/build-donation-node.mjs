import { build } from 'esbuild'
import { mkdir, writeFile } from 'node:fs/promises'
import { createHash } from 'node:crypto'
const result = await build({ entryPoints: ['electron/donation-node-main.ts'], bundle: true, platform: 'node', target: 'node22', format: 'cjs', outfile: 'dist-electron/donation-node.js', write: false, sourcemap: false, metafile: true, legalComments: 'none', minify: true })
for (const output of Object.values(result.metafile.outputs)) for (const item of output.imports) if (item.external && !item.path.startsWith('node:')) throw new Error(`Unexpected donor runtime dependency: ${item.path}`)
const bytes = result.outputFiles[0].contents
if (bytes.length > 512 * 1024) throw new Error('Donor bundle exceeds the reviewed transfer limit.')
await mkdir('dist-electron', { recursive: true })
await writeFile('dist-electron/donation-node.js', bytes)
console.log(`Donation node: ${bytes.length} bytes; sha256 ${createHash('sha256').update(bytes).digest('hex')}`)
