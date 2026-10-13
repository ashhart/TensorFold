// Run: jiti electron/agent/remote-static.test.ts
//
// Static renderer serving (VST-035): traversal guards, SPA fallback, content types,
// cache policy. Injected fs — no disk.
import assert from 'node:assert/strict'
import path from 'node:path'
import { loadStaticFile, resolveStaticPath, staticContentType, type StaticFs } from './remote-static'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('remote-static tests:')

const ROOT = path.resolve('/srv/dist')

function fakeFs(files: Record<string, string>): StaticFs {
  const table = new Map(Object.entries(files).map(([p, content]) => [path.resolve(ROOT, p), content]))
  return {
    async stat(filePath) {
      if (!table.has(path.resolve(filePath))) throw new Error('ENOENT')
      return { isFile: () => true }
    },
    async lstat(filePath) {
      if (!table.has(path.resolve(filePath))) throw new Error('ENOENT')
      return { isFile: () => true }
    },
    async readFile(filePath) {
      const content = table.get(path.resolve(filePath))
      if (content == null) throw new Error('ENOENT')
      return Buffer.from(content)
    },
  }
}

{
  assert.equal(resolveStaticPath(ROOT, '/'), path.join(ROOT, 'index.html'))
  assert.equal(resolveStaticPath(ROOT, '/assets/app.js'), path.join(ROOT, 'assets', 'app.js'))
  assert.equal(resolveStaticPath(ROOT, '/../secrets.txt'), null, '.. traversal is refused')
  assert.equal(resolveStaticPath(ROOT, '/%2e%2e/%2e%2e/etc/passwd'), null, 'encoded traversal is refused')
  assert.equal(resolveStaticPath(ROOT, '/a%00.js'), null, 'NUL bytes are refused')
  assert.equal(resolveStaticPath('', '/index.html'), null, 'no root → serving disabled')
  ok('resolveStaticPath: root-confined, traversal and NUL refused')
}

{
  assert.equal(staticContentType('index.html'), 'text/html; charset=utf-8')
  assert.equal(staticContentType('assets/app.3fa9c2d1.js'), 'text/javascript; charset=utf-8')
  assert.equal(staticContentType('style.css'), 'text/css; charset=utf-8')
  assert.equal(staticContentType('unknown.bin'), 'application/octet-stream')
  ok('staticContentType: common renderer asset types')
}

{
  const fs = fakeFs({
    'index.html': '<html>vontra</html>',
    'assets/app.3fa9c2d1.js': 'console.log(1)',
    'vite.svg': '<svg/>',
  })
  const index = await loadStaticFile(ROOT, '/', fs)
  assert.equal(index?.content.toString(), '<html>vontra</html>')
  assert.equal(index?.cacheControl, 'no-cache', 'index must revalidate')

  const asset = await loadStaticFile(ROOT, '/assets/app.3fa9c2d1.js', fs)
  assert.equal(asset?.contentType, 'text/javascript; charset=utf-8')
  assert.match(asset?.cacheControl ?? '', /immutable/, 'hashed assets cache hard')

  const spa = await loadStaticFile(ROOT, '/settings', fs)
  assert.equal(spa?.content.toString(), '<html>vontra</html>', 'client routes fall back to index.html')

  const missingAsset = await loadStaticFile(ROOT, '/assets/gone.js', fs)
  assert.equal(missingAsset, null, 'a missing ASSET 404s instead of serving HTML')

  const escape = await loadStaticFile(ROOT, '/../../etc/passwd', fs)
  assert.equal(escape, null)
  ok('loadStaticFile: index, hashed assets, SPA fallback for routes only')
}

console.log(`  ${passed} passed`)
