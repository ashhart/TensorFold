import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
import { readFileSync, existsSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
const mobileRoot = fileURLToPath(new URL('../mobile/', import.meta.url))
const require = createRequire(path.join(mobileRoot, 'package.json'))
const { getDefaultConfig } = require('expo/metro-config')
const baseline = getDefaultConfig(mobileRoot)
const configured = require(path.join(mobileRoot, 'metro.config.js'))
const shared = path.resolve(mobileRoot, '../shared')
assert.ok(configured.watchFolders.includes(shared), 'actual shared runtime source is visible to Metro')
for (const folder of baseline.watchFolders) assert.ok(configured.watchFolders.includes(folder), 'preserves Expo discovered workspaces')
assert.deepEqual(configured.resolver.nodeModulesPaths, baseline.resolver.nodeModulesPaths)
assert.deepEqual(configured.resolver.sourceExts, baseline.resolver.sourceExts)
assert.equal(configured.resolver.disableHierarchicalLookup, baseline.resolver.disableHierarchicalLookup)
assert.equal(configured.transformer.babelTransformerPath, baseline.transformer.babelTransformerPath)
const facade = readFileSync(path.join(mobileRoot, 'src/lib/watch-relay.ts'), 'utf8')
for (const [, reference] of facade.matchAll(/from ['"]([^'"]*shared\/watch-relay)['"]/g)) {
  assert.ok(existsSync(path.resolve(mobileRoot, 'src/lib', reference + '.ts')), 'facade import resolves to the source actually included in Metro')
}
const runtime = readFileSync(path.join(shared, 'watch-relay.ts'), 'utf8')
assert.doesNotMatch(runtime, /^import .*from ['"](?:node:|.*mobile\/|.*electron\/|react)/m, 'neutral runtime cannot pull host or phone native modules into the other bundle')
console.log('Mobile shared watch runtime resolves while preserving Expo resolver and transformer defaults')
