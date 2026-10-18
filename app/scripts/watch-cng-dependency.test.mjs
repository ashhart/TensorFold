import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
const require = createRequire(import.meta.url)
const mobileRequire = createRequire(new URL('../mobile/package.json', import.meta.url))
const xcode = mobileRequire('xcode')
const { ensureWatchTargetDependency } = require('../mobile/plugins/with-tensorfold-watch.cjs')
assert.equal(typeof ensureWatchTargetDependency, 'function', 'CNG must explicitly install the host-to-watch build dependency even when Expo generated no dependency sections')
const project = xcode.project('synthetic.pbxproj')
project.hash = {project: {rootObject: 'PROJECT', rootObject_comment: 'Project object', objects: {
  PBXNativeTarget: {
    HOST: {isa: 'PBXNativeTarget', name: 'TensorFold', dependencies: [{value: 'CUSTOM', comment: 'PBXTargetDependency'}]}, HOST_comment: 'TensorFold',
    WATCH: {isa: 'PBXNativeTarget', name: 'TensorFoldWatch', dependencies: []}, WATCH_comment: 'TensorFoldWatch',
    EXTRA: {isa: 'PBXNativeTarget', name: 'CustomExtension', dependencies: []}, EXTRA_comment: 'CustomExtension',
  },
}}}
// The xcode package silently skips addTargetDependency when both sections are absent.
project.addTargetDependency('HOST', ['WATCH'])
assert.equal(project.hash.project.objects.PBXNativeTarget.HOST.dependencies.length, 1, 'fixture reproduces the generated-project omission')
ensureWatchTargetDependency(project, 'HOST', 'WATCH')
const objects = project.hash.project.objects
const added = objects.PBXNativeTarget.HOST.dependencies.find(item => item.value !== 'CUSTOM')
assert.ok(added)
assert.equal(objects.PBXTargetDependency[added.value].target, 'WATCH')
const proxy = objects.PBXContainerItemProxy[objects.PBXTargetDependency[added.value].targetProxy]
assert.equal(proxy.remoteGlobalIDString, 'WATCH')
assert.equal(proxy.containerPortal, 'PROJECT')
assert.deepEqual(objects.PBXNativeTarget.WATCH.dependencies, [], 'watch cannot depend backwards on the iPhone')
const first = JSON.stringify(project.hash)
ensureWatchTargetDependency(project, 'HOST', 'WATCH')
assert.equal(JSON.stringify(project.hash), first, 'repeated CNG preserves existing/custom dependencies without duplicating the watch')
assert.ok(objects.PBXNativeTarget.HOST.dependencies.some(item => item.value === 'CUSTOM'))
assert.throws(() => ensureWatchTargetDependency(project, 'MISSING', 'WATCH'), /companion target/)
assert.throws(() => ensureWatchTargetDependency(project, 'HOST', 'HOST'), /distinct/)
console.log('Watch CNG dependency: actual xcode absent-section regression, embedding graph, direction, custom dependencies and idempotency passed.')
