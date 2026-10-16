import assert from 'node:assert/strict'
import { installDefaultTrustedTypesPolicy } from './trusted-types'

let rules: any
const scope = {
  location: { href: 'file:///Applications/TensorFold Studio/dist/index.html' },
  trustedTypes: { createPolicy: (name: string, value: any) => { assert.equal(name, 'default'); rules = value } },
}
assert.equal(installDefaultTrustedTypesPolicy(scope), true)
// Fail closed: every HTML sink throws, even benign-looking markup. No regex
// "sanitizing" that a payload like `java\tscript:` could slip through.
assert.throws(() => rules.createHTML('<div>safe</div>'), /Blocked HTML sink/)
assert.throws(() => rules.createHTML('<img onerror=alert(1)>'), /Blocked HTML sink/)
assert.throws(() => rules.createHTML('<a href="java\tscript:alert(1)">x</a>'), /Blocked HTML sink/)
assert.throws(() => rules.createScript('alert(1)'), /Inline script/)
assert.match(rules.createScriptURL('./assets/app.js'), /^file:/)
assert.throws(() => rules.createScriptURL('https://evil.example/app.js'), /Cross-origin/)
assert.equal(installDefaultTrustedTypesPolicy({}), false)
// An existing default policy is left alone (createPolicy would throw on a duplicate).
assert.equal(installDefaultTrustedTypesPolicy({
  trustedTypes: { createPolicy: () => { throw new Error('must not re-register') }, getPolicyNames: () => ['default'] },
}), true)
console.log('trusted types policy tests passed')
