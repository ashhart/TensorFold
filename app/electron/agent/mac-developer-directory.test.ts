import assert from 'node:assert/strict'
import { resolveDeveloperReadRoot } from './mac-developer-directory'

const link = '/var/db/xcode_select_link'
const clt = '/Library/Developer/CommandLineTools'
const selected = '/Applications/Selected Xcode.app/Contents/Developer'
let queries = 0
const io = (paths: Record<string, string>, choice: string | null = selected) => ({
  realpath(value: string) {
    if (!(value in paths)) throw new Error('Missing fixture path')
    return paths[value]
  },
  selectedPath() { queries++; return choice },
})

assert.equal(resolveDeveloperReadRoot(io({ [selected]: selected, [clt]: clt })), '/Applications/Selected Xcode.app')
assert.equal(queries, 1, 'missing legacy link asks xcode-select before choosing installed Command Line Tools')
queries = 0
assert.equal(resolveDeveloperReadRoot(io({ [link]: selected })), '/Applications/Selected Xcode.app')
assert.equal(queries, 0, 'an existing selection link keeps the direct lookup')
assert.equal(resolveDeveloperReadRoot(io({ [clt]: clt }, clt)), clt)
assert.equal(resolveDeveloperReadRoot(io({ [clt]: clt }, null)), clt)
assert.equal(resolveDeveloperReadRoot(io({ [clt]: clt }, '/missing/Developer')), clt)
assert.equal(resolveDeveloperReadRoot(io({ [clt]: clt }, 'relative/Developer')), clt)
assert.equal(resolveDeveloperReadRoot(io({}, null)), null)
assert.equal(resolveDeveloperReadRoot({ ...io({ [clt]: clt }), selectedPath() { throw new Error('Unavailable') } }), clt)
assert.equal(resolveDeveloperReadRoot(io({ [selected]: '/opt/developer' })), '/opt/developer')
console.log('developer directory discovery: 9 cases passed')
