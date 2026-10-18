import assert from 'node:assert/strict'
import { composerBudget } from '../../lib/composer-budget'
import { find, harness, jsx, load, text } from '../reading/reading-test-harness'
const hooks = harness()
const { ContextBudget } = load('src/components/composer/ContextBudget.tsx', { react: hooks.hooks, 'react/jsx-runtime': jsx })
const budget = composerBudget({ text: 'x'.repeat(1000), attachments: [{ id: 'image', name: 'one.png', kind: 'image', mime: 'image/png', status: 'needs-reselection' }], messages: [], windowTokens: 1024, windowVerified: true, running: true })
const render = () => hooks.render(() => ContextBudget({ budget }))
let result = render()
assert.ok(text(result).includes('Partial'))
assert.equal(find(result, node => node.type === 'section'), undefined)
find(result, node => node.type === 'button').props.onClick()
result = render()
assert.ok(text(result).includes('Current draft')); assert.ok(text(result).includes('Reserved output')); assert.ok(text(result).includes('Not reported'))
assert.ok(text(result).includes('snapshot is stale')); assert.ok(text(result).includes('Image token costs are unknown'))
assert.ok(text(result).includes('exceeds the selected context window'))
assert.ok(text(result).includes('does not guarantee admission'))
hooks.cleanup()
console.log('Actual context-budget disclosure exposes partial, stale, unready-image, missing reply-limit and over-window conditions without claiming exact admission')
