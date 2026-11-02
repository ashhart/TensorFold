import assert from 'node:assert/strict'
import { createAskShortcut } from './shortcut'

const registered = new Map<string, () => void>()
let toggles = 0
const shortcut = createAskShortcut({ register: (key, callback) => { if (key === 'Control+X') return false; registered.set(key, callback); return true }, unregister: key => { registered.delete(key) } }, () => toggles++)
shortcut.apply({ enabled: false, shortcut: 'Control+Space' })
assert.equal(registered.size, 0)
shortcut.apply({ enabled: true, shortcut: 'Control+Space' }); registered.get('Control+Space')!()
assert.equal(toggles, 1)
assert.throws(() => shortcut.apply({ enabled: true, shortcut: 'Control+X' }), /already in use/)
assert.equal(registered.has('Control+Space'), true, 'A rejected replacement preserves the working shortcut')
shortcut.apply({ enabled: true, shortcut: 'Control+Shift+Space' })
assert.equal(registered.has('Control+Space'), false)
assert.equal(registered.has('Control+Shift+Space'), true)
shortcut.apply({ enabled: false, shortcut: 'Control+Shift+Space' })
assert.equal(registered.size, 0)
shortcut.dispose(); assert.equal(registered.size, 0)
console.log('Quick Ask shortcut conflict, replacement and teardown checks passed')
