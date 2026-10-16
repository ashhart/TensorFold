// Run: jiti src/lib/work-pane.test.ts
//
// Work pane rules: tabs per thread, the default tab, and when the pane is open.
import assert from 'node:assert/strict'
import { defaultWorkPaneTab, resolveWorkPaneTab, workPaneOpen, workPaneTabsFor } from './work-pane'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('work-pane tests:')

{
  assert.deepEqual(workPaneTabsFor(true), ['changes', 'terminal', 'files', 'preview', 'details'])
  assert.deepEqual(workPaneTabsFor(false), ['preview', 'details'])
  ok('folder tabs only exist for threads with a folder')
}

{
  assert.equal(defaultWorkPaneTab({ hasFolder: true, hasChanges: true }), 'changes')
  assert.equal(defaultWorkPaneTab({ hasFolder: true, hasChanges: false }), 'files')
  assert.equal(defaultWorkPaneTab({ hasFolder: false, hasChanges: false }), 'details')
  ok('default is Changes with changes, otherwise Files; Details without a folder')
}

{
  assert.equal(resolveWorkPaneTab('terminal', { hasFolder: true, hasChanges: true }), 'terminal')
  assert.equal(resolveWorkPaneTab('terminal', { hasFolder: false, hasChanges: false }), 'details')
  assert.equal(resolveWorkPaneTab(null, { hasFolder: true, hasChanges: false }), 'files')
  ok('a picked tab wins while it is available')
}

{
  assert.equal(workPaneOpen({ threadId: 'a', hasFolder: true, rightPanelHidden: false, chatPaneOpenId: null }), true)
  assert.equal(workPaneOpen({ threadId: 'a', hasFolder: true, rightPanelHidden: true, chatPaneOpenId: 'a' }), false)
  assert.equal(workPaneOpen({ threadId: 'a', hasFolder: false, rightPanelHidden: false, chatPaneOpenId: null }), false)
  assert.equal(workPaneOpen({ threadId: 'a', hasFolder: false, rightPanelHidden: true, chatPaneOpenId: 'a' }), true)
  assert.equal(workPaneOpen({ threadId: 'b', hasFolder: false, rightPanelHidden: false, chatPaneOpenId: 'a' }), false)
  ok('folder threads follow the toggle; chats without a folder start closed')
}

console.log(`\n${passed} work-pane tests passed`)
