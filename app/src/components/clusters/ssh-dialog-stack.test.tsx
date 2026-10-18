import assert from 'node:assert/strict'
import clsx from 'clsx'
import { installFixtureDom } from '../navigation/accessibility-fixture'
import { find, harness, jsx, load, nodes, text } from '../reading/reading-test-harness'
const dom = installFixtureDom()
;(dom.body as any).style = { overflow: 'scroll' }
dom.doc.addEventListener = dom.win.addEventListener.bind(dom.win)
dom.doc.removeEventListener = dom.win.removeEventListener.bind(dom.win)
try {
  const drawerHooks = harness(), sshHooks = harness()
  sshHooks.hooks.useLayoutEffect = sshHooks.hooks.useEffect
  const dialogs = load('src/components/navigation/useAccessibleDialog.ts', { react: sshHooks.hooks })
  const { ModelsDrawer } = load('src/components/models/ModelsDrawer.tsx', { react: drawerHooks.hooks, 'react/jsx-runtime': jsx, 'react-dom': { createPortal: (node: any) => node }, '../ui': { Button: 'button' }, '../navigation/useAccessibleDialog': dialogs })
  const state: any = { sshTrustPrompt: null, sshSecretPrompt: null, resolveSshTrust: (value: boolean) => decisions.push(value), resolveSshSecret: (value: string | null) => secrets.push(value) }
  const decisions: boolean[] = [], secrets: Array<string | null> = []
  const { SshPrompts } = load('src/components/SshPrompts.tsx', { react: sshHooks.hooks, 'react/jsx-runtime': jsx, 'react-dom': { createPortal: (node: any) => node }, './navigation/useAccessibleDialog': dialogs,
    '../lib/store': { useStore: (select: any) => select(state) }, '../lib/icons': { Icon: 'icon' }, clsx })
  let closed = 0, blocked = false
  const renderDrawer = () => drawerHooks.render(() => ModelsDrawer({ title: 'Compute and clusters', labelId: 'test-clusters', closeDisabled: blocked, onClose: () => { closed++ }, children: jsx.jsx('button', { id: 'test-inspect', children: 'Inspect' }, undefined) }))
  dom.trigger.focus(); dom.mount(renderDrawer()); drawerHooks.flush()
  const inspect = dom.doc.getElementById('test-inspect')!; inspect.focus()
  state.sshTrustPrompt = { promptId: 'synthetic-trust', host: 'example.invalid', port: 22, keyType: 'ssh-ed25519', fingerprintSha256: 'synthetic-fingerprint', changed: false }
  const trust = sshHooks.render(SshPrompts), trustNodes = dom.mount(trust); sshHooks.flush()
  assert.equal(String(find(trust, node => node.props?.role === 'dialog').props['aria-modal']), 'true')
  assert.equal(dom.doc.activeElement.tagName, 'button')
  assert.equal(dialogs.hasAccessibleDialogAbove(find(renderDrawer(), node => node.props?.role === 'dialog').ref.current), true)
  const cancel = find(trust, node => node.type === 'button' && text(node) === 'Cancel').ref.current
  assert.equal(dom.doc.activeElement, cancel, 'An unknown SSH key defaults focus to refusal, not trust')
  dom.listeners.filter(entry => entry.name === 'focusin').forEach(entry => entry.callback({ target: cancel }))
  assert.equal(dom.doc.activeElement, cancel, 'The underlying drawer cannot steal security-dialog focus')
  assert.equal(dom.key('Tab').prevented, true)
  dom.key('Escape'); assert.deepEqual(decisions, [false]); assert.equal(closed, 0, 'Escape cancels only the SSH approval, never its underlying workflow')
  state.sshTrustPrompt = null; sshHooks.render(SshPrompts); sshHooks.flush()
  assert.equal(dom.doc.activeElement, inspect, 'Closing the SSH dialog restores its drawer control')
  dom.body.children = dom.body.children.filter(child => !trustNodes.includes(child))
  blocked = true; renderDrawer(); dom.key('Escape'); assert.equal(closed, 0, 'A pending cluster operation still blocks drawer dismissal')
  blocked = false; renderDrawer(); dom.key('Escape'); assert.equal(closed, 1)
  state.sshSecretPrompt = { promptId: 'synthetic-secret', kind: 'password', label: 'Synthetic connection' }
  const secret = sshHooks.render(SshPrompts); dom.mount(secret); sshHooks.flush()
  const password = find(secret, node => node.type === 'input'); assert.equal(password.props.type, 'password')
  dom.key('Escape'); assert.deepEqual(secrets, [null], 'Escape never submits a password or accepts host trust')
  state.sshSecretPrompt = null; sshHooks.render(SshPrompts); sshHooks.flush(); sshHooks.cleanup(); drawerHooks.cleanup()
  assert.equal((dom.body as any).style.overflow, 'scroll')
  assert.equal(dom.doc.activeElement, dom.trigger)
  console.log('Actual ModelsDrawer and global SSH prompts pass stacked focus, default refusal, Escape ownership, focus restoration, busy close guard and password cancellation without sending credentials')
} finally { dom.restore() }
