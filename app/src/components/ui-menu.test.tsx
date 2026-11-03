import assert from 'node:assert/strict'
import clsx from 'clsx'
import { find, harness, jsx, load } from './reading/reading-test-harness'

const originalDocument = Object.getOwnPropertyDescriptor(globalThis, 'document')
const originalWindow = Object.getOwnPropertyDescriptor(globalThis, 'window')
const listeners = new Map<string, (event: any) => void>()
Object.defineProperty(globalThis, 'document', { configurable: true, value: {
  body: {}, addEventListener: (name: string, callback: any) => listeners.set(name, callback),
  removeEventListener: (name: string, callback: any) => { if (listeners.get(name) === callback) listeners.delete(name) },
} })
Object.defineProperty(globalThis, 'window', { configurable: true, value: {
  innerWidth: 1024, innerHeight: 768,
  requestAnimationFrame: () => 1, cancelAnimationFrame() {}, addEventListener() {}, removeEventListener() {},
} })

try {
  for (const controlled of [false, true]) {
    const hooks = harness()
    hooks.hooks.useCallback = (callback: any) => callback
    hooks.hooks.useLayoutEffect = hooks.hooks.useEffect
    const { Menu } = load('src/components/ui.tsx', {
      react: hooks.hooks, 'react/jsx-runtime': jsx, 'react-dom': { createPortal: (node: any) => node }, clsx,
      '../lib/icons': { Icon: 'icon-fixture' },
    })
    let open = false, focus = 0, content: any, triggerOpen = false
    const changes: boolean[] = []
    const props = {
      ...(controlled ? { onOpenChange: (value: boolean) => { changes.push(value); open = value } } : {}),
      trigger: (value: boolean) => { triggerOpen = value; return jsx.jsx('button', { children: 'Choose model' }, null) },
      children: (close: () => void) => { content = { close }; return jsx.jsx('div', { role: 'dialog' }, null) },
    }
    if (controlled) Object.defineProperty(props, 'open', { get: () => open })
    const render = () => {
      const tree = hooks.render(() => Menu(props))
      tree.ref.current = { contains: (node: unknown) => node === 'trigger', querySelector: () => ({ focus: () => focus++ }),
        getBoundingClientRect: () => ({ top: 600, bottom: 632, left: 400, right: 600 }) }
      hooks.flush()
      return tree
    }
    let tree = render()
    assert.equal(triggerOpen, false)
    assert.equal(find(tree, node => node.props.role === 'dialog'), undefined)
    find(tree, node => node.type === 'div' && node.props.onClick).props.onClick()
    tree = render()
    assert.equal(triggerOpen, true)
    assert.ok(find(tree, node => node.props.role === 'dialog'))
    listeners.get('mousedown')!({ target: 'trigger' })
    assert.ok(find(render(), node => node.props.role === 'dialog'), 'Trigger clicks are not mistaken for outside clicks')
    let prevented = false
    listeners.get('keydown')!({ key: 'Escape', defaultPrevented: false, preventDefault() { prevented = true } })
    tree = render()
    assert.ok(prevented)
    assert.equal(focus, 1, 'Keyboard dismissal restores focus to the model trigger')
    assert.equal(find(tree, node => node.props.role === 'dialog'), undefined)
    if (controlled) open = true
    else find(tree, node => node.type === 'div' && node.props.onClick).props.onClick()
    tree = render()
    assert.ok(find(tree, node => node.props.role === 'dialog'), 'A controlled menu can open from another composer button')
    content.close()
    tree = render()
    assert.equal(find(tree, node => node.props.role === 'dialog'), undefined, 'Successful selection closes the controlled menu')
    find(tree, node => node.type === 'div' && node.props.onClick).props.onClick()
    render()
    listeners.get('mousedown')!({ target: 'outside' })
    tree = render()
    assert.equal(find(tree, node => node.props.role === 'dialog'), undefined, 'Outside dismissal updates the owner of a controlled menu')
    if (controlled) assert.deepEqual(changes, [true, false, false, true, false])
    hooks.cleanup()
    assert.equal(listeners.size, 0)
  }
} finally {
  if (originalDocument) Object.defineProperty(globalThis, 'document', originalDocument)
  else delete (globalThis as any).document
  if (originalWindow) Object.defineProperty(globalThis, 'window', originalWindow)
  else delete (globalThis as any).window
}
console.log('Actual controlled and standalone menus preserve trigger toggling, external opening, selection close, Escape focus and outside dismissal.')
