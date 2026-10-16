// Markdown link override: chat links open via the preload bridge's openExternal
// (main-process scheme check) instead of an in-window navigation the
// will-navigate guard would silently block.
import assert from 'node:assert/strict'

// Browser shim (installed BEFORE importing the module under test, which pulls
// in ./ipc and picks its transport at import time).
const opened: string[] = []
Object.defineProperty(globalThis, 'window', {
  configurable: true,
  value: {
    vontra: { openExternal: async (url: string) => { opened.push(url) } },
    dispatchEvent: () => true,
  },
})

const { MarkdownLink } = await import('./markdown-link')

{
  const el = MarkdownLink({ href: 'https://example.com/docs', children: 'docs' })
  assert.equal(el.type, 'a')
  assert.equal(el.props.href, 'https://example.com/docs')
  let prevented = false
  el.props.onClick({ preventDefault: () => { prevented = true } })
  assert.equal(prevented, true)
  assert.deepEqual(opened, ['https://example.com/docs'])
  console.log('  ✓ link click calls openExternal and prevents navigation')
}

{
  opened.length = 0
  const el = MarkdownLink({ href: undefined, children: 'bare' })
  let prevented = false
  el.props.onClick({ preventDefault: () => { prevented = true } })
  assert.equal(prevented, true)
  assert.deepEqual(opened, [])
  console.log('  ✓ href-less anchor is swallowed without opening anything')
}

console.log('markdown link tests passed')
