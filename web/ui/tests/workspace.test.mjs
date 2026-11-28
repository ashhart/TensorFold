import assert from 'node:assert/strict'
import { test } from 'node:test'
import { routeOf, modelTab, SHORTCUTS, loopbackEngine } from '../src/navigation.ts'
import { createServer } from 'vite'
import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

test('opening and old Home links start a new chat; old feature shortcuts survive', () => {
  for (const hash of ['', '#', '#home', '#unknown']) assert.equal(routeOf(hash), 'chat')
  assert.equal(routeOf('#chat/saved'), 'chat')
  for (const address of ['http://localhost:8080', 'http://127.0.0.1:8080', 'http://127.0.0.2', 'http://[::1]:8080']) assert.equal(loopbackEngine(address), true)
  for (const address of [undefined, 'bad address', 'http://engine.example:8080']) assert.equal(loopbackEngine(address), false)
  assert.deepEqual(SHORTCUTS, ['chat', 'models', 'chat', 'teach', 'connect', 'bench', 'monitor', 'fleet', 'settings'])
  for (const [hash, tab] of [['#models', 'mine'], ['#models/discover', 'discover'], ['#models/learning', 'learning'], ['#teach', 'learning'], ['#models/api', 'api'], ['#connect', 'api'], ['#models/owner/checkpoint', 'mine']]) assert.equal(modelTab(hash), tab)
})

test('actual workspace components render the desktop hierarchy and retained feature links', async () => {
  const local = new Map()
  globalThis.localStorage = { getItem: (k) => local.get(k) ?? null, setItem: (k, v) => local.set(k, v) }
  globalThis.location = new URL('http://localhost/')
  globalThis.matchMedia = () => ({ matches: false })
  const server = await createServer({ server: { middlewareMode: true, watch: null }, appType: 'custom' })
  try {
    const App = (await server.ssrLoadModule('/src/App.tsx')).default
    let html = renderToStaticMarkup(React.createElement(App))
    assert.doesNotMatch(html, /<section[^>]*aria-label="Conversations"/)
    assert.match(html, /role="log" aria-label="Conversation messages" aria-busy="false"/)
    assert.match(html, /aria-label="Search conversations"/)
    assert.match(html, /> New chat</)
    assert.match(html, /aria-label="Message"/)
    assert.match(html, /aria-describedby="chat-engine-status"/)
    assert.match(html, /aria-label="Hide sidebar"/)
    assert.doesNotMatch(html, />Home<|Fold the menu|class="starters"|<h1>Chat<|class="brand"/)
    const Workspace = (await server.ssrLoadModule('/src/ModelWorkspace.tsx')).default
    const noop = () => {}
    const ctx = { engine: null, machine: null, settings: null, toast: noop, refreshEngine: noop }
    globalThis.location.hash = '#models'
    html = renderToStaticMarkup(React.createElement(Workspace, ctx))
    for (const label of ['My models', 'Discover', 'Learning', 'API', 'Benchmarks', 'Compute and clusters', 'Details']) assert.ok(html.includes(label), label)
    assert.match(html, /<table[^>]*aria-label="Engine statistics"/)
    for (const name of ['State', 'Memory', 'Context']) assert.ok(html.includes('<th scope="col">' + name + '</th>'))
    assert.doesNotMatch(html, /Workers/)
    assert.match(html, /href="#monitor"/)
    assert.match(html, /href="#fleet"/)
    assert.match(html, /href="#bench"/)
    for (const screen of ['Monitor', 'Bench']) {
      const Screen = (await server.ssrLoadModule(`/src/screens/${screen}.tsx`)).default
      const markup = renderToStaticMarkup(React.createElement(Screen, { ...ctx, decodeHistory: [] }))
      assert.ok(markup.includes('<a class="back-link" href="#models"><span aria-hidden="true">← </span>Models</a>'), screen)
    }
    const Markdown = (await server.ssrLoadModule('/src/Markdown.tsx')).default
    const code = ['```js', 'one()', '```', '', '```js', 'two()', '```'].join('\n')
    const reply = renderToStaticMarkup(React.createElement(Markdown, { text: code, message: 4, onCopy: noop }))
    assert.match(reply, /Copy code block at line 1 in message 4/)
    assert.match(reply, /Copy code block at line 5 in message 4/)
    const Settings = (await server.ssrLoadModule('/src/screens/Settings.tsx')).default
    globalThis.location.hash = '#settings/about'
    html = renderToStaticMarkup(React.createElement(Settings, { ...ctx, machine: { budget: { default_fraction: .7, raised_fraction: .85 }, backend: 'none', os: 'test', ram_bytes: 1024 }, settings: { extra_args: [], thinking: 'default', context: null, currency: 'USD', theme: 'system' } }))
    assert.match(html, /Settings sections/)
    assert.ok(html.includes('<a class="back-link" href="#chat"><span aria-hidden="true">← </span>Back to app</a>'))
    assert.match(html, /href="#settings\/storage"/)
    assert.match(html, /Native engine/)
  } finally {
    await server.close()
  }
})

test('browser startup ignores durable URL tokens and redeems one-time codes', async () => {
  const saved = { fetch: globalThis.fetch, location: globalThis.location, history: globalThis.history, localStorage: globalThis.localStorage }
  const stored = new Map()
  const server = await createServer({ server: { middlewareMode: true, watch: null }, appType: 'custom' })
  let requests = 0
  let cleaned = ''
  globalThis.localStorage = { getItem: (k) => stored.get(k) ?? null, setItem: (k, v) => stored.set(k, v) }
  globalThis.history = { replaceState: (_state, _unused, url) => { cleaned = String(url) } }
  globalThis.fetch = async () => { requests += 1; return { ok: true, json: async () => ({ token: 'one-session-token' }) } }
  try {
    const { redeemFromAddress, webToken } = await server.ssrLoadModule('/src/api.ts')
    globalThis.location = new URL('http://localhost/?token=durable-secret')
    await redeemFromAddress()
    assert.equal(requests, 0)
    assert.equal(webToken(), null)
    globalThis.location = new URL('http://localhost/?code=one-use')
    await redeemFromAddress()
    assert.equal(requests, 1)
    assert.equal(webToken(), 'one-session-token')
    assert.doesNotMatch(cleaned, /code=|token=/)
  } finally {
    try { await server.close() } finally { Object.assign(globalThis, saved) }
  }
})
