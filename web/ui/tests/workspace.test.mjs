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
    assert.match(html, /aria-label="Search conversations"/)
    assert.match(html, /> New chat</)
    assert.match(html, /aria-label="Message"/)
    assert.match(html, /aria-label="Hide sidebar"/)
    assert.doesNotMatch(html, />Home<|Fold the menu|class="starters"|<h1>Chat<|class="brand"/)
    const Workspace = (await server.ssrLoadModule('/src/ModelWorkspace.tsx')).default
    const noop = () => {}
    const ctx = { engine: null, machine: null, settings: null, toast: noop, refreshEngine: noop }
    globalThis.location.hash = '#models'
    html = renderToStaticMarkup(React.createElement(Workspace, ctx))
    for (const label of ['My models', 'Discover', 'Learning', 'API', 'Benchmarks', 'Compute and clusters', 'Details']) assert.ok(html.includes(label), label)
    assert.doesNotMatch(html, /Workers/)
    assert.match(html, /href="#monitor"/)
    assert.match(html, /href="#fleet"/)
    assert.match(html, /href="#bench"/)
    const Settings = (await server.ssrLoadModule('/src/screens/Settings.tsx')).default
    globalThis.location.hash = '#settings/about'
    html = renderToStaticMarkup(React.createElement(Settings, { ...ctx, machine: { budget: { default_fraction: .7, raised_fraction: .85 }, backend: 'none', os: 'test', ram_bytes: 1024 }, settings: { extra_args: [], thinking: 'default', context: null, currency: 'USD', theme: 'system' } }))
    assert.match(html, /Settings sections/)
    assert.match(html, /Back to app/)
    assert.match(html, /href="#settings\/storage"/)
    assert.match(html, /Native engine/)
  } finally {
    await server.close()
  }
})
