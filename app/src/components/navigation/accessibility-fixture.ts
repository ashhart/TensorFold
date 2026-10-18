import React from 'react'

// Deliberately small synthetic DOM for deterministic hook and handler tests.
// Browser layout and assistive-technology behavior still need integrated QA.
export class FixtureElement {
  children: FixtureElement[] = []
  parentElement: FixtureElement | null = null
  attrs: Record<string, string> = {}
  inert = false
  hidden = false
  disabled = false
  tabIndex: number
  connected = true
  constructor(readonly tagName: string, readonly doc: any) {
    this.tabIndex = ['button', 'input', 'select', 'textarea'].includes(tagName) ? 0 : -1
  }
  get isConnected(): boolean { return this.connected && (this.tagName === 'body' || !!this.parentElement?.isConnected) }
  append(...children: FixtureElement[]) { for (const child of children) { this.children.push(child); child.parentElement = this } }
  contains(target: unknown): boolean { return target === this || this.children.some((child) => child.contains(target)) }
  focus() { this.doc.activeElement = this }
  matches(selector: string): boolean { return selector === ':disabled' && this.disabled }
  closest(selector: string): FixtureElement | null {
    if (selector.includes('[inert]') && this.inert || selector.includes('[hidden]') && this.hidden || selector.includes('[aria-hidden="true"]') && this.attrs['aria-hidden'] === 'true') return this
    return this.parentElement?.closest(selector) ?? null
  }
  getClientRects() { return this.closest('[hidden]') ? [] : [{}] }
  querySelectorAll<T = FixtureElement>(_selector: string): T[] {
    return this.children.flatMap((child) => [child, ...child.querySelectorAll<FixtureElement>(_selector)]).filter((child) => ['button', 'input', 'select', 'textarea', 'a'].includes(child.tagName) || 'tabindex' in child.attrs || child.attrs.contenteditable === 'true') as T[]
  }
  scrollIntoView() {}
}

export function installFixtureDom() {
  const originals = new Map(['window', 'document', 'HTMLElement'].map((key) => [key, Object.getOwnPropertyDescriptor(globalThis, key)]))
  const doc: any = { activeElement: null }
  const body = new FixtureElement('body', doc)
  const root = new FixtureElement('div', doc); root.attrs.id = 'root'
  const trigger = new FixtureElement('button', doc)
  root.append(trigger); body.append(root); doc.body = body; doc.activeElement = trigger
  const listeners: { name: string; callback: any; capture: boolean }[] = []
  const win: any = {
    addEventListener(name: string, callback: any, capture = false) { listeners.push({ name, callback, capture }) },
    removeEventListener(name: string, callback: any) { const index = listeners.findIndex((entry) => entry.name === name && entry.callback === callback); if (index >= 0) listeners.splice(index, 1) },
    dispatchEvent(event: any) { listeners.filter((entry) => entry.name === event.type).forEach((entry) => entry.callback(event)) },
  }
  const walk = (node: FixtureElement): FixtureElement[] => [node, ...node.children.flatMap(walk)]
  doc.getElementById = (id: string) => walk(body).find((node) => node.attrs.id === id) ?? null
  doc.querySelector = (selector: string) => selector.includes('role="dialog"') ? walk(body).find((node) => node.attrs.role === 'dialog' && node.attrs['aria-modal'] === 'true') ?? null : null
  for (const [key, value] of Object.entries({ window: win, document: doc, HTMLElement: FixtureElement })) Object.defineProperty(globalThis, key, { configurable: true, value })
  function element(tag: string) { return new FixtureElement(tag, doc) }
  function mount(tree: any, parent = body): FixtureElement[] {
    if (!tree || typeof tree !== 'object') return []
    if (typeof tree.type !== 'string') return React.Children.toArray(tree.props?.children).flatMap((child) => mount(child, parent))
    const node = element(tree.type)
    for (const [key, value] of Object.entries(tree.props)) {
      if (key === 'tabIndex') { node.tabIndex = Number(value); node.attrs.tabindex = String(value) }
      if (key === 'disabled') node.disabled = Boolean(value)
      if (key === 'hidden') node.hidden = Boolean(value)
      if (key === 'id' || key === 'role' || key.startsWith('aria-')) node.attrs[key] = String(value)
    }
    parent.append(node)
    if (tree.ref && typeof tree.ref === 'object') tree.ref.current = node
    React.Children.toArray(tree.props.children).forEach((child) => mount(child, node))
    return [node]
  }
  function key(keyName: string, extra: Record<string, any> = {}) {
    const event = { key: keyName, prevented: false, stopped: false, preventDefault() { this.prevented = true }, stopPropagation() { this.stopped = true }, stopImmediatePropagation() { this.stopped = true }, ...extra }
    for (const entry of listeners.filter((entry) => entry.name === 'keydown' && entry.capture)) { entry.callback(event); if (event.stopped) break }
    return event
  }
  return { body, root, trigger, doc, win, listeners, element, mount, key, restore() {
    for (const [name, descriptor] of originals) { if (descriptor) Object.defineProperty(globalThis, name, descriptor); else Reflect.deleteProperty(globalThis, name) }
  } }
}
