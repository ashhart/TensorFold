import { languageDirection } from '../../../shared/languages'
import { normalizeUiPreferences } from './model'

const messageSelector = '[data-ui-message], .prose-tf, .select-text.whitespace-pre-wrap, textarea, input:not([type]), input[type="text"], input[type="search"], [contenteditable="true"]'
const codeSelector = 'pre, code, .syn-code, .xterm, [data-ui-code], textarea.font-mono, [contenteditable="true"].font-mono, input[type="url"], input[type="email"]'

/** Stateless DOM projection; cleanup restores previous attributes and only our CSS variables. */
export function applyUiPreferences(input: unknown, doc: Document = document): () => void {
  const preference = normalizeUiPreferences(input), root = doc.documentElement
  const attributes = ['lang', 'dir', 'data-ui-density', 'data-ui-preferences']
  const previous = attributes.map((name) => [name, root.getAttribute(name)] as const)
  const variables = ['--tf-chat-scale', '--tf-code-scale', '--tf-reading-leading', '--tf-density-gap', '--tf-row-height']
  const styles = variables.map((name) => [name, root.style.getPropertyValue(name), root.style.getPropertyPriority(name)] as const)
  root.setAttribute('lang', preference.locale)
  root.setAttribute('dir', preference.direction === 'auto' ? languageDirection(preference.locale) : preference.direction)
  root.setAttribute('data-ui-preferences', 'true'); root.setAttribute('data-ui-density', preference.density)
  root.style.setProperty('--tf-chat-scale', String(preference.chatScale)); root.style.setProperty('--tf-code-scale', String(preference.codeScale))
  root.style.setProperty('--tf-reading-leading', preference.density === 'compact' ? '1.48' : '1.62')
  root.style.setProperty('--tf-density-gap', preference.density === 'compact' ? '0.5rem' : '0.875rem')
  root.style.setProperty('--tf-row-height', preference.density === 'compact' ? '2rem' : '2.5rem')
  const directions = new Map<Element, string | null>()
  const setDirection = (element: Element, direction: 'auto' | 'ltr') => {
    if (!directions.has(element)) directions.set(element, element.getAttribute('dir'))
    element.setAttribute('dir', direction)
  }
  const project = (parent: ParentNode & { matches?: Element['matches'] }) => {
    if (parent.matches?.(messageSelector)) setDirection(parent as Element, 'auto')
    parent.querySelectorAll(messageSelector).forEach((element) => setDirection(element, 'auto'))
    if (parent.matches?.(codeSelector)) setDirection(parent as Element, 'ltr')
    parent.querySelectorAll(codeSelector).forEach((element) => setDirection(element, 'ltr'))
  }
  project(doc)
  const Observer = doc.defaultView?.MutationObserver
  const observer = Observer ? new Observer((records) => {
    for (const record of records) for (const node of record.addedNodes) if (node.nodeType === 1) project(node as Element)
    for (const element of directions.keys()) if (!element.isConnected) directions.delete(element)
  }) : null
  observer?.observe(root, { subtree: true, childList: true })
  return () => {
    observer?.disconnect()
    for (const [name, value] of previous) { if (value === null) root.removeAttribute(name); else root.setAttribute(name, value) }
    for (const [name, value, priority] of styles) { if (!value) root.style.removeProperty(name); else root.style.setProperty(name, value, priority) }
    for (const [element, direction] of directions) { if (direction === null) element.removeAttribute('dir'); else element.setAttribute('dir', direction) }
  }
}
