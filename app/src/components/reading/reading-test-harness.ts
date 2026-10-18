import { reviewedLanguageImport } from '../preferences/language-test-fixture'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import React from 'react'
import ts from 'typescript'
export const jsx = { jsx: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }), jsxs: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }), Fragment: React.Fragment }
export function load(file: string, imports: Record<string, unknown>) {
  const module = { exports: {} as any }
  const output = ts.transpileModule(readFileSync(file, 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true } }).outputText
  new Function('require', 'module', 'exports', output)((id: string) => { const reviewed = reviewedLanguageImport(id, imports); if (reviewed.matched) return reviewed.value;  assert.ok(id in imports, `Reviewed production import: ${id}`); return imports[id] }, module, module.exports)
  return module.exports
}
export function harness() {
  const values: any[] = [], deps: any[][] = [], effects: (() => void)[] = [], cleanups: any[] = []
  let cursor = 0
  const hooks: any = { ...React,
    memo: (component: any) => component,
    useState(initial: any) { const index = cursor++; if (!(index in values)) values[index] = typeof initial === 'function' ? initial() : initial; return [values[index], (next: any) => { values[index] = typeof next === 'function' ? next(values[index]) : next }] },
    useRef(initial: any) { const index = cursor++; return values[index] ??= { current: initial } },
    useMemo(build: any) { cursor++; return build() },
    useId() { return `reading-fixture-${cursor++}` },
    useEffect(effect: any, next: any[]) { const index = cursor++; if (!deps[index] || !next || next.some((value, i) => value !== deps[index][i])) { effects.push(() => { cleanups[index]?.(); cleanups[index] = effect() }); deps[index] = next } },
  }
  return { hooks, render<T>(component: () => T): T { cursor = 0; return component() }, flush() { while (effects.length) effects.shift()!() }, cleanup() { cleanups.forEach((cleanup) => cleanup?.()) } }
}
export function nodes(tree: any): any[] { return tree && typeof tree === 'object' ? [tree, ...React.Children.toArray(tree.props?.children).flatMap(nodes)] : [] }
export const find = (tree: any, test: (node: any) => boolean): any => nodes(tree).find(test)
export const text = (tree: any): string => typeof tree === 'string' || typeof tree === 'number' ? String(tree) : React.Children.toArray(tree?.props?.children).map(text).join(' ').replace(/\s+/g, ' ').trim()
export const settle = async () => { for (let i = 0; i < 12; i++) await Promise.resolve() }
