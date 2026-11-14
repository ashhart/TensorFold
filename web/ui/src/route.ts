// Use screen/subselection links; replacing the selection preserves Back navigation between screens.
import { useEffect, useRef } from 'react'

export const screenOf = (hash = location.hash) => hash.replace(/^#/, '').split('/')[0]


export function subOf(screen: string, hash = location.hash): string | null {
  const [s, ...rest] = hash.replace(/^#/, '').split('/')
  if (s !== screen || !rest.length || !rest.join('')) return null
  try { return rest.map(decodeURIComponent).join('/') } catch { return null }
}

export const linkTo = (screen: string, sub?: string | null) => `#${screen}${sub ? `/${sub.split('/').map(encodeURIComponent).join('/')}` : ''}`


export function showSub(screen: string, sub: string | null) {
  if (screenOf() !== screen) return
  const h = linkTo(screen, sub)
  if (location.hash !== h) history.replaceState(null, '', h)
}


export function useSub(screen: string, apply: (sub: string) => void) {
  const fn = useRef(apply)
  fn.current = apply
  useEffect(() => {
    const on = () => { const s = subOf(screen); if (s) fn.current(s) }
    on()
    addEventListener('hashchange', on)
    return () => removeEventListener('hashchange', on)
  }, [screen])
}
