// Every view a link: `#screen/what`, such as #bench/<run>, #chat/<conversation>, #models/<owner>/<name>.
// A screen reads its part of the address when it opens and when the address changes, and writes it back when the
// selection changes, replacing the address rather than adding to the history, so Back still leaves the screen.
import { useEffect, useRef } from 'react'

export const screenOf = (hash = location.hash) => hash.replace(/^#/, '').split('/')[0]

/** What the address names on ``screen``, or null when it names nothing there (or another screen). */
export function subOf(screen: string, hash = location.hash): string | null {
  const [s, ...rest] = hash.replace(/^#/, '').split('/')
  if (s !== screen || !rest.length || !rest.join('')) return null
  try { return rest.map(decodeURIComponent).join('/') } catch { return null }
}

export const linkTo = (screen: string, sub?: string | null) => `#${screen}${sub ? `/${sub.split('/').map(encodeURIComponent).join('/')}` : ''}`

/** Put ``sub`` in the address while ``screen`` is the one shown; a hidden screen leaves the address alone. */
export function showSub(screen: string, sub: string | null) {
  if (screenOf() !== screen) return
  const h = linkTo(screen, sub)
  if (location.hash !== h) history.replaceState(null, '', h)
}

/** Call ``apply`` with what the address names on ``screen``, now and whenever the address changes. */
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
