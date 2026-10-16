import React from 'react'
import { api } from './ipc'

// react-markdown `a` override for model-produced chat links: open through the
// main-process scheme check (openExternal) instead of an in-window navigation
// the will-navigate guard would silently block. Plain createElement (no JSX) so
// the behavior stays unit-testable under the jiti runner.
export function MarkdownLink({ href, children, ...props }: any) {
  return React.createElement(
    'a',
    {
      ...props,
      href,
      onClick: (e: React.MouseEvent) => {
        e.preventDefault()
        if (href) void api.openExternal(href).catch(() => {})
      },
    },
    children,
  )
}
