import { useLayoutEffect, useRef, type RefObject } from 'react'
import type { QuickAskApi } from '../../../shared/quick-ask'

export function useAskPanelSize(api: QuickAskApi, panel: RefObject<HTMLElement>, dialog: boolean, report: (error: unknown) => void) {
  const last = useRef(0)
  useLayoutEffect(() => {
    if (!panel.current) return
    let frame = 0
    const update = () => {
      cancelAnimationFrame(frame)
      frame = requestAnimationFrame(() => {
        const height = dialog ? 600 : Math.min(680, Math.max(88, Math.ceil(panel.current!.getBoundingClientRect().height)))
        if (last.current === height) return
        last.current = height
        void api.resize?.(height).catch(report)
      })
    }
    const observer = new ResizeObserver(update)
    observer.observe(panel.current); update()
    return () => { observer.disconnect(); cancelAnimationFrame(frame) }
  }, [api, panel, dialog, report])
}
