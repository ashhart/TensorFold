import { useCallback, useEffect, useRef, useState } from 'react'
import { api, type ConversationRow } from './api'

export function useConversations(toast: (text: string, kind?: 'ok' | 'bad') => void) {
  const [convId, setConvId] = useState<string | null>(null)
  const [convs, setConvs] = useState<ConversationRow[]>([])
  const [q, setQ] = useState('')
  const [more, setMore] = useState(false)
  const requested = useRef(0)
  const shown = useRef(0)
  const refreshList = useCallback((query = q) => {
    const gen = ++requested.current
    return api.conversations(query).then((r) => {
      if (gen !== requested.current) return
      shown.current = gen; setConvs(r.conversations); setMore(!!r.more)
    }).catch(() => {})
  }, [q])
  const loadMore = () => {
    const gen = requested.current
    if (shown.current !== gen) return Promise.resolve()
    return api.conversations(q, convs.filter((c) => !c.pinned).length).then((r) => {
      if (gen !== requested.current) return
      setConvs((old) => [...old, ...r.conversations.filter((c) => !old.some((o) => o.id === c.id))]); setMore(!!r.more)
    }).catch((e) => toast(String(e), 'bad'))
  }
  useEffect(() => { const t = setTimeout(() => refreshList(), 250); return () => clearTimeout(t) }, [refreshList])
  return { convId, setConvId, convs, q, setQ, more, refreshList, loadMore }
}
export type ConversationIndex = ReturnType<typeof useConversations>
