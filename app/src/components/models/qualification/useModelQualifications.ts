import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../../../lib/ipc'
import { useStore } from '../../../lib/store'
import type { ModelQualificationApi, ModelQualificationReport } from '../../../../shared/model-fit'

/** Evidence belongs to a settings-provider snapshot, transport and optional profile/host scope. */
export function useModelQualifications(client: ModelQualificationApi = api, scopeKey: string | null = null) {
  const providers = useStore(state => state.settings?.providers)
  const owner = useRef({ providers, client, scopeKey })
  if (owner.current.providers !== providers || owner.current.client !== client || owner.current.scopeKey !== scopeKey) owner.current = { providers, client, scopeKey }
  const currentOwner = owner.current
  const [snapshot, setSnapshot] = useState<{ owner: typeof currentOwner; reports: ModelQualificationReport[] }>({ owner: currentOwner, reports: [] })
  const [failure, setFailure] = useState({ owner: currentOwner, message: '' })
  const request = useRef(0)
  const refresh = useCallback(async () => {
    if (owner.current !== currentOwner) return
    const current = ++request.current
    setSnapshot({ owner: currentOwner, reports: [] })
    try {
      const found = await client.modelQualificationReports()
      if (current === request.current && owner.current === currentOwner) { setSnapshot({ owner: currentOwner, reports: found }); setFailure({ owner: currentOwner, message: '' }) }
    } catch {
      if (current === request.current && owner.current === currentOwner) { setSnapshot({ owner: currentOwner, reports: [] }); setFailure({ owner: currentOwner, message: 'Capability checks could not be loaded; capabilities remain unknown.' }) }
    }
  }, [client, currentOwner])
  useEffect(() => { void refresh(); const changed = () => { void refresh() }; window.addEventListener('tensorfold:model-qualification', changed); return () => { request.current++; window.removeEventListener('tensorfold:model-qualification', changed) } }, [refresh])
  return { ownership: currentOwner, reports: snapshot.owner === currentOwner ? snapshot.reports : [], error: failure.owner === currentOwner ? failure.message : '', refresh }
}
