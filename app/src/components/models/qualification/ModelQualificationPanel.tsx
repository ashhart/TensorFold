import { currentIntlLocale } from '../../../lib/preferences/current-locale'
import { useAuthoredText } from '../../../lib/preferences/useAuthoredText'
import { useFlowText } from '../../../lib/preferences/useFlowText'
import { useEffect, useRef, useState } from 'react'
import { MODEL_CHECKS, currentModelQualification, modelQualificationKey } from '../../../../shared/model-fit'
import type { ModelCheck, ModelQualificationApi, ModelQualificationInput, ModelQualificationReport, ModelQualificationRoute } from '../../../../shared/model-fit'
import { api } from '../../../lib/ipc'
import { Button } from '../../ui'
import { useModelQualifications } from './useModelQualifications'

export function ModelQualificationPanel({ route, disabled = false, unavailableReason, threadId, client = api }: { route: ModelQualificationRoute; disabled?: boolean; unavailableReason?: string; threadId?: string; client?: ModelQualificationApi }) {
  const tx = useAuthoredText()
  const { t, authored, known, locale } = useFlowText()
  const routeKey = modelQualificationKey(route)
  const { reports, error: loadError, refresh, ownership } = useModelQualifications(client, routeKey)
  const [checks, setChecks] = useState<ModelCheck[]>(['text', 'cancellation'])
  const [report, setReport] = useState<ModelQualificationReport | null>(null)
  const reportOwner = useRef(ownership)
  const latestOwner = useRef(ownership)
  latestOwner.current = ownership
  const visibleReport = reportOwner.current === ownership && report?.key === routeKey ? report : null
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<'' | 'qualification.error' | 'qualification.cancelError'>('')
  const active = useRef<ModelQualificationInput | null>(null)
  const epoch = useRef(0)
  const key = modelQualificationKey(route)
  useEffect(() => { reportOwner.current = ownership; setReport(currentModelQualification(reports, route)); setError('') }, [key, reports, ownership])
  useEffect(() => () => { epoch.current++; if (active.current) void client.modelQualificationCancel(active.current.requestId).catch(() => {}) }, [client, key])
  const run = async () => {
    if (busy || active.current || disabled || !checks.length) return
    const current = ++epoch.current
    const requestedOwner = ownership
    const input: ModelQualificationInput = { ...route, requestId: crypto.randomUUID(), checks: [...checks], ...(threadId ? { threadId } : {}) }
    active.current = input
    setBusy(true); setError('')
    try {
      const found = await client.modelQualificationCheck(input)
      if (current === epoch.current && latestOwner.current === requestedOwner) { reportOwner.current = requestedOwner; setReport(found); await refresh(); window.dispatchEvent(new Event('tensorfold:model-qualification')) }
    } catch { if (current === epoch.current) setError('qualification.error') }
    finally { if (active.current?.requestId === input.requestId) { active.current = null; setBusy(false) } }
  }
  const cancel = async () => {
    if (!active.current) return
    try { await client.modelQualificationCancel(active.current.requestId) }
    catch { setError('qualification.cancelError') }
  }
  return <section aria-label={t("qualification.label")} aria-busy={busy} className="space-y-3 rounded-xl border border-line p-4">
    <div><h3 className="text-[14px] font-medium text-ink">{t("qualification.title")}</h3><p className="mt-1 text-[12px] leading-relaxed text-ink-2">{t("qualification.help")}</p></div>
    <p className="break-all text-[12px] text-ink-3">{t('qualification.route', { provider: route.providerId, account: route.accountId ?? t('qualification.default'), model: route.model })}</p>
    <div className="flex flex-wrap gap-x-4 gap-y-2">{MODEL_CHECKS.map(check => <label key={check} className="flex items-center gap-2 text-[13px] text-ink-2"><input type="checkbox" checked={checks.includes(check)} disabled={busy || disabled} onChange={event => setChecks(current => event.target.checked ? [...current, check] : current.filter(item => item !== check))} />{t(`qualification.${check}`)}</label>)}</div>
    {disabled && <p className="text-[12px] text-ink-3">{unavailableReason ? authored(unavailableReason) : t('qualification.blocked')}</p>}
    <div className="flex flex-wrap gap-2"><Button size="sm" disabled={disabled || busy || !checks.length} onClick={() => void run()}>{t(busy ? 'qualification.busy' : 'qualification.run')}</Button>{busy && <Button size="sm" variant="ghost" onClick={() => void cancel()}>{t("qualification.cancel")}</Button>}<Button size="sm" variant="ghost" disabled={busy} onClick={() => void refresh()}>{t("qualification.refresh")}</Button></div>
    {(error || loadError) && <p role="alert" className="text-[12px] text-danger">{error ? t(error) : authored(loadError)}</p>}
    {busy && <p role="status" className="text-[12px] text-ink-3">{t("qualification.wait")}</p>}
    <dl className="space-y-2">{MODEL_CHECKS.map(check => {
      const result = visibleReport?.checks[check]
      return <div key={check} className="rounded-lg bg-card-2 px-3 py-2"><dt className="text-[13px] font-medium text-ink">{t(`qualification.${check}`)} · {t(`qualification.${busy && checks.includes(check) ? 'requested' : result?.state ?? 'unknown'}`)}</dt><dd className="mt-1 text-[12px] leading-relaxed text-ink-2">{result?.message ? <>{!known(result.message) && <span>{t('common.diagnostic')}: </span>}{authored(result.message)}</> : t('qualification.none')}</dd></div>
    })}</dl>
    {visibleReport && <p className="text-[12px] text-ink-3">{t('qualification.date', { date: new Date(visibleReport.checkedAt).toLocaleString(currentIntlLocale()), probe: visibleReport.probeRevision, route: visibleReport.revision })}</p>}
    <p className="text-[12px] leading-relaxed text-ink-3">{t("qualification.limitations")}</p>
  </section>
}
