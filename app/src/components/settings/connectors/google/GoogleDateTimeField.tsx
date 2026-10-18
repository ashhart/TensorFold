import { useAuthoredText } from '../../../../lib/preferences/useAuthoredText'
import { Input } from '../../../ui'
import { wallTimeCandidates } from './google-calendar-time'

export function GoogleDateTimeField({ label, value, timeZone, disabled, onChange }: { label: string; value: string; timeZone: string; disabled: boolean; onChange: (value: string) => void }) {
  const tx = useAuthoredText()
  const local = value.slice(0, 19)
  const candidates = wallTimeCandidates(local, timeZone)
  const chooseWall = (next: string) => {
    const options = wallTimeCandidates(next, timeZone)
    onChange(options.length === 1 ? options[0] : next.length === 16 ? `${next}:00` : next)
  }
  return <div className="space-y-1.5">
    <label className="block text-[13px] text-ink-2">{label}<Input aria-label={label} type="datetime-local" step="1" value={/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}/.test(local) ? local : ''} disabled={disabled} onChange={(event) => chooseWall(event.target.value)} className="mt-1" /></label>
    {candidates.length > 1 ? <label className="block text-[12px] text-warn">{tx("This time occurs twice; choose its UTC offset")}<select aria-label={tx("{value0} daylight-saving offset", { value0: label })} value={candidates.includes(value) ? value : ''} disabled={disabled} onChange={(event) => onChange(event.target.value)} className="mt-1 block w-full rounded-lg border border-line bg-card-2 px-2 py-1.5 text-[13px] text-ink"><option value="">{tx("Choose which occurrence")}</option>{candidates.map((candidate) => <option key={candidate} value={candidate}>{candidate.slice(19)} · {timeZone}</option>)}</select></label> : candidates.length === 1 ? <p className="text-[12px] text-ink-3">{timeZone} {tx("· UTC")}{candidates[0].slice(19)}</p> : local && <p className="text-[12px] text-warn">{tx("This local time is invalid or skipped by daylight saving; choose another time.")}</p>}
    <details className="text-[12px] text-ink-3"><summary className="cursor-pointer">{tx("Exact time and offset")}</summary><Input aria-label={tx("{value0} exact timestamp", { value0: label })} value={value} disabled={disabled} onChange={(event) => onChange(event.target.value)} className="mt-1" spellCheck={false} /></details>
  </div>
}
