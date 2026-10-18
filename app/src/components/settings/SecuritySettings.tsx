import { currentIntlLocale } from '../../lib/preferences/current-locale'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { BrandIcon } from '../../lib/brand'
import { Button, Field, Toggle } from '../ui'
import { Section, SettingsPane, type SettingsSectionContext } from './shared'
import { ConfidentialTermsSection } from './ConfidentialTermsSection'

export function SecuritySettings({ ctx }: { ctx: SettingsSectionContext }) {
  const tx = useAuthoredText()
  const {
    draft,
    patch,
    controllerBusy,
    controllerStatus,
    commitController,
    refreshControllerStatus,
    enableController,
  } = ctx

  return (
<SettingsPane>
        <ConfidentialTermsSection draft={draft} patch={patch} />
        <Section title={tx("Controller")} desc={tx("Attach your current app/window to a chat or task, using local Accessibility capture when available.")}>
          <div className="space-y-3">
            <div className="rounded-xl border border-line bg-card p-3.5 shadow-card">
              <div className="mb-3 flex items-start justify-between gap-3">
                <div className="flex min-w-0 items-start gap-2.5">
                  <BrandIcon brand="controller" size={32} />
                  <div className="min-w-0">
                    <div className="text-[13px] font-medium text-ink">{tx("Controller")}</div>
                    <div className="mt-0.5 text-[13px] leading-snug text-ink-3">
                      {controllerStatus?.message ?? tx("Attach foreground window text as context when you ask.")}
                    </div>
                  </div>
                </div>
                <Toggle label={tx("Computer controller")} checked={draft.controller.enabled} onChange={(enabled) => commitController({ ...draft.controller, enabled, lastError: enabled ? draft.controller.lastError : null })} />
              </div>
              <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
                <Field label={tx("Accessibility permission")} hint={controllerStatus?.accessibilityTrusted ? tx("Granted") : tx("Needed on macOS")}>
                  <div className="flex gap-2">
                    <Button variant="default" icon="monitor" onClick={enableController} disabled={controllerBusy}>
                      {controllerBusy ? tx("Opening...") : tx("Enable Controller")}
                    </Button>
                    <Button variant="ghost" icon="search" onClick={refreshControllerStatus}>{tx("Check")}</Button>
                  </div>
                </Field>
                <Field label={tx("Text capture")} hint={draft.controller.includeAccessibilityText ? tx("On") : tx("Off")}>
                  <div className="flex items-center gap-3 rounded-lg border border-line bg-card-2 px-3 py-2.5">
                    <span className="flex-1 text-[13px] leading-snug text-ink-3">{tx("Include readable UI text exposed by the target app, including off-screen text when Accessibility exposes it.")}</span>
                    <Toggle label={tx("Include accessibility text")} checked={draft.controller.includeAccessibilityText} onChange={(includeAccessibilityText) => commitController({ ...draft.controller, includeAccessibilityText })} />
                  </div>
                </Field>
                <Field label={tx("Capture size")} hint={tx("{value0}k chars", { value0: Math.round(draft.controller.maxTextChars / 1000) })}>
                  <input
                    type="range" min={4_000} max={80_000} step={2_000} value={draft.controller.maxTextChars}
                    onChange={(e) => commitController({ ...draft.controller, maxTextChars: parseInt(e.target.value, 10) })}
                    className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-ink-5 accent-[var(--accent)]"
                  />
                  <div className="mt-1 flex justify-between text-[12px] text-ink-4"><span>{tx("4k")}</span><span>{tx("keeps pasted context controlled")}</span><span>{tx("80k")}</span></div>
                </Field>
                <Field label={tx("Last capture")}>
                  <div className="rounded-lg border border-line bg-card-2 px-3 py-2.5 text-[13px] text-ink-3">
                    {draft.controller.lastCapturedAt ? new Date(draft.controller.lastCapturedAt).toLocaleString(currentIntlLocale()) : tx("Not used yet")}
                  </div>
                </Field>
              </div>
              {(draft.controller.lastError || (controllerStatus && !controllerStatus.canCapture && draft.controller.enabled)) && (
                <div className="mt-2.5 rounded-lg border border-warn/30 bg-warn/5 px-3 py-2 text-[13px] text-warn">
                  {draft.controller.lastError || controllerStatus?.message}
                </div>
              )}
            </div>
          </div>
        </Section>
              </SettingsPane>

  )
}
