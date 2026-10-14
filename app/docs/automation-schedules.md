# Automation schedules

Automations run on the scheduler host while TensorFold is open. Manual runs start only
when requested; intervals measure elapsed minutes, and daily schedules use a wall-clock
time in a named timezone.

Enabling a never-run interval schedules its first run one complete interval later, at
the time shown in its next-run display. **Run now** is the explicit immediate action.
Restart preserves an already committed first-run time; a legacy interval with no such
time receives a new future first run. Missed first fires follow the same Run once or
bounded Catch up each policy as subsequent interval runs.

New daily schedules require an explicit IANA timezone, such as `Europe/London`,
`America/New_York` or `UTC`. The editor initially offers the current device's timezone
and lets you choose another. It shows the next scheduled time before saving or enabling.
The selected timezone stays fixed when the device or scheduler host changes timezone.
An invalid identifier is rejected rather than replaced with the host's local timezone.

During a daylight-saving change, a daily time that does not exist is skipped for that
calendar date. If the clock repeats a time, the schedule uses its first occurrence once.
An overdue daily run is coalesced into one run when the host becomes available; it does
not replay every missed day. Interval schedules keep their existing Run once or capped
Catch up each policy.

Existing daily schedules without a saved timezone are labelled **Legacy host-local
timezone**. They retain the scheduler host's local-time interpretation, including the
previous behavior for a nonexistent clock time, until you deliberately select a timezone.
Saving an unrelated edit does not silently migrate them. Their calendar-day calculation
now preserves the chosen hour across daylight-saving changes instead of adding a fixed
24 hours. A legacy schedule's exact clock time cannot be previewed from a different
device without knowing the host timezone, so the editor asks you to pin a zone for that
preview.

The plain-language builder recognizes explicit IANA identifiers and `UTC`, and rejects
invalid identifiers or explicit timezone abbreviations such as `PST` or `CST` instead of
silently interpreting them in another timezone. Review the resulting daily time, timezone,
model, instruction and delivery before enabling. The scheduler supports manual, interval
and daily schedules; it does not implement cron expressions or weekday-only calendars.

If a persisted daily timezone is invalid, that automation shows a schedule diagnostic
and cannot fire automatically until repaired. Other valid schedules continue to run, and
repairing the timezone computes a new future run without dispatching a duplicate.
