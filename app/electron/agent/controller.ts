import { systemPreferences } from 'electron'
import { execFile } from 'node:child_process'
import type { ControllerSettings, ControllerSnapshot, ControllerStatus } from './types'
import { controllerActionScript, controllerActionTarget, normalizeControllerActionInput, type ControllerActionInput, type ControllerActionResult } from './controller-actions'

const DEFAULT_MAX_TEXT = 4_000

export function buildControllerStatus(settings: ControllerSettings): ControllerStatus {
  const supported = process.platform === 'darwin'
  const accessibilityTrusted = supported ? systemPreferences.isTrustedAccessibilityClient(false) : false
  const enabled = !!settings.enabled
  return {
    platform: process.platform,
    supported,
    enabled,
    accessibilityTrusted,
    canCapture: supported && enabled && accessibilityTrusted,
    message: !supported
      ? 'Controller is currently implemented for macOS Accessibility.'
      : !enabled
        ? 'Controller is disabled; no foreground window capture runs passively.'
        : accessibilityTrusted
          ? 'Controller can attach foreground window context after an explicit user click.'
          : 'Enable Accessibility permission for TensorFold in macOS System Settings.',
  }
}

export function requestControllerAccess(settings: ControllerSettings): ControllerStatus {
  if (process.platform === 'darwin') systemPreferences.isTrustedAccessibilityClient(true)
  return buildControllerStatus({ ...settings, enabled: true })
}

function runOsa(script: string): Promise<string> {
  return new Promise((resolve, reject) => {
    execFile('/usr/bin/osascript', ['-e', script], { timeout: 7000, maxBuffer: 2_000_000 }, (err, stdout, stderr) => {
      if (err) reject(new Error((stderr || stdout || String(err)).trim()))
      else resolve(stdout || '')
    })
  })
}

function captureScript(appName: string): string {
  const safeAppName = appName.replace(/\\/g, '\\\\').replace(/"/g, '\\"')
  return `
on cleaned(v)
  try
    set t to v as text
    if t is "missing value" then return ""
    set AppleScript's text item delimiters to tab
    set parts to text items of t
    set AppleScript's text item delimiters to " "
    set t to parts as text
    set AppleScript's text item delimiters to linefeed
    set parts to text items of t
    set AppleScript's text item delimiters to " "
    set t to parts as text
    set AppleScript's text item delimiters to ""
    if (count of t) > 260 then set t to text 1 thru 260 of t
    return t
  on error
    return ""
  end try
end cleaned

on addChunk(chunks, v)
  set t to my cleaned(v)
  if t is "" then return chunks
  if chunks does not contain t then set end of chunks to t
  return chunks
end addChunk

on collectElement(el, depth, chunks)
  if depth > 5 then return chunks
  try
    set chunks to my addChunk(chunks, name of el)
  end try
  try
    set chunks to my addChunk(chunks, value of el)
  end try
  try
    set chunks to my addChunk(chunks, description of el)
  end try
  try
    set chunks to my addChunk(chunks, title of el)
  end try
  if (count of chunks) > 180 then return chunks
  try
    repeat with child in UI elements of el
      set chunks to my collectElement(child, depth + 1, chunks)
      if (count of chunks) > 180 then exit repeat
    end repeat
  end try
  return chunks
end collectElement

tell application "System Events"
  set excludedNames to {"${safeAppName}", "Electron"}
  set targetProcess to missing value
  try
    set frontProcess to first application process whose frontmost is true
    set frontName to name of frontProcess
    if excludedNames does not contain frontName then set targetProcess to frontProcess
  end try
  if targetProcess is missing value then
    repeat with proc in application processes
      try
        set procName to name of proc
        if excludedNames does not contain procName and visible of proc is true and background only of proc is false and (count of windows of proc) > 0 then
          set targetProcess to proc
          exit repeat
        end if
      end try
    end repeat
  end if
  if targetProcess is missing value then error "No readable foreground window found."

  set appName to name of targetProcess
  set bundleId to ""
  try
    set bundleId to bundle identifier of targetProcess
  end try
  set windowTitle to ""
  set chunks to {}
  try
    set w to front window of targetProcess
    try
      set windowTitle to name of w
    end try
    set chunks to my collectElement(w, 0, chunks)
  end try
  set AppleScript's text item delimiters to linefeed
  set bodyText to chunks as text
  set AppleScript's text item delimiters to ""
  return appName & tab & bundleId & tab & windowTitle & linefeed & bodyText
end tell
`
}

export async function captureControllerSnapshot(settings: ControllerSettings, appName = 'TensorFold'): Promise<ControllerSnapshot> {
  const status = buildControllerStatus(settings)
  if (!settings.enabled) throw new Error('Controller is disabled.')
  if (!status.supported) throw new Error(status.message)
  if (!status.accessibilityTrusted) throw new Error(status.message)

  const raw = await runOsa(captureScript(appName))
  const [head = '', ...body] = raw.trimEnd().split('\n')
  const [capturedApp = '', bundleId = '', windowTitle = ''] = head.split('\t')
  const maxChars = Math.max(2_000, settings.maxTextChars || DEFAULT_MAX_TEXT)
  const fullText = settings.includeAccessibilityText ? body.join('\n').trim() : ''
  const truncated = fullText.length > maxChars
  const text = truncated ? fullText.slice(0, maxChars) : fullText
  const warning = !settings.includeAccessibilityText
    ? 'Text capture is off; only app/window metadata was captured.'
    : fullText ? null : 'No accessible text was exposed by this window.'
  return {
    appName: capturedApp || 'Unknown app',
    bundleId: bundleId || null,
    windowTitle: windowTitle || 'Untitled window',
    capturedAt: Date.now(),
    platform: process.platform,
    text,
    textCharCount: fullText.length,
    truncated,
    source: 'accessibility',
    warning,
  }
}

export async function runControllerAction(settings: ControllerSettings, input: ControllerActionInput, ownAppName = 'TensorFold'): Promise<ControllerActionResult> {
  const status = buildControllerStatus(settings)
  if (!settings.enabled) throw new Error('Controller is disabled.')
  if (!status.supported) throw new Error(status.message)
  if (!status.accessibilityTrusted) throw new Error(status.message)

  const normalized = normalizeControllerActionInput(input as unknown as Record<string, unknown>, { ownAppName })
  await runOsa(controllerActionScript(normalized))
  return {
    ok: true,
    action: normalized.action,
    appName: normalized.appName,
    target: controllerActionTarget(normalized),
    executedAt: Date.now(),
    note: normalized.note,
  }
}
