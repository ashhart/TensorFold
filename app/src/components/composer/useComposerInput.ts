import { useEffect, useLayoutEffect, useMemo, useRef, useState, type Dispatch, type RefObject, type SetStateAction } from 'react'
import { useStore } from '../../lib/store'
import { api } from '../../lib/ipc'
import type { Settings, Thread, Skill, Command, ControllerSnapshot } from '../../lib/types'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { currentIntlLocale } from '../../lib/preferences/current-locale'
import { COMPOSER_INSERT_EVENT } from '../../lib/browser-preview'
import { insertTextareaText } from '../../lib/textarea-editing'
import { buildTerminalReference } from '../../lib/terminal-context'
import { BUILTIN_CAPABILITIES, capabilityInstruction, capabilityToken, type BuiltInCapability } from '../../lib/capabilities'
import { dictationSupported, startDictation } from '../../lib/voice'
import type { useConversationDraft } from '../../lib/use-conversation-draft'
import type { ComposerWorkspace } from './useComposerWorkspace'

function formatControllerSnapshot(snapshot: ControllerSnapshot): string {
  const lines = [
    '<controller_window permission="explicit-user-attach">',
    `Captured: ${new Date(snapshot.capturedAt).toLocaleString(currentIntlLocale())}`,
    'Permission: User clicked Controller and chose to insert this foreground-window capture.',
    'Capture scope: foreground macOS Accessibility snapshot; not passive screen scraping.',
    `App: ${snapshot.appName}${snapshot.bundleId ? ` (${snapshot.bundleId})` : ''}`,
    `Window: ${snapshot.windowTitle}`,
    `Source: ${snapshot.source}${snapshot.truncated ? ' (truncated)' : ''}`,
  ]
  if (snapshot.warning) lines.push(`Warning: ${snapshot.warning}`)
  if (snapshot.text) lines.push('', 'Accessible text:', snapshot.text)
  lines.push('</controller_window>')
  return lines.join('\n')
}

interface ComposerInputOptions {
  mode: 'home' | 'thread'
  thread?: Thread
  seed?: { text: string; n: number }
  settings: Settings
  draft: ReturnType<typeof useConversationDraft>
  work: ComposerWorkspace
  taRef: RefObject<HTMLTextAreaElement>
  addFiles: (selection: FileList | File[], replacingId?: string) => Promise<void>
  setDragOver: Dispatch<SetStateAction<boolean>>
}

export function useComposerInput({ mode, thread, seed, settings, draft, work, taRef, addFiles, setDragOver }: ComposerInputOptions) {
  const tx = useAuthoredText()
  const { text, setText } = draft
  const { workCwd, workspace, activeTerminalSession } = work
  const isChat = mode === 'thread' && thread?.kind === 'chat'
  const pendingMention = useStore((s) => s.pendingMention)
  const composerSeed = useStore((s) => s.composerSeed)
  const [files, setFiles] = useState<string[]>([])
  const filesCwd = useRef('')
  const [mention, setMention] = useState<{ query: string; start: number } | null>(null)
  const [mIdx, setMIdx] = useState(0)

  // $-skill / /-command picker state
  const [skills, setSkills] = useState<Skill[]>([])
  const [commands, setCommands] = useState<Command[]>([])
  const [pick, setPick] = useState<{ kind: '/' | '$'; query: string; start: number } | null>(null)
  const [pIdx, setPIdx] = useState(0)
  const [controllerModal, setControllerModal] = useState(false)
  const [controllerPreview, setControllerPreview] = useState<ControllerSnapshot | null>(null)
  const [controllerBusy, setControllerBusy] = useState(false)
  const [controllerError, setControllerError] = useState('')
  const [dictating, setDictating] = useState(false)
  const [dictationError, setDictationError] = useState('')
  const dictationStopRef = useRef<(() => void) | null>(null)
  useEffect(() => { api.getSkills().then(setSkills).catch(() => {}); api.getCommands().then(setCommands).catch(() => {}) }, [])
  useEffect(() => () => { dictationStopRef.current?.() }, [])

  const ensureFiles = () => {
    const workspaceKey = `${workspace?.kind ?? 'local'}:${workspace?.kind === 'ssh' ? workspace.hostId : ''}:${workCwd}`
    if (filesCwd.current === workspaceKey && files.length) return
    filesCwd.current = workspaceKey
    if (!workCwd) { setFiles([]); return }
    if (workCwd) api.listFiles(workCwd, workspace)
      .then((next) => { if (filesCwd.current === workspaceKey) setFiles(next) })
      .catch(() => { if (filesCwd.current === workspaceKey) setFiles([]) })
  }

  const items = useMemo(() => {
    if (!mention || thread?.bot?.executionTarget === 'vm') return []
    const q = mention.query.toLowerCase()
    const score = (f: string) => {
      const base = (f.split('/').pop() || '').toLowerCase()
      if (!q) return 0
      if (base.startsWith(q)) return 0
      if (base.includes(q)) return 1
      if (f.toLowerCase().includes(q)) return 2
      return 3
    }
    return files
      .filter((f) => !q || f.toLowerCase().includes(q))
      .sort((a, b) => score(a) - score(b) || a.length - b.length)
      .slice(0, 8)
  }, [mention, files, thread?.bot?.executionTarget])
  const capabilityItems = useMemo(() => {
    if (!mention || thread?.bot?.executionTarget === 'vm') return []
    const q = mention.query.toLowerCase()
    return BUILTIN_CAPABILITIES
      .filter((capability) => !q || capability.name.toLowerCase().includes(q) || capability.mention.includes(q) || capability.description.toLowerCase().includes(q))
      .slice(0, 8)
  }, [mention, thread?.bot?.executionTarget])
  useEffect(() => {
    const total = capabilityItems.length + items.length
    if (total > 0 && mIdx >= total) setMIdx(0)
  }, [capabilityItems.length, items.length, mIdx])

  // Built-in slash commands (thread-scoped) shown alongside user commands.
  const builtins = useMemo<{ name: string; description: string; argHint?: string }[]>(
    () => (mode === 'thread' && !thread?.bot
      ? [{ name: 'compact', description: tx("Summarize the conversation to reclaim context") },
         { name: 'goal', description: tx("Set a goal the agent verifies against"), argHint: '<text…>' },
         { name: 'computer-use', description: tx("Let the agent operate Mac apps for this task, with approval per app"), argHint: '<task… | stop>' }]
      : []),
    [mode, thread?.bot, tx],
  )
  const pickItems = useMemo<{ name: string; description?: string; argHint?: string }[]>(() => {
    if (!pick) return []
    const q = pick.query.toLowerCase()
    if (pick.kind === '$') return skills.filter((s) => s.enabled && (!q || s.name.toLowerCase().includes(q))).slice(0, 8)
    return [...builtins, ...commands].filter((c) => !q || c.name.toLowerCase().includes(q)).slice(0, 8)
  }, [pick, skills, commands, builtins])
  useLayoutEffect(() => {
    const ta = taRef.current
    if (!ta) return
    ta.style.height = 'auto'
    ta.style.height = Math.min(ta.scrollHeight, 224) + 'px'
  }, [text])

  useEffect(() => { taRef.current?.focus() }, [mode, thread?.id])
  useEffect(() => { if (seed && seed.text) { setText(seed.text); taRef.current?.focus() } }, [seed?.n])
  // Injected composer text (VST-040 ask-why → chat reference): insert at the caret, focus.
  useEffect(() => {
    if (!composerSeed?.text) return
    const ta = taRef.current
    if (ta) setText(insertTextareaText(ta, composerSeed.text))
    useStore.setState({ composerSeed: null })
  }, [composerSeed?.n])
  // File explorer → reference a file in the composer.
  useEffect(() => {
    if (!pendingMention) return
    const ta = taRef.current
    if (ta) setText(insertTextareaText(ta, `@${pendingMention.path} `))
    useStore.setState({ pendingMention: null })
  }, [pendingMention?.n]) // eslint-disable-line
  const insertAtCaret = (insert: string) => {
    const ta = taRef.current
    if (!ta) return
    const caret = ta.selectionStart
    const pad = caret > 0 && !/\s/.test(text[caret - 1]) ? '\n\n' : ''
    setText(insertTextareaText(ta, pad + insert))
  }
  useEffect(() => {
    const onInsert = (event: Event) => {
      const detail = (event as CustomEvent<{ text?: string }>).detail
      if (typeof detail?.text === 'string' && detail.text.trim()) insertAtCaret(detail.text)
    }
    window.addEventListener(COMPOSER_INSERT_EVENT, onInsert)
    return () => window.removeEventListener(COMPOSER_INSERT_EVENT, onInsert)
  }, [text])
  const detectMention = (v: string, caret: number) => {
    const m = /(?:^|\s)@([^\s@]*)$/.exec(v.slice(0, caret))
    if (m) { setMention({ query: m[1], start: caret - m[1].length - 1 }); setMIdx(0); ensureFiles() }
    else if (mention) setMention(null)
  }
  // $skill anywhere; /command only as the leading token.
  const detectPick = (v: string, caret: number) => {
    const before = v.slice(0, caret)
    const sm = /(?:^|\s)\$([\w-]*)$/.exec(before)
    if (sm) { setPick({ kind: '$', query: sm[1], start: caret - sm[1].length - 1 }); setPIdx(0); return }
    const cm = /^\/([\w-]*)$/.exec(before)
    if (cm) { setPick({ kind: '/', query: cm[1], start: 0 }); setPIdx(0); return }
    if (pick) setPick(null)
  }
  const onChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setText(e.target.value)
    const caret = e.target.selectionStart ?? e.target.value.length
    if (!isChat) detectMention(e.target.value, caret)
    detectPick(e.target.value, caret)
  }
  const selectMention = (file: string) => {
    if (!mention) return
    const ta = taRef.current!
    const caret = ta.selectionStart ?? text.length
    const insert = `@${file} `
    setText(insertTextareaText(ta, insert, mention.start, Math.max(caret, ta.selectionEnd)))
    setMention(null)
    requestAnimationFrame(() => { const pos = mention.start + insert.length; ta.focus(); ta.setSelectionRange(pos, pos) })
  }
  const selectCapability = (capability: BuiltInCapability) => {
    if (!mention) return
    const ta = taRef.current!
    const caret = ta.selectionStart ?? text.length
    const insert = `${capabilityToken(capability)}\n${capabilityInstruction(capability)}\n\n`
    setText(insertTextareaText(ta, insert, mention.start, Math.max(caret, ta.selectionEnd)))
    setMention(null)
    requestAnimationFrame(() => { const pos = mention.start + insert.length; ta.focus(); ta.setSelectionRange(pos, pos) })
  }
  // Pick a skill (inserts its body inline) or a command (completes the /name).
  const selectPick = (name: string) => {
    if (!pick) return
    const ta = taRef.current!
    const caret = ta.selectionStart ?? text.length
    let insert: string
    if (pick.kind === '$') { const s = skills.find((x) => x.name === name); insert = (s?.body?.trim() || `$${name}`) + ' ' }
    else insert = `/${name} `
    setText(insertTextareaText(ta, insert, pick.start, Math.max(caret, ta.selectionEnd)))
    setPick(null)
    requestAnimationFrame(() => { const pos = pick.start + insert.length; ta.focus(); ta.setSelectionRange(pos, pos) })
  }
  // Reference terminal output (VST-039): pull the active terminal's backlog and insert a
  // bounded, ANSI-stripped reference at the caret so the agent can act on what just ran.
  const referenceTerminalOutput = async () => {
    if (!activeTerminalSession) return
    const originalEditor = taRef.current
    let backlog = ''
    // `terminalBacklog` is host-local. Never probe it for an Orchestrator session: a same-id local
    // PTY would leak the wrong output, and legacy remote rows intentionally fail closed.
    if (activeTerminalSession.hostLocation !== 'orchestrator') {
      try { backlog = await api.terminalBacklog(activeTerminalSession.id) } catch { /* keep empty → honest "no output" note */ }
    }
    const reference = buildTerminalReference(activeTerminalSession.title || 'terminal', backlog)
    const ta = taRef.current
    if (!ta || ta !== originalEditor) return
    const caret = ta?.selectionStart ?? text.length
    const pad = caret > 0 && text[caret - 1] && !/\s/.test(text[caret - 1]) ? '\n' : ''
    setText(insertTextareaText(ta, pad + reference))
    requestAnimationFrame(() => { const pos = caret + pad.length + reference.length; ta?.focus(); ta?.setSelectionRange(pos, pos) })
  }
  const onAttach = () => {
    const ta = taRef.current
    if (!ta) return
    const caret = ta.selectionStart ?? text.length
    const pad = caret > 0 && !/\s/.test(text[caret - 1]) ? ' ' : ''
    const insert = pad + '@'
    setText(insertTextareaText(ta, insert))
    ensureFiles()
    const pos = caret + insert.length
    setMention({ query: '', start: pos - 1 }); setMIdx(0)
    requestAnimationFrame(() => { ta.focus(); ta.setSelectionRange(pos, pos) })
  }
  // "Use a skill" from the overflow menu: type a $ at the caret and open the skill picker.
  const startSkillPick = () => {
    const ta = taRef.current
    if (!ta) return
    const caret = ta.selectionStart ?? text.length
    const pad = caret > 0 && !/\s/.test(text[caret - 1]) ? ' ' : ''
    const insert = pad + '$'
    setText(insertTextareaText(ta, insert))
    const pos = caret + insert.length
    setPick({ kind: '$', query: '', start: pos - 1 }); setPIdx(0)
    requestAnimationFrame(() => { ta.focus(); ta.setSelectionRange(pos, pos) })
  }
  const attachControllerSnapshot = async () => {
    setControllerBusy(true)
    setControllerError('')
    try {
      const snapshot = await api.captureController()
      setControllerPreview(snapshot)
      setControllerModal(true)
    } catch (e: any) {
      setControllerError(String(e?.message || e || 'Controller could not capture the current window.').replace(/^Error invoking remote method '[^']+':\s*/, '').replace(/^Error:\s*/, ''))
      setControllerModal(true)
    } finally {
      setControllerBusy(false)
    }
  }
  const onController = async () => {
    setControllerBusy(true)
    setControllerError('')
    try {
      const status = await api.getControllerStatus()
      if (!status.canCapture) { setControllerModal(true); return }
      await attachControllerSnapshot()
    } catch (e: any) {
      setControllerError(String(e?.message || e || 'Controller is unavailable.').replace(/^Error invoking remote method '[^']+':\s*/, '').replace(/^Error:\s*/, ''))
      setControllerModal(true)
    } finally {
      setControllerBusy(false)
    }
  }
  const enableControllerAndCapture = async () => {
    setControllerBusy(true)
    setControllerError('')
    try {
      const status = await api.enableController()
      if (!status.canCapture) {
        setControllerError(status.message)
        return
      }
    } catch (e: any) {
      setControllerError(String(e?.message || e || 'Controller could not be enabled.').replace(/^Error invoking remote method '[^']+':\s*/, '').replace(/^Error:\s*/, ''))
      return
    } finally {
      setControllerBusy(false)
    }
    await attachControllerSnapshot()
  }
  const insertControllerPreview = () => {
    if (!controllerPreview) return
    insertAtCaret(formatControllerSnapshot(controllerPreview))
    setControllerPreview(null)
    setControllerModal(false)
  }
  const toggleDictation = () => {
    if (dictating) {
      dictationStopRef.current?.()
      dictationStopRef.current = null
      setDictating(false)
      return
    }
    setDictationError('')
    if (!dictationSupported(settings.voice)) {
      setDictationError(tx("Speech-to-text is not available in this runtime."))
      return
    }
    setDictating(true)
    dictationStopRef.current = startDictation(settings.voice, {
      onText: (spoken) => {
        setText((current) => `${current}${current && !/\s$/.test(current) ? ' ' : ''}${spoken}`)
        taRef.current?.focus()
      },
      onError: (message) => setDictationError(message),
      onEnd: () => {
        dictationStopRef.current = null
        setDictating(false)
      },
    })
  }
  // Files are read as explicit attachments; directories retain the workspace-reference flow.
  const onDrop = (e: React.DragEvent) => {
    const items = Array.from(e.dataTransfer.items || [])
    const files = Array.from(e.dataTransfer.files || [])
    if (!items.length && !files.length) return
    e.preventDefault(); setDragOver(false)
    const refs: string[] = []
    const selectedFiles: File[] = []
    const n = Math.max(items.length, files.length)
    for (let i = 0; i < n; i++) {
      const entry = (items[i] as any)?.webkitGetAsEntry?.()
      const f = files[i] as any
      const name: string | undefined = entry?.name || f?.name
      if (!name) continue
      const abs: string | undefined = f?.path // Electron exposes the absolute path
      const isDir = !!entry?.isDirectory
      if (!isDir && f) { selectedFiles.push(f); continue }
      let ref = abs || name
      if (abs && workCwd && abs.startsWith(workCwd)) ref = abs.slice(workCwd.length).replace(/^[/\\]/, '') || name
      refs.push(`@${ref}${isDir ? '/' : ''}`)
    }
    if (selectedFiles.length) void addFiles(selectedFiles)
    if (refs.length) setText((t) => `${t}${t && !t.endsWith(' ') ? ' ' : ''}${refs.join(' ')} `)
    taRef.current?.focus()
  }
  useEffect(() => () => { dictationStopRef.current?.(); dictationStopRef.current = null }, [draft.scopeKey])

  return { skills, commands, mention, setMention, mIdx, setMIdx, items, capabilityItems,
    pick, setPick, pIdx, setPIdx, pickItems, selectMention, selectCapability, selectPick,
    onChange, onAttach, startSkillPick, referenceTerminalOutput, onDrop,
    controllerModal, setControllerModal, controllerPreview, setControllerPreview, controllerBusy, controllerError,
    onController, enableControllerAndCapture, insertControllerPreview, dictating, dictationError, toggleDictation }
}

export type ComposerInput = ReturnType<typeof useComposerInput>
