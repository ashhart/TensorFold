// Behavioral tests for CLI providers as room seats (room-cli-seat.ts + the rooms.ts routing):
// a CLI-backed seat runs its turn as a one-shot confined CLI invocation, aborts kill the
// child process group, a hung CLI seat is skipped on the per-turn timeout, and HTTP seats
// still route to streamChat.
import assert from 'node:assert/strict'
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, realpathSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { isCliSeatProvider, runCliSeatTurn } from './room-cli-seat.ts'
import { runRoom } from './rooms.ts'

const ok = (msg: string) => console.log(`ok - ${msg}`)

const provider = (id: string, extra: Record<string, unknown> = {}): any => ({ id, name: id, baseUrl: 'http://local/v1', apiKey: '', kind: 'local', models: [], ...extra })
const seat = (id: string, name: string, providerId: string, model: string): any => ({ id, name, role: name, providerId, providerAccountId: null, model })
const makeRoom = (seats: any[]): any => ({
  id: 'room1', name: 'Room', problem: '', seats, deliberationEffort: 'quick',
  events: [], memory: [], memorySummary: '', state: 'queued', finalText: '', seatHeartbeats: {},
  collaborationMode: 'huddle', tickets: [],
})
const project: any = { id: 'p1', name: 'P', cwd: '/tmp/room', workspace: null }

// A fake streamChat that records calls and replays scripted replies per model.
function recordingChat(scripts: Record<string, string[]>) {
  const calls: { model: string }[] = []
  const q: Record<string, string[]> = Object.fromEntries(Object.entries(scripts).map(([m, a]) => [m, [...a]]))
  const chat = async function* (p: any) {
    calls.push({ model: p.model })
    const reply = (q[p.model] && q[p.model].shift()) || 'noted\nACTION: YIELD'
    yield { type: 'content' as const, text: reply }
  }
  return { chat: chat as any, calls }
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

async function waitFor(cond: () => boolean, timeoutMs = 5_000): Promise<boolean> {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    if (cond()) return true
    await sleep(25)
  }
  return cond()
}

function pidAlive(pid: number): boolean {
  try {
    process.kill(pid, 0)
    return true
  } catch (error: any) {
    return error?.code !== 'ESRCH'
  }
}

// CLI seats run planning-only: each turn gets a throwaway private folder in the OS temp dir.
const planning = realpathSync(tmpdir())
const ownPlanningDirs = () => readdirSync(planning).filter((d) => d.startsWith(`tf-plan-${process.pid}-`)).map((d) => path.join(planning, d))

async function run() {
  // 1) DETECTION — the five one-shot CLI transports are seat-CLI; HTTP and the OpenCode/Kilo
  // server protocols (no read-only prompt bridge) are not.
  {
    for (const transport of ['codex-cli', 'claude-cli', 'pi-cli', 'cursor-cli', 'gemini-cli']) {
      assert.equal(isCliSeatProvider(provider('p', { transport })), true, `${transport} is a CLI seat transport`)
    }
    assert.equal(isCliSeatProvider(provider('p')), false, 'no transport means HTTP')
    assert.equal(isCliSeatProvider(provider('p', { transport: 'chat-completions' })), false, 'chat-completions stays on streamChat')
    assert.equal(isCliSeatProvider(provider('p', { transport: 'opencode' })), false, 'opencode has no read-only prompt bridge')
    assert.equal(isCliSeatProvider(provider('p', { transport: 'kilo' })), false, 'kilo has no read-only prompt bridge')
    assert.equal(isCliSeatProvider(null), false)
    assert.equal(isCliSeatProvider(undefined), false)
    ok('CLI seat provider detection')
  }

  // 2) ACCUMULATION — content chunks join into the reply and stream as deltas; reasoning
  // status lines and resume cursors are dropped.
  {
    const source = async function* () {
      yield { type: 'reasoning', text: 'Claude thinking\n' }
      yield { type: 'content', text: 'Part one. ' }
      yield { type: 'content', text: 'Part two.\nACTION: SPEAK done' }
      yield { type: 'resume_cursor', cursor: { sessionId: 'x' } }
    }
    const deltas: string[] = []
    const reply = await runCliSeatTurn(
      { provider: provider('p', { transport: 'claude-cli' }), model: 'm', messages: [], signal: new AbortController().signal },
      (text) => deltas.push(text),
      source as any,
    )
    assert.equal(reply, 'Part one. Part two.\nACTION: SPEAK done', 'content chunks accumulate into the reply')
    assert.deepEqual(deltas, ['Part one. ', 'Part two.\nACTION: SPEAK done'], 'deltas stream in order')
    ok('runCliSeatTurn accumulates content and streams deltas')
  }

  // 3) REAL DISPATCH — a fake `claude` binary proves the one-shot is confined: plan
  // permission mode, no tools, no --resume, system prompt kept out of argv, room context on
  // stdin, run from a planning folder. It can't write beside itself (the seatbelt), so it
  // reports what it saw inside its reply.
  {
    const dir = mkdtempSync(path.join(tmpdir(), 'vontra-room-cli-seat-'))
    const fakeBin = path.join(dir, 'claude')
    writeFileSync(fakeBin, `#!/usr/bin/env node
const fs = require('node:fs')
const path = require('node:path')
let blocked = 'no'
try { fs.writeFileSync(path.join(__dirname, 'args.json'), '[]') } catch (e) { blocked = e.code }
let stdin = ''
process.stdin.on('data', (chunk) => { stdin += String(chunk) })
process.stdin.on('end', () => {
  const report = Buffer.from(JSON.stringify({ args: process.argv.slice(2), stdin, cwd: process.cwd(), blocked })).toString('base64')
  process.stdout.write(JSON.stringify({ type: 'system', subtype: 'init', session_id: 'seat-1' }) + '\\n')
  process.stdout.write(JSON.stringify({ type: 'stream_event', event: { type: 'content_block_delta', delta: { type: 'text_delta', text: 'CLI seat reply' } } }) + '\\n')
  process.stdout.write(JSON.stringify({ type: 'stream_event', event: { type: 'content_block_delta', delta: { type: 'text_delta', text: '\\nREPORT ' + report } } }) + '\\n')
  process.stdout.write(JSON.stringify({ type: 'result', subtype: 'success', is_error: false, result: 'CLI seat reply', session_id: 'seat-1' }) + '\\n')
})
`)
    chmodSync(fakeBin, 0o755)

    const systemPrimer = 'PRIMER_SECRET: you are Alpha. End every reply with exactly one line: ACTION: <VERB> <content>.'
    const reply = await runCliSeatTurn({
      provider: provider('pC', { transport: 'claude-cli', cliPath: fakeBin }),
      model: 'claude-code-default',
      messages: [
        { role: 'system', content: systemPrimer },
        { role: 'user', content: '[moderator] Alpha, your turn.' },
      ] as any,
      cwd: null,
      signal: new AbortController().signal,
    }, () => {})

    const [seatReply, encoded] = reply.split('\nREPORT ')
    assert.equal(seatReply, 'CLI seat reply', 'the CLI text output is the seat reply')
    const report = JSON.parse(Buffer.from(encoded, 'base64').toString('utf8')) as { args: string[]; stdin: string; cwd: string; blocked: string }
    const args = report.args
    assert.equal(report.blocked, 'EPERM', 'the seatbelt blocks writes outside the planning folder')
    assert.ok(report.cwd.startsWith(path.join(planning, `tf-plan-${process.pid}-`)), 'the seat runs from a planning folder')
    assert.deepEqual(args.slice(0, 2), ['--tools', ''], 'claude tools are switched off')
    const modeAt = args.indexOf('--permission-mode')
    assert.equal(args[modeAt + 1], 'plan', 'readonly rooms map claude to plan mode (no edits)')
    assert.equal(args.includes('--resume'), false, 'a seat turn never resumes an agentic session')
    assert.equal(args.some((arg) => arg.includes('PRIMER_SECRET')), false, 'the system primer stays out of argv')
    const stdin = report.stdin
    assert.ok(stdin.includes('Alpha, your turn'), 'the room context reaches the CLI prompt')
    assert.ok(!stdin.includes('PRIMER_SECRET'), 'the system primer goes through the prompt file, not stdin')
    ok('claude dispatch is a confined one-shot (plan mode, no resume, primer via file)')
  }

  // 4) ROUTING — a CLI seat runs through cliSeatTurn (with the room primer and ACTION
  // protocol in its messages) while an HTTP seat still routes to streamChat; the room
  // finalizes through the CLI seat's ACTION line.
  {
    const { chat, calls } = recordingChat({
      mB: ['I propose we cache results in a Map.\nACTION: SPEAK'],
    })
    const cliCalls: { model: string; messages: { role: string; content: string }[] }[] = []
    const deltas: any[] = []
    const events: any[] = []
    const room = makeRoom([seat('s1', 'Beta', 'pH', 'mB'), seat('s2', 'Alpha', 'pC', 'mA')])
    await runRoom({
      room, project,
      providers: [provider('pH'), provider('pC', { transport: 'claude-cli' })],
      problem: 'Speed up lookups?', temperature: 0.2, reasoningEffort: 'medium',
      emit: (e) => { events.push(e); if (e.type === 'delta') deltas.push(e) },
      signal: new AbortController().signal, maxTurns: 6, chat,
      cliSeatTurn: async (params, onDelta) => {
        cliCalls.push({ model: params.model, messages: params.messages.map((m: any) => ({ role: m.role, content: String(m.content ?? '') })) })
        const text = 'Agreed, a Map cache is right.'
        onDelta(text)
        return `${text}\nACTION: FINAL Use a Map cache for O(1) lookups.`
      },
    })

    assert.equal(cliCalls.length, 1, 'the CLI seat ran through the CLI turn path')
    assert.equal(cliCalls[0].model, 'mA', 'the CLI seat kept its own model')
    assert.ok(cliCalls[0].messages.some((m) => m.role === 'system' && m.content.includes('ACTION: <VERB>')),
      'the CLI seat prompt carries the primer with the ACTION protocol')
    assert.ok(cliCalls[0].messages.some((m) => m.content.includes('cache results in a Map')),
      "the CLI seat saw the HTTP seat's broadcast")
    assert.ok(calls.every((c) => c.model === 'mB') && calls.length > 0, 'the HTTP seat still routes to streamChat')
    assert.ok(deltas.some((d) => d.seat === 'Alpha' && d.chunk.includes('Agreed')), 'CLI seat deltas stream to the room')
    assert.ok(events.some((e) => e.type === 'final'), 'the room reached FINAL via the CLI seat')
    assert.equal(room.state, 'complete')
    ok('CLI seats route to the one-shot path; HTTP seats stay on streamChat')
  }

  // 5) TIMEOUT — a hung CLI seat is skipped on the per-turn timeout; the HTTP seat finalizes.
  {
    const { chat } = recordingChat({
      mB: ['Alpha is stuck; I will decide.\nACTION: FINAL Proceed with plan B.'],
    })
    const events: any[] = []
    const room = makeRoom([seat('s1', 'Alpha', 'pC', 'mA'), seat('s2', 'Beta', 'pH', 'mB')])
    await runRoom({
      room, project,
      providers: [provider('pC', { transport: 'claude-cli' }), provider('pH')],
      problem: 'Decide', temperature: 0.2, reasoningEffort: 'medium',
      emit: (e) => events.push(e),
      signal: new AbortController().signal, maxTurns: 6, turnTimeoutMs: 40, chat,
      cliSeatTurn: (params) => new Promise<string>((_, rej) => {
        params.signal?.addEventListener('abort', () => rej(new Error('aborted')), { once: true })
      }),
    })

    assert.ok(events.some((e) => e.type === 'error' && /timed out/i.test(e.message)), 'the hung CLI seat timed out and was skipped')
    assert.ok(events.some((e) => e.type === 'final'), 'the HTTP seat still finalized')
    assert.equal(room.state, 'complete')
    ok('a hung CLI seat is skipped on the per-turn timeout')
  }

  // 6) ABORT — aborting the room mid CLI turn kills the whole child process group
  // (leader plus descendants), not just the CLI leader.
  {
    const dir = mkdtempSync(path.join(tmpdir(), 'vontra-room-cli-abort-'))
    const binDir = path.join(dir, 'bin')
    mkdirSync(binDir)
    const fakeBin = path.join(binDir, 'claude')
    // A fake `claude` that spawns a descendant in its process group and then hangs. Its
    // planning folder is the only place it may write.
    writeFileSync(fakeBin, `#!/bin/sh
sleep 60 &
echo $! > child.pid
exec sleep 60
`)
    chmodSync(fakeBin, 0o755)
    const pidFile = () => ownPlanningDirs().map((d) => path.join(d, 'child.pid')).find((f) => existsSync(f))

    const ctrl = new AbortController()
    const events: any[] = []
    const room = makeRoom([seat('s1', 'Alpha', 'pC', 'claude-code-default')])
    const run = runRoom({
      room, project: { ...project, cwd: path.join(dir, 'ws') },
      providers: [provider('pC', { transport: 'claude-cli', cliPath: fakeBin })],
      problem: 'Think forever', temperature: 0.2, reasoningEffort: 'medium',
      emit: (e) => events.push(e),
      signal: ctrl.signal, maxTurns: 4,
    })

    assert.ok(await waitFor(() => !!pidFile()), 'the CLI child spawned its descendant')
    const childPid = Number(readFileSync(pidFile()!, 'utf8').trim())
    assert.ok(childPid > 0 && pidAlive(childPid), 'the descendant is alive before abort')
    ctrl.abort()
    await run

    assert.equal(room.state, 'stopped', 'the room stops on abort')
    assert.ok(await waitFor(() => !pidAlive(childPid)), 'abort killed the CLI process group (descendant is gone)')
    ok('room abort kills the CLI child process group')
  }

  console.log('room-cli-seat tests passed')
}

run().catch((error) => {
  console.error(error)
  process.exit(1)
})
