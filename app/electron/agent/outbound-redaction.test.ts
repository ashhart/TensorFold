// Run: npx tsx electron/agent/outbound-redaction.test.ts
import assert from 'node:assert/strict'
import { PlaceholderVault, RedactionRun, createStreamRehydrator, describeWithheld, withheldTotal } from './outbound-redaction'
import type { ApiMessage } from './types'

const ctx = {
  workspace: '/Users/jdoe/code/acme-portal',
  homeDir: '/Users/jdoe',
  username: 'jdoe',
  hostname: 'jdoe-mbp.local',
  terms: ['Acme Corp', 'Project Falcon'],
}

function run(text: string, vault = new PlaceholderVault()) {
  const r = new RedactionRun(vault, ctx)
  return { text: r.text(text), tally: r.tally, vault }
}

// PII kinds each become a stable placeholder and leave the surrounding text alone.
{
  const { text, tally } = run([
    'Mail jane.roe@example.com or call +44 20 7946 0958 / (555) 010-0199.',
    'Server 192.0.2.10 and 2001:db8::8a2e:370:7334; loopback 127.0.0.1 stays.',
    'Card 4111 1111 1111 1111, IBAN GB82 WEST 1234 5698 7654 32 today.',
    'SSN 123-45-6789, NI AB 12 34 56 C.',
    'Repo at /Users/jdoe/code/acme-portal/src/app.ts, notes in /Users/jdoe/notes.md, other /Users/bsmith/x.',
    'Host jdoe-mbp.local (jdoe-mbp), user jdoe, for ACME corp on project falcon.',
  ].join('\n'))
  for (const value of ['jane.roe@example.com', '7946', '010-0199', '192.0.2.10', '2001:db8::8a2e', '4111', 'WEST 1234', '123-45-6789', 'AB 12 34', '/Users/jdoe', 'bsmith', 'jdoe-mbp', 'ACME corp', 'project falcon']) {
    assert.ok(!text.includes(value), `withheld: ${value}\n${text}`)
  }
  assert.match(text, /\[EMAIL_1\]/)
  assert.match(text, /\[WORKSPACE_1\]\/src\/app\.ts/)
  assert.match(text, /\[HOME_1\]\/notes\.md/)
  assert.match(text, /\/Users\/\[USER_\d\]\/x/)
  assert.match(text, /127\.0\.0\.1 stays/)
  assert.match(text, /user \[USER_\d\]/)
  assert.equal(tally.email, 1)
  assert.equal(tally.phone, 2)
  assert.equal(tally.ip, 2)
  assert.equal(tally.card, 1)
  assert.equal(tally.iban, 1)
  assert.equal(tally.ssn, 1)
  assert.equal(tally.nino, 1)
  assert.equal(tally.term, 2)
  assert.equal(tally.workspace, 1)
}

// Luhn-invalid numbers, timestamps, dates and versions are not cards, phones or IDs.
{
  const { text, tally } = run('Build 1728650000000 at 2026-10-11 10:55, node 22.23.2, id 4111 1111 1111 1112.')
  assert.equal(text, 'Build 1728650000000 at 2026-10-11 10:55, node 22.23.2, id 4111 1111 1111 1112.')
  assert.equal(withheldTotal(tally), 0)
}

// Terms are word-bounded and case-insensitive; placeholders are never re-redacted.
{
  const vault = new PlaceholderVault()
  const r = new RedactionRun(vault, { terms: ['Acme', 'TERM', 'email'] })
  assert.equal(r.text('acme, Acmeville, ACME.'), '[TERM_1], Acmeville, [TERM_1].')
  assert.equal(r.text('write to a@example.com'), 'write to [EMAIL_1]')
}

// Secrets are numbered per pass and never rehydrated; credential files are blocked whole.
{
  const { text, tally, vault } = run('use sk-proj-abcdefghijklmnopqrstuvwxyz123456 here')
  assert.equal(text, 'use [SECRET_1] here')
  assert.equal(tally.secret, 1)
  assert.equal(vault.rehydrate(text), text)
  const key = ['-----BEGIN OPENSSH', 'PRIVATE KEY-----\nAAAA\n-----END OPENSSH', 'PRIVATE KEY-----'].join(' ')
  assert.match(run(`here is my key\n${key}`).text, /^\[Withheld: this part held a private key/)
  const envFile = 'API_KEY=abcdef123456\nDB_PASSWORD=hunter2hunter2\nAUTH_TOKEN=zzzzzzzzzzzz'
  const blocked = run(envFile)
  assert.match(blocked.text, /credential file/)
  assert.equal(blocked.tally.credential, 1)
}

// Messages: tool results dropped, tool calls summarized, images withheld, roles valid.
{
  const vault = new PlaceholderVault()
  const r = new RedactionRun(vault, ctx)
  const messages: ApiMessage[] = [
    { role: 'system', content: 'Workspace: /Users/jdoe/code/acme-portal' },
    { role: 'user', content: [{ type: 'text', text: 'look at this from jane.roe@example.com' }, { type: 'image_url', image_url: { url: 'data:image/png;base64,AAAA' } }] },
    { role: 'assistant', content: null, tool_calls: [{ id: 'c1', type: 'function', function: { name: 'read_file', arguments: '{"path":"/Users/jdoe/code/acme-portal/.env"}' } }, { id: 'c2', type: 'function', function: { name: 'read_file', arguments: '{}' } }] },
    { role: 'tool', tool_call_id: 'c1', content: 'SECRET FILE CONTENT' },
    { role: 'tool', tool_call_id: 'c2', content: 'more file content' },
    { role: 'assistant', content: 'Done reading.' },
  ]
  const out = r.messages(messages)
  assert.deepEqual(out.map((m) => m.role), ['system', 'user', 'assistant'])
  assert.equal(out[0].content, 'Workspace: [WORKSPACE_1]')
  assert.match(String(out[1].content), /\[EMAIL_1\]\n\[Image withheld/)
  assert.match(String(out[2].content), /Used tools: read_file x2\. Arguments and results were withheld/)
  assert.match(String(out[2].content), /Done reading\.$/)
  assert.ok(!JSON.stringify(out).includes('FILE CONTENT'))
  assert.ok(out.every((m) => !m.tool_calls && m.role !== 'tool'))
  assert.equal(r.tally.tool_output, 2)
  assert.equal(r.tally.tool_call, 2)
  assert.equal(r.tally.image, 1)
  assert.equal(describeWithheld(r.tally), '2 tool calls, 2 tool results, 1 workspace path, 1 email, 1 image')
}

// Stable across passes: the same value keeps its placeholder in a later request.
{
  const vault = new PlaceholderVault()
  const first = new RedactionRun(vault, ctx).text('ping bob@example.com then jane.roe@example.com')
  const second = new RedactionRun(vault, ctx).text('jane.roe@example.com again')
  assert.equal(first, 'ping [EMAIL_1] then [EMAIL_2]')
  assert.equal(second, '[EMAIL_2] again')
  assert.equal(vault.rehydrate('Reply to [EMAIL_2] and [EMAIL_9].'), 'Reply to jane.roe@example.com and [EMAIL_9].')
}

// Streaming rehydration survives a placeholder split across chunks.
{
  const vault = new PlaceholderVault()
  new RedactionRun(vault, ctx).text('edit /Users/jdoe/code/acme-portal/a.ts for Acme Corp')
  const s = createStreamRehydrator(vault)
  const out = ['Open [WORK', 'SPACE_1]/a.ts for [', 'TERM_1', '] and [link](x) [TE'].map((c) => s.push(c)).join('') + s.flush()
  assert.equal(out, 'Open /Users/jdoe/code/acme-portal/a.ts for Acme Corp and [link](x) [TE')
}

console.log('outbound-redaction tests passed')
