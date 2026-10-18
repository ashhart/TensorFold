import assert from 'node:assert/strict'
import { mkdtemp, readFile, rm } from 'node:fs/promises'
import { spawnSync } from 'node:child_process'
import os from 'node:os'
import path from 'node:path'
const profile = await mkdtemp(path.join(os.tmpdir(), 'tensorfold-quoted-drafts-'))
process.env.VONTRA_HOME = profile
const { writeComposerDraft, readComposerDraft, composerDraftFile } = await import('./composer-drafts')
const scope = { surface: 'chat' as const, conversationId: 'synthetic-quote-owner', hostId: null, incognito: false }
const quote = { id: 'synthetic-quote', sourceItemId: 'synthetic-source', sourceRevision: 'a'.repeat(64), sourceText: 'SYNTHETIC-QUOTE-ORIGINAL', text: 'SYNTHETIC-QUOTE-EDITED' }
try {
  await writeComposerDraft({ scope, expectedRevision: 0, text: 'Unfinished synthetic reply', attachments: [], quotes: [quote] })
  assert.deepEqual((await readComposerDraft(scope)).quotes, [quote])
  const raw = await readFile(composerDraftFile(scope), 'utf8')
  assert.match(raw, /safe-json:v1:/); assert.ok(!raw.includes(quote.text) && !raw.includes(quote.sourceText))
  const result = spawnSync(process.execPath, ['-e', `const assert=require('node:assert/strict'); const {createJiti}=require('jiti'); createJiti(process.cwd()+'/quotation-restart.cjs').import('./electron/store/composer-drafts').then(async m=>{const value=await m.readComposerDraft(${JSON.stringify(scope)});assert.deepEqual(value.quotes,${JSON.stringify([quote])});assert.equal((await m.readComposerDraft({...${JSON.stringify(scope)},conversationId:'synthetic-other'})).quotes,undefined);console.log('protected quote restart passed')}).catch(e=>{console.error(e);process.exitCode=1})`], { cwd: process.cwd(), env: process.env, encoding: 'utf8', timeout: 15_000 })
  assert.equal(result.status, 0, result.stderr); assert.match(result.stdout, /restart passed/)
} finally { await rm(profile, { recursive: true, force: true }) }
console.log('Protected quoted drafts recover in a fresh process without leaking source text into disk JSON or another conversation')
