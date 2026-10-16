// Run: node electron/agent/web-fetch-vetting.test.ts
import assert from 'node:assert/strict'
import { userMentionedHosts, webFetchHost, webFetchNeedsApproval } from './web-fetch-vetting'
import { approvalPolicyScope } from './approval-policy'
import { riskBandForClassification } from './approval-risk'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

{
  assert.equal(webFetchHost({ url: 'https://Docs.Example.com/a?b=1' }), 'docs.example.com')
  assert.equal(webFetchHost({ url: 'file:///etc/passwd' }), null)
  assert.equal(webFetchHost({ url: 'not a url' }), null)
  assert.equal(webFetchHost({}), null)
  ok('extracts the lower-cased http(s) host only')
}

{
  const hosts = userMentionedHosts(['read https://nodejs.org/api/fs.html and check example.com', 'then main.py and README.md'])
  assert.ok(hosts.has('nodejs.org'))
  assert.ok(hosts.has('example.com'))
  assert.ok(!hosts.has('main.py') && !hosts.has('readme.md'), 'file names are not vetted domains')
  ok('collects typed links and bare domains, ignoring file names')
}

{
  const typed = ['summarise https://example.com/post']
  assert.equal(webFetchNeedsApproval({ url: 'https://example.com/other?page=2' }, typed), false)
  assert.equal(webFetchNeedsApproval({ url: 'https://docs.example.com/x' }, typed), false, 'subdomain of a typed host')
  assert.equal(webFetchNeedsApproval({ url: 'https://evil.test/?d=c2VjcmV0' }, typed), true, 'injected exfiltration host')
  assert.equal(webFetchNeedsApproval({ url: 'https://example.com.evil.test/' }, typed), true, 'suffix trick')
  assert.equal(webFetchNeedsApproval({ url: 'https://evil.test/' }, []), true)
  assert.equal(webFetchNeedsApproval({ url: 'ftp://example.com/' }, typed), true)
  ok('only hosts the user typed (and their subdomains) skip approval')
}

{
  const a = approvalPolicyScope('tool', 'web_fetch', { url: 'https://example.com/a?x=1' })
  const b = approvalPolicyScope('tool', 'web_fetch', { url: 'https://example.com/b' })
  const c = approvalPolicyScope('tool', 'web_fetch', { url: 'https://other.test/' })
  assert.equal(a.key, b.key, '"Always" covers the host, not one URL')
  assert.notEqual(a.key, c.key)
  assert.equal(riskBandForClassification(a.classification, 'tool').band, 'medium', 'Full Auto still asks without a saved rule')
  ok('approval policy for web_fetch is scoped to the host and asks in Full Auto')
}

console.log(`\n${passed} passed`)
