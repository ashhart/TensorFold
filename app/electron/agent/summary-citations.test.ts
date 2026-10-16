import assert from 'node:assert/strict'
import { ensureSummaryCitations } from './summary-citations'
import type { ConnectorSourceRef } from './types'

const sources: ConnectorSourceRef[] = [
  { id: 'email:mail:one', connectorId: 'email', sourceId: 'mail', itemId: 'one', title: 'One', path: '/mail/one.eml', excerpt: 'First message.' },
  { id: 'docs:docs:roadmap', connectorId: 'docs', sourceId: 'docs', itemId: 'roadmap', title: 'roadmap.md', path: '/repo/roadmap.md', excerpt: 'Roadmap item.' },
]

{
  const result = ensureSummaryCitations({ summary: 'Two relevant things changed.', sources })
  assert.match(result.summary, /Sources:/)
  assert.match(result.summary, /email:mail:one/)
  assert.match(result.summary, /docs:docs:roadmap/)
  assert.deepEqual(result.sources, sources)
}

{
  const result = ensureSummaryCitations({ summary: 'Already cited.\n\nSources:\n- email:mail:one | One', sources: [sources[0]] })
  assert.equal(result.summary.match(/email:mail:one/g)?.length, 1)
}

{
  const result = ensureSummaryCitations({ summary: 'Nothing matched.', sources: [] })
  assert.match(result.summary, /Sources: none\./)
}

console.log('summary citation tests passed')
