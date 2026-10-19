import assert from 'node:assert/strict'
import { HF_ORIGIN, treeTotal } from './hub'

const revision = 'c'.repeat(40)
const repo = 'Example/Weights'
const route = `${HF_ORIGIN}/api/models/${repo}/tree/${revision}`

function source(body: unknown, link?: string) {
  return { fetch: (async () => new Response(JSON.stringify(body), {
    headers: { 'content-type': 'application/json', ...(link ? { link } : {}) },
  })) as typeof fetch }
}

for (const size of [undefined, null, -1, 1.5, '12', Number.MAX_SAFE_INTEGER + 1]) {
  await assert.rejects(
    treeTotal(source([{ type: 'file', path: 'weights', size }]), repo, revision),
    /file size/i,
    `unverified size ${String(size)} must not turn into a successful zero-byte estimate`,
  )
}
await assert.rejects(
  treeTotal(source([{ type: 'file', size: Number.MAX_SAFE_INTEGER }, { type: 'file', size: 1 }]), repo, revision),
  /size|safe integer/i,
)
for (const link of [
  '<https://outside.example/next>; rel="next"',
  `<${HF_ORIGIN}/api/models/Example/Other/tree/${revision}>; rel="next"`,
  `<${HF_ORIGIN}/api/models/${repo}/tree/${'a'.repeat(40)}>; rel="next"`,
  `<${route}?recursive=false&cursor=2>; rel="next"`,
  `<${route}?recursive=&cursor=2>; rel="next"`,
  `<${route}?recursive=true&recursive=false&cursor=2>; rel="next"`,
]) {
  let fetched = 0
  await assert.rejects(treeTotal({ fetch: (async () => {
    fetched++
    return new Response(JSON.stringify([{ type: 'file', size: 10 }]), {
      headers: { 'content-type': 'application/json', ...(fetched === 1 ? { link } : {}) },
    })
  }) as typeof fetch }, repo, revision), /pagination|next link/i)
  assert.equal(fetched, 1, 'an invalid continuation must be rejected before fetching it')
}

let pages = 0
await assert.rejects(treeTotal({ fetch: (async () => {
  pages++
  return new Response(JSON.stringify([{ type: 'file', path: `weight-${pages}`, size: 10 }]), {
    headers: { 'content-type': 'application/json', link: `<${route}?cursor=${pages}>; rel="next"` },
  })
}) as typeof fetch }, repo, revision), /pagination|page limit/i)
assert.equal(pages, 20, 'hitting the bounded pagination budget cannot certify a partial total')

let calls = 0
const complete = await treeTotal({ fetch: (async (input: string | URL | Request) => {
  assert.equal(new URL(String(input)).searchParams.get('recursive'), 'true', 'every page must include nested model files')
  calls++
  return new Response(JSON.stringify([{ type: 'file', size: calls === 1 ? 10 : 32 }]), {
    headers: { 'content-type': 'application/json', ...(calls === 1 ? { link: `<${route}?cursor=2>; rel=next` } : {}) },
  })
}) as typeof fetch }, repo, revision)
assert.deepEqual(complete, { bytes: 42, files: 2 }, 'complete same-revision pagination still computes the exact sum')
for (const body of [[], [{ type: 'file', size: 0 }]]) {
  await assert.rejects(treeTotal(source(body), repo, revision), /no non-empty model files/)
}
console.log('Hub file-list completeness regressions passed')
