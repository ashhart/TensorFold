import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { test } from 'node:test'

test('hosted context sources end with newlines and use single-line comments', () => {
  for (const file of ['hosted-context.ts', 'hosted-context.test.ts']) {
    const source = readFileSync(new URL(file, import.meta.url), 'utf8')
    assert.ok(source.endsWith('\n'), `${file} needs a final newline`)
    assert.doesNotMatch(source, /\/\*\*[\s\S]*?\n[\s\S]*?\*\//, `${file} needs single-line comments`)
    for (const line of source.split('\n').filter((line) => line.trimStart().startsWith('//'))) {
      assert.ok(line.length <= 120, `${file} comment exceeds 120 columns`)
    }
  }
})
