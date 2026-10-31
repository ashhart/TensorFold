// node --experimental-strip-types --test tests/
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { rank, score } from '../src/fuzzy.ts'

const labels = ['Go to Bench', 'Go to Models', 'Start Nemotron 3.5 Lightning 30B-A3B', 'Model: Qwen3.8-27B',
  'Conversation: Paramètres réseau', 'Use the dark look', 'Run: quick score, Nemotron, 10 Oct']

test('pieces, letters in order, one typo, accents', () => {
  assert.ok(score('bench', 'Go to Bench') != null)
  assert.ok(score('bnch', 'Go to Bench') != null)
  assert.ok(score('nemtoron', 'Start Nemotron 3.5') != null)          // two letters swapped
  assert.ok(score('nemotrn', 'Start Nemotron 3.5') != null)           // one left out
  assert.ok(score('parametres', 'Conversation: Paramètres réseau') != null)
  assert.equal(score('xyz', 'Go to Bench'), null)
  assert.equal(score('ab', 'Go to Bench'), null)                       // too short for letters in order alone
})

test('every word must match, best first', () => {
  assert.deepEqual(rank('nemo start', labels, (x) => x), ['Start Nemotron 3.5 Lightning 30B-A3B'])
  assert.equal(rank('nemotron', labels, (x) => x)[0], 'Start Nemotron 3.5 Lightning 30B-A3B')
  assert.equal(rank('dark', labels, (x) => x)[0], 'Use the dark look')
  assert.equal(rank('', labels, (x) => x).length, labels.length)
})

test('a word start beats the middle of a word', () => {
  assert.ok(score('mod', 'Go to Models') > score('mod', 'Run: a remodel'))
})
