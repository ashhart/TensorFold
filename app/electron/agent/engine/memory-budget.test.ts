import assert from 'node:assert/strict'
import { loweredEngineMemoryBudget } from './memory-budget'

const GIB = 1024 ** 3
for (const gib of [8, 16, 24, 32, 64]) assert.equal(loweredEngineMemoryBudget(128 * GIB, String(gib * GIB)), gib * GIB)
assert.equal(loweredEngineMemoryBudget(8 * GIB, 64 * GIB), 8 * GIB, 'qualification cannot raise physical memory')
for (const invalid of [undefined, null, NaN, Infinity, -1, 0, '8 GB', '3.5', '-1', 'Infinity']) assert.equal(loweredEngineMemoryBudget(8 * GIB, invalid), 8 * GIB)
console.log('Qualification memory budgets only lower physical memory and reject malformed values')
