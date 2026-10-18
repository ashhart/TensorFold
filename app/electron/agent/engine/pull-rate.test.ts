import assert from 'node:assert/strict'
import { TransferRateEstimator } from './pull'

const rate = new TransferRateEstimator()
assert.equal(rate.sample(1024 ** 3, 0), null, 'existing cached bytes are not this session throughput')
assert.equal(rate.sample(1024 ** 3 + 1000, 1000), null, 'a single short sample is insufficient')
assert.equal(rate.sample(1024 ** 3 + 2000, 2000), 1000)
assert.equal(rate.sample(1024 ** 3 + 2000, 12_000), null, 'stalled transfers stop promising an arrival time')
assert.equal(rate.sample(0, 13_000), null, 'cache shrink resets the measurement')
assert.equal(rate.sample(2000, 15_000), 1000)
assert.equal(rate.sample(3000, 100), null, 'clock rollback resets the measurement')
assert.equal(rate.sample(5000, 2100), 1000)
assert.equal(rate.sample(Number.NaN, 3100), null)
assert.equal(rate.sample(-1, 3100), null)
console.log('Download rate: resumed cache baseline, warmup, stall, shrink and clock reset passed')
