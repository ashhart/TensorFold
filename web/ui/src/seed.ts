// TensorFold derives an unseeded request's seed from its prompt, on purpose (its docs/api.md: an evaluation run twice
// repeats its draws). In Chat, asking the same thing again at a temperature above zero should give a new answer, so
// the page draws a seed for each reply when the preset fixes none; the reply keeps it, to be reproduced by hand.
type Sampling = { temperature?: number | null; seed?: number | null }

/** A seed for this reply, or null when the preset fixes one or the reply is greedy (temperature 0). */
export function drawSeed(p: Sampling, random: () => number = Math.random): number | null {
  if (p.seed != null || p.temperature === 0) return null
  return Math.floor(random() * 2 ** 31)
}
