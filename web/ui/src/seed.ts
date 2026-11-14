// Draw a seed for each sampled chat reply and record it so the same reply can be reproduced.
type Sampling = { temperature?: number | null; seed?: number | null }


export function drawSeed(p: Sampling, random: () => number = Math.random): number | null {
  if (p.seed != null || p.temperature === 0) return null
  return Math.floor(random() * 2 ** 31)
}
