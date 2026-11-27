// What the strength field shows after the engine refused a change: the snapshot's own value, not the attempt.
export type ControlVectorStrength = { shown: string; error: string }

export const strengthOnRefusal = (snapshot: { alpha: number } | null, message: string): ControlVectorStrength => ({
  shown: snapshot ? String(snapshot.alpha) : '',
  error: message,
})
