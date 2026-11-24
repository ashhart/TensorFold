//! A drafter context's bytes: the taps ring, and the drafter's K/V caches when the slot drafts.
const q = @import("tensorfold").qwen27;

pub fn bytes(d: q.drafter.Any, drafting: bool) usize {
    const g = d.ring();
    const m = switch (d) {
        .dflash2 => |m| m,
        .dspine => |m| return m.contextBytes(drafting),
    };
    const c = m.graph.config;
    var caches: usize = 0;
    if (drafting) {
        for (0..c.layers) |i| caches += 2 * (@as(usize, c.layerWindow(i)) + 16) * c.kvWidth() * 2;
    }
    return @as(usize, g.window) * g.stride + caches;
}
