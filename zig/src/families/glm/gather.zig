//! Gather tiles using the sorted row reduction, with each projection's own affine format.
const std = @import("std");
const mtl = @import("metal");
const affine = @import("affine.zig");
const Affine = affine.Affine;
const Ref = affine.Ref;

pub const names = [2][:0]const u8{ "glm_gather_glu", "glm_gather_down" };
pub const Dims = extern struct {
    k: i32,
    n: i32,
    group: i32,
    bits: i32,
    up_group: i32,
    up_bits: i32,
    pairs: i32,
    repeat: i32,
    experts: i32,
    ldo: i32,
    lim: f32,
};
pub const Io = struct { x: Ref, offsets: Ref, order: Ref, out: Ref };

pub fn supported(q: Affine) bool {
    return q.more == 0 and (q.group == 32 or q.group == 64) and
        std.mem.indexOfScalar(u8, &affine.row_bits, q.bits) != null and
        q.n > 0 and q.n % 4 == 0 and q.k > 0 and q.k % q.group == 0;
}

/// Upper bound on nonempty four-row tiles, including expert boundaries; extra tiles return without a write.
pub fn tiles(pairs: u32, experts: u32) u32 {
    return if (pairs == 0 or experts == 0) 0 else @min(pairs, (pairs + 3) / 4 + experts - 1);
}

/// The production encoder or a host recorder, with identical bindings and constants.
pub fn encode(e: anytype, pipe: mtl.Pipeline, gate: Affine, up: ?Affine, io: Io, pairs: u32, experts: u32, repeat: u32, ldo: u32, lim: f32) void {
    if (pairs == 0 or experts == 0) return;
    const u = up orelse gate;
    std.debug.assert(supported(gate) and supported(u));
    std.debug.assert(up == null or (gate.n == u.n and gate.k == u.k));
    e.setPipeline(pipe);
    inline for (.{ io.x, gate.w, gate.s, gate.b }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
    e.setValue(Dims{
        .k = @intCast(gate.k),
        .n = @intCast(gate.n),
        .group = gate.group,
        .bits = gate.bits,
        .up_group = u.group,
        .up_bits = u.bits,
        .pairs = @intCast(pairs),
        .repeat = @intCast(repeat),
        .experts = @intCast(experts),
        .ldo = @intCast(ldo),
        .lim = lim,
    }, 4);
    inline for (.{ io.out, io.offsets, io.order, u.w, u.s, u.b }, 5..) |r, i| e.setBuffer(r.buf, r.off, i);
    e.dispatchGroups(mtl.Size.of((gate.n + 7) / 8, tiles(pairs, experts), 1), mtl.Size.of(64, 1, 1));
}
