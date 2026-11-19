//! One pre-M5 projection tree at every window and chunk height.
const mtl = @import("metal");
const simd = @import("../../core/simd_qmm_bits.zig");
const wide = @import("wide.zig");
const Affine = @import("affine.zig").Affine;
pub const Pipelines = simd.Pipelines;

/// Other group sizes and small tails keep the identical row operator in both callers.
pub fn encode(k: anytype, e: mtl.ComputeEncoder, q: Affine, io: wide.Io) bool {
    const p = if (k.ordered) |*v| @constCast(v) else return false;
    for (0..@as(usize, q.more) + 1) |i| {
        const r = q.run(i);
        const out = io.out.at(@as(usize, r.row) * 2);
        if (r.group != 64 or r.n % 8 != 0 or q.k % 64 != 0 or io.ldx % 8 != 0) {
            wide.dense(&k.wide.?, e, .{ .w = r.w, .s = r.s, .b = r.b, .n = r.n, .k = q.k, .bits = r.bits, .group = r.group }, .{
                .in = io.in,
                .ldx = io.ldx,
                .out = out,
                .ldo = io.ldo,
                .rows = io.rows,
            });
            continue;
        }
        // All variants are warmed at load; command encoding cannot compile or allocate a pipeline.
        const c = simd.callAt(p, r.bits, r.n, q.k, io.rows, io.ldx, io.ldo, true, 0) catch |err| switch (err) {
            error.SimdThreads => simd.callAt(p, r.bits, r.n, q.k, io.rows, io.ldx, io.ldo, false, 0) catch unreachable,
            else => unreachable,
        };
        simd.encode(e, c, .{ .w = r.w.buf, .w_off = r.w.off, .scales = r.s.buf, .s_off = r.s.off, .biases = r.b.buf, .b_off = r.b.off }, io.in.buf, io.in.off, out.buf, out.off);
    }
    return true;
}
