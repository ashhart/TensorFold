//! The chunk recurrence also owns pre-M5 windows and accepted-prefix replay.
const mtl = @import("metal");
const std = @import("std");
const Part = @import("profile.zig").Part;
const wts = @import("weights.zig");
const Ref = wts.Ref;
const ordered = @import("ordered_projection.zig");
const size = mtl.Size.of;
fn bind(e: mtl.ComputeEncoder, start: usize, refs: anytype) void {
    inline for (refs, 0..) |v, i| e.setBuffer(v.buf, v.off, start + i);
}
fn shape(e: mtl.ComputeEncoder, index: usize, values: anytype) void {
    var dims: [8]i32 = @splat(0);
    inline for (values, 0..) |v, i| dims[i] = @intCast(v);
    e.setBytes(std.mem.asBytes(&dims), index);
}

pub fn encode(x: anytype, e: mtl.ComputeEncoder, ki: usize, w: *const wts.Kda, proj: Ref, pitch: u32, y: Ref, scratch: Ref, M: u32, block_rows: u32) void {
    const c = x.c;
    const k = x.k;
    const L = &x.s.kda[ki];
    const cur = L.cur;
    const H = c.kda_heads;
    const W = c.kdaWidth();
    const tp = c.tp > 1; // TP2: this Mac's heads
    const fa = 3 * W; // the projection row: q, k, v, then f_a, g_a and beta
    var r0: u32 = 0;
    while (r0 < M) : (r0 += block_rows) {
        const m = @min(block_rows, M - r0);
        const plane = @as(usize, m) * W * 2; // [m, heads * dim] bf16
        const q = scratch;
        const kk = q.at(plane);
        const v = kk.at(plane);
        const sy = v.at(plane);
        const g = sy.at(plane); // fp32
        const gate = g.at(2 * plane);
        const beta = gate.at(plane);
        if (x.pskip & Part.bit("kda_pre") == 0) {
            const at = @as(usize, r0) * pitch * 2; // the block's projection rows
            _ = ordered.encode(k, e, w.f_b, .{ .in = proj.at(at + @as(usize, fa) * 2), .ldx = pitch, .out = sy, .ldo = W, .rows = m }); // f_b's rows into sy's plane, read before the scan writes it
            _ = ordered.encode(k, e, w.g_b, .{ .in = proj.at(at + @as(usize, fa + c.kda_dim) * 2), .ldx = pitch, .out = gate, .ldo = W, .rows = m });
            e.setPipeline(if (tp) k.kda_prep_tp else k.kda_prep);
            bind(e, 0, .{proj});
            shape(e, 1, .{ M, pitch, r0 });
            bind(e, 2, .{ L.cs[cur], w.conv_w, sy, w.a, w.dt_bias });
            e.setValue(c.lower_bound, 7);
            bind(e, 8, .{ q, kk, v, g, beta });
            e.dispatchGroups(size(m, H / 4, 1), size(128, 1, 1));
        }
        if (x.pskip & Part.bit("kda_scan") == 0) {
            e.setPipeline(if (tp) k.kda_scan_tp else k.kda_scan);
            bind(e, 0, .{ q, kk, v, g, beta, L.st[if (r0 == 0) cur else 1 - cur], L.st[1 - cur], sy });
            e.setValue(@as(i32, @intCast(m)), 8);
            e.dispatchGroups(size(c.kda_dim / 64, H, 1), size(512, 1, 1));
        }
        if (x.pskip & Part.bit("kda_post") != 0) continue;
        e.setPipeline(if (tp) k.kda_post_tp else k.kda_post);
        bind(e, 0, .{ sy, gate, w.o_norm });
        e.setValue(c.eps, 3);
        bind(e, 4, .{ y.at(@as(usize, r0) * W * 2), proj });
        shape(e, 6, .{ M, pitch }); // every block writes the next chunk's conv window, the same rows
        bind(e, 7, .{ L.cs[cur], L.cs[1 - cur] });
        e.dispatchGroups(size(m, H, 1), size(32, 1, 1));
    }
}
