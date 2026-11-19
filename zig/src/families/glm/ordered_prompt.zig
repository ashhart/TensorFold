//! Chunk projections retain their tiles; MLA and routed experts keep the window operators.
const std = @import("std");
const mtl = @import("metal");
const fwd = @import("forward.zig");
const st = @import("state.zig");
const wts = @import("weights.zig");
const Ref = wts.Ref;
const Prompt = @import("prompt.zig").Prompt;
const ordered = @import("ordered_projection.zig");

pub fn mla(p: *const Prompt, x: *const fwd.Ctx, e: mtl.ComputeEncoder, mi: usize, w: *const wts.Mla, in: Ref, M: u32, pos: u32) void {
    const c = x.c;
    const pitch = std.mem.alignForward(u32, c.xProj(), 64);
    if (fwd.on(x, "mla_proj")) {
        _ = ordered.encode(x.k, e, w.x_proj, .{ .in = in, .ldx = c.hidden, .out = p.xp, .ldo = pitch, .rows = M });
        fwd.rms(x, e, p.xp, w.q_norm, p.qr, M, c.q_lora, pitch, c.q_lora, c.eps);
        _ = ordered.encode(x.k, e, w.qr_proj, .{ .in = p.qr, .ldx = c.q_lora, .out = p.qp, .ldo = c.qrProj(), .rows = M });
    }
    if (fwd.on(x, "mla_cache")) fwd.mlaCache(x, e, mi, w, in, p.xp, pitch, p.iw, M, pos);
    if (fwd.on(x, "mla_absorb")) fwd.absorb(x, e, w, p.qp, p.ql, M);
    var r: u32 = 0;
    const lat = @as(usize, c.mla_heads) * c.kv_lora * 2;
    while (r < M) : (r += st.max_rows) {
        const n = @min(st.max_rows, M - r);
        var sc = x.sc.*;
        sc.qp = p.qp.at(@as(usize, r) * c.qrProj() * 2);
        sc.ql = p.ql.at(r * lat);
        sc.att = p.att.at(r * lat);
        sc.iw = p.iw.at(@as(usize, r) * c.i_heads * 2);
        var y = x.*;
        y.sc = &sc;
        fwd.attend(&y, e, mi, .{ .s = x.s, .row0 = 0, .rows = n, .pos = pos + r });
    }
    if (fwd.on(x, "mla_unabs")) fwd.unabsorb(x, e, w, p.att, p.vals, M);
    if (fwd.on(x, "mla_out")) _ = ordered.encode(x.k, e, w.o_proj, .{ .in = p.vals, .ldx = c.mla_heads * c.v_dim, .out = p.streams.branch, .ldo = c.hidden, .rows = M });
}

pub fn moe(p: *const Prompt, x: *const fwd.Ctx, e: mtl.ComputeEncoder, w: *const wts.Moe, in: Ref, M: u32) void {
    var r: u32 = 0;
    while (r < M) : (r += st.max_rows) {
        const n = @min(st.max_rows, M - r);
        var sc = x.sc.*;
        sc.branch = p.streams.branch.at(@as(usize, r) * x.c.hidden * 2);
        var y = x.*;
        y.sc = &sc;
        fwd.moe(&y, e, w, in.at(@as(usize, r) * x.c.hidden * 2), n);
    }
}
