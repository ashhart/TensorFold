//! The GPU-side rounds' buffers an engine sets up once at load.
const fz = @import("replay.zig");
const engine = @import("engine.zig");
const Engine = engine.Engine;
const MAXR = fz.MAXR;
const CS_ROW = fz.CS_ROW;
const SO_ROW = fz.SO_ROW;
const PLE_TAIL = fz.PLE_TAIL;
const WIDE = fz.WIDE;
const CAP = fz.CAP;
const i32Buf = fz.i32Buf;

/// DeltaNet states (kept, window rows), PLE tail pair, arena, token ring, per-width row counts, MTP slots.
pub fn rounds(e: *Engine) !void {
    const r = e.r;
    const m = e.m;
    const n_lin: usize = 36;
    e.g_cs = try r.buffer(n_lin * CS_ROW);
    e.g_so = try r.buffer(n_lin * SO_ROW);
    e.o_cs = try r.buffer(n_lin * MAXR * CS_ROW);
    e.o_so = try r.buffer(n_lin * MAXR * SO_ROW);
    var gi: usize = 0;
    for (&m.layers) |*L| if (L.linear) {
        L.cs[0] = .{ .b = e.g_cs, .off = gi * CS_ROW };
        L.cs[1] = .{ .b = e.o_cs, .off = gi * MAXR * CS_ROW };
        L.so[0] = .{ .b = e.g_so, .off = gi * SO_ROW };
        L.so[1] = .{ .b = e.o_so, .off = gi * MAXR * SO_ROW };
        gi += 1;
    };
    if (r.gdn_kept != null) m.recs = .{ .{ .b = try r.buffer(n_lin * fz.gdn_step.RECORD) }, .{ .b = try r.buffer(n_lin * fz.gdn_step.RECORD) } };
    m.marks = try fz.Marks.init(r, n_lin);
    e.cins = .{ m.ple.cin, .{ .b = try r.buffer((PLE_TAIL + MAXR) * WIDE * 2) } };
    r.ar = .{ .b = try r.buffer(4 * fz.AR_WORDS) };
    e.ring = try r.buffer(fz.RING_WORDS * 4 * engine.RING);
    e.hist = try r.buffer((CAP + 64) * 4);
    e.wids = .{ .b = try r.buffer(64) };
    for (1..MAXR + 1) |n| {
        e.rows_w[n] = try i32Buf(r, &.{@intCast(n)});
        e.mdims_w[n] = try i32Buf(r, &.{ @intCast(n), 16, 0, 0, 0, 0, 0, 0 });
    }
    e.host = .{ .rows = m.t.rows, .mdims = m.t.mdims, .pos8 = m.t.pos8, .nk8 = m.t.nk8, .kvmeta = m.t.kvmeta, .slots = m.mtp.slots };
    e.gpu_slots = m.mtp.slots;
    for (1..MAXR + 1) |n| { // the head absorbing a window of n rows: slot MAXR - 1 + n
        const sl = &e.gpu_slots[MAXR - 1 + n];
        try m.mtpMeta(sl, n);
        sl.pos8 = .{ .b = r.ar.b, .off = fz.AR_ABS_POS * 4 };
        sl.nk8 = .{ .b = r.ar.b, .off = fz.AR_ABS_NK * 4 };
        sl.kvmeta = .{ .b = r.ar.b, .off = fz.AR_ABS_KV * 4 };
    }
    for (1..MAXR - 1) |j| { // chained draft j: slot j
        try m.mtpMeta(&e.gpu_slots[j], 1);
        const b = (fz.AR_CHAIN + (j - 1) * fz.AR_CHAIN_STRIDE) * 4;
        e.gpu_slots[j].pos8 = .{ .b = r.ar.b, .off = b };
        e.gpu_slots[j].nk8 = .{ .b = r.ar.b, .off = b + 16 * 4 };
        e.gpu_slots[j].kvmeta = .{ .b = r.ar.b, .off = b + 32 * 4 };
    }
}
