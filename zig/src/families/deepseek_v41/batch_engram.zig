//! Row windows' Engram inputs: each slot's staged rows and the warm jobs (op 56).

const std = @import("std");
const eh = @import("engram_host.zig");
const ft = @import("forward_types.zig");
const hooks = @import("forward_hooks.zig");
const rowtab = @import("rowtab.zig");
const iface = @import("draft/iface.zig");
const batch = @import("batch.zig");
const Batch = batch.Batch;
const max_warm = batch.Batch.max_warm;
const max_warm_ids = batch.Batch.max_warm_ids;
const op_engram_warm = batch.Batch.op_engram_warm;

/// TF_DSV41_ENGRAM_WARM: commit and draft-end Engram reads as gate warm jobs on every rank; reads only.
pub fn warmEngram(b: *Batch, asks: []const iface.Warm) !void {
    const f = b.f;
    if (f.gate == null or f.engram == null) return;
    if (b.ss.leads()) {
        var msg: [2 + max_warm * (3 + max_warm_ids)]i64 = undefined;
        var at: usize = 2;
        var n: usize = 0;
        for (asks) |a| {
            if (n == max_warm or a.ids.len == 0 or a.slot >= b.ss.n) continue;
            const m = @min(a.ids.len, max_warm_ids);
            msg[at..][0..3].* = .{ a.slot, @intCast(a.start), @intCast(m) };
            for (a.ids[0..m], msg[at + 3 ..][0..m]) |t, *y| y.* = t;
            at += 3 + m;
            n += 1;
        }
        if (n == 0) return;
        msg[0] = op_engram_warm;
        msg[1] = @intCast(n);
        try f.link.?.send(msg[0..at]);
    }
    b.warmLocal(asks);
}

/// op 56's asks into `asks` (their ids in `ids`); the count.
pub fn parseWarm(msg: []const i64, asks: *[max_warm]iface.Warm, ids: *[max_warm * max_warm_ids]u32) !usize {
    if (msg.len < 2) return error.BadPlan;
    const n: usize = @intCast(msg[1]);
    if (n > max_warm) return error.BadPlan;
    var at: usize = 2;
    for (0..n) |i| {
        if (at + 3 > msg.len) return error.BadPlan;
        const m: usize = @intCast(msg[at + 2]);
        if (m == 0 or m > max_warm_ids or at + 3 + m > msg.len) return error.BadPlan;
        const out = ids[i * max_warm_ids ..][0..m];
        for (msg[at + 3 ..][0..m], out) |x, *y| y.* = @intCast(x);
        asks[i] = .{ .slot = @intCast(msg[at]), .start = @intCast(msg[at + 1]), .ids = out };
        at += 3 + m;
    }
    if (at != msg.len) return error.BadPlan;
    return n;
}

/// This rank's warm job for `asks` (the leader's or op 56's), the same items on every rank.
pub fn warmLocal(b: *Batch, asks: []const iface.Warm) void {
    const f = b.f;
    const g = f.gate orelse return;
    const h = f.engram orelse return;
    var items: [max_warm]ft.Item = undefined;
    var bufs: [max_warm][2 * eh.max_ngram + 64]u32 = undefined;
    var n: usize = 0;
    for (asks) |a| {
        if (n == max_warm or a.ids.len == 0 or a.slot >= b.ss.n) continue;
        const items_n = @min(a.ids.len, max_warm_ids);
        const lb = warmLookback(b.ss.state(a.slot).*, b.pend[a.slot], a.start, h.max_ngram - 1, &bufs[n]) orelse continue;
        items[n] = .{ .ids = a.ids[0..items_n], .tail = lb };
        n += 1;
    }
    if (n > 0) hooks.set.gate.?.warm(g, items[0..n]);
}

/// Each Engram layer's window rows through the forward's source; padding rows copy the last real row.
pub fn stageEngram(b: *Batch, mix: []const rowtab.Seg, ids: []const u32, R: u32) !void {
    const f = b.f;
    f.gated = false;
    const h = f.engram orelse return;
    if (b.stages.items.len == 0) return;
    // the table opens on the forward's first Engram window (a prompt always comes first); never index rows instead
    if (std.c.getenv("TF_DSV41_ENGRAM_DIR") != null and f.engram_rows == null) return error.EngramNotOpen;
    const src: ?eh.Source = try f.engramSource(h); // the gate's locked table while it lives
    const n = rowtab.totalRows(mix);
    const dim: usize = f.cfg.engram_head_dim;
    // TF_DSV41_ENGRAM_GATE: the round to the gate's worker; the window's engram_rows steps launch the gate kernel
    {
        var items: [64]ft.Item = undefined;
        var a: usize = 0;
        for (mix, 0..) |m, i| {
            const st = &b.ss.states[m.slot];
            items[i] = .{ .ids = ids[a .. a + m.rows], .tail = st.tail[0..st.tail_len] };
            a += m.rows;
        }
        f.gated = mix.len <= items.len and try f.armEngram(items[0..mix.len], R);
        if (f.gated) return;
    }
    // every slot's rows on every Engram layer in flight first, so the waits below cost one read latency
    if (src != null) {
        var a: usize = 0;
        for (mix) |m| {
            const st = &b.ss.states[m.slot];
            f.issueEngram(ids[a .. a + m.rows], st.tail[0..st.tail_len]);
            a += m.rows;
        }
    }
    for (b.stages.items) |*stg| {
        const cols = stg.cols;
        const out = try stg.begin(R);
        var a: usize = 0;
        for (mix) |m| {
            const st = &b.ss.states[m.slot];
            try h.windowRows(b.gpa, src, ids[a .. a + m.rows], st.tail[0..st.tail_len], stg.layer, f.comm.rank(), f.comm.world(), dim, out[a * cols .. (a + m.rows) * cols]);
            a += m.rows;
        }
        for (n..R) |t| @memcpy(out[t * cols ..][0..cols], out[(n - 1) * cols ..][0..cols]);
        try stg.commit(f.runner.stream, R);
    }
}

/// The lookback a warm at `start` reads after: the committed tail plus the pending rows `start` implies kept, or null.
pub fn warmLookback(st: anytype, pend: anytype, start: u64, keep_n: usize, buf: []u32) ?[]const u32 {
    const pending: ?[]const u32 = if (pend) |p| (if (p.start == st.pos) p.ids[0..p.n] else null) else null;
    return eh.lookbackAt(st.tail[0..st.tail_len], st.pos, pending, start, keep_n, buf);
}
