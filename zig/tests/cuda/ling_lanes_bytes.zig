//! The Ling engine's steps against their twins byte for byte, and the plain pass's logits against float64.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const ling = @import("ling_cuda");
const rig = @import("ling_lanes_rig.zig");
const frig = @import("ling_forward_rig.zig");
const synth = @import("ling_forward_synth.zig");

const Cover = rig.Cover;
const Engine = rig.Engine;
const F = ling.forward;

/// Kept-state ids by mark, as a direct engine pass keeps them.
const Marked = struct {
    e: *Engine,
    ids: [8]u64 = undefined,
    n: usize = 0,
    fn at(ptr: *anyopaque, pos: u32) void {
        _ = pos;
        const m: *Marked = @ptrCast(@alignCast(ptr));
        m.ids[m.n] = m.e.keepState() catch return;
        m.n += 1;
    }
};

fn logitsRows(e: *Engine, a: std.mem.Allocator, ptr: u64, rows: usize) ![]u8 {
    const out = try a.alloc(u8, rows * e.c.vocab * 2);
    try e.stream.synchronize();
    const b: cuda.DeviceBuffer = .{ .d = e.d, .ptr = ptr, .len = out.len };
    try b.download(0, out);
    return out;
}

/// A fresh pass of `ids` on `slot` with `rows`-row chunks (and `marks`), or resumed from `from`; its first token.
fn pass(e: *Engine, slot: u32, ids: []const u32, rows: usize, from: ?ling.engine.Resume, marks: ?ling.engine.Marks) !u32 {
    try e.bind(slot);
    try e.begin(null);
    e.chunk = rows;
    return e.prefill(ids, from, null, marks);
}

pub fn engineChecks(c: *Cover, e: *Engine, prompt: []const u32, w: []const u32) !void {
    const a = c.a;
    const keep_chunk = e.chunk;
    defer e.chunk = keep_chunk;
    const t10 = try pass(e, 10, prompt, 512, null, null);
    const t11 = try pass(e, 11, prompt, 37, null, null);
    const t12 = try pass(e, 12, prompt, 16, null, null);
    try c.expect(t10 == t11 and t10 == t12, "first tokens: whole {d}, 37-row chunks {d}, 16-row chunks {d}", .{ t10, t11, t12 });
    try rig.sameSeq(c, e, "prompt in 37-row chunks == whole", 11, 10);
    try rig.sameSeq(c, e, "prompt in 16-row chunks == whole", 12, 10);
    var mk: Marked = .{ .e = e };
    const marks = [_]u32{ 101, 263, 349 };
    _ = try pass(e, 13, prompt, 48, null, .{ .at = &marks, .ctx = &mk, .call = Marked.at });
    try c.expect(mk.n == marks.len, "a pass kept {d} states at marks 101, 263, 349", .{mk.n});
    for (mk.ids[0..mk.n], marks[0..mk.n]) |id, at| {
        const t = try pass(e, 14, prompt, 37, .{ .id = id, .at = at }, null);
        try c.expect(t == t10, "resumed at {d}: first token {d}, the whole pass's {d}", .{ at, t, t10 });
        try rig.sameSeq(c, e, try std.fmt.allocPrint(a, "resumed at {d} == whole", .{at}), 14, 10);
        e.dropState(id);
    }
    // a 9-row window, keep 4, a 5-row window == 9 single rows on the twin (tokens, logits, states)
    var g: [10]u32 = undefined;
    g[0] = t11;
    const lone = try a.alloc([]u8, 9);
    try e.bind(11);
    for (0..9) |r| {
        try e.verify(g[r .. r + 1], g[r + 1 .. r + 2]);
        lone[r] = try logitsRows(e, a, e.bd.logits, 1);
        try e.keep(1);
    }
    try e.bind(10);
    var got: [9]u32 = undefined;
    try e.verify(g[0..9], &got);
    const win = try logitsRows(e, a, e.bd.logits, 9);
    var same_rows = std.mem.eql(u32, &got, g[1..10]);
    for (0..9) |r| same_rows = same_rows and std.mem.eql(u8, win[r * lone[r].len ..][0..lone[r].len], lone[r]);
    try e.keep(4);
    try e.verify(g[4..9], got[0..5]);
    const win2 = try logitsRows(e, a, e.bd.logits, 5);
    same_rows = same_rows and std.mem.eql(u32, got[0..5], g[5..10]);
    for (0..5) |r| same_rows = same_rows and std.mem.eql(u8, win2[r * lone[4 + r].len ..][0..lone[4 + r].len], lone[4 + r]);
    try e.keep(5);
    try c.expect(same_rows, "a 9-row window and, after keeping 4, a 5-row window: tokens and logits == 9 single rows", .{});
    try rig.sameSeq(c, e, "9-row window, keep 4, 5-row window == 9 single rows", 10, 11);
    try sharedChecks(c, e, prompt, w);
    if (e.mtp != null) try mtpChain(c, e, prompt, w);
    for ([_]u32{ 10, 11, 12, 13, 14 }) |s| e.release(s);
}

/// MTP chains of 3 and of 1 on twins: the same first draft, and after the same windows the chain's rows rolled back.
fn mtpChain(c: *Cover, e: *Engine, prompt: []const u32, w: []const u32) !void {
    var t: [2]u32 = undefined;
    var d: [2][3]u32 = undefined;
    for (0..2) |k| {
        t[k] = try pass(e, @intCast(40 + k), prompt[0..200], 37, null, null);
        try e.draft(true, &.{t[k]}, if (k == 0) 3 else 1, &d[k]);
    }
    var ok = t[0] == t[1] and d[0][0] == d[1][0];
    var pend = t[0];
    var chained: usize = 0;
    for (0..6) |round| {
        var ids: [8]u32 = undefined;
        var out: [2][8]u32 = undefined;
        const rows = 2 + round % 6;
        ids[0] = pend;
        for (1..rows) |r| ids[r] = w[(round * 11 + r) % w.len];
        const keep = 1 + (round * 3) % rows;
        for (0..2) |k| {
            try e.bind(@intCast(40 + k));
            try e.verify(ids[0..rows], out[k][0..rows]);
            try e.keep(keep);
            try e.draft(false, out[k][0..keep], if (k == 0) 3 else 1, &d[k]);
        }
        ok = ok and std.mem.eql(u32, out[0][0..rows], out[1][0..rows]) and d[0][0] == d[1][0];
        { // the chain's second draft by hand on the twin: d0 with the newest kept row's MTP output as its hidden row
            const h = &e.mtp.?;
            const hd = &e.seqs.items[41].?.hd.?;
            const D: usize = e.c.hidden;
            const f = h.forward(&e.m, e.stream, &h.bm);
            try h.absorb(&e.m, e.stream, &h.bm, hd, @ptrCast(d[1][0..1]), h.bm.normed + (keep - 1) * D * 2);
            try f.final(1);
            try f.logitsOf(0, 1);
            var second: [1]u32 = undefined;
            try e.draw(h.bm.logits, 1, hd.pos + 1, &second);
            try hd.setPos(e.stream, hd.pos - 1);
            ok = ok and second[0] == d[0][1];
            chained += 1;
        }
        pend = out[0][keep - 1];
    }
    try c.expect(ok, "MTP chains of 3 and of 1 over 6 windows (kept 1 to 7 rows): the same tokens and first drafts; {d} second drafts == by hand", .{chained});
    const sa = e.seqs.items[40].?;
    const sb = e.seqs.items[41].?;
    try c.expect(sa.hd.?.pos == sa.st.pos + 2 and sb.hd.?.pos == sb.st.pos, "the chain of 3 holds its 2 chained rows past the kept ones (MTP at {d}, the sequence at {d})", .{ sa.hd.?.pos, sa.st.pos });
    for (0..2) |k| { // one more window, both drafting 1: the chain's rows rolled back, the twins byte for byte
        var out: [2]u32 = undefined;
        try e.bind(@intCast(40 + k));
        try e.verify(&.{ pend, w[3] }, &out);
        try e.keep(2);
        try e.draft(false, &out, 1, &d[k]);
    }
    try rig.sameSeq(c, e, "after chains of 3 and of 1, a window and a chain of 1 each", 40, 41);
    for (0..2) |k| e.release(@intCast(40 + k));
}

/// Two streams' windows in one shared forward against lone twins over 8 rounds: logits, tokens, then states.
fn sharedChecks(c: *Cover, e: *Engine, prompt: []const u32, w: []const u32) !void {
    const a = c.a;
    const pa = prompt[0..120];
    const pb = prompt[50..250];
    var pend: [2]u32 = undefined;
    for ([_]u32{ 20, 21, 22, 23 }, 0..) |slot, k| pend[k % 2] = try pass(e, slot, if (k % 2 == 0) pa else pb, 37, null, null);
    var ok = true;
    var rows_seen: usize = 0;
    for (0..8) |round| {
        var ids: [2][16]u32 = undefined;
        var outs: [2][16]u32 = undefined;
        var twin: [2][16]u32 = undefined;
        var parts: [2]ling.engine.Part = undefined;
        for (0..2) |k| {
            const rows = 1 + (round * 5 + k * 7) % 16;
            ids[k][0] = pend[k];
            for (1..rows) |r| ids[k][r] = w[(round * 13 + k * 31 + r) % w.len];
            parts[k] = .{ .slot = @intCast(20 + k), .ids = ids[k][0..rows], .out = outs[k][0..rows] };
        }
        try e.verifyShared(&parts);
        const sh = try logitsRows(e, a, e.bs.logits, parts[0].ids.len + parts[1].ids.len);
        var row0: usize = 0;
        for (0..2) |k| {
            const rows = parts[k].ids.len;
            try e.bind(@intCast(22 + k));
            try e.verify(parts[k].ids, twin[k][0..rows]);
            const lo = try logitsRows(e, a, e.bd.logits, rows);
            const V2 = e.c.vocab * 2;
            if (!std.mem.eql(u8, sh[row0 * V2 ..][0 .. rows * V2], lo) or !std.mem.eql(u32, outs[k][0..rows], twin[k][0..rows])) ok = false;
            row0 += rows;
            rows_seen += rows;
            const keep = 1 + (round * 3 + k) % rows;
            for ([_]u32{ @intCast(20 + k), @intCast(22 + k) }) |s| {
                try e.bind(s);
                try e.keep(keep);
            }
            pend[k] = outs[k][keep - 1];
        }
    }
    try c.expect(ok, "shared windows: {d} rows of 2 streams over 8 rounds (1 to 16 rows each) == lone twins' logits and tokens", .{rows_seen});
    try rig.sameSeq(c, e, "shared stream 0 after 8 rounds == its lone twin", 20, 22);
    try rig.sameSeq(c, e, "shared stream 1 after 8 rounds == its lone twin", 21, 23);
    for ([_]u32{ 20, 21, 22, 23 }) |s| e.release(s);
}

/// The plain pass serially (whole prompt, then one row at a time) with its picks; files for ling_forward.py --ids.
pub fn serial(c: *Cover, gpu: check.Gpu, e: *Engine, p: []const u32, w: []const u32, out: []const u8, ref_path: ?[]const u8) !void {
    const a = c.a;
    const ids = try a.alloc(i32, p.len + w.len - 1);
    for (ids, 0..) |*x, k| x.* = @intCast(if (k < p.len) p[k] else w[k - p.len]);
    const T = ids.len;
    const P = p.len;
    const V: usize = e.c.vocab;
    const top: usize = e.c.topk;
    var nm: usize = 0;
    for (e.main.layers) |l| nm += @intFromBool(l.ffn == .moe);
    var r: frig.Rig = .{ .gpu = gpu, .a = a, .s = e.stream, .m = &e.m, .g = &e.g };
    defer r.deinit();
    try e.bind(30);
    try e.begin(null);
    const st = &e.cur.st;
    const logits = try a.alloc(u16, T * V);
    const picks = try a.alloc(i32, nm * T * top);
    const pp = try r.zeros(nm * P * (top + 1) * 4);
    @memcpy(logits[0 .. P * V], try r.run(&e.bp, &.{.{ .st = st, .row0 = 0, .rows = P }}, ids[0..P], pp));
    const sp = try frig.sortedPicks(a, try r.back(pp, nm * P * (top + 1) * 4), nm, P, top);
    for (0..nm) |l| @memcpy(picks[l * T * top ..][0 .. P * top], sp[l * P * top ..][0 .. P * top]);
    try r.fwd(&e.bp, null).commit(st, P);
    const dp = try r.zeros(nm * (top + 1) * 4);
    for (P..T) |t| {
        @memcpy(logits[t * V ..][0..V], try r.run(&e.bd, &.{.{ .st = st, .row0 = 0, .rows = 1 }}, ids[t .. t + 1], dp));
        const one = try frig.sortedPicks(a, try r.back(dp, nm * (top + 1) * 4), nm, 1, top);
        for (0..nm) |l| @memcpy(picks[(l * T + t) * top ..][0..top], one[l * top ..][0..top]);
        try r.fwd(&e.bd, null).commit(st, 1);
    }
    e.release(30);
    var agree: usize = 0;
    for (P - 1..T) |t| {
        var best: usize = 0;
        for (0..V) |j| if (frig.bfOf(logits[t * V + j]) > frig.bfOf(logits[t * V + best])) {
            best = j;
        };
        const next: u32 = if (t + 1 < T) @intCast(ids[t + 1]) else w[w.len - 1];
        agree += @intFromBool(best == next);
    }
    try c.expect(agree == T - P + 1, "the lanes' greedy reply == the serial forward's argmax at all {d} of {d} rows", .{ agree, T - P + 1 });
    var name: [512]u8 = undefined;
    try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = try std.fmt.bufPrint(&name, "{s}.ids", .{out}), .data = std.mem.sliceAsBytes(ids) });
    try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = try std.fmt.bufPrint(&name, "{s}.picks", .{out}), .data = std.mem.sliceAsBytes(picks) });
    const path = ref_path orelse return;
    const ref = try synth.readRef(a, gpu.io, path);
    try c.expect(ref.T == T and ref.V == V and ref.forced and std.mem.eql(i32, ref.ids, ids), "reference: {d} tokens routed on these picks, the same ids", .{ref.T});
    const flipped = try a.alloc(bool, T);
    @memset(flipped, false);
    const nf = frig.flips(picks, ref.picks, nm, T, 0, T, top, flipped);
    const d64 = frig.distance(logits, ref.l64, T, V, flipped);
    const dbf = frig.distance(logits, ref.lbf, T, V, flipped);
    var bd: f64 = 0;
    var bw: f64 = 0;
    for (ref.lbf, ref.l64) |x, y| {
        bd += (x - y) * (x - y);
        bw += y * y;
    }
    const base = @sqrt(bd / bw);
    std.debug.print("RESULT   served logits vs float64 (same picks): rel L2 {e:.3}, worst row {e:.3}, argmax {d}/{d}; vs bf16 points {e:.3}; bf16 points vs float64 {e:.3}\n", .{ d64.rel, d64.worst_row, d64.argmax, T, dbf.rel, base });
    if (e.mtp != null) try mtpLogits(c, gpu, e, ids, path);
    try c.expect(nf == 0 and d64.rel <= 1.5 * base and dbf.rel <= base, "served logits within 1.5x the bf16 points' rel L2 of float64 ({e:.3} <= 1.5 x {e:.3}), within it of the bf16 points", .{ d64.rel, base });
}

/// The MTP layer's logits at every row but the last (hidden row t, token t + 1) against ling_forward.py --mtp REF.mtp.
fn mtpLogits(c: *Cover, gpu: check.Gpu, e: *Engine, ids: []const i32, ref_path: []const u8) !void {
    const a = c.a;
    var name: [512]u8 = undefined;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(gpu.io, try std.fmt.bufPrint(&name, "{s}.mtp", .{ref_path}), a, .limited(1 << 30));
    if (bytes.len < 20 or !std.mem.eql(u8, bytes[0..8], "TFLINGM2")) return error.BadFixture;
    const dims = std.mem.bytesAsSlice(u32, bytes[8..20]);
    const R: usize = dims[0];
    const V: usize = dims[1];
    const K: usize = dims[2];
    const T = ids.len;
    try c.expect(R == T - 1 and V == e.c.vocab and K == e.c.topk, "MTP reference: {d} rows", .{R});
    const m64 = try a.alloc(f64, R * V);
    const mbf = try a.alloc(f64, R * V);
    const theirs = try a.alloc(i32, R * K);
    var at: usize = 20;
    for ([_][]u8{ std.mem.sliceAsBytes(m64), std.mem.sliceAsBytes(mbf), std.mem.sliceAsBytes(theirs) }) |dst| {
        @memcpy(dst, bytes[at..][0..dst.len]);
        at += dst.len;
    }
    const h = &e.mtp.?;
    var r: frig.Rig = .{ .gpu = gpu, .a = a, .s = e.stream, .m = &e.m, .g = &e.g };
    defer r.deinit();
    try e.bind(31);
    try e.begin(null);
    const q = e.seqs.items[31].?;
    const main = e.forward(&e.bp);
    const seg = [_]F.Seg{.{ .st = &q.st, .row0 = 0, .rows = T }};
    try main.embed(ids);
    try main.stack(&seg);
    try main.final(T); // e.bp.normed: the hidden rows, as a prompt pass leaves them
    const pk = try r.zeros(R * (K + 1) * 4);
    const f = h.forward(&e.m, e.stream, &e.bp);
    try f.embed(ids[1..]);
    try h.step(f, &q.hd.?, R, e.bp.normed, pk);
    try f.final(R);
    try f.logitsOf(0, R);
    const raw = try r.back(e.bp.logits, R * V * 2);
    const got = try a.alloc(u16, R * V);
    @memcpy(std.mem.sliceAsBytes(got), raw);
    try main.commit(&q.st, T);
    _ = try pass(e, 32, @ptrCast(ids), 37, null, null); // a prompt pass in 37-row chunks: its MTP rows the same
    try rig.sameSeq(c, e, "a prompt pass's MTP rows in 37-row chunks == the serial MTP rows", 32, 31);
    e.release(31);
    e.release(32);
    const ours = try frig.sortedPicks(a, try r.back(pk, R * (K + 1) * 4), 1, R, K);
    const flipped = try a.alloc(bool, R);
    @memset(flipped, false);
    const nf = frig.flips(ours, theirs, 1, R, 0, R, K, flipped);
    const d64 = frig.distance(got, m64, R, V, flipped);
    const dbf = frig.distance(got, mbf, R, V, flipped);
    var bd: f64 = 0;
    var bw: f64 = 0;
    for (0..R) |t| if (!flipped[t]) for (mbf[t * V ..][0..V], m64[t * V ..][0..V]) |x, y| {
        bd += (x - y) * (x - y);
        bw += y * y;
    };
    const base = @sqrt(bd / bw); // the bf16 points' own distance on the same rows (their routing may differ there too)
    std.debug.print("RESULT   MTP logits vs float64: rel L2 {e:.3} ({e:.3} without {d} rows whose picks differ), argmax {d}/{d}; vs bf16 points {e:.3}; bf16 points vs float64 on those rows {e:.3}\n", .{ d64.rel, d64.rel_kept, nf, d64.argmax, R, dbf.rel, base });
    try c.expect(d64.rel_kept <= 1.5 * base, "MTP logits within 1.5x the bf16 points' rel L2 of float64 on rows routed alike ({e:.3} <= 1.5 x {e:.3})", .{ d64.rel_kept, base });
}
