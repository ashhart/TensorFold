//! Qwen3.8 Flash Next (qwen4_exp) XPU Engine adapter for `native/xpu_lane.zig` and standalone `xpu-flashnext`.
const std = @import("std");
pub const ple = @import("xpu_ple.zig");
pub const ld = @import("xpu_load.zig");
pub const fwd = @import("xpu_forward.zig");

pub fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

pub fn nowSec() f64 {
    return @as(f64, @floatFromInt(nowNs())) * 1e-9;
}

fn median(v: []f64) f64 {
    std.mem.sort(f64, v, {}, std.sort.asc(f64));
    return v[v.len / 2];
}

pub const Options = struct {
    context: usize = 16384,
    tp: usize = 4,
    mtp: bool = true,
};

pub const Engine = struct {
    pub const Sequence = fwd.Seq;
    pub const Snapshot = fwd.Snap;

    gpa: std.mem.Allocator,
    r: *fwd.Runner,
    max_len: usize,
    load_seconds: f64,
    head: ?u8 = null,
    own: fwd.Seq = undefined,
    seqs: [64]?*fwd.Seq = @splat(null),
    m: struct { pos: u32 = 0 } = .{},

    pub fn init(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, o: Options) !*Engine {
        const t0 = nowSec();
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        const runner = try fwd.Runner.initTp(gpa, io, dir, o.context, o.tp, o.mtp);
        e.* = .{
            .gpa = gpa,
            .r = runner,
            .max_len = o.context,
            .load_seconds = nowSec() - t0,
            .head = if (runner.has_mtp) 1 else null,
            .own = runner.own,
            .m = .{ .pos = 0 },
        };
        e.own.slot = 0;
        e.seqs[0] = &e.own;
        e.r.bind(&e.own);
        return e;
    }

    pub fn deinit(e: *Engine) void {
        const gpa = e.gpa;
        e.r.own = e.own;
        e.r.cur = &e.r.own;
        e.r.deinit();
        gpa.destroy(e);
    }

    pub fn vocab(_: *const Engine) usize {
        return ld.vocab;
    }

    pub fn isEos(e: *const Engine, token: u32) bool {
        return @as(i64, @intCast(token)) == e.r.eos;
    }

    pub fn allocatedBytesPerRank(e: *const Engine) usize {
        return e.r.devs[0].allocated;
    }

    pub fn newSeq(e: *Engine) !*Sequence {
        for (1..e.seqs.len) |idx| if (e.seqs[idx] == null) {
            const s = try e.r.newSeq();
            s.slot = @intCast(idx);
            e.seqs[idx] = s;
            return s;
        };
        return error.TooManyStreams;
    }

    pub fn freeSeq(e: *Engine, s: *Sequence) void {
        if (s == &e.own) return;
        if (e.r.cur == s) e.bind(&e.own);
        if (s.slot < e.seqs.len and e.seqs[s.slot] == s) e.seqs[s.slot] = null;
        e.r.freeSeq(s);
    }

    pub fn slotOf(_: *const Engine, s: *const Sequence) u32 {
        return s.slot;
    }

    pub fn bind(e: *Engine, s: *Sequence) void {
        e.r.cur.pos = e.m.pos;
        e.r.bind(s);
        e.m.pos = s.pos;
    }

    pub fn reset(e: *Engine) !void {
        try e.r.resetSeq(e.r.cur);
        e.m.pos = 0;
    }

    pub fn sync(e: *Engine) !void {
        try e.r.syncAll();
    }

    pub fn feed(e: *Engine, token: u32, logits: bool) !void {
        try e.r.forwardPass(&.{token}, true, if (logits) .last else .none, null);
        e.m.pos = e.r.cur.pos;
    }

    pub fn stepRow(e: *Engine, token: u32) !void {
        try e.r.forwardPass(&.{token}, true, .last, null);
        e.m.pos = e.r.cur.pos;
    }

    pub fn verifyRows(e: *Engine, ids: []const u32) !void {
        try e.r.forwardPass(ids, false, .all, null);
    }

    pub fn verifyMulti(e: *Engine, slots_in: []const u32, wins: []const []const u32) !void {
        var sqs: [8]*Sequence = undefined;
        if (slots_in.len > sqs.len) return error.WindowTooWide;
        for (slots_in, 0..) |sl, i| sqs[i] = e.seqs[sl] orelse return error.UnknownStream;
        try e.r.verifyMulti(sqs[0..slots_in.len], wins);
    }

    pub fn keepRows(e: *Engine, c: u32) !void {
        try e.r.keepRows(c);
        e.m.pos = e.r.cur.pos;
    }

    pub fn argmax(e: *Engine) !u32 {
        return @intCast(e.r.host_tok[0]);
    }

    pub fn rowTokens(e: *Engine, out: []i32) !void {
        @memcpy(out, e.r.host_tok[0..out.len]);
    }

    pub fn candidates(e: *Engine, k: usize, v_out: *[64]f32, id_out: *[64]u32) !void {
        try e.r.rowCandidates(0, k, v_out, id_out);
    }

    pub fn rowCandidates(e: *Engine, row: u32, k: usize, v_out: *[64]f32, id_out: *[64]u32) !void {
        try e.r.rowCandidates(row, k, v_out, id_out);
    }

    pub fn fetchLogits(e: *Engine, out: []f32) !void {
        try e.r.fetchRowLogits(out[0..ld.vocab]);
    }

    pub fn rowLogits(e: *Engine, out: []f32) !void {
        try e.r.fetchRowLogits(out);
    }

    pub fn prefillWindows(e: *Engine, head_ids: []const u32, rows: u32) !void {
        const step = @min(@as(usize, @max(rows, 1)), ld.chunk_rows);
        var t: usize = 0;
        while (t < head_ids.len) {
            const n = @min(step, head_ids.len - t);
            try e.r.forwardPass(head_ids[t .. t + n], true, .none, null);
            e.m.pos = e.r.cur.pos;
            t += n;
        }
    }

    pub fn prefillSpan(e: *Engine, prompt: []const u32, from: usize, to: usize, rows: u32) !void {
        if (e.m.pos != from) return error.PositionMismatch;
        const step = @min(@as(usize, @max(rows, 1)), ld.chunk_rows);
        var save_dsts: [ld.max_tp]*anyopaque = undefined;
        for (0..e.r.tp) |rk| save_dsts[rk] = e.r.acts[rk].prompt_streams;

        var t: usize = from;
        while (t < to) {
            const n = @min(step, to - t);
            const last = (t + n == prompt.len);
            try e.r.forwardPass(prompt[t .. t + n], true, if (last) .last else .none, save_dsts);
            e.m.pos = e.r.cur.pos;
            const absorb = if (last) n - 1 else n;
            if (absorb > 0 and e.r.has_mtp) {
                _ = try e.r.forwardMtp(prompt[t + 1 .. t + 1 + absorb], save_dsts, t, false, false);
            }
            t += n;
        }
    }

    pub fn draftFirst(e: *Engine, first: u32, depth: u32, out: []u32) !usize {
        var st: [ld.max_tp]*anyopaque = undefined;
        for (0..e.r.tp) |rk| st[rk] = e.r.cur.ranks[rk].last_streams;
        const pk = try e.r.forwardMtp(&.{first}, st, e.m.pos - 1, true, depth > 0);
        return e.chainDrafts(pk, depth, out);
    }

    pub fn draftKept(e: *Engine, follow: []const u32, start: u32, depth: u32, out: []u32) !usize {
        var st: [ld.max_tp]*anyopaque = undefined;
        for (0..e.r.tp) |rk| st[rk] = e.r.cur.ranks[rk].verify_streams;
        const pk = try e.r.forwardMtp(follow, st, start, true, depth > 0);
        return e.chainDrafts(pk, depth, out);
    }

    fn chainDrafts(e: *Engine, first: ?fwd.Runner.MtpPick, depth: u32, out: []u32) !usize {
        if (depth == 0) return 0;
        const n = @min(@as(usize, depth), out.len);
        out[0] = first.?.token;
        var mtp_st: [ld.max_tp]*anyopaque = undefined;
        for (0..e.r.tp) |rk| mtp_st[rk] = e.r.cur.ranks[rk].mtp_out;
        var pos: usize = e.m.pos;
        for (1..n) |i| {
            const pk = (try e.r.forwardMtp(out[i - 1 .. i], mtp_st, pos, true, true)).?;
            out[i] = pk.token;
            pos += 1;
        }
        return n;
    }

    pub fn snapBytes(e: *const Engine, at: u32) u64 {
        return e.r.snapBytes(at);
    }

    pub fn snapSave(e: *Engine, at: u32) !*Snapshot {
        e.r.cur.pos = e.m.pos;
        return e.r.snapSave(at);
    }

    pub fn snapRestore(e: *Engine, sn: *const Snapshot) !void {
        try e.r.snapRestore(sn);
        e.m.pos = e.r.cur.pos;
    }

    pub fn snapDrop(e: *Engine, sn: *Snapshot) void {
        e.r.snapDrop(sn);
    }

    pub fn timeRounds(e: *Engine, ms: []f64) !f64 {
        e.r.ngram.waitWarm();
        var ids: [64]u32 = undefined;
        for (&ids, 0..) |*x, i| x.* = 1000 + @as(u32, @intCast(i)) * 37;
        try e.reset();
        try e.prefillSpan(&ids, 0, ids.len, 64);
        var g: [ld.max_verify_rows]i32 = undefined;
        var runs: [5]f64 = undefined;
        for (ms, 1..) |*out, width| {
            for (&runs) |*rv| {
                const base_pos = e.m.pos;
                const t0 = nowNs();
                try e.verifyRows(ids[0..width]);
                try e.rowTokens(g[0..width]);
                e.r.cur.history = e.r.cur.win_history;
                e.r.cur.win_open = false;
                e.r.cur.pos = base_pos;
                rv.* = @as(f64, @floatFromInt(nowNs() - t0)) / 1e6;
            }
            out.* = median(&runs);
        }
        var d: [1]u32 = undefined;
        _ = try e.draftFirst(ids[0], 1, &d);
        var mtp_st: [ld.max_tp]*anyopaque = undefined;
        for (0..e.r.tp) |rk| mtp_st[rk] = e.r.cur.ranks[rk].mtp_out;
        for (&runs) |*rv| {
            const t0 = nowNs();
            _ = try e.r.forwardMtp(&d, mtp_st, e.m.pos, true, true);
            rv.* = @as(f64, @floatFromInt(nowNs() - t0)) / 1e6;
        }
        try e.reset();
        return median(&runs);
    }
};

test "flashnext xpu: TP={1,2,4} sharding geometry and invalid degree refusal" {
    try std.testing.expectError(error.InvalidTpDegree, ld.TpShape.forTp(0));
    try std.testing.expectError(error.InvalidTpDegree, ld.TpShape.forTp(3));
    const s1 = try ld.TpShape.forTp(1);
    const s2 = try ld.TpShape.forTp(2);
    const s4 = try ld.TpShape.forTp(4);
    try std.testing.expectEqual(@as(usize, 512), s1.local_e);
    try std.testing.expectEqual(@as(usize, 256), s2.local_e);
    try std.testing.expectEqual(@as(usize, 128), s4.local_e);
    try std.testing.expectEqual(@as(usize, 2), s1.nkv);
    try std.testing.expectEqual(@as(usize, 1), s2.nkv);
    try std.testing.expectEqual(@as(usize, 1), s4.nkv);
    try std.testing.expectEqual(ld.vocab, s4.vocab_per * 4);
}

test "flashnext xpu: token IDs at or past vocab are refused before upload" {
    try fwd.ops.checkIds(&.{ 0, 1, ld.vocab - 1 });
    try std.testing.expectError(error.InvalidTokenId, fwd.ops.checkIds(&.{ld.vocab}));
    try std.testing.expectError(error.InvalidTokenId, fwd.ops.checkIds(&.{ 42, ld.vocab + 7 }));
}

test "flashnext xpu: PLE n-gram hash resets across EOS boundary" {
    const h = try ple.Hash.init(.{
        .vocab = 248320,
        .ngram = 3,
        .per = 2,
        .base = 1009,
        .divisor = 2,
        .shards = 4,
        .seed = 9973,
        .eos = 248044,
        .ple_dim = 2560,
        .ple_index = 1,
    });
    var a: [4]i64 = undefined;
    var b: [4]i64 = undefined;
    try ple.lastIds(h, &.{ 111, 248044 }, 77, &a);
    try ple.lastIds(h, &.{ 248044, 248044 }, 77, &b);
    try std.testing.expectEqualSlices(i64, &b, &a);
}
