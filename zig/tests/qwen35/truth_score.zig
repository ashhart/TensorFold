//! Candidate logits scored against a high-precision truth .npy as tools/truth/score.py scores them, a block at a time.

const std = @import("std");
const npy = @import("npy");
const Kind = @import("logit_kind.zig").Kind;

/// One row's numbers.
const Row = struct { kl: f64, agree: bool, max_abs: f64, nll_truth: f64, nll_cand: f64 };

fn elem(wide: bool, bytes: []const u8, i: usize) f64 {
    return if (wide) @bitCast(std.mem.readInt(u64, bytes[i * 8 ..][0..8], .little)) else @as(f32, @bitCast(std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little)));
}

/// `truth` and `cand` rows (the candidate's first `cols` logits), `target` the token the row predicts, if any.
fn scoreRow(wide: bool, truth: []const u8, kind: Kind, cand: []const u8, cols: usize, target: ?u32) Row {
    var mt: f64 = -std.math.inf(f64);
    var mc: f64 = -std.math.inf(f64);
    var at: usize = 0;
    var ac: usize = 0;
    var max_abs: f64 = 0;
    for (0..cols) |i| {
        const t = elem(wide, truth, i);
        const c = kind.at(cand[i * kind.size() ..]);
        if (t > mt) {
            mt = t;
            at = i;
        }
        if (c > mc) {
            mc = c;
            ac = i;
        }
        max_abs = @max(max_abs, @abs(t - c));
    }
    // KL = sum p (t - c) - (lse_t - lse_c), with p the truth's softmax
    var zt: f64 = 0;
    var zc: f64 = 0;
    var a: f64 = 0;
    for (0..cols) |i| {
        const t = elem(wide, truth, i);
        const c = kind.at(cand[i * kind.size() ..]);
        const et = @exp(t - mt);
        zt += et;
        zc += @exp(c - mc);
        a += et * (t - c);
    }
    const lse_t = mt + @log(zt);
    const lse_c = mc + @log(zc);
    var r: Row = .{ .kl = a / zt - (lse_t - lse_c), .agree = at == ac, .max_abs = max_abs, .nll_truth = 0, .nll_cand = 0 };
    if (target) |g| {
        r.nll_truth = lse_t - elem(wide, truth, g);
        r.nll_cand = lse_c - kind.at(cand[g * kind.size() ..]);
    }
    return r;
}

pub const Score = struct {
    rows: usize,
    kl_mean: f64,
    kl_max: f64,
    /// Percent of rows whose top logit is the truth's.
    top1: f64,
    max_abs: f64,
    ppl: f64,
    ppl_truth: f64,
};

pub const Scorer = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    file: std.Io.File,
    wide: bool,
    /// Where the data starts in the file, the rows, and the logits a row.
    offset: u64,
    rows: usize,
    cols: usize,
    kind: Kind,
    /// The width of a candidate row (at least `cols`: the padded vocabulary).
    width: usize,
    /// The token each row predicts (the ids shifted by one; the last row has none).
    ids: []const u32,
    raw: []u8,
    scored: []Row,
    seen: usize = 0,
    threads: usize,

    pub const block = 32;

    pub fn init(gpa: std.mem.Allocator, io: std.Io, path: []const u8, ids: []const u32, kind: Kind, width: usize) !Scorer {
        const file = try std.Io.Dir.cwd().openFile(io, path, .{});
        errdefer file.close(io);
        var head: [1024]u8 = undefined;
        const n = try file.readPositionalAll(io, &head, 0);
        const a = try npy.parse(head[0..n]);
        if (a.rank != 2) return error.BadTruth;
        const wide = std.mem.eql(u8, a.descr, "<f8");
        if (!wide and !std.mem.eql(u8, a.descr, "<f4")) return error.BadTruth;
        if (a.shape[0] != ids.len or a.shape[1] > width) return error.TruthShape;
        const raw = try gpa.alloc(u8, block * a.shape[1] * @as(usize, if (wide) 8 else 4));
        errdefer gpa.free(raw);
        const scored = try gpa.alloc(Row, ids.len);
        return .{ .gpa = gpa, .io = io, .file = file, .wide = wide, .offset = n - a.data.len, .rows = a.shape[0], .cols = a.shape[1], .kind = kind, .width = width, .ids = ids, .raw = raw, .scored = scored, .threads = @min(16, std.Thread.getCpuCount() catch 1) };
    }

    pub fn deinit(s: *Scorer) void {
        s.gpa.free(s.scored);
        s.gpa.free(s.raw);
        s.file.close(s.io);
    }

    /// Starts a new pass over the rows.
    pub fn reset(s: *Scorer) void {
        s.seen = 0;
    }

    const Work = struct {
        s: *const Scorer,
        first: usize,
        bytes: []const u8,
        count: usize,
        from: usize,
        step: usize,

        fn go(w: Work) void {
            const s = w.s;
            const elem_size: usize = if (s.wide) 8 else 4;
            var i = w.from;
            while (i < w.count) : (i += w.step) {
                const row = w.first + i;
                const target: ?u32 = if (row + 1 < s.ids.len) s.ids[row + 1] else null;
                s.scored[row] = scoreRow(s.wide, s.raw[i * s.cols * elem_size ..][0 .. s.cols * elem_size], s.kind, w.bytes[i * s.width * s.kind.size() ..][0 .. s.width * s.kind.size()], s.cols, target);
            }
        }
    };

    /// Scores rows `first..first + count` (candidate `bytes`, `width` logits a row) against the truth's.
    pub fn put(ctx: *anyopaque, first: usize, count: usize, bytes: []const u8) anyerror!void {
        const s: *Scorer = @ptrCast(@alignCast(ctx));
        if (first + count > s.rows) return error.TruthShape;
        const elem_size: usize = if (s.wide) 8 else 4;
        const n = count * s.cols * elem_size;
        if (try s.file.readPositionalAll(s.io, s.raw[0..n], s.offset + first * s.cols * elem_size) != n) return error.TruthShort;
        var spawned: [16]?std.Thread = @splat(null);
        const t = @min(s.threads, count);
        const base: Work = .{ .s = s, .first = first, .bytes = bytes, .count = count, .from = 0, .step = t };
        for (1..t) |k| {
            var w = base;
            w.from = k;
            spawned[k] = std.Thread.spawn(.{}, Work.go, .{w}) catch null;
        }
        Work.go(base);
        for (1..t) |k| if (spawned[k]) |th| th.join() else {
            var w = base;
            w.from = k;
            Work.go(w);
        };
        s.seen += count;
    }

    pub fn result(s: *const Scorer) !Score {
        if (s.seen != s.rows) return error.MissingRows;
        var kl: f64 = 0;
        var kl_max: f64 = 0;
        var agree: usize = 0;
        var max_abs: f64 = 0;
        var nt: f64 = 0;
        var nc: f64 = 0;
        for (s.scored) |r| {
            kl += r.kl;
            kl_max = @max(kl_max, r.kl);
            agree += @intFromBool(r.agree);
            max_abs = @max(max_abs, r.max_abs);
            nt += r.nll_truth;
            nc += r.nll_cand;
        }
        const n: f64 = @floatFromInt(s.rows);
        const targets: f64 = @floatFromInt(s.rows - 1);
        return .{ .rows = s.rows, .kl_mean = kl / n, .kl_max = kl_max, .top1 = 100.0 * @as(f64, @floatFromInt(agree)) / n, .max_abs = max_abs, .ppl = @exp(nc / targets), .ppl_truth = @exp(nt / targets) };
    }
};

test {
    _ = @import("logit_kind.zig");
}

test "a row scored against itself has no divergence" {
    var truth: [4 * 8]u8 = undefined;
    var cand: [4 * 4]u8 = undefined;
    for ([_]f32{ 1.5, -2, 0.25, 3 }, 0..) |v, i| {
        std.mem.writeInt(u64, truth[i * 8 ..][0..8], @bitCast(@as(f64, v)), .little);
        std.mem.writeInt(u32, cand[i * 4 ..][0..4], @bitCast(v), .little);
    }
    const r = scoreRow(true, &truth, .f32, &cand, 4, 3);
    try std.testing.expect(@abs(r.kl) < 1e-12);
    try std.testing.expect(r.agree);
    try std.testing.expectEqual(@as(f64, 0), r.max_abs);
    try std.testing.expectApproxEqAbs(r.nll_truth, r.nll_cand, 1e-12);
}

test "the divergence of a shifted row is the log-sum-exp gap" {
    // truth (0, 0) is uniform, the candidate (0, ln 3) puts 3/4 on the second
    var truth: [2 * 4]u8 = undefined;
    var cand: [2 * 4]u8 = undefined;
    std.mem.writeInt(u32, truth[0..4], @bitCast(@as(f32, 0)), .little);
    std.mem.writeInt(u32, truth[4..8], @bitCast(@as(f32, 0)), .little);
    std.mem.writeInt(u32, cand[0..4], @bitCast(@as(f32, 0)), .little);
    std.mem.writeInt(u32, cand[4..8], @bitCast(@as(f32, @floatCast(@log(3.0)))), .little);
    const r = scoreRow(false, &truth, .f32, &cand, 2, null);
    const want = 0.5 * @log(0.5 / 0.25) + 0.5 * @log(0.5 / 0.75);
    try std.testing.expectApproxEqAbs(want, r.kl, 1e-6);
    try std.testing.expect(!r.agree);
}
