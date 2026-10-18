//! FastH3's tile top-k (h3_topk, CUDA) under the contract: its bytes against a CPU oracle with the same comparisons.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;

const c = contract.spec;
const g = contract.gates;

fn noSums(_: c.Shape) usize {
    return 0;
}

const rule = "a NaN score ranks as -inf, ties go to the lower tile, so every row keeps exactly n tiles";

/// Rows: query video tiles; n: tiles kept a row; k: video tiles; FastH3's 280 video tiles at 864x480 served.
pub fn spec() c.Spec {
    return .{
        .name = "cuda h3 tile top-k h3_topk",
        .accum = .none,
        .depth = noSums,
        .out = .raw,
        .nan = .{ .stated = rule },
        .tiles = .{ .{}, .{}, .{ .size = 32 } },
        .served = &.{ .{ .rows = 280, .n = 56, .k = 280 }, .{ .rows = 8, .n = 3, .k = 12 } },
    };
}

const heads = 2;
const prefix = 8;

fn rank(v: f32) f32 {
    return if (std.math.isNan(v)) -std.math.inf(f32) else v;
}

pub const TopK = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    shape: c.Shape = undefined,
    scores: []f32 = &.{},
    s: u64 = 0,
    out: u64 = 0,
    msg: [200]u8 = undefined,

    pub fn deinit(t: *TopK) void {
        t.rig.gpu.gpa.free(t.scores);
        t.rig.deinit();
    }

    pub fn kernel(t: *TopK) g.Kernel {
        return .{ .spec = spec(), .ctx = t, .vt = &vt };
    }

    const vt = g.hooks(TopK);

    fn tiles(t: *const TopK) usize {
        return prefix + t.shape.k;
    }

    pub fn prepare(t: *TopK, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n > shape.k or shape.rows > shape.k) return false;
        t.rig.reset();
        t.shape = shape;
        const nt = t.tiles();
        const a = t.rig.gpu.gpa;
        a.free(t.scores);
        t.scores = &.{};
        t.scores = try a.alloc(f32, heads * nt * nt);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // eighths: many ties
        for (t.scores) |*v| v.* = @as(f32, @floatFromInt(rng.uintLessThan(u32, 64))) / 8 - 4;
        if (fill == .nonfinite) {
            const at = [_]usize{ (prefix + 1) * nt + prefix + 2, (prefix + 2) * nt + nt - 1, (prefix + 3) * nt + prefix };
            const put = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };
            for (at, put) |p, v| t.scores[p] = v;
        }
        t.s = try t.rig.input(std.mem.sliceAsBytes(t.scores));
        t.out = try t.rig.output(t.count());
        return true;
    }

    pub fn run(t: *TopK) !void {
        var args: cuda.Args = .{};
        args.add(t.s);
        args.add(t.out);
        args.add(@as(c_int, @intCast(t.tiles())));
        args.add(@as(c_int, prefix));
        args.add(@as(c_int, @intCast(t.shape.n)));
        try cuda.launch.launch(t.f, .{ .grid = .{ .x = @intCast(t.shape.rows), .y = heads }, .block = .{ .x = 32 } }, t.rig.s, &args);
        try t.rig.s.synchronize();
    }

    pub fn count(t: *TopK) usize {
        return heads * t.shape.k * (prefix + t.shape.n) * 4;
    }

    pub fn reference(_: *TopK, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    /// Each launched row's chosen tiles against the oracle: the prefix, then each tile fewer than `n` score above.
    pub fn check(t: *TopK, at: g.Gate) !?[]const u8 {
        const nt = t.tiles();
        const nv = t.shape.k;
        const keep = t.shape.n;
        const bytes = try t.rig.bytes(t.out, t.count());
        const got: []align(1) const i32 = std.mem.bytesAsSlice(i32, bytes);
        var planted: usize = 0;
        for (0..heads) |h| for (0..t.shape.rows) |q| {
            const row = t.scores[(h * nt + prefix + q) * nt + prefix ..][0..nv];
            const out = got[(h * nv + q) * (prefix + keep) ..][0 .. prefix + keep];
            for (0..prefix) |i| if (out[i] != i) return try std.fmt.bufPrint(&t.msg, "head {d} row {d}: prefix slot {d} holds {d}", .{ h, q, i, out[i] });
            var placed: usize = prefix;
            for (row, 0..) |raw, j| {
                if (!std.math.isFinite(raw)) planted += 1;
                const mine = rank(raw);
                var above: usize = 0;
                for (row, 0..) |other, i| above += @intFromBool(rank(other) > mine or (rank(other) == mine and i < j));
                if (above >= keep) continue;
                if (placed >= out.len or out[placed] != prefix + j) return try std.fmt.bufPrint(&t.msg, "head {d} row {d}: tile {d} kept, slot {d} differs", .{ h, q, j, placed });
                placed += 1;
            }
            if (placed != out.len) return try std.fmt.bufPrint(&t.msg, "head {d} row {d}: {d} tiles kept, not {d}", .{ h, q, placed - prefix, keep });
        };
        if (at == .nonfinite and planted == 0) return try std.fmt.bufPrint(&t.msg, "no planted score in a launched row: no proof", .{});
        return null;
    }
};
