//! DeepSeek-V4.1's indexer top-512 under the kernel contract on HIP: Metal's shapes, rows and controls (topk512.hip).
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const dsv41 = hip.launches.dsv41;

const base = "dsv41 topk512 ";
const rule = "NaN ranks as -inf, -0 as +0, ties go to the lower index; row r keeps min(512, visible[r]) indices, ascending, then -1";

/// The kernel, then its controls: NaN above +inf (nonfinite), ties high and last score dropped (shapes), input write.
pub const registrations = [_]struct { fault: c_int, name: []const u8 }{
    .{ .fault = 0, .name = base ++ "tf_dsv41_topk512" },
    .{ .fault = 1, .name = base ++ "control: NaN ranked above +inf" },
    .{ .fault = 2, .name = base ++ "control: ties go to the higher index" },
    .{ .fault = 3, .name = base ++ "control: the last score dropped" },
    .{ .fault = 4, .name = base ++ "control: writes into its input" },
};

fn noSums(_: c.Shape) usize {
    return 0;
}

const served = blk: {
    const rows = [_]usize{ 1, 4, 64, 4096, 8192 };
    const widths = [_]usize{ 128, 511, 512, 513, 1024, 4096, 16384 };
    var out: [rows.len * widths.len + 4]c.Shape = undefined;
    var n: usize = 0;
    out[n] = .{ .rows = 8, .n = 512, .k = 1024 };
    n += 1;
    for (rows) |r| for (widths) |w| {
        if (r >= 4096 and w > 1024 and w < 16384) continue;
        if (r == 8192 and w > 4096) continue;
        out[n] = .{ .rows = r, .n = 512, .k = w };
        n += 1;
    };
    out[n] = .{ .rows = 1, .n = 512, .k = 131072 };
    out[n + 1] = .{ .rows = 1, .n = 512, .k = 1048576 };
    out[n + 2] = .{ .rows = 4, .n = 512, .k = 1048576 };
    break :blk out[0 .. n + 3].*;
};

pub fn spec(name: []const u8) c.Spec {
    return .{
        .name = name,
        .accum = .none,
        .depth = noSums,
        .out = .raw,
        .nan = .{ .stated = rule },
        .tiles = .{ .{ .size = 4096 }, .{}, .{ .size = 16384, .step = 128, .min = 128 } },
        .served = &served,
    };
}

fn rank(v: f32) f32 {
    return if (std.math.isNan(v)) -std.math.inf(f32) else v;
}

const Order = struct {
    row: []const f32,
    fn before(o: Order, a: u32, b: u32) bool {
        const x = rank(o.row[a]);
        const y = rank(o.row[b]);
        return x > y or (x == y and a < b);
    }
};

pub const TopK = struct {
    rig: rig_.Rig,
    l: *const hip.Launcher,
    fault: c_int,
    shape: c.Shape = undefined,
    scores: []f32 = &.{},
    vis: []i32 = &.{},
    plants: usize = 0,
    s: u64 = 0,
    v: u64 = 0,
    out: u64 = 0,
    msg: [200]u8 = undefined,

    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher, fault: c_int) TopK {
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .l = l, .fault = fault };
    }

    pub fn deinit(t: *TopK) void {
        t.free();
        t.rig.deinit();
    }

    fn free(t: *TopK) void {
        t.rig.a.free(t.scores);
        t.rig.a.free(t.vis);
        t.scores = &.{};
        t.vis = &.{};
    }

    pub fn kernel(t: *TopK, name: []const u8) g.Kernel {
        const control: ?g.Gate = switch (t.fault) {
            1 => .nonfinite,
            2, 3 => .shapes,
            4 => .repeat,
            else => null,
        };
        return .{ .spec = spec(name), .ctx = t, .vt = &vt, .control = control };
    }

    const vt = g.hooks(TopK);

    /// Metal's rows (visible whole, half + 1, 512, 513, 0, 1; five value styles; +inf past visible) and five plants.
    pub fn prepare(t: *TopK, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != 512 or shape.k < 1 or !dsv41.fits(shape.rows, shape.k)) return false;
        t.free();
        t.rig.reset();
        t.shape = shape;
        const k = shape.k;
        const a = t.rig.a;
        t.scores = try a.alloc(f32, shape.rows * k);
        t.vis = try a.alloc(i32, shape.rows);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (0..shape.rows) |r| {
            const row = t.scores[r * k ..][0..k];
            const vis: usize = if (r < 4) k else switch (r % 16) {
                1 => k / 2 + 1,
                6 => @min(k, 512),
                7 => @min(k, 513),
                9 => 0,
                13 => @min(k, 1),
                else => k,
            };
            t.vis[r] = @intCast(vis);
            for (row, 0..) |*x, i| x.* = if (i >= vis) std.math.inf(f32) else switch (r % 5) {
                0 => @as(f32, @floatFromInt(rng.uintLessThan(u32, 512))) / 16 - 16,
                1 => 1.5,
                2 => (rng.float(f32) - 0.5) * 1e6,
                3 => -@as(f32, @floatFromInt(i / 3)),
                else => switch (rng.uintLessThan(u32, 8)) {
                    0 => 1.0,
                    1...4 => 0.0,
                    else => -0.0,
                },
            };
        }
        t.plants = 0;
        if (fill == .nonfinite) {
            const at = [_][2]usize{ .{ 0, k / 3 }, .{ 1, k / 2 }, .{ 2, k - 1 }, .{ 3, 0 }, .{ 3, k - 1 } };
            const put = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), std.math.inf(f32), -std.math.inf(f32) };
            for (at, put) |p, v| if (p[0] < shape.rows) {
                t.scores[p[0] * k + p[1]] = v;
                t.plants += 1;
            };
        }
        t.s = try t.rig.input(std.mem.sliceAsBytes(t.scores));
        t.v = try t.rig.input(std.mem.sliceAsBytes(t.vis));
        t.out = try t.rig.output(t.count());
        return true;
    }

    pub fn count(t: *TopK) usize {
        return t.shape.rows * dsv41.keep * @sizeOf(i32);
    }

    /// Rows r0..r0+rows of the prepared call, as one launch.
    pub fn launch(t: *TopK, r0: usize, rows: usize) !void {
        const k = t.shape.k;
        try dsv41.topk512(t.l, t.s + r0 * k * 4, t.v + r0 * 4, t.out + r0 * dsv41.keep * 4, rows, k, t.fault, t.rig.s.handle);
    }

    pub fn run(t: *TopK) !void {
        try t.launch(0, t.shape.rows);
        try t.rig.s.synchronize();
    }

    pub fn reference(_: *TopK, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    /// The CPU stable sort against each row: all rows up to 4M scores, else a stride of rows and the last.
    pub fn check(t: *TopK, at: g.Gate) !?[]const u8 {
        const k = t.shape.k;
        const a = t.rig.a;
        const raw = try t.rig.bytes(t.out, t.count());
        const got = try a.alloc(i32, raw.len / @sizeOf(i32));
        defer a.free(got);
        @memcpy(std.mem.sliceAsBytes(got), raw);
        const stride = @max(1, t.shape.rows * k / (4 << 20));
        const idx = try a.alloc(u32, k);
        defer a.free(idx);
        var nonfinite: usize = 0;
        for (0..t.shape.rows) |r| {
            if (r >= 8 and r % stride != 0 and r + 1 != t.shape.rows) continue;
            const vis: usize = @intCast(t.vis[r]);
            const row = t.scores[r * k ..][0..vis];
            for (row) |x| nonfinite += @intFromBool(!std.math.isFinite(x));
            for (idx[0..vis], 0..) |*x, i| x.* = @intCast(i);
            std.sort.pdq(u32, idx[0..vis], Order{ .row = row }, Order.before);
            const keep = @min(dsv41.keep, vis);
            std.mem.sort(u32, idx[0..keep], {}, std.sort.asc(u32));
            const out = got[r * dsv41.keep ..][0..dsv41.keep];
            for (out, 0..) |have, i| {
                const want: i32 = if (i < keep) @intCast(idx[i]) else -1;
                if (have != want) return try std.fmt.bufPrint(&t.msg, "row {d} (visible {d} of {d}) slot {d}: got {d}, stable rank gives {d}", .{ r, vis, k, i, have, want });
            }
        }
        if (at == .nonfinite and (t.plants < 5 or nonfinite < 5)) return try std.fmt.bufPrint(&t.msg, "{d} plants: no proof", .{t.plants});
        return null;
    }
};
