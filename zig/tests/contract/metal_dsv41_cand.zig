//! DeepSeek-V4.1's candidate blocks under the kernel contract on Metal: stable-sort check, controls, splits, timing.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");

const c = contract.spec;
const g = contract.gates;
const base = "dsv41 cand ";
const inf = std.math.inf(f32);
const rule = "NaN ranks as -inf, -0 as +0; block maxima over 8 keys, the last visible block pinned first, ties to the lower block";

fn noSums(_: c.Shape) usize {
    return 0;
}

/// Shapes are (rows, topk, blocks); keys are 8 a block less 3 when blocks is odd; stride 4 over when blocks % 8 == 5.
const served = blk: {
    const rows = [_]usize{ 1, 4, 64, 4096, 8192 };
    const blocks = [_]usize{ 64, 255, 2048, 2049, 2053, 16384 };
    var out: [rows.len * blocks.len + 12]c.Shape = undefined;
    var n: usize = 0;
    out[n] = .{ .rows = 8, .n = 2048, .k = 3000 };
    n += 1;
    for (rows) |r| for (blocks) |b| {
        if (r * b > (1 << 21) or (r > 64 and b > 255 and b < 16384)) continue;
        out[n] = .{ .rows = r, .n = 2048, .k = b };
        n += 1;
    };
    for ([_]c.Shape{
        .{ .rows = 16, .n = 1, .k = 300 },
        .{ .rows = 16, .n = 100, .k = 5000 },
        .{ .rows = 16, .n = 2047, .k = 2048 },
        .{ .rows = 5, .n = 2048, .k = 2049 },
        .{ .rows = 1, .n = 2048, .k = 131071 },
        .{ .rows = 1, .n = 2048, .k = 131072 },
        .{ .rows = 4, .n = 2048, .k = 131072 },
    }) |s| {
        out[n] = s;
        n += 1;
    }
    break :blk out[0..n].*;
};

pub fn spec(name: []const u8) c.Spec {
    return .{
        .name = name,
        .accum = .none,
        .depth = noSums,
        .out = .raw,
        .nan = .{ .stated = rule },
        .tiles = .{ .{}, .{}, .{ .size = 4096, .step = 64, .min = 64 } },
        .served = &served,
    };
}

fn rank(v: f32) f32 {
    return if (std.math.isNan(v)) -inf else v;
}

const Order = struct {
    maxima: []const f32,
    fn before(o: Order, a: u32, b: u32) bool {
        return o.maxima[a] > o.maxima[b] or (o.maxima[a] == o.maxima[b] and a < b);
    }
};

pub const Cand = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    ctl: u8,
    full: bool = false,
    shape: c.Shape = undefined,
    keys: usize = 0,
    stride: usize = 0,
    scores: []f32 = &.{},
    vis: []i32 = &.{},
    plants: usize = 0,
    s: rig_.Ref = undefined,
    v: rig_.Ref = undefined,
    kd: rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    msg: [200]u8 = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, ctl: u8) !Cand {
        const text = try std.fmt.allocPrint(a, "#define TF_CTL {d}\n{s}", .{ ctl, sources.dsv41.cand });
        defer a.free(text);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try rig_.pipeline(device, text, "tf_cand"), .ctl = ctl };
    }

    pub fn deinit(t: *Cand) void {
        t.a.free(t.scores);
        t.a.free(t.vis);
        t.pipe.deinit();
        t.rig.deinit();
    }

    pub fn kernel(t: *Cand, name: []const u8) g.Kernel {
        const control: ?g.Gate = switch (t.ctl) {
            1 => .nonfinite,
            2, 3 => .shapes,
            4 => .repeat,
            else => null,
        };
        return .{ .spec = spec(name), .ctx = t, .vt = &vt, .control = control };
    }

    const vt = g.hooks(Cand);

    pub fn prepare(t: *Cand, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n < 1 or shape.n > 2048 or shape.k < 1) return false;
        t.rig.reset();
        t.shape = shape;
        const nb = shape.k;
        t.keys = 8 * nb - (if (nb % 2 == 1) @as(usize, 3) else 0);
        t.stride = nb + (if (nb % 8 == 5) @as(usize, 4) else 0);
        const keys = t.keys;
        t.a.free(t.scores);
        t.a.free(t.vis);
        t.scores = &.{};
        t.vis = &.{};
        t.scores = try t.a.alloc(f32, shape.rows * keys);
        t.vis = try t.a.alloc(i32, shape.rows);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (0..shape.rows) |r| {
            const row = t.scores[r * keys ..][0..keys];
            const vis: usize = if (t.full or r < 4) keys else switch (r % 16) {
                1 => keys / 2 + 1,
                6 => @min(keys, 8 * 2048),
                7 => @min(keys, 8 * 2048 + 1),
                9 => 0,
                13 => @min(keys, 1),
                else => keys,
            };
            t.vis[r] = @intCast(vis);
            const style: usize = if (t.full) 0 else r % 5;
            for (row, 0..) |*x, i| x.* = if (i >= vis) -inf else switch (style) {
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
            // an all-NaN block, a lone +inf, an all -inf block, a lone NaN and an all-NaN last block (which is pinned)
            const at = [_][2]usize{ .{ 0, nb / 3 }, .{ 1, nb / 2 }, .{ 2, nb - 2 }, .{ 3, 0 }, .{ 4, nb - 1 }, .{ 3, nb - 1 } };
            const put = [_]f32{ std.math.nan(f32), inf, -inf, std.math.nan(f32), std.math.nan(f32), std.math.nan(f32) };
            for (at, put, 0..) |p, v, n| if (p[0] < shape.rows and p[1] < nb and t.vis[p[0]] > 0) {
                const row = t.scores[p[0] * keys ..][0..keys];
                const whole = n != 1 and n != 3;
                for (row[8 * p[1] .. @min(keys, 8 * p[1] + (if (whole) @as(usize, 8) else 1))]) |*x| x.* = v;
                t.plants += 1;
            };
        }
        t.s = try t.rig.input(std.mem.sliceAsBytes(t.scores));
        t.v = try t.rig.input(std.mem.sliceAsBytes(t.vis));
        t.kd = try t.rig.scratch(shape.rows * t.stride * 4);
        t.out = try t.rig.output(shape.rows * t.stride);
        return true;
    }

    /// Rows r0..r0+rows of the prepared call, as one dispatch.
    pub fn encode(t: *Cand, e: anytype, r0: usize, rows: usize) void {
        const args = [3]u32{ @intCast(t.keys), @intCast(t.stride), @intCast(t.shape.n) };
        e.setPipeline(t.pipe);
        e.setBuffer(t.s.buf, t.s.off + r0 * t.keys * 4, 1);
        e.setBuffer(t.v.buf, t.v.off + r0 * 4, 2);
        e.setBuffer(t.kd.buf, t.kd.off + r0 * t.stride * 4, 3);
        e.setBuffer(t.out.buf, t.out.off + r0 * t.stride, 4);
        e.setValue(args, 0);
        e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(1024, 1, 1));
    }

    pub fn run(t: *Cand) !void {
        @memset(t.rig.bytes(t.out, t.count()), 0x55);
        const pass = t.rig.begin();
        t.encode(pass.e, 0, t.shape.rows);
        try t.rig.finish(pass);
    }

    pub fn count(t: *Cand) usize {
        return t.shape.rows * t.stride;
    }

    pub fn reference(_: *Cand, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    /// The CPU stable sort against each row: all rows up to 512K blocks, else a stride of rows and the last.
    pub fn check(t: *Cand, at: g.Gate) !?[]const u8 {
        const nb = t.shape.k;
        const keys = t.keys;
        const got = t.rig.bytes(t.out, t.count());
        const step = @max(1, t.shape.rows * nb / (1 << 19));
        const idx = try t.a.alloc(u32, nb);
        defer t.a.free(idx);
        const bm = try t.a.alloc(f32, nb);
        defer t.a.free(bm);
        const inset = try t.a.alloc(bool, nb);
        defer t.a.free(inset);
        var nonfinite: usize = 0;
        for (0..t.shape.rows) |r| {
            if (r >= 8 and r % step != 0 and r + 1 != t.shape.rows) continue;
            const row = t.scores[r * keys ..][0..keys];
            for (row) |x| nonfinite += @intFromBool(!std.math.isFinite(x));
            for (bm, idx, 0..) |*m, *x, j| {
                m.* = -inf;
                for (row[8 * j .. @min(keys, 8 * j + 8)]) |v| m.* = @max(m.*, rank(v));
                x.* = @intCast(j);
            }
            const vis: usize = @intCast(t.vis[r]);
            if (vis >= 1 and (vis - 1) / 8 < nb) bm[(vis - 1) / 8] = inf;
            std.sort.pdq(u32, idx, Order{ .maxima = bm }, Order.before);
            const keep = @min(t.shape.n, nb);
            const out = got[r * t.stride ..][0..t.stride];
            @memset(inset, false);
            for (idx[0..keep]) |j| inset[j] = bm[j] > -inf;
            for (out, 0..) |have, j| {
                const w: u8 = if (j < nb and inset[j]) 1 else 0;
                if (have != w) return try std.fmt.bufPrint(&t.msg, "row {d} (visible {d} keys, {d} blocks, top {d}) block {d}: got {d}, stable rank gives {d}", .{ r, vis, nb, t.shape.n, j, have, w });
            }
        }
        if (at == .nonfinite and (t.plants < 5 or nonfinite < 5)) return try std.fmt.bufPrint(&t.msg, "{d} plants: no proof", .{t.plants});
        return null;
    }
};

/// Every registered variant, the launch-split probe after the gates, the glue_tmp equality and the timing.
pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    probes: std.ArrayList(*Cand) = .empty,

    const list = [_]struct { ctl: u8, name: []const u8 }{
        .{ .ctl = 0, .name = base ++ "tf_cand" },
        .{ .ctl = 1, .name = base ++ "control: NaN ranked above +inf" },
        .{ .ctl = 2, .name = base ++ "control: ties go to the higher block" },
        .{ .ctl = 3, .name = base ++ "control: the newest block not pinned" },
        .{ .ctl = 4, .name = base ++ "control: writes into its input" },
    };

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a, .device = device, .queue = queue };
        for (list) |entry| {
            if (only) |text| if (std.mem.indexOf(u8, entry.name, text) == null) continue;
            const p = try a.create(Cand);
            p.* = try Cand.init(a, device, queue, entry.ctl);
            try s.probes.append(a, p);
            try kernels.append(a, p.kernel(entry.name));
        }
        return s;
    }

    pub fn destroy(s: *Suite) void {
        for (s.probes.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
        s.a.destroy(s);
    }

    /// Rows split into launches of 1, 7 and rows/3, and one launch a row, equal the single launch byte for byte.
    pub fn invariance(s: *Suite, seed: u64) !usize {
        const p = if (s.probes.items.len > 0) s.probes.items[0] else return 0;
        if (p.ctl != 0) return 0;
        var bad: usize = 0;
        var launches: usize = 0;
        for ([_]c.Shape{ .{ .rows = 64, .n = 2048, .k = 16384 }, .{ .rows = 33, .n = 2048, .k = 2053 }, .{ .rows = 5, .n = 2048, .k = 131071 } }) |shape| {
            if (!try p.prepare(shape, .random, seed)) return error.Refused;
            try p.run();
            const whole = try s.a.dupe(u8, p.rig.bytes(p.out, p.count()));
            defer s.a.free(whole);
            for ([_]usize{ 1, 7, shape.rows / 3 }) |step| {
                @memset(p.rig.bytes(p.out, p.count()), 0x55);
                const pass = p.rig.begin();
                var r0: usize = 0;
                while (r0 < shape.rows) : (r0 += step) {
                    p.encode(pass.e, r0, @min(step, shape.rows - r0));
                    launches += 1;
                }
                try p.rig.finish(pass);
                if (!std.mem.eql(u8, whole, p.rig.bytes(p.out, p.count()))) {
                    bad += 1;
                    std.debug.print("FAIL launch split {d}x{d}: {d} rows a launch differs from one launch\n", .{ shape.rows, shape.k, step });
                }
            }
        }
        const word = if (bad == 0) "PASS" else "FAIL";
        std.debug.print("{s} launch split {s}: 3 shapes at 1, 7 and rows/3 rows a launch ({d} launches) byte-equal to one launch, {d} differ\n", .{ word, base, launches, bad });
        return bad;
    }

    /// The O(blocks^2) glue_tmp kernel and this one give the same mask bytes over glue_tmp's envelope.
    pub fn equality(s: *Suite, seed: u64) !usize {
        const p = if (s.probes.items.len > 0) s.probes.items[0] else return 0;
        if (p.ctl != 0) return 0;
        const old = try rig_.pipeline(s.device, sources.dsv41.glue_tmp, "tf_dsv41_g_cand");
        defer old.deinit();
        var bad: usize = 0;
        var cases: usize = 0;
        const blocks = [_]usize{ 1, 7, 64, 255, 2048, 2049, 2053, 3000, 4100 };
        const tops = [_]usize{ 2048, 100, 1 };
        for (blocks, 0..) |nb, bi| for (tops, 0..) |top, ti| {
            const shape: c.Shape = .{ .rows = 1 + (bi * 3 + ti) % 9, .n = top, .k = nb };
            if (!try p.prepare(shape, .random, seed + 100 * bi + ti)) return error.Refused;
            try p.run();
            const bs = try p.rig.scratch(shape.rows * p.stride * 4);
            const want = try p.rig.output(p.count());
            const pass = p.rig.begin();
            const e = pass.e;
            e.setPipeline(old);
            e.setBuffer(p.s.buf, p.s.off, 0);
            e.setBuffer(p.v.buf, p.v.off, 1);
            e.setBuffer(bs.buf, bs.off, 2);
            e.setBuffer(want.buf, want.off, 3);
            e.setValue([3]u32{ @intCast(p.keys), @intCast(p.stride), @intCast(top) }, 4);
            e.dispatchGroups(mtl.Size.of(shape.rows, 1, 1), mtl.Size.of(256, 1, 1));
            try p.rig.finish(pass);
            cases += 1;
            const at = std.mem.indexOfDiff(u8, p.rig.bytes(p.out, p.count()), p.rig.bytes(want, p.count()));
            if (at) |i| {
                bad += 1;
                std.debug.print("FAIL glue_tmp equality {d} rows, {d} blocks, top {d}: first difference at byte {d}\n", .{ shape.rows, nb, top, i });
            }
        };
        const word = if (bad == 0) "PASS" else "FAIL";
        std.debug.print("{s} equality {s}: {d} cases of 1..9 rows, 1..4100 blocks and top 1, 100, 2048 byte-equal to glue_tmp's tf_dsv41_g_cand, {d} differ\n", .{ word, base, cases, bad });
        return bad;
    }

    /// Median microseconds a launch, over launches queued back to back, at full visible length.
    pub fn timing(s: *Suite, io: std.Io) !void {
        var p = try Cand.init(s.a, s.device, s.queue, 0);
        defer p.deinit();
        p.full = true;
        for ([_]usize{ 16384, 131072 }) |blocks| for ([_]usize{ 1, 16 }) |rows| {
            if (!try p.prepare(.{ .rows = rows, .n = 2048, .k = blocks }, .random, 7)) return error.Refused;
            const reps: usize = if (rows == 1) 32 else 4;
            var samples: [9]f64 = undefined;
            for (&samples) |*out| {
                const t0 = std.Io.Clock.awake.now(io);
                const pass = p.rig.begin();
                for (0..reps) |_| p.encode(pass.e, 0, rows);
                try p.rig.finish(pass);
                const ns: f64 = @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds());
                out.* = ns / 1000.0 / @as(f64, @floatFromInt(reps));
            }
            std.mem.sort(f64, &samples, {}, std.sort.asc(f64));
            const med = samples[samples.len / 2];
            std.debug.print("TIME {s}: blocks {d} rows {d}: {d:.1} us a launch, {d:.3} us a row (median of {d} x {d} launches)\n", .{ base, blocks, rows, med, med / @as(f64, @floatFromInt(rows)), samples.len, reps });
        };
    }
};
