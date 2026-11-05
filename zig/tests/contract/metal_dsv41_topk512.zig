//! DeepSeek-V4.1's indexer top-512 over fp32 score rows under the kernel contract on Metal.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");

const c = contract.spec;
const g = contract.gates;
const base = "dsv41 topk512 ";
const rule = "NaN ranks as -inf, -0 as +0, ties go to the lower index; row r keeps min(512, visible[r]) indices, ascending, then -1";

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
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    ctl: u8,
    full: bool = false,
    shape: c.Shape = undefined,
    scores: []f32 = &.{},
    vis: []i32 = &.{},
    plants: usize = 0,
    s: rig_.Ref = undefined,
    v: rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    msg: [200]u8 = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, ctl: u8) !TopK {
        const text = try std.fmt.allocPrint(a, "#define TF_CTL {d}\n{s}", .{ ctl, sources.dsv41.topk512 });
        defer a.free(text);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try rig_.pipeline(device, text, "tf_topk512"), .ctl = ctl };
    }

    pub fn deinit(t: *TopK) void {
        t.a.free(t.scores);
        t.a.free(t.vis);
        t.pipe.deinit();
        t.rig.deinit();
    }

    pub fn kernel(t: *TopK, name: []const u8) g.Kernel {
        const control: ?g.Gate = switch (t.ctl) {
            1 => .nonfinite,
            2, 3 => .shapes,
            4 => .repeat,
            else => null,
        };
        return .{ .spec = spec(name), .ctx = t, .vt = &vt, .control = control };
    }

    const vt = g.hooks(TopK);

    pub fn prepare(t: *TopK, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != 512 or shape.k < 1) return false;
        t.rig.reset();
        t.shape = shape;
        const k = shape.k;
        t.a.free(t.scores);
        t.a.free(t.vis);
        t.scores = &.{};
        t.vis = &.{};
        t.scores = try t.a.alloc(f32, shape.rows * k);
        t.vis = try t.a.alloc(i32, shape.rows);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (0..shape.rows) |r| {
            const row = t.scores[r * k ..][0..k];
            const vis: usize = if (t.full or r < 4) k else switch (r % 16) {
                1 => k / 2 + 1,
                6 => @min(k, 512),
                7 => @min(k, 513),
                9 => 0,
                13 => @min(k, 1),
                else => k,
            };
            t.vis[r] = @intCast(vis);
            const style: usize = if (t.full) 0 else r % 5;
            for (row, 0..) |*x, i| x.* = if (i >= vis) std.math.inf(f32) else switch (style) {
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
        t.out = try t.rig.output(shape.rows * 512 * @sizeOf(i32));
        return true;
    }

    /// Rows r0..r0+rows of the prepared call, as one dispatch.
    pub fn encode(t: *TopK, e: anytype, r0: usize, rows: usize) void {
        const args = [1]u32{@intCast(t.shape.k)};
        e.setPipeline(t.pipe);
        e.setBuffer(t.s.buf, t.s.off + r0 * t.shape.k * 4, 1);
        e.setBuffer(t.v.buf, t.v.off + r0 * 4, 2);
        e.setBuffer(t.out.buf, t.out.off + r0 * 512 * 4, 3);
        e.setValue(args, 0);
        e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(1024, 1, 1));
    }

    pub fn run(t: *TopK) !void {
        @memset(t.rig.bytes(t.out, t.count()), 0x55);
        const pass = t.rig.begin();
        t.encode(pass.e, 0, t.shape.rows);
        try t.rig.finish(pass);
    }

    pub fn count(t: *TopK) usize {
        return t.shape.rows * 512 * @sizeOf(i32);
    }

    pub fn reference(_: *TopK, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    /// The CPU stable sort against each row: all rows up to 4M scores, else a stride of rows and the last.
    pub fn check(t: *TopK, at: g.Gate) !?[]const u8 {
        const k = t.shape.k;
        const got = std.mem.bytesAsSlice(i32, t.rig.bytes(t.out, t.count()));
        const stride = @max(1, t.shape.rows * k / (4 << 20));
        const idx = try t.a.alloc(u32, k);
        defer t.a.free(idx);
        var nonfinite: usize = 0;
        for (0..t.shape.rows) |r| {
            if (r >= 8 and r % stride != 0 and r + 1 != t.shape.rows) continue;
            const vis: usize = @intCast(t.vis[r]);
            const row = t.scores[r * k ..][0..vis];
            for (row) |x| nonfinite += @intFromBool(!std.math.isFinite(x));
            for (idx[0..vis], 0..) |*x, i| x.* = @intCast(i);
            std.sort.pdq(u32, idx[0..vis], Order{ .row = row }, Order.before);
            const keep = @min(512, vis);
            std.mem.sort(u32, idx[0..keep], {}, std.sort.asc(u32));
            const out = got[r * 512 ..][0..512];
            for (out, 0..) |have, i| {
                const want: i32 = if (i < keep) @intCast(idx[i]) else -1;
                if (have != want) return try std.fmt.bufPrint(&t.msg, "row {d} (visible {d} of {d}) slot {d}: got {d}, stable rank gives {d}", .{ r, vis, k, i, have, want });
            }
        }
        if (at == .nonfinite and (t.plants < 5 or nonfinite < 5)) return try std.fmt.bufPrint(&t.msg, "{d} plants: no proof", .{t.plants});
        return null;
    }
};

/// Every registered variant, the launch-split probe after the gates, and the timing.
pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    probes: std.ArrayList(*TopK) = .empty,

    const list = [_]struct { ctl: u8, name: []const u8 }{
        .{ .ctl = 0, .name = base ++ "tf_topk512" },
        .{ .ctl = 1, .name = base ++ "control: NaN ranked above +inf" },
        .{ .ctl = 2, .name = base ++ "control: ties go to the higher index" },
        .{ .ctl = 3, .name = base ++ "control: the last score dropped" },
        .{ .ctl = 4, .name = base ++ "control: writes into its input" },
    };

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a, .device = device, .queue = queue };
        for (list) |entry| {
            if (only) |text| if (std.mem.indexOf(u8, entry.name, text) == null) continue;
            const p = try a.create(TopK);
            p.* = try TopK.init(a, device, queue, entry.ctl);
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
        for ([_]c.Shape{ .{ .rows = 64, .n = 512, .k = 16384 }, .{ .rows = 33, .n = 512, .k = 700 }, .{ .rows = 5, .n = 512, .k = 131072 } }) |shape| {
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

    /// Median microseconds a launch, over launches queued back to back, at full visible length.
    pub fn timing(s: *Suite, io: std.Io) !void {
        var p = try TopK.init(s.a, s.device, s.queue, 0);
        defer p.deinit();
        p.full = true;
        for ([_]usize{ 16384, 131072 }) |width| for ([_]usize{ 1, 4096 }) |rows| {
            if (!try p.prepare(.{ .rows = rows, .n = 512, .k = width }, .random, 7)) return error.Refused;
            const reps: usize = if (rows == 1) 64 else 4;
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
            std.debug.print("TIME {s}: width {d} rows {d}: {d:.1} us a launch, {d:.3} us a row (median of {d} x {d} launches)\n", .{ base, width, rows, med, med / @as(f64, @floatFromInt(rows)), samples.len, reps });
        };
    }
};
