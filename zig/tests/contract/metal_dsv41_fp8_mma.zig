//! DeepSeek V4.1 FP8 prompt MMA under the contract: exact groups, release fixture, chunk invariance, timing.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const ref = @import("dsv41_fp8_ref.zig");

const c = contract.spec;
const g = contract.gates;
const fp8 = tf.dsv41_fp8_mma;
const base = "dsv41_fp8_mma ";

pub const Variant = struct { out: fp8.Out = .f32, fixture: bool = false, ctl: u8 = 0 };

const real = [_]c.Shape{
    .{ .rows = 16, .n = 1280, .k = 5120 }, // wq_a
    .{ .rows = 1, .n = 512, .k = 5120 }, // wkv, compressor
    .{ .rows = 4, .n = 512, .k = 8192 }, // wo_b's K at a few columns
    .{ .rows = 8, .n = 5120, .k = 2304 }, // shared w2
    .{ .rows = 16, .n = 128, .k = 512 }, // indexer wk
    .{ .rows = 16, .n = 256, .k = 1280 }, // wq_b's K
    .{ .rows = 16, .n = 1024, .k = 4096 }, // wo_a
    .{ .rows = 8, .n = 256, .k = 6144 }, // engram wkv's K
};
const served = blk: {
    var out: [real.len + 4]c.Shape = undefined;
    for (real, 0..) |s, i| out[i] = s;
    for ([_]usize{ 1, 7, 9, 16 }, 0..) |r, i| out[real.len + i] = .{ .rows = r, .n = 64, .k = 1536 };
    break :blk out;
};

pub fn spec(v: Variant, name: []const u8) c.Spec {
    return .{
        .name = name,
        .accum = .f32,
        .depth = ref.depth,
        .out = if (v.out == .f32) .f32 else .bf16,
        .tiles = if (v.fixture) .{ .{}, .{}, .{} } else .{ .{}, .{ .size = 32, .step = 32, .min = 32 }, .{ .size = 256, .step = 32, .min = 32 } },
        .served = if (v.fixture) &.{ref.fixture.shape} else &served,
    };
}

pub const Proj = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    v: Variant,
    name: []const u8,
    pipes: fp8.Pipelines,
    f: ?ref.Fp8 = null,
    fill: g.Fill = .random,
    real_fixture: bool = false,
    x: rig_.Ref = undefined,
    w: rig_.Ref = undefined,
    s: rig_.Ref = undefined,
    y: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, v: Variant, name: []const u8) !Proj {
        var p: Proj = .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .v = v, .name = name, .pipes = undefined };
        if (v.ctl == 0) {
            p.pipes = try fp8.Pipelines.load(device);
        } else {
            const text = try std.fmt.allocPrint(a, "#define TF_CTL {d}\n{s}", .{ v.ctl, sources.deepseek_v41_fp8_mma });
            defer a.free(text);
            p.pipes = try fp8.Pipelines.loadSource(device, text);
        }
        return p;
    }

    pub fn deinit(p: *Proj) void {
        p.drop();
        p.pipes.deinit();
        p.rig.deinit();
    }

    fn drop(p: *Proj) void {
        if (p.f) |f| f.deinit();
        p.f = null;
        p.rig.reset();
    }

    pub fn kernel(p: *Proj) g.Kernel {
        const control: ?g.Gate = switch (p.v.ctl) {
            1 => .nonfinite,
            2 => .shapes,
            3 => .sums,
            4 => .repeat,
            else => null,
        };
        return .{ .spec = spec(p.v, p.name), .ctx = p, .vt = &vt, .control = control };
    }

    const vt = g.hooks(Proj);

    fn call(p: *const Proj, rows: usize) fp8.Call {
        const f = p.f.?;
        return .{ .k = @intCast(f.shape.k), .n = @intCast(f.shape.n), .rows = @intCast(rows), .out = p.v.out };
    }

    pub fn prepare(p: *Proj, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        p.drop();
        fp8.check(.{ .k = @intCast(shape.k), .n = @intCast(shape.n), .rows = @intCast(shape.rows) }) catch return false;
        const f = try ref.Fp8.init(p.a, shape);
        p.f = f;
        p.fill = fill;
        p.real_fixture = p.v.fixture and fill == .random and std.meta.eql(shape, ref.fixture.shape);
        if (p.real_fixture) try ref.fixture.load(f, null) else f.fill(fill, seed);
        try p.upload(f);
        return true;
    }

    fn upload(p: *Proj, f: ref.Fp8) !void {
        const bytes = f.shape.rows * f.shape.n * c.size(spec(p.v, p.name).out);
        p.x = try p.rig.input(std.mem.sliceAsBytes(f.x));
        p.w = try p.rig.input(f.codes);
        p.s = try p.rig.input(f.scales);
        p.y = try p.rig.output(bytes);
    }

    fn at(r: rig_.Ref, extra: usize) fp8.Ref {
        return .{ .buf = r.buf, .off = r.off + extra };
    }

    pub fn encode(p: *Proj, e: anytype, rows: usize, offset: usize) !void {
        const f = p.f.?;
        const xoff = 2 * offset * f.shape.k;
        const yoff = c.size(spec(p.v, p.name).out) * offset * f.shape.n;
        return fp8.encode(&p.pipes, e, at(p.x, xoff), at(p.w, 0), at(p.s, 0), at(p.y, yoff), p.call(rows));
    }

    pub fn run(p: *Proj) !void {
        const pass = p.rig.begin();
        try p.encode(pass.e, p.f.?.shape.rows, 0);
        try p.rig.finish(pass);
    }

    pub fn count(p: *Proj) usize {
        return p.f.?.outputs();
    }

    pub fn reference(p: *Proj, want: []f64, mag: []f64) !void {
        const f = p.f.?;
        if (!p.real_fixture) return f.reference(want, mag, p.v.out == .bf16);
        try ref.npyF64(ref.fixture.ref_npy, want);
        f.magnitudes(mag);
    }

    /// Random inputs must reproduce the fp8_rows order bit for bit; stress inputs keep the generic bound.
    pub fn bounds(p: *Proj, _: []const f64, _: []const f64, allow: []f64) !void {
        if (p.real_fixture or p.fill == .stress) return;
        @memset(allow, 0);
    }

    pub fn simulate(p: *Proj, acc: c.Accum, order: g.Order, out: []f64) !void {
        return p.f.?.simulate(acc, order, out);
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    probes: std.ArrayList(*Proj) = .empty,

    const list = [_]struct { v: Variant, name: []const u8 }{
        .{ .v = .{}, .name = base ++ "f32" },
        .{ .v = .{ .out = .bf16 }, .name = base ++ "bf16" },
        .{ .v = .{ .fixture = true }, .name = base ++ "release wkv fixture" },
        .{ .v = .{ .ctl = 1 }, .name = base ++ "control: non-finite inputs flushed" },
        .{ .v = .{ .ctl = 2 }, .name = base ++ "control: tile roots added in a chain" },
        .{ .v = .{ .ctl = 3 }, .name = base ++ "control: group roots rounded to fp16" },
        .{ .v = .{ .ctl = 4 }, .name = base ++ "control: writes past its output" },
    };

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a, .device = device, .queue = queue };
        for (list) |entry| {
            if (only) |text| if (std.mem.indexOf(u8, entry.name, text) == null) continue;
            const p = try a.create(Proj);
            p.* = try Proj.init(a, device, queue, entry.v, entry.name);
            try s.probes.append(a, p);
            try kernels.append(a, p.kernel());
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

    fn at(r: rig_.Ref) fp8.Ref {
        return .{ .buf = r.buf, .off = r.off };
    }

    /// A whole launch equals arbitrary row chunks and leading-row offsets at every requested size.
    pub fn invariance(s: *Suite, seed: u64) !usize {
        const p = if (s.probes.items.len > 0) s.probes.items[0] else return 0;
        if (p.v.ctl != 0 or p.v.out != .f32) return 0;
        var bad: usize = 0;
        var launches: usize = 0;
        for ([_]c.Shape{ .{ .rows = 8192, .n = 64, .k = 5120 }, .{ .rows = 2048, .n = 64, .k = 1536 } }) |shape| {
            if (!try p.prepare(shape, .random, seed)) return error.Refused;
            const whole = try s.a.alloc(u8, shape.rows * shape.n * 4);
            defer s.a.free(whole);
            try p.run();
            @memcpy(whole, p.rig.bytes(p.y, whole.len));
            for ([_]usize{ 64, 512, 2048, 8192 }) |rows| {
                if (rows > shape.rows) continue;
                for ([_]usize{ 0, 1, 37 }) |offset| {
                    if (offset + rows > shape.rows) continue;
                    @memset(p.rig.bytes(p.y, whole.len), 0xff);
                    const pass = p.rig.begin();
                    try p.encode(pass.e, rows, offset);
                    try p.rig.finish(pass);
                    launches += 1;
                    const got = p.rig.bytes(p.y, (offset + rows) * shape.n * 4)[offset * shape.n * 4 ..];
                    const want = whole[offset * shape.n * 4 ..][0 .. rows * shape.n * 4];
                    if (!std.mem.eql(u8, want, got)) {
                        bad += 1;
                        std.debug.print("FAIL {s}chunk invariance: rows {d} offset {d}\n", .{ base, rows, offset });
                    }
                }
            }
        }
        std.debug.print("{s} {s}chunk invariance: 64/512/2048/8192 rows, offsets 0/1/37, {d} launches, {d} differ\n", .{ if (bad == 0) "PASS" else "FAIL", base, launches, bad });
        return bad;
    }

    /// Per projection class: the worst error against float64 must not exceed the fp8_rows order's.
    pub fn accuracy(s: *Suite, seed: u64) !usize {
        const p = if (s.probes.items.len > 0) s.probes.items[0] else return 0;
        if (p.v.ctl != 0 or p.v.out != .f32) return 0;
        var bad: usize = 0;
        for (real) |shape| {
            if (!try p.prepare(shape, .random, seed)) return error.Refused;
            try p.run();
            const f = p.f.?;
            const got: [*]const f32 = @ptrCast(@alignCast(p.rig.bytes(p.y, f.outputs() * 4).ptr));
            const leaves = try s.a.alloc(f32, ref.levelsOf(shape.k / 32));
            defer s.a.free(leaves);
            var worst_mma: f64 = 0;
            var worst_rows: f64 = 0;
            var equal: usize = 0;
            for (0..shape.rows) |r| for (0..shape.n) |j| {
                const o = f.one(r, j, leaves);
                const v = got[r * shape.n + j];
                worst_mma = @max(worst_mma, @abs(@as(f64, v) - o.exact));
                worst_rows = @max(worst_rows, @abs(@as(f64, o.oracle) - o.exact));
                if (v == o.oracle) equal += 1;
            };
            const fail = worst_mma > worst_rows;
            if (fail) bad += 1;
            std.debug.print("{s} {s}vs fp8_rows {d}x{d}x{d}: worst error {e:.3} (fp8_rows {e:.3}), {d}/{d} bits equal\n", .{ if (fail) "FAIL" else "PASS", base, shape.rows, shape.n, shape.k, worst_mma, worst_rows, equal, f.outputs() });
        }
        return bad;
    }

    /// TFLOPS at the two prompt sizes on the hidden-5120 shared-w1 shape.
    pub fn timing(s: *Suite, io: std.Io) !void {
        var p = try Proj.init(s.a, s.device, s.queue, .{}, "timing");
        defer p.deinit();
        for ([_]usize{ 2048, 8192 }) |rows| {
            const shape: c.Shape = .{ .rows = rows, .n = 2304, .k = 5120 };
            if (!try p.prepare(shape, .random, 7)) return error.Refused;
            var samples: [7]f64 = undefined;
            for (&samples) |*out| {
                const t0 = std.Io.Clock.awake.now(io);
                const pass = p.rig.begin();
                try fp8.encode(&p.pipes, pass.e, at(p.x), at(p.w), at(p.s), at(p.y), p.call(rows));
                try p.rig.finish(pass);
                const ns: f64 = @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds());
                out.* = ns / 1000.0;
            }
            std.mem.sort(f64, &samples, {}, std.sort.asc(f64));
            const us = samples[samples.len / 2];
            const flops = 2.0 * @as(f64, @floatFromInt(shape.rows * shape.n * shape.k));
            std.debug.print("TIME {s}: K 5120 N 2304 R {d}: {d:.1} us median of {d}, {d:.3} TFLOPS\n", .{ base, rows, us, samples.len, flops / us / 1e6 });
        }
    }
};
