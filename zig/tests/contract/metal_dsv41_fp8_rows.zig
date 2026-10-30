//! DeepSeek V4.1's FP8 dense rows under the contract on Metal: exact bytes, real fixture, controls, invariance, timing.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const ref = @import("dsv41_fp8_ref.zig");

const c = contract.spec;
const g = contract.gates;
const fp8 = tf.dsv41_fp8_rows;
const base = "dsv41_fp8_rows ";

/// What a registration varies: the output, the split, one launch a row, the real fixture, a planted fault.
pub const Variant = struct { out: fp8.Out = .f32, split: u32 = 1, solo: bool = false, fixture: bool = false, ctl: u8 = 0 };

const real = [_]c.Shape{
    .{ .rows = 16, .n = 1280, .k = 5120 }, // wq_a
    .{ .rows = 1, .n = 512, .k = 5120 }, // wkv, compressor
    .{ .rows = 4, .n = 512, .k = 8192 }, // wo_b's K at a few columns
    .{ .rows = 8, .n = 5120, .k = 2304 }, // shared w2
    .{ .rows = 16, .n = 128, .k = 512 }, // indexer wk
};
const served = blk: {
    var out: [real.len + 16]c.Shape = undefined;
    for (real, 0..) |s, i| out[i] = s;
    for (1..17) |r| out[real.len + r - 1] = .{ .rows = r, .n = 64, .k = 1536 }; // every row count
    break :blk out;
};

pub fn spec(v: Variant, name: []const u8) c.Spec {
    return .{
        .name = name,
        .accum = .f32,
        .depth = ref.depth,
        .out = if (v.out == .f32) .f32 else .bf16,
        .tiles = if (v.fixture) .{ .{}, .{}, .{} } else .{ .{ .size = 16 }, .{ .size = 32, .step = 32, .min = 32 }, .{ .size = 256, .step = 32, .min = 32 } },
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
    scratch: rig_.Ref = undefined,
    y: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, v: Variant, name: []const u8) !Proj {
        var p: Proj = .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .v = v, .name = name, .pipes = undefined };
        if (v.ctl == 0) {
            p.pipes = try fp8.Pipelines.load(device);
        } else {
            const text = try std.fmt.allocPrint(a, "#define TF_CTL {d}\n{s}", .{ v.ctl, sources.deepseek_v41_fp8_rows });
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

    fn call(p: *const Proj, rows: usize, split: u32) fp8.Call {
        const f = p.f.?;
        return .{ .k = @intCast(f.shape.k), .n = @intCast(f.shape.n), .rows = @intCast(rows), .split = split, .out = p.v.out };
    }

    pub fn prepare(p: *Proj, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        p.drop();
        fp8.check(.{ .k = @intCast(shape.k), .n = @intCast(shape.n), .rows = @intCast(shape.rows), .split = p.v.split }) catch return false;
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
        p.scratch = try p.rig.scratch(fp8.scratchBytes(p.call(f.shape.rows, 4))); // split 4 has the most tiles
        p.y = try p.rig.output(bytes);
    }

    fn at(r: rig_.Ref, extra: usize) fp8.Ref {
        return .{ .buf = r.buf, .off = r.off + extra };
    }

    pub fn encode(p: *Proj, e: anytype, rows: usize, split: u32) !void {
        const f = p.f.?;
        const es = c.size(spec(p.v, p.name).out);
        if (!p.v.solo) return fp8.encode(&p.pipes, e, at(p.x, 0), at(p.w, 0), at(p.s, 0), at(p.scratch, 0), at(p.y, 0), p.call(rows, split));
        for (0..rows) |r| try fp8.encode(&p.pipes, e, at(p.x, 2 * r * f.shape.k), at(p.w, 0), at(p.s, 0), at(p.scratch, 0), at(p.y, es * r * f.shape.n), p.call(1, split));
    }

    pub fn run(p: *Proj) !void {
        const pass = p.rig.begin();
        try p.encode(pass.e, p.f.?.shape.rows, p.v.split);
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

    /// Exact order: every output must equal the oracle's bits. The real fixture keeps the float64 bound.
    pub fn bounds(p: *Proj, _: []const f64, _: []const f64, allow: []f64) !void {
        if (p.real_fixture or p.fill == .stress) return;
        @memset(allow, 0);
    }

    pub fn simulate(p: *Proj, acc: c.Accum, order: g.Order, out: []f64) !void {
        return p.f.?.simulate(acc, order, out);
    }
};

/// Every registered variant, the probes that run after the gates, and the timing of real shapes.
pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    probes: std.ArrayList(*Proj) = .empty,

    const list = [_]struct { v: Variant, name: []const u8 }{
        .{ .v = .{}, .name = base ++ "f32 split 1" },
        .{ .v = .{ .split = 4 }, .name = base ++ "f32 split 4" },
        .{ .v = .{ .out = .bf16 }, .name = base ++ "bf16 split 1" },
        .{ .v = .{ .out = .bf16, .split = 4 }, .name = base ++ "bf16 split 4" },
        .{ .v = .{ .solo = true }, .name = base ++ "one launch a row" },
        .{ .v = .{ .fixture = true }, .name = base ++ "release wkv fixture" },
        .{ .v = .{ .ctl = 1 }, .name = base ++ "control: non-finite inputs flushed" },
        .{ .v = .{ .ctl = 2 }, .name = base ++ "control: tile roots added in a chain" },
        .{ .v = .{ .ctl = 3 }, .name = base ++ "control: sums rounded to fp16" },
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

    /// R rows alone equal the first R rows of the 16-row launch at each split, on three shapes; returns mismatches.
    pub fn invariance(s: *Suite, seed: u64) !usize {
        const p = if (s.probes.items.len > 0) s.probes.items[0] else return 0;
        if (p.v.ctl != 0 or p.v.split != 1 or p.v.out != .f32) return 0;
        var bad: usize = 0;
        var launches: usize = 0;
        for ([_]c.Shape{ .{ .rows = 16, .n = 1280, .k = 5120 }, .{ .rows = 16, .n = 128, .k = 512 }, .{ .rows = 16, .n = 512, .k = 8192 } }) |shape| {
            if (!try p.prepare(shape, .random, seed)) return error.Refused;
            const n = shape.n;
            const whole = try s.a.alloc(u8, 16 * n * 4);
            defer s.a.free(whole);
            try p.run();
            @memcpy(whole, p.rig.bytes(p.y, whole.len));
            for ([_]u32{ 1, 4 }) |split| for (1..17) |r| {
                @memset(p.rig.bytes(p.y, whole.len), 0xff);
                const pass = p.rig.begin();
                try p.encode(pass.e, r, split);
                try p.rig.finish(pass);
                launches += 1;
                if (!std.mem.eql(u8, whole[0 .. r * n * 4], p.rig.bytes(p.y, r * n * 4))) {
                    bad += 1;
                    std.debug.print("FAIL row invariance {d}x{d}x{d}: R={d} split {d} differs from the 16-row launch\n", .{ shape.rows, n, shape.k, r, split });
                }
            };
            // the one-launch kernels: every block kind and row tier
            for ([_]fp8.Kind{ .a, .b, .c }) |kind| for (1..17) |r| {
                @memset(p.rig.bytes(p.y, whole.len), 0xff);
                const pass = p.rig.begin();
                const call: fp8.Call = .{ .k = @intCast(shape.k), .n = @intCast(n), .rows = @intCast(r), .split = if (kind == .a) 1 else 4, .out = .f32 };
                try fp8.encodeWholeAs(&p.pipes, pass.e, at(p.x), at(p.w), at(p.s), at(p.y), call, .{}, kind);
                try p.rig.finish(pass);
                launches += 1;
                if (!std.mem.eql(u8, whole[0 .. r * n * 4], p.rig.bytes(p.y, r * n * 4))) {
                    bad += 1;
                    std.debug.print("FAIL row invariance {d}x{d}x{d}: R={d} one launch {s} differs from the 16-row launch\n", .{ shape.rows, n, shape.k, r, @tagName(kind) });
                }
            };
        }
        const word = if (bad == 0) "PASS" else "FAIL";
        std.debug.print("{s} row invariance {s}: R=1..16 at split 1 and 4 and one launch of blocks a, b and c on 3 shapes ({d} launches) byte-equal to the 16-row launch, {d} differ\n", .{ word, base, launches, bad });
        return bad;
    }

    /// Median microseconds a launch pair and the bytes it reads: weights cycled over enough copies to miss every cache.
    pub fn timing(s: *Suite, io: std.Io) !void {
        var p = try Proj.init(s.a, s.device, s.queue, .{}, "timing");
        defer p.deinit();
        const shapes = [_][2]u32{ .{ 5120, 2304 }, .{ 2304, 5120 }, .{ 8192, 5120 } };
        for (shapes) |kn| for ([_]u32{ 1, 4, 16 }) |rows| for ([_]u32{ 1, 4 }) |split| {
            const shape: c.Shape = .{ .rows = rows, .n = kn[1], .k = kn[0] };
            if (!try p.prepare(shape, .random, 7)) return error.Refused;
            const wbytes = shape.n * shape.k;
            const copies = @max(2, (768 << 20) / wbytes);
            const ws = try s.a.alloc(rig_.Ref, copies);
            defer s.a.free(ws);
            for (ws) |*w| w.* = try p.rig.input(p.f.?.codes);
            var samples: [9]f64 = undefined;
            for (&samples) |*out| {
                const t0 = std.Io.Clock.awake.now(io);
                const pass = p.rig.begin();
                for (ws) |w| try fp8.encode(&p.pipes, pass.e, at(p.x), at(w), at(p.s), at(p.scratch), at(p.y), p.call(rows, split));
                try p.rig.finish(pass);
                const ns: f64 = @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds());
                out.* = ns / 1000.0 / @as(f64, @floatFromInt(copies));
            }
            std.mem.sort(f64, &samples, {}, std.sort.asc(f64));
            const us = samples[samples.len / 2];
            const read: f64 = @floatFromInt(wbytes + wbytes / 1024 + rows * shape.k * 2);
            std.debug.print("TIME {s}: K {d} N {d} R {d} split {d}: {d:.1} us median of {d} x {d} cold launches, {d:.0} GB/s over {d:.1} MB read\n", .{ base, shape.k, shape.n, rows, split, us, samples.len, copies, read / us / 1000.0, read / 1e6 });
        };
    }

    fn at(r: rig_.Ref) fp8.Ref {
        return .{ .buf = r.buf, .off = r.off };
    }
};
