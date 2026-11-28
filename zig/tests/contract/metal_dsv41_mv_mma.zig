//! DeepSeek V4.1 prompt MMA matvecs under the contract on Metal: exact bytes vs mv, controls, invariance, timing.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const ref = @import("dsv41_mv_ref.zig");

const c = contract.spec;
const g = contract.gates;
const mv = tf.dsv41_metal.mv;
const mv_mma = tf.dsv41_metal.mv_mma;
const base = "dsv41 mv mma ";

pub const Variant = struct { out: mv_mma.Out = .f32, ctl: u8 = 0 };

const served = [_]c.Shape{
    .{ .rows = 16, .n = 64, .k = 5120 },
    .{ .rows = 16, .n = 32, .k = 5120 },
    .{ .rows = 16, .n = 384, .k = 5120 },
    .{ .rows = 4, .n = 512, .k = 5120 },
    .{ .rows = 16, .n = 128, .k = 512 },
    .{ .rows = 40, .n = 96, .k = 2048 },
    .{ .rows = 300, .n = 33, .k = 1024 },
    .{ .rows = 17, .n = 33, .k = 289 },
    .{ .rows = 5, .n = 33, .k = 97 },
    .{ .rows = 3, .n = 7, .k = 31 },
    .{ .rows = 2, .n = 4099, .k = 5120 },
    .{ .rows = 1, .n = 16, .k = 512 },
};

pub fn spec(v: Variant, name: []const u8) c.Spec {
    return .{
        .name = name,
        .accum = .f32,
        .depth = ref.depth,
        .out = if (v.out == .f32) .f32 else .bf16,
        .tiles = .{ .{ .size = 16 }, .{ .size = 16 }, .{ .size = 32 } },
        .served = &served,
    };
}

pub const Proj = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    v: Variant,
    name: []const u8,
    pipes: mv_mma.Pipelines,
    f: ?ref.Mv = null,
    fill: g.Fill = .random,
    x: rig_.Ref = undefined,
    w: rig_.Ref = undefined,
    y: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, v: Variant, name: []const u8) !Proj {
        var p: Proj = .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .v = v, .name = name, .pipes = undefined };
        if (v.ctl == 0) {
            p.pipes = try mv_mma.Pipelines.load(device);
        } else {
            const text = try std.fmt.allocPrint(a, "#define TF_CTL {d}\n{s}", .{ v.ctl, sources.dsv41.mv_mma });
            defer a.free(text);
            p.pipes = try mv_mma.Pipelines.loadSource(device, text);
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

    fn es(p: *const Proj) usize {
        return if (p.v.out == .f32) 4 else 2;
    }

    pub fn prepare(p: *Proj, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        p.drop();
        const f = try ref.Mv.init(p.a, shape, false);
        p.f = f;
        p.fill = fill;
        f.fill(fill, seed, false);
        p.x = try p.rig.input(std.mem.sliceAsBytes(f.x));
        p.w = try p.rig.input(f.wBytes());
        p.y = try p.rig.output(f.outputs() * p.es());
        return true;
    }

    fn at(r: rig_.Ref, extra: usize) mv_mma.Ref {
        return .{ .buf = r.buf, .off = r.off + extra };
    }

    fn call(p: *const Proj, rows: usize) mv_mma.Call {
        const f = p.f.?;
        return .{ .k = @intCast(f.shape.k), .n = @intCast(f.shape.n), .rows = @intCast(rows), .out = p.v.out };
    }

    /// Rows r0..r0+rows of the prepared call as one launch.
    pub fn encode(p: *Proj, e: anytype, r0: usize, rows: usize) !void {
        const f = p.f.?;
        try mv_mma.encode(&p.pipes, e, at(p.x, 2 * r0 * f.shape.k), at(p.w, 0), at(p.y, p.es() * r0 * f.shape.n), p.call(rows));
    }

    pub fn run(p: *Proj) !void {
        const pass = p.rig.begin();
        try p.encode(pass.e, 0, p.f.?.shape.rows);
        try p.rig.finish(pass);
    }

    pub fn count(p: *Proj) usize {
        return p.f.?.outputs();
    }

    pub fn reference(p: *Proj, want: []f64, mag: []f64) !void {
        return p.f.?.reference(want, mag, p.v.out == .bf16);
    }

    /// Exact order: every output must equal the oracle's bits.
    pub fn bounds(p: *Proj, _: []const f64, _: []const f64, allow: []f64) !void {
        if (p.fill == .stress) return;
        @memset(allow, 0);
    }

    pub fn simulate(p: *Proj, acc: c.Accum, order: g.Order, out: []f64) !void {
        return p.f.?.simulate(acc, order, out);
    }

    fn splitsEqual(p: *Proj, whole: []const u8, steps: []const usize, label: []const u8, bad: *usize, launches: *usize) !void {
        const f = p.f.?;
        for (steps) |step| {
            @memset(p.rig.bytes(p.y, whole.len), 0xff);
            const pass = p.rig.begin();
            var r0: usize = 0;
            while (r0 < f.shape.rows) : (r0 += step) {
                try p.encode(pass.e, r0, @min(step, f.shape.rows - r0));
                launches.* += 1;
            }
            try p.rig.finish(pass);
            if (!std.mem.eql(u8, whole, p.rig.bytes(p.y, whole.len))) {
                bad.* += 1;
                std.debug.print("FAIL {s} {d}x{d}x{d}: {d} rows a launch differs from one launch\n", .{ label, f.shape.rows, f.shape.n, f.shape.k, step });
            }
        }
    }

    fn prefixesEqual(p: *Proj, whole: []const u8, bad: *usize, launches: *usize) !void {
        const f = p.f.?;
        for (1..@min(f.shape.rows, 32) + 1) |r| {
            const len = r * f.shape.n * p.es();
            @memset(p.rig.bytes(p.y, whole.len), 0xff);
            const pass = p.rig.begin();
            try p.encode(pass.e, 0, r);
            try p.rig.finish(pass);
            launches.* += 1;
            if (!std.mem.eql(u8, whole[0..len], p.rig.bytes(p.y, len))) {
                bad.* += 1;
                std.debug.print("FAIL row invariance {d}x{d}x{d}: R={d} differs from the whole launch\n", .{ f.shape.rows, f.shape.n, f.shape.k, r });
            }
        }
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    probes: std.ArrayList(*Proj) = .empty,

    const list = [_]struct { v: Variant, name: []const u8 }{
        .{ .v = .{}, .name = base ++ "bf16 weights, f32 out" },
        .{ .v = .{ .out = .bf16 }, .name = base ++ "bf16 weights, bf16 out" },
        .{ .v = .{ .ctl = 1 }, .name = base ++ "control: non-finite inputs flushed" },
        .{ .v = .{ .ctl = 2 }, .name = base ++ "control: a partial last K step dropped" },
        .{ .v = .{ .ctl = 3 }, .name = base ++ "control: sums rounded to fp16" },
        .{ .v = .{ .ctl = 4 }, .name = base ++ "control: writes past its output" },
    };

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a, .device = device, .queue = queue };
        for (list) |entry| {
            if (only) |text| if (std.mem.indexOf(u8, entry.name, text) == null) continue;
            const p = try a.create(Proj);
            errdefer a.destroy(p);
            p.* = try Proj.init(a, device, queue, entry.v, entry.name);
            errdefer p.deinit();
            try @import("contract").register(a, &s.probes, &kernels, p, p.kernel());
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

    fn find(s: *Suite, out: mv_mma.Out) ?*Proj {
        for (s.probes.items) |p| if (p.v.ctl == 0 and p.v.out == out) return p;
        return null;
    }

    /// Launch splits and row prefixes equal one whole launch, byte for byte.
    pub fn invariance(s: *Suite, seed: u64) !usize {
        var bad: usize = 0;
        var launches: usize = 0;
        var cases: usize = 0;
        const shapes = [_]c.Shape{
            .{ .rows = 300, .n = 384, .k = 5120 },
            .{ .rows = 40, .n = 65, .k = 513 },
            .{ .rows = 33, .n = 129, .k = 95 },
            .{ .rows = 33, .n = 35, .k = 289 },
        };
        for (s.probes.items) |p| {
            if (p.v.ctl != 0) continue;
            for (shapes) |shape| {
                if (!try p.prepare(shape, .random, seed)) return error.Refused;
                try p.run();
                const whole = try s.a.dupe(u8, p.rig.bytes(p.y, p.f.?.outputs() * p.es()));
                defer s.a.free(whole);
                try p.splitsEqual(whole, &.{ 1, 5, 7, 16, 17, 32 }, "chunk invariance", &bad, &launches);
                try p.prefixesEqual(whole, &bad, &launches);
                cases += 1;
            }
        }
        const word = if (bad == 0) "PASS" else "FAIL";
        std.debug.print("{s} row, split and chunk invariance {s}: {d} variant x shape cases ({d} launches) byte-equal to one launch, {d} differ\n", .{ word, base, cases, launches, bad });
        return bad;
    }

    /// Byte-for-byte equality against mv.metal across prompt, odd and K-boundary shapes.
    pub fn equality(s: *Suite, seed: u64) !usize {
        var bad: usize = 0;
        var cases: usize = 0;
        const shapes = [_]c.Shape{
            .{ .rows = 1, .n = 384, .k = 5120 },  .{ .rows = 16, .n = 32, .k = 5120 },
            .{ .rows = 16, .n = 64, .k = 5120 },  .{ .rows = 16, .n = 384, .k = 5120 },
            .{ .rows = 17, .n = 512, .k = 5120 }, .{ .rows = 300, .n = 64, .k = 5120 },
            .{ .rows = 40, .n = 128, .k = 512 },  .{ .rows = 3, .n = 64, .k = 31 },
            .{ .rows = 5, .n = 33, .k = 1 },      .{ .rows = 7, .n = 100, .k = 95 },
            .{ .rows = 16, .n = 64, .k = 97 },    .{ .rows = 2, .n = 1001, .k = 127 },
            .{ .rows = 17, .n = 5, .k = 129 },    .{ .rows = 16, .n = 32, .k = 255 },
            .{ .rows = 16, .n = 32, .k = 256 },   .{ .rows = 16, .n = 32, .k = 257 },
            .{ .rows = 9, .n = 33, .k = 513 },
        };
        var mv_pipes = try mv.Pipelines.load(s.device);
        defer mv_pipes.deinit();
        for ([_]mv_mma.Out{ .f32, .bf16 }) |out| {
            const p = s.find(out) orelse continue;
            for (shapes, 0..) |shape, si| for ([_]g.Fill{ .random, .nonfinite }) |fill| {
                if (fill == .nonfinite and shape.rows < 5) continue;
                if (!try p.prepare(shape, fill, seed + si)) return error.Refused;
                try p.run();
                const want = try p.rig.output(p.f.?.outputs() * p.es());
                const pass = p.rig.begin();
                try mv.encode(&mv_pipes, pass.e, .{ .buf = p.x.buf, .off = p.x.off }, .{ .buf = p.w.buf, .off = p.w.off }, .{ .buf = want.buf, .off = want.off }, .{
                    .k = @intCast(shape.k),
                    .n = @intCast(shape.n),
                    .rows = @intCast(shape.rows),
                    .out = if (out == .f32) .f32 else .bf16,
                });
                try p.rig.finish(pass);
                cases += 1;
                const n = p.f.?.outputs() * p.es();
                if (std.mem.indexOfDiff(u8, p.rig.bytes(p.y, n), p.rig.bytes(want, n))) |i| {
                    bad += 1;
                    std.debug.print("FAIL mv equality {d}x{d}x{d} {t} {t}: first difference at byte {d}\n", .{ shape.rows, shape.n, shape.k, out, fill, i });
                }
            };
        }
        const word = if (bad == 0) "PASS" else "FAIL";
        std.debug.print("{s} equality {s}: {d} cases byte-equal to tf_dsv41_mv, {d} differ\n", .{ word, base, cases, bad });
        return bad;
    }

    fn median(samples: []f64) f64 {
        std.mem.sort(f64, samples, {}, std.sort.asc(f64));
        return samples[samples.len / 2];
    }

    /// Prompt chunk comparison of mv_mma vs mv over release shapes.
    pub fn timing(s: *Suite, io: std.Io) !void {
        const p = s.find(.f32) orelse return;
        var mv_pipes = try mv.Pipelines.load(s.device);
        defer mv_pipes.deinit();
        const shapes = [_]c.Shape{
            .{ .rows = 2048, .n = 512, .k = 5120 },
            .{ .rows = 4096, .n = 512, .k = 5120 },
            .{ .rows = 4096, .n = 384, .k = 5120 },
            .{ .rows = 4096, .n = 64, .k = 5120 },
            .{ .rows = 4096, .n = 32, .k = 5120 },
            .{ .rows = 4096, .n = 128, .k = 512 },
        };
        for (shapes) |shape| {
            if (!try p.prepare(shape, .random, 7)) return error.Refused;
            for ([_]bool{ true, false }) |is_old| {
                var samples: [9]f64 = undefined;
                for (&samples) |*out| {
                    const t0 = std.Io.Clock.awake.now(io);
                    const pass = p.rig.begin();
                    for (0..8) |_| {
                        if (is_old) {
                            try mv.encode(&mv_pipes, pass.e, .{ .buf = p.x.buf, .off = p.x.off }, .{ .buf = p.w.buf, .off = p.w.off }, .{ .buf = p.y.buf, .off = p.y.off }, .{
                                .k = @intCast(shape.k),
                                .n = @intCast(shape.n),
                                .rows = @intCast(shape.rows),
                                .out = .f32,
                            });
                        } else {
                            try p.encode(pass.e, 0, shape.rows);
                        }
                    }
                    try p.rig.finish(pass);
                    out.* = @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1000.0 / 8.0;
                }
                const us = median(&samples);
                const flops = 2.0 * @as(f64, @floatFromInt(shape.rows * shape.n * shape.k));
                std.debug.print("TIME {s}{s} {d}x{d}x{d}: {d:.1} us median of {d} x 8 launches, {d:.2} TFLOP/s\n", .{ base, if (is_old) "mv" else "mv_mma", shape.rows, shape.n, shape.k, us, samples.len, flops / us / 1e6 });
            }
        }
    }
};
