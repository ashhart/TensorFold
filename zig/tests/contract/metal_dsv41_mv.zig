//! DeepSeek V4.1's small matvecs under the contract on Metal: exact bytes, controls, invariance, equality, head.
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
const base = "dsv41 mv ";
const head_shape: c.Shape = .{ .rows = 16, .n = 129280, .k = 5120 };

/// What a registration varies: the weight type, the output, one launch a row, a planted fault.
pub const Variant = struct { w: mv.Weight = .bf16, out: mv.Out = .f32, solo: bool = false, ctl: u8 = 0 };

const served = [_]c.Shape{
    .{ .rows = 16, .n = 64, .k = 5120 }, // indexer weights_proj
    .{ .rows = 16, .n = 384, .k = 5120 }, // router gate
    .{ .rows = 1, .n = 384, .k = 5120 },
    .{ .rows = 4, .n = 512, .k = 5120 }, // compressor wkv and wgate
    .{ .rows = 16, .n = 128, .k = 512 }, // indexer wk
    .{ .rows = 2, .n = 4099, .k = 5120 }, // a head slice with an odd column count
    .{ .rows = 40, .n = 96, .k = 2048 }, // prompt chunks
    .{ .rows = 300, .n = 33, .k = 1024 },
    .{ .rows = 5, .n = 33, .k = 97 },
    .{ .rows = 3, .n = 7, .k = 31 },
    .{ .rows = 8, .n = 512, .k = 5120 }, // compressor wkv at a verify round's rows
    .{ .rows = 1, .n = 8197, .k = 544 }, // head-wide geometries at each row tier
    .{ .rows = 2, .n = 8197, .k = 544 },
    .{ .rows = 3, .n = 8197, .k = 544 },
    .{ .rows = 5, .n = 8197, .k = 544 },
} ++ everyRow();

fn everyRow() [16]c.Shape {
    var out: [16]c.Shape = undefined;
    for (&out, 1..) |*s, r| s.* = .{ .rows = r, .n = 48, .k = 1088 };
    return out;
}

pub fn spec(v: Variant, name: []const u8) c.Spec {
    return .{
        .name = name,
        .accum = .f32,
        .depth = ref.depth,
        .out = if (v.out == .f32) .f32 else .bf16,
        .tiles = .{ .{ .size = 16 }, .{}, .{ .size = 32 } },
        .served = &served,
    };
}

pub const Proj = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    v: Variant,
    name: []const u8,
    pipes: mv.Pipelines,
    f: ?ref.Mv = null,
    fill: g.Fill = .random,
    x: rig_.Ref = undefined,
    w: rig_.Ref = undefined,
    y: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, v: Variant, name: []const u8) !Proj {
        var p: Proj = .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .v = v, .name = name, .pipes = undefined };
        if (v.ctl == 0) {
            p.pipes = try mv.Pipelines.load(device);
        } else {
            const text = try std.fmt.allocPrint(a, "#define TF_CTL {d}\n{s}", .{ v.ctl, sources.dsv41.mv });
            defer a.free(text);
            p.pipes = try mv.Pipelines.loadSource(device, text);
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
        const f = try ref.Mv.init(p.a, shape, p.v.w == .f32);
        p.f = f;
        p.fill = fill;
        f.fill(fill, seed, false);
        p.x = try p.rig.input(std.mem.sliceAsBytes(f.x));
        p.w = try p.rig.input(f.wBytes());
        p.y = try p.rig.output(f.outputs() * p.es());
        return true;
    }

    fn at(r: rig_.Ref, extra: usize) mv.Ref {
        return .{ .buf = r.buf, .off = r.off + extra };
    }

    fn call(p: *const Proj, rows: usize) mv.Call {
        const f = p.f.?;
        return .{ .k = @intCast(f.shape.k), .n = @intCast(f.shape.n), .rows = @intCast(rows), .w = p.v.w, .out = p.v.out };
    }

    /// Rows r0..r0+rows of the prepared call as one launch.
    pub fn encode(p: *Proj, e: anytype, r0: usize, rows: usize) !void {
        const f = p.f.?;
        try mv.encode(&p.pipes, e, at(p.x, 2 * r0 * f.shape.k), at(p.w, 0), at(p.y, p.es() * r0 * f.shape.n), p.call(rows));
    }

    pub fn run(p: *Proj) !void {
        const pass = p.rig.begin();
        const rows = p.f.?.shape.rows;
        if (p.v.solo) {
            for (0..rows) |r| try p.encode(pass.e, r, 1);
        } else {
            try p.encode(pass.e, 0, rows);
        }
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

    /// The prepared call's launches in chunks of `step` rows leave the same bytes as `whole`.
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

    /// Row prefixes 1..min(rows, 16) alone leave the first rows of the whole launch's bytes.
    fn prefixesEqual(p: *Proj, whole: []const u8, bad: *usize, launches: *usize) !void {
        const f = p.f.?;
        for (1..@min(f.shape.rows, 16) + 1) |r| {
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

fn glueTmp(device: mtl.Device, out: mv.Out) !mtl.Pipeline {
    return rig_.pipeline(device, sources.dsv41.glue_tmp, if (out == .f32) "tf_dsv41_g_mv_f32" else "tf_dsv41_g_mv_bf16");
}

/// glue_tmp's launch: x [rows][k], w [n][k] bf16 into y: one simdgroup an output.
fn encodeOld(old: mtl.Pipeline, e: anytype, p: *const Proj, y: rig_.Ref) void {
    const f = p.f.?;
    e.setPipeline(old);
    e.setBuffer(p.x.buf, p.x.off, 0);
    e.setBuffer(p.w.buf, p.w.off, 1);
    e.setBuffer(y.buf, y.off, 2);
    e.setValue([2]u32{ @intCast(f.shape.k), @intCast(f.shape.n) }, 3);
    e.dispatchGroups(mtl.Size.of(f.shape.n, f.shape.rows, 1), mtl.Size.of(32, 1, 1));
}

/// Every registered variant, the probes that run after the gates, and the head's shape and timing.
pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    probes: std.ArrayList(*Proj) = .empty,

    const list = [_]struct { v: Variant, name: []const u8 }{
        .{ .v = .{}, .name = base ++ "bf16 weights, f32 out" },
        .{ .v = .{ .out = .bf16 }, .name = base ++ "bf16 weights, bf16 out" },
        .{ .v = .{ .w = .f32 }, .name = base ++ "f32 weights, f32 out" },
        .{ .v = .{ .w = .f32, .out = .bf16 }, .name = base ++ "f32 weights, bf16 out" },
        .{ .v = .{ .solo = true }, .name = base ++ "one launch a row" },
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

    /// The plain probe for a weight type and output, or null when not registered.
    fn find(s: *Suite, w: mv.Weight, out: mv.Out) ?*Proj {
        for (s.probes.items) |p| if (p.v.ctl == 0 and !p.v.solo and p.v.w == w and p.v.out == out) return p;
        return null;
    }

    /// Launch splits and row prefixes of every variant equal the one launch, byte for byte.
    pub fn invariance(s: *Suite, seed: u64) !usize {
        var bad: usize = 0;
        var launches: usize = 0;
        var cases: usize = 0;
        for (s.probes.items) |p| {
            if (p.v.ctl != 0 or p.v.solo) continue;
            for ([_]c.Shape{ .{ .rows = 300, .n = 384, .k = 5120 }, .{ .rows = 40, .n = 65, .k = 513 }, .{ .rows = 33, .n = 129, .k = 95 } }) |shape| {
                if (!try p.prepare(shape, .random, seed)) return error.Refused;
                try p.run();
                const whole = try s.a.dupe(u8, p.rig.bytes(p.y, p.f.?.outputs() * p.es()));
                defer s.a.free(whole);
                try p.splitsEqual(whole, &.{ 1, 5, 7, 16, 17 }, "chunk invariance", &bad, &launches);
                try p.prefixesEqual(whole, &bad, &launches);
                cases += 1;
            }
        }
        const word = if (bad == 0) "PASS" else "FAIL";
        std.debug.print("{s} row, split and chunk invariance {s}: {d} variant x shape cases ({d} launches) byte-equal to one launch, {d} differ\n", .{ word, base, cases, launches, bad });
        return bad;
    }

    /// Bytes equal glue_tmp's over its envelope; fp32 weights holding bf16 values equal the bf16 weights.
    pub fn equality(s: *Suite, seed: u64) !usize {
        var bad: usize = 0;
        var cases: usize = 0;
        const shapes = [_]c.Shape{
            .{ .rows = 1, .n = 384, .k = 5120 }, .{ .rows = 16, .n = 384, .k = 5120 }, .{ .rows = 3, .n = 64, .k = 31 },
            .{ .rows = 5, .n = 33, .k = 1 },     .{ .rows = 7, .n = 100, .k = 95 },    .{ .rows = 16, .n = 64, .k = 97 },
            .{ .rows = 17, .n = 5, .k = 129 },   .{ .rows = 40, .n = 128, .k = 512 },  .{ .rows = 300, .n = 64, .k = 5120 },
            .{ .rows = 2, .n = 1001, .k = 127 }, .{ .rows = 9, .n = 4099, .k = 5120 },
        };
        for ([_]mv.Out{ .f32, .bf16 }) |out| {
            const p = s.find(.bf16, out) orelse continue;
            const old = try glueTmp(s.device, out);
            defer old.deinit();
            for (shapes, 0..) |shape, si| for ([_]g.Fill{ .random, .nonfinite }) |fill| {
                if (fill == .nonfinite and shape.rows < 5) continue;
                if (!try p.prepare(shape, fill, seed + si)) return error.Refused;
                try p.run();
                const want = try p.rig.output(p.f.?.outputs() * p.es());
                const pass = p.rig.begin();
                encodeOld(old, pass.e, p, want);
                try p.rig.finish(pass);
                cases += 1;
                const n = p.f.?.outputs() * p.es();
                if (std.mem.indexOfDiff(u8, p.rig.bytes(p.y, n), p.rig.bytes(want, n))) |i| {
                    bad += 1;
                    std.debug.print("FAIL glue_tmp equality {d}x{d}x{d} {t} {t}: first difference at byte {d}\n", .{ shape.rows, shape.n, shape.k, out, fill, i });
                }
            };
        }
        for ([_]mv.Out{ .f32, .bf16 }) |out| {
            const wide = s.find(.f32, out) orelse continue;
            const narrow = s.find(.bf16, out) orelse continue;
            for (shapes[0..9], 0..) |shape, si| {
                if (!try wide.prepare(shape, .random, seed + si)) return error.Refused;
                wide.f.?.fill(.random, seed + si, true);
                @memcpy(wide.rig.bytes(wide.w, wide.f.?.wBytes().len), wide.f.?.wBytes());
                try wide.run();
                const f = wide.f.?;
                const bits = try s.a.alloc(u16, shape.n * shape.k);
                defer s.a.free(bits);
                for (bits, f.wf) |*b, v| b.* = c.toBf16(v);
                narrow.drop();
                narrow.f = try ref.Mv.init(s.a, shape, false);
                @memcpy(narrow.f.?.x, f.x);
                @memcpy(narrow.f.?.wb, bits);
                narrow.x = try narrow.rig.input(std.mem.sliceAsBytes(f.x));
                narrow.w = try narrow.rig.input(std.mem.sliceAsBytes(bits));
                narrow.y = try narrow.rig.output(f.outputs() * narrow.es());
                try narrow.run();
                cases += 1;
                const n = f.outputs() * wide.es();
                if (!std.mem.eql(u8, wide.rig.bytes(wide.y, n), narrow.rig.bytes(narrow.y, n))) {
                    bad += 1;
                    std.debug.print("FAIL f32 weights equality {d}x{d}x{d} {t}: differs from the same values as bf16\n", .{ shape.rows, shape.n, shape.k, out });
                }
            }
        }
        const word = if (bad == 0) "PASS" else "FAIL";
        std.debug.print("{s} equality {s}: {d} cases byte-equal to glue_tmp's tf_dsv41_g_mv_f32 and _bf16 (bf16 weights) or to the bf16 weights of the same values (fp32 weights), {d} differ\n", .{ word, base, cases, bad });
        return bad;
    }

    /// The LM head's shape: prefixes, splits, sampled columns against the oracle, glue_tmp's bytes; `time` adds timing.
    pub fn head(s: *Suite, io: std.Io, seed: u64, time: bool) !usize {
        const p = s.find(.bf16, .f32) orelse return 0;
        if (!try p.prepare(head_shape, .random, seed)) return error.Refused;
        const f = p.f.?;
        var bad: usize = 0;
        var launches: usize = 0;
        try p.run();
        const len = f.outputs() * 4;
        const whole = try s.a.dupe(u8, p.rig.bytes(p.y, len));
        defer s.a.free(whole);
        try p.splitsEqual(whole, &.{ 1, 7, 16 }, "head chunk invariance", &bad, &launches);
        try p.prefixesEqual(whole, &bad, &launches);
        var sampled: usize = 0;
        var j: usize = 0;
        while (j < f.shape.n) : (j += 4099) {
            for (0..f.shape.rows) |r| {
                const want: u32 = @bitCast(f.one(r, j).oracle);
                const got = std.mem.readInt(u32, whole[4 * (r * f.shape.n + j) ..][0..4], .little);
                sampled += 1;
                if (got != want) {
                    bad += 1;
                    std.debug.print("FAIL head oracle row {d} column {d}: {x} where the fp32 order gives {x}\n", .{ r, j, got, want });
                }
            }
        }
        const last = f.shape.n - 1;
        for (0..f.shape.rows) |r| {
            const want: u32 = @bitCast(f.one(r, last).oracle);
            if (std.mem.readInt(u32, whole[4 * (r * f.shape.n + last) ..][0..4], .little) != want) bad += 1;
            sampled += 1;
        }
        const old = try glueTmp(s.device, .f32);
        defer old.deinit();
        const want = try p.rig.output(len);
        const pass = p.rig.begin();
        encodeOld(old, pass.e, p, want);
        try p.rig.finish(pass);
        if (std.mem.indexOfDiff(u8, whole, p.rig.bytes(want, len))) |i| {
            bad += 1;
            std.debug.print("FAIL head glue_tmp equality: first difference at byte {d}\n", .{i});
        }
        const word = if (bad == 0) "PASS" else "FAIL";
        std.debug.print("{s} head {s}{d}x{d}x{d}: {d} launches byte-equal to one launch, {d} sampled outputs equal the fp32 order, whole output equals glue_tmp, {d} problems\n", .{ word, base, f.shape.rows, f.shape.n, f.shape.k, launches, sampled, bad });
        if (time) try s.timing(io, p, old);
        return bad;
    }

    fn median(samples: []f64) f64 {
        std.mem.sort(f64, samples, {}, std.sort.asc(f64));
        return samples[samples.len / 2];
    }

    /// Microseconds a launch over cold weights: the weight buffer cycled over enough copies to miss every cache.
    fn timing(s: *Suite, io: std.Io, p: *Proj, old: mtl.Pipeline) !void {
        const f = p.f.?;
        const wbytes = f.wBytes();
        const copies = @max(2, (768 << 20) / wbytes.len);
        const ws = try s.a.alloc(rig_.Ref, copies);
        defer s.a.free(ws);
        for (ws) |*w| w.* = try p.rig.input(wbytes);
        const cuts = [_]struct { rows: usize, n: usize }{ .{ .rows = 1, .n = f.shape.n }, .{ .rows = 16, .n = f.shape.n } };
        for (cuts) |cut| for ([_]bool{ false, true }) |is_old| {
            var samples: [9]f64 = undefined;
            for (&samples) |*out| {
                const t0 = std.Io.Clock.awake.now(io);
                const pass = p.rig.begin();
                for (ws) |w| {
                    p.w = w;
                    if (is_old) encodeOldRows(old, pass.e, p, cut.rows) else try p.encode(pass.e, 0, cut.rows);
                }
                try p.rig.finish(pass);
                const ns: f64 = @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds());
                out.* = ns / 1000.0 / @as(f64, @floatFromInt(copies));
            }
            const us = median(&samples);
            const read: f64 = @floatFromInt(wbytes.len + cut.rows * f.shape.k * 2);
            std.debug.print("TIME {s}{s} head K {d} N {d} R {d}: {d:.1} us median of {d} x {d} cold launches, {d:.0} GB/s over {d:.1} MB read\n", .{ base, if (is_old) "glue_tmp" else "mv", f.shape.k, f.shape.n, cut.rows, us, samples.len, copies, read / us / 1000.0, read / 1e6 });
        };
        try s.promptTiming(io, p, old);
    }

    fn encodeOldRows(old: mtl.Pipeline, e: anytype, p: *const Proj, rows: usize) void {
        const f = p.f.?;
        e.setPipeline(old);
        e.setBuffer(p.x.buf, p.x.off, 0);
        e.setBuffer(p.w.buf, p.w.off, 1);
        e.setBuffer(p.y.buf, p.y.off, 2);
        e.setValue([2]u32{ @intCast(f.shape.k), @intCast(f.shape.n) }, 3);
        e.dispatchGroups(mtl.Size.of(f.shape.n, rows, 1), mtl.Size.of(32, 1, 1));
    }

    /// Prompt chunks of the router gate and the indexer projection: mv against glue_tmp, weights warm as in a layer.
    fn promptTiming(s: *Suite, io: std.Io, _: *Proj, old: mtl.Pipeline) !void {
        var p = try Proj.init(s.a, s.device, s.queue, .{}, "timing");
        defer p.deinit();
        for ([_]c.Shape{ .{ .rows = 512, .n = 384, .k = 5120 }, .{ .rows = 2048, .n = 384, .k = 5120 }, .{ .rows = 2048, .n = 64, .k = 5120 } }) |shape| {
            if (!try p.prepare(shape, .random, 7)) return error.Refused;
            for ([_]bool{ false, true }) |is_old| {
                var samples: [9]f64 = undefined;
                for (&samples) |*out| {
                    const t0 = std.Io.Clock.awake.now(io);
                    const pass = p.rig.begin();
                    for (0..8) |_| if (is_old) encodeOldRows(old, pass.e, &p, shape.rows) else try p.encode(pass.e, 0, shape.rows);
                    try p.rig.finish(pass);
                    out.* = @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1000.0 / 8.0;
                }
                const us = median(&samples);
                const flops = 2.0 * @as(f64, @floatFromInt(shape.rows * shape.n * shape.k));
                std.debug.print("TIME {s}{s} prompt chunk {d}x{d}x{d}: {d:.1} us median of {d} x 8 launches, {d:.2} TFLOP/s\n", .{ base, if (is_old) "glue_tmp" else "mv", shape.rows, shape.n, shape.k, us, samples.len, flops / us / 1e6 });
            }
        }
    }
};
