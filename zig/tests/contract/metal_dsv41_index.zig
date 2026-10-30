//! DeepSeek V4.1's indexer scores under the contract on Metal: exact fp32 order, masks, controls, invariance, timing.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const ref = @import("dsv41_index_ref.zig");

const c = contract.spec;
const g = contract.gates;
const ix = tf.dsv41_index;
const base = "dsv41 ix ";

/// What a registration varies: how a row's keys are bounded, and a planted fault.
pub const Variant = struct { kind: ref.Kind = .dense, ctl: u8 = 0 };

// Pairs (rows x keys) stay near a million: the float64 reference costs ~4,000 products a pair.
const wide = [_]c.Shape{
    .{ .rows = 8, .n = 256, .k = 128 }, // first: the small gates use it
    .{ .rows = 1, .n = 128, .k = 128 },
    .{ .rows = 1, .n = 129, .k = 128 },
    .{ .rows = 1, .n = 1000, .k = 128 },
    .{ .rows = 3, .n = 513, .k = 128 },
    .{ .rows = 16, .n = 4097, .k = 128 },
    .{ .rows = 33, .n = 1024, .k = 128 },
    .{ .rows = 64, .n = 4096, .k = 128 },
    .{ .rows = 1024, .n = 1024, .k = 128 },
    .{ .rows = 8192, .n = 128, .k = 128 },
    .{ .rows = 1, .n = 131072, .k = 128 },
    .{ .rows = 2, .n = 131072, .k = 128 },
};
const steps = [_]c.Shape{
    .{ .rows = 8, .n = 256, .k = 128 },
    .{ .rows = 1, .n = 128, .k = 128 },
    .{ .rows = 1, .n = 1000, .k = 128 },
    .{ .rows = 2, .n = 4097, .k = 128 },
    .{ .rows = 3, .n = 513, .k = 128 },
    .{ .rows = 8, .n = 8192, .k = 128 },
    .{ .rows = 4, .n = 65536, .k = 128 },
    .{ .rows = 1, .n = 131072, .k = 128 },
};

pub fn spec(v: Variant, name: []const u8) c.Spec {
    return .{ .name = name, .accum = .f32, .depth = ref.depth, .out = .f32, .served = if (v.kind == .decode) &steps else &wide };
}

pub const Proj = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    v: Variant,
    name: []const u8,
    pipes: ix.Pipelines,
    f: ?ref.Ix = null,
    q: rig_.Ref = undefined,
    qs: rig_.Ref = undefined,
    w: rig_.Ref = undefined,
    ik: rig_.Ref = undefined,
    cm: rig_.Ref = undefined,
    s: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, v: Variant, name: []const u8) !Proj {
        var p: Proj = .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .v = v, .name = name, .pipes = undefined };
        if (v.ctl == 0) {
            p.pipes = try ix.Pipelines.load(device);
        } else {
            const text = try std.fmt.allocPrint(a, "#define TF_CTL {d}\n{s}", .{ v.ctl, sources.deepseek_v41_index });
            defer a.free(text);
            p.pipes = try ix.Pipelines.loadSource(device, text);
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

    /// Rows r0.. of the prepared call, with chunk 0 the helper's own.
    fn call(p: *const Proj, r0: usize, nrows: usize, chunk: u32) ix.Call {
        const f = p.f.?;
        return .{
            .rows = @intCast(nrows),
            .keys = @intCast(f.shape.n),
            .start = f.geo.start + @as(u32, @intCast(r0)),
            .ratio = f.geo.ratio,
            .chunk = chunk,
            .mask_stride = f.geo.mask_stride,
            .w_scale = ref.w_scale,
        };
    }

    pub fn prepare(p: *Proj, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        p.drop();
        if (shape.k != ref.dim) return false;
        ix.check(.{ .rows = @intCast(shape.rows), .keys = @intCast(shape.n) }) catch return false;
        var f = try ref.Ix.init(p.a, shape, p.v.kind);
        f.fill(fill, seed);
        p.f = f;
        p.q = try p.rig.input(f.q);
        p.qs = try p.rig.input(f.qs);
        p.w = try p.rig.input(std.mem.sliceAsBytes(f.w));
        p.ik = try p.rig.input(f.ik);
        p.cm = try p.rig.input(f.cm);
        p.s = try p.rig.output(f.outputs() * 4);
        return true;
    }

    fn at(r: rig_.Ref, extra: usize) ix.Ref {
        return .{ .buf = r.buf, .off = r.off + extra };
    }

    /// Rows r0..r0+nrows as one launch: every input offset by its rows, the output the same.
    pub fn encode(p: *Proj, e: anytype, r0: usize, nrows: usize, chunk: u32) !void {
        const f = p.f.?;
        const n = f.shape.n;
        const cm = at(p.cm, r0 * f.geo.mask_stride);
        try ix.encode(&p.pipes, e, at(p.q, r0 * ref.qrow), at(p.qs, r0 * ref.qsrow), at(p.w, r0 * ref.heads * 2), at(p.ik, 0), at(p.s, r0 * n * 4), cm, p.call(r0, nrows, chunk));
    }

    pub fn run(p: *Proj) !void {
        const pass = p.rig.begin();
        try p.encode(pass.e, 0, p.f.?.shape.rows, 0);
        try p.rig.finish(pass);
    }

    pub fn count(p: *Proj) usize {
        return p.f.?.outputs();
    }

    pub fn reference(p: *Proj, want: []f64, mag: []f64) !void {
        return p.f.?.reference(want, mag);
    }

    /// Exact order: every score must equal the oracle's bits, so no allowance at all.
    pub fn bounds(_: *Proj, _: []const f64, _: []const f64, allow: []f64) !void {
        @memset(allow, 0);
    }

    pub fn simulate(p: *Proj, acc: c.Accum, order: g.Order, out: []f64) !void {
        return p.f.?.simulate(acc, order, out);
    }
};

/// Every registered variant, the invariance probes that run after the gates, and the timing.
pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    probes: std.ArrayList(*Proj) = .empty,

    const list = [_]struct { v: Variant, name: []const u8 }{
        .{ .v = .{}, .name = base ++ "dense" },
        .{ .v = .{ .kind = .causal }, .name = base ++ "causal prompt" },
        .{ .v = .{ .kind = .masked }, .name = base ++ "candidate-masked" },
        .{ .v = .{ .kind = .decode }, .name = base ++ "decode" },
        .{ .v = .{ .ctl = 1 }, .name = base ++ "control: relu turns a NaN score into 0" },
        .{ .v = .{ .ctl = 2 }, .name = base ++ "control: heads folded in reverse, head 0 dropped" },
        .{ .v = .{ .ctl = 3 }, .name = base ++ "control: group sums in fp16" },
        .{ .v = .{ .ctl = 4 }, .name = base ++ "control: writes past its last output" },
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

    fn launch(p: *Proj, r0: usize, nrows: usize, chunk: u32) !void {
        const pass = p.rig.begin();
        try p.encode(pass.e, r0, nrows, chunk);
        try p.rig.finish(pass);
    }

    fn poison(p: *Proj) void {
        @memset(p.rig.bytes(p.s, p.count() * 4), 0xff);
    }

    fn report(word: []const u8, what: []const u8, kind: ref.Kind, bad: usize, launches: usize) void {
        std.debug.print("{s} {s} {s}{t}: {d} launches byte-equal to the reference launch, {d} differ\n", .{ word, what, base, kind, launches, bad });
    }

    /// Row counts, chunk sizes and prompt-vs-decode launches equal one 2048-row launch byte for byte; returns misses.
    pub fn invariance(s: *Suite, seed: u64) !usize {
        var bad_all: usize = 0;
        for (s.probes.items) |p| {
            if (p.v.ctl != 0 or p.v.kind == .decode) continue;
            var bad: usize = 0;
            var launches: usize = 0;
            const shape: c.Shape = .{ .rows = 2048, .n = 1000, .k = 128 };
            if (!try p.prepare(shape, .random, seed)) return error.Refused;
            const n = shape.n;
            try launch(p, 0, shape.rows, 0);
            const whole = try s.a.dupe(u8, p.rig.bytes(p.s, p.count() * 4));
            defer s.a.free(whole);
            for ([_]usize{ 1, 3, 16, 2048 }) |r| {
                poison(p);
                try launch(p, 0, r, 0);
                launches += 1;
                if (!std.mem.eql(u8, whole[0 .. r * n * 4], p.rig.bytes(p.s, r * n * 4))) bad += 1;
            }
            report(if (bad == 0) "PASS" else "FAIL", "row invariance", p.v.kind, bad, launches);
            bad_all += bad;
            bad = 0;
            launches = 0;
            for ([_]u32{ 1, 7, 64, 255, 256, 1000 }) |chunk| {
                poison(p);
                try launch(p, 0, shape.rows, chunk);
                launches += 1;
                if (!std.mem.eql(u8, whole, p.rig.bytes(p.s, whole.len))) bad += 1;
            }
            report(if (bad == 0) "PASS" else "FAIL", "chunk invariance (1,7,64,255,256,1000 keys)", p.v.kind, bad, launches);
            bad_all += bad;
            bad = 0;
            poison(p);
            const pass = p.rig.begin();
            for (0..shape.rows) |r| try p.encode(pass.e, r, 1, 0);
            try p.rig.finish(pass);
            launches = 1;
            if (!std.mem.eql(u8, whole, p.rig.bytes(p.s, whole.len))) bad += 1;
            report(if (bad == 0) "PASS" else "FAIL", "prompt vs one decode launch a row", p.v.kind, bad, launches);
            bad_all += bad;
        }
        return bad_all;
    }

    /// Median microseconds a launch over launches queued back to back, every key live.
    pub fn timing(s: *Suite, io: std.Io) !void {
        var p = try Proj.init(s.a, s.device, s.queue, .{}, "timing");
        defer p.deinit();
        const cases = [_][2]usize{ .{ 1, 8192 }, .{ 1, 32768 }, .{ 1, 131072 }, .{ 8192, 32768 } };
        for (cases) |case| {
            if (!try p.prepare(.{ .rows = case[0], .n = case[1], .k = 128 }, .random, 7)) return error.Refused;
            const reps: usize = if (case[0] == 1) 32 else 2;
            var samples: [9]f64 = undefined;
            for (&samples) |*out| {
                const t0 = std.Io.Clock.awake.now(io);
                const pass = p.rig.begin();
                for (0..reps) |_| try p.encode(pass.e, 0, case[0], 0);
                try p.rig.finish(pass);
                const ns: f64 = @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds());
                out.* = ns / 1000.0 / @as(f64, @floatFromInt(reps));
            }
            std.mem.sort(f64, &samples, {}, std.sort.asc(f64));
            const us = samples[samples.len / 2];
            const pairs: f64 = @floatFromInt(case[0] * case[1]);
            std.debug.print("TIME {s}: rows {d} keys {d}: {d:.1} us a launch, {d:.2} ns a (row, key) pair (median of {d} x {d})\n", .{ base, case[0], case[1], us, us * 1000.0 / pairs, samples.len, reps });
        }
    }
};
