//! DeepSeek V4.1's 2-16 row EXL3 projection and its gate/up SwiGLU under the contract: the bits of what they replace.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const row = @import("metal_dsv41_exl3_row.zig");

const c = contract.spec;
const g = contract.gates;
const ex = tf.exl3_projection;
const fu = tf.dsv41_metal.fused;
const fmt = ex.format;
const base = "dsv41 exl3 rows ";
const limit = tf.dsv41_metal.experts.swiglu_limit;

const kept = "low[i] += (high[i] - (next - part)) + (product - part);";
const up_clamp = "tf_exl3_clip(float(bfloat(uo)), -limit, limit)";
const rule = "the replaced chain's bytes on the same inputs: a NaN and both infinities in x reach the outputs as its do";

fn sh(rows: usize, n: usize, k: usize) c.Shape {
    return .{ .rows = rows, .n = n, .k = k };
}

// wq_a, wkv, wq_b, a wo_a group, wo_b, the shared expert's w1/w3 and w2, ix_wq_b: verify windows and several streams
const served_rows = [_]c.Shape{ sh(16, 1280, 5120), sh(3, 512, 5120), sh(8, 32768, 1280), sh(2, 1024, 4096), sh(2, 5120, 8192), sh(16, 2304, 5120), sh(5, 5120, 2304), sh(16, 4096, 1280) };

/// P1's registration at 2 to 16 rows, all in one launch through `tf_exl3_matmul_rows`.
fn rowsSpec(v: row.Variant, name: []const u8) c.Spec {
    var s = row.spec(v, name);
    s.served = &served_rows;
    s.tiles = .{ .{ .size = 4, .step = 1, .min = 2 }, .{ .size = 256, .step = 128, .min = 128 }, .{ .size = 256, .step = 128, .min = 128 } };
    return s;
}

fn runRows(ctx: *anyopaque) anyerror!void {
    const p: *row.Proj = @ptrCast(@alignCast(ctx));
    const s = p.shape;
    const pass = p.rig.begin();
    try ex.encode(&p.pipes, pass.e, p.m, .{ .buffer = p.x.buf, .offset = p.x.off }, .{ .buffer = p.y.buf, .offset = p.y.off }, .{ .buffer = p.scratch.buf, .offset = p.scratch.off }, .{ .count = s.rows, .x_stride = s.k, .y_stride = s.n, .output = p.v.out });
    try p.rig.finish(pass);
}

const vt_rows = blk: {
    var v = g.hooks(row.Proj);
    v.run = runRows;
    break :blk v;
};

/// The rows kernels of `p` rebuilt from the source without the TwoSum error term of their row body.
fn plantRows(a: std.mem.Allocator, device: mtl.Device, p: *row.Proj) !void {
    const at = std.mem.indexOf(u8, sources.core_exl3_rows, kept) orelse return error.ControlNotPlanted;
    const text = try std.mem.concat(a, u8, &.{ sources.core_exl3_rows[0..at], sources.core_exl3_rows[at + kept.len ..] });
    defer a.free(text);
    const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
    defer lib.deinit();
    for (&p.pipes.rows.?, ex.row_groups) |*pl, rg| {
        var buf: [32]u8 = undefined;
        const planted = try mtl.Pipeline.init(device, lib, rg.name(&buf), false);
        pl.deinit();
        pl.* = planted;
    }
}

/// A gate/up registration: each matrix's width and codebook, biases, and whether it is the planted control.
pub const PairVariant = struct { gate: struct { u16, fmt.Codebook }, up: struct { u16, fmt.Codebook }, bias: bool = false, ctl: bool = false };

fn noDepth(_: c.Shape) usize {
    return 0;
}

// the shared expert's w1 and w3 at decode: 1 to 16 rows
const served_pair = [_]c.Shape{ sh(1, 2304, 5120), sh(2, 2304, 5120), sh(4, 2304, 5120), sh(16, 2304, 5120) };

fn pairSpec(name: []const u8) c.Spec {
    return .{
        .name = name,
        // its K sums are tf_exl3_matmul_row's and _rows', which their registrations hold to fp32 at gate 4
        .accum = .none,
        .depth = noDepth,
        .out = .bf16,
        .replaces = "tf_exl3_matmul_hoisted on w1 and on w3, then tf_dsv41_f_swiglu",
        .nan = .{ .stated = rule },
        .tiles = .{ .{ .size = 4, .step = 1, .min = 1 }, .{ .size = 256, .step = 128, .min = 128 }, .{ .size = 256, .step = 128, .min = 128 } },
        .served = &served_pair,
    };
}

pub const Pair = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    v: PairVariant,
    name: []const u8,
    pipes: ex.Pipelines,
    q: ex.gate_up.Pipelines,
    fused: *const fu.Pipelines,
    shape: c.Shape = undefined,
    m: [2]ex.Matrix = undefined,
    x: rig_.Ref = undefined,
    y: rig_.Ref = undefined,
    yq: rig_.Ref = undefined,
    scratch: rig_.Ref = undefined,
    gu: rig_.Ref = undefined,
    main_y: rig_.Ref = undefined,
    main_scratch: rig_.Ref = undefined,
    msg: [160]u8 = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, fused: *const fu.Pipelines, v: PairVariant, name: []const u8) !Pair {
        var pipes = try ex.Pipelines.load(device);
        errdefer pipes.deinit();
        pipes.gpu_cores = std.math.maxInt(u32);
        var q = try ex.gate_up.Pipelines.load(device);
        errdefer q.deinit();
        if (v.ctl) {
            const at = std.mem.indexOf(u8, sources.core_exl3_rows, up_clamp) orelse return error.ControlNotPlanted;
            const text = try std.mem.concat(a, u8, &.{ sources.core_exl3_rows[0..at], "float(bfloat(uo))", sources.core_exl3_rows[at + up_clamp.len ..] });
            defer a.free(text);
            const planted = try rig_.pipeline(device, text, "tf_exl3_swiglu");
            q.swiglu.deinit();
            q.swiglu = planted;
        }
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .v = v, .name = name, .pipes = pipes, .q = q, .fused = fused };
    }

    pub fn deinit(p: *Pair) void {
        p.rig.deinit();
        p.q.deinit();
        p.pipes.deinit();
    }

    pub fn kernel(p: *Pair) g.Kernel {
        return .{ .spec = pairSpec(p.name), .ctx = p, .vt = &vt_pair, .control = if (p.v.ctl) .shapes else null };
    }

    const vt_pair = g.hooks(Pair);

    fn view(r: rig_.Ref, extra: usize) ex.View {
        return .{ .buffer = r.buf, .offset = r.off + extra };
    }

    /// Random words, fp16 scales and biases, bf16 x in [-1, 1); non-finite: x[1][3] NaN, x[2][5] +Inf, x[2][9] -Inf.
    pub fn prepare(p: *Pair, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 16 or shape.k % 128 != 0 or shape.n % 128 != 0) return false;
        p.rig.reset();
        p.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const widths = [2]struct { u16, fmt.Codebook }{ p.v.gate, p.v.up };
        for (&p.m, widths) |*m, wb| {
            const width = fmt.Width{ .halves = wb[0] };
            const words = try p.a.alloc(i16, (shape.k / 16) * (shape.n / 16) * width.tileWords());
            defer p.a.free(words);
            for (words) |*w| w.* = rng.int(i16);
            const scales = try p.a.alloc(f16, shape.k + 2 * shape.n);
            defer p.a.free(scales);
            for (scales) |*s| s.* = @floatCast(rng.float(f32) * 2 - 1);
            const t = try p.rig.input(std.mem.sliceAsBytes(words));
            const su = try p.rig.input(std.mem.sliceAsBytes(scales[0..shape.k]));
            const sv = try p.rig.input(std.mem.sliceAsBytes(scales[shape.k..][0..shape.n]));
            m.* = .{ .trellis = view(t, 0), .suh = view(su, 0), .svh = view(sv, 0), .k = shape.k, .n = shape.n, .width = width, .codebook = wb[1] };
            if (p.v.bias) {
                for (scales[shape.k + shape.n ..]) |*b| b.* = @floatCast(rng.float(f32) * 8 - 4);
                m.bias = view(try p.rig.input(std.mem.sliceAsBytes(scales[shape.k + shape.n ..])), 0);
            }
        }
        const x = try p.a.alloc(u16, shape.rows * shape.k);
        defer p.a.free(x);
        for (x) |*v| v.* = c.toBf16(rng.float(f32) * 2 - 1);
        if (fill == .nonfinite and shape.rows > 2) {
            x[shape.k + 3] = 0x7fc0;
            x[2 * shape.k + 5] = 0x7f80;
            x[2 * shape.k + 9] = 0xff80;
        }
        p.x = try p.rig.input(std.mem.sliceAsBytes(x));
        const out = shape.rows * shape.n * 2;
        p.y = try p.rig.output(out);
        p.yq = try p.rig.output(out);
        p.scratch = try p.rig.scratch(try ex.gate_up.scratchBytes(shape.k, shape.n, shape.rows));
        p.gu = try p.rig.scratch(2 * out);
        p.main_y = try p.rig.scratch(2 * out);
        p.main_scratch = try p.rig.scratch(try ex.scratchBytes(shape.k, shape.n, shape.rows));
        return true;
    }

    fn rows(p: *const Pair) ex.Rows {
        return .{ .count = p.shape.rows, .x_stride = p.shape.k, .y_stride = p.shape.n };
    }

    pub fn run(p: *Pair) !void {
        const pass = p.rig.begin();
        try ex.gate_up.encode(&p.pipes, &p.q, pass.e, p.m[0], p.m[1], view(p.x, 0), view(p.y, 0), view(p.yq, 0), view(p.scratch, 0), p.rows(), limit);
        try p.rig.finish(pass);
    }

    /// Main's chain: w1 and w3 through tf_exl3_matmul_hoisted to bf16, then tf_dsv41_f_swiglu; y then yq.
    pub fn replaced(p: *Pair) ![]const u8 {
        var main = p.pipes;
        main.row = null;
        main.rows = null;
        const len = p.shape.rows * p.shape.n;
        const pass = p.rig.begin();
        for (p.m, 0..) |m, i| {
            try ex.encode(&main, pass.e, m, view(p.x, 0), view(p.gu, i * len * 2), view(p.main_scratch, 0), p.rows());
        }
        const ref = struct {
            fn of(r: rig_.Ref, extra: usize) fu.Ref {
                return .{ .buf = r.buf, .off = r.off + extra };
            }
        };
        fu.swiglu(p.fused, pass.e, ref.of(p.gu, 0), ref.of(p.gu, len * 2), ref.of(p.main_y, 0), ref.of(p.main_y, len * 2), @intCast(len), limit);
        try p.rig.finish(pass);
        return p.rig.bytes(p.main_y, 4 * len);
    }

    /// Gate 1: the replaced chain's bytes, with some output non-finite to prove the planted values reach it.
    pub fn check(p: *Pair, _: g.Gate) !?[]const u8 {
        const got = try p.a.dupe(u8, try p.rig.outputBytes());
        defer p.a.free(got);
        const want = try p.replaced();
        if (std.mem.indexOfDiff(u8, got, want)) |at| return try std.fmt.bufPrint(&p.msg, "byte {d} of {d} differs from the replaced chain's", .{ at, got.len });
        for (0..p.count()) |i| if (c.classOf(c.read(.bf16, got, i)) != .finite) return null;
        return try std.fmt.bufPrint(&p.msg, "no output is non-finite: the planted values prove nothing", .{});
    }

    pub fn count(p: *Pair) usize {
        return 2 * p.shape.rows * p.shape.n;
    }

    /// A replacement is held to the replaced chain's bytes at every gate that compares; it has no float64 outputs.
    pub fn reference(_: *Pair, _: []f64, _: []f64) !void {
        return error.ReplacementOnly;
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    rows: std.ArrayList(*row.Proj) = .empty,
    pairs: std.ArrayList(*Pair) = .empty,
    fused: ?*fu.Pipelines = null,

    const rows_list = [_]struct { v: row.Variant, name: []const u8 }{
        .{ .v = .{ .halves = 4, .book = .mcg }, .name = base ++ "tf_exl3_matmul_rows mcg 2-bit" },
        .{ .v = .{ .halves = 5, .book = .mul1 }, .name = base ++ "tf_exl3_matmul_rows mul1 2.5-bit" },
        .{ .v = .{ .halves = 8, .book = .mul1, .out = .f32 }, .name = base ++ "tf_exl3_matmul_rows mul1 4-bit fp32 out" },
        .{ .v = .{ .halves = 12, .book = .inst3 }, .name = base ++ "tf_exl3_matmul_rows 3inst 6-bit" },
        .{ .v = .{ .halves = 16, .book = .mcg }, .name = base ++ "tf_exl3_matmul_rows mcg 8-bit" },
        .{ .v = .{ .halves = 6, .book = .mul1, .out = .f32, .ctl = true }, .name = base ++ "control: the rows body's TwoSum error term dropped" },
    };
    const pair_list = [_]struct { v: PairVariant, name: []const u8 }{
        .{ .v = .{ .gate = .{ 5, .mul1 }, .up = .{ 4, .mcg } }, .name = base ++ "tf_exl3_swiglu w1 mul1 2.5-bit, w3 mcg 2-bit" },
        .{ .v = .{ .gate = .{ 12, .inst3 }, .up = .{ 3, .mul1 }, .bias = true }, .name = base ++ "tf_exl3_swiglu w1 3inst 6-bit, w3 mul1 1.5-bit, biases" },
        .{ .v = .{ .gate = .{ 6, .mul1 }, .up = .{ 6, .mul1 }, .ctl = true }, .name = base ++ "control: tf_exl3_swiglu's up clamp dropped" },
    };

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a };
        errdefer s.destroy();
        for (rows_list) |entry| {
            if (only) |text| if (std.mem.indexOf(u8, entry.name, text) == null) continue;
            const p = try a.create(row.Proj);
            errdefer a.destroy(p);
            p.* = try row.Proj.init(a, device, queue, .{ .halves = entry.v.halves, .book = entry.v.book, .out = entry.v.out }, entry.name);
            errdefer p.deinit();
            // every served count through the rows kernels, whatever the machine
            p.pipes.gpu_cores = std.math.maxInt(u32);
            if (entry.v.ctl) try plantRows(a, device, p);
            try contract.register(a, &s.rows, kernels, p, .{ .spec = rowsSpec(entry.v, entry.name), .ctx = p, .vt = &vt_rows, .control = if (entry.v.ctl) .shapes else null });
        }
        for (pair_list) |entry| {
            if (only) |text| if (std.mem.indexOf(u8, entry.name, text) == null) continue;
            if (s.fused == null) {
                const f = try a.create(fu.Pipelines);
                f.* = fu.Pipelines.load(device) catch |err| {
                    a.destroy(f);
                    return err;
                };
                s.fused = f;
            }
            const p = try a.create(Pair);
            errdefer a.destroy(p);
            p.* = try Pair.init(a, device, queue, s.fused.?, entry.v, entry.name);
            errdefer p.deinit();
            try contract.register(a, &s.pairs, kernels, p, p.kernel());
        }
        return s;
    }

    pub fn destroy(s: *Suite) void {
        for (s.rows.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        for (s.pairs.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.rows.deinit(s.a);
        s.pairs.deinit(s.a);
        if (s.fused) |f| {
            f.deinit();
            s.a.destroy(f);
        }
        s.a.destroy(s);
    }
};

test "dsv41 exl3 suite: a failed allocation at each step of the rows create, rows and pairs growth included, frees all" {
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var fail_index: usize = 0;
    while (true) : (fail_index += 1) {
        var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = fail_index });
        const a = failing.allocator();
        var kernels: std.ArrayList(g.Kernel) = .empty;
        defer kernels.deinit(a);
        // one rows probe and one pair, so the pair's failures unwind a registered rows probe
        const s = Suite.create(a, device, queue, &kernels, "mcg 2-bit") catch |err| {
            try std.testing.expectEqual(error.OutOfMemory, err);
            continue;
        };
        defer s.destroy();
        try std.testing.expectEqual(@as(usize, 1), s.rows.items.len);
        try std.testing.expectEqual(@as(usize, 1), s.pairs.items.len);
        try std.testing.expect(fail_index >= 6);
        break;
    }
}
