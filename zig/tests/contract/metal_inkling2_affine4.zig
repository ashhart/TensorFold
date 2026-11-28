//! Inkling 2 Mini MLX 4-bit affine g64 Metal kernels: 0-ULP exact baseline, FP64 evidence, and 8 red controls.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const c = contract.spec;
const g = contract.gates;
const Affine = contract.affine.Affine;
const rig_ = @import("rig.zig");
const host = @import("tensorfold").inkling2_metal_host.affine4;
const glue_plan = @import("inkling2_glue_plan");
const controls = @import("metal_inkling2_affine4_controls.zig");

const Kind = controls.Kind;
const hidden: usize = host.hidden;
const words_hidden: usize = hidden / 8;
const groups_hidden: usize = hidden / 64;

fn baseName(kind: Kind) []const u8 {
    return switch (kind) {
        .embed_norm => "metal inkling2 affine4 embed_norm tf_ink2_embed_norm_q4",
        .dense => "metal inkling2 affine4 dense tf_ink2_dense_q4",
        .logits => "metal inkling2 affine4 logits tf_ink2_logits_q4",
    };
}

fn label(kind: Kind, control: ?controls.Mutation) []const u8 {
    return if (control) |m| m.name() else baseName(kind);
}

fn q4Depth(s: c.Shape) usize {
    return s.k / 32 + 16;
}

const embed_shapes = [_]c.Shape{
    .{ .rows = 1, .n = hidden, .k = hidden },
    .{ .rows = 5, .n = hidden, .k = hidden },
    .{ .rows = 9, .n = hidden, .k = hidden },
};

const dense_shapes = [_]c.Shape{
    .{ .rows = 1, .n = 24, .k = hidden },
    .{ .rows = 1, .n = 3072, .k = hidden },
    .{ .rows = 5, .n = 1024, .k = hidden },
    .{ .rows = 9, .n = hidden, .k = 3072 },
    .{ .rows = 17, .n = 384, .k = hidden },
};

const logits_shapes = [_]c.Shape{
    .{ .rows = 1, .n = 129, .k = hidden },
    .{ .rows = 5, .n = 256, .k = hidden },
    .{ .rows = 9, .n = 131, .k = hidden },
};

pub fn spec(kind: Kind, control: ?controls.Mutation) c.Spec {
    return switch (kind) {
        .embed_norm => .{
            .name = label(kind, control),
            .accum = .f32,
            .depth = q4Depth,
            .ops = 96,
            .out = .bf16,
            .replaces = "affine4_ref.embedNormBytes (exact dequant/tree-sum/sqrt-div baseline)",
            .tiles = .{ .{ .size = 1 }, .{}, .{} },
            .served = &embed_shapes,
            .nan = .propagate,
            .sums = .{ .f16 = 0, .bf16 = 0, .why = "nonlinear RMSNorm with bf16 endpoints has no uniform lower-precision separability factor" },
        },
        .dense => .{
            .name = label(kind, control),
            .accum = .f32,
            .depth = q4Depth,
            .out = .bf16,
            .replaces = "affine4_ref.denseBytes (exact dequant/pair-acc/tree-sum baseline)",
            .tiles = .{ .{ .size = 4 }, .{ .size = 8 }, .{ .size = 64, .step = 64, .min = 64 } },
            .served = &dense_shapes,
            .nan = .propagate,
            .sums = .{},
        },
        .logits => .{
            .name = label(kind, control),
            .accum = .f32,
            .depth = q4Depth,
            .ops = 96,
            .mids = 1,
            .out = .f32,
            .store = .bf16,
            .replaces = "affine4_ref.logitsBytes (exact dequant/pair-acc/tree-sum/softcap30 baseline)",
            .tiles = .{ .{ .size = 4 }, .{ .size = 8 }, .{} },
            .served = &logits_shapes,
            .nan = .{ .stated = "NaN propagates to NaN; +/-Inf saturates to +/-30 through softcap30" },
            .sums = .{ .f16 = 0, .bf16 = 0, .why = "bf16 rounding before tanh softcap30 has no uniform lower-precision separability factor" },
        },
    };
}

fn hr(ref: rig_.Ref) host.Ref {
    return .{ .buf = ref.buf, .off = ref.off };
}

/// Plant cancelling terms in col 0 (warp tree order) and col 1 (pair association order) for row 0.
fn plantDenseOrderProbes(aff: *Affine) void {
    const k = aff.shape.k;
    const groups = k / 64;
    if (aff.shape.n >= 1 and k >= 64) {
        @memset(aff.codes[0..k], 0);
        @memset(aff.scales[0..groups], 0);
        @memset(aff.biases[0..groups], 0);
        aff.scales[0] = c.toBf16(512.0);
        aff.codes[0] = 8;
        aff.codes[2] = 8;
        aff.codes[32] = 8;
        aff.x[0] = c.toBf16(4096.0);
        aff.x[1] = 0;
        aff.x[2] = c.toBf16(0x1p-13);
        aff.x[3] = 0;
        aff.x[32] = c.toBf16(-4096.0);
        aff.x[33] = 0;
    }
    if (aff.shape.n >= 2 and k >= 128) {
        @memset(aff.codes[k .. 2 * k], 0);
        @memset(aff.scales[groups .. 2 * groups], 0);
        @memset(aff.biases[groups .. 2 * groups], 0);
        aff.scales[groups + 0] = c.toBf16(0x1p-13);
        aff.codes[k + 0] = 1;
        aff.scales[groups + 1] = c.toBf16(512.0);
        aff.codes[k + 64] = 8;
        aff.codes[k + 65] = 8;
        aff.x[64] = c.toBf16(4096.0);
        aff.x[65] = c.toBf16(-4096.0);
    }
}

pub const Probe = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    rig: rig_.Rig,
    kind: Kind,
    control: ?controls.Mutation,
    pipes: host.Pipelines,
    shape: c.Shape = undefined,
    aff: ?Affine = null,
    ids: []u32 = &.{},
    words: []u32 = &.{},
    scales: []u16 = &.{},
    biases: []u16 = &.{},
    norm_bf: []u16 = &.{},
    embed_rows: []f32 = &.{},
    norm_w: []f32 = &.{},
    refs: [5]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kind: Kind, control: ?controls.Mutation) !Probe {
        const text = try controls.apply(a, control orelse .none);
        defer a.free(text);
        const pipes = try host.Pipelines.loadSource(device, text);
        errdefer pipes.deinit();
        return .{
            .a = a,
            .arena = .init(a),
            .rig = .{ .a = a, .device = device, .queue = queue },
            .kind = kind,
            .control = control,
            .pipes = pipes,
        };
    }

    fn clear(p: *Probe) void {
        if (p.aff) |aff| aff.deinit();
        p.aff = null;
        p.ids = &.{};
        p.words = &.{};
        p.scales = &.{};
        p.biases = &.{};
        p.norm_bf = &.{};
        p.embed_rows = &.{};
        p.norm_w = &.{};
        p.rig.reset();
        _ = p.arena.reset(.retain_capacity);
    }

    pub fn deinit(p: *Probe) void {
        p.clear();
        p.arena.deinit();
        p.pipes.deinit();
        p.rig.deinit();
    }

    const vt = blk: {
        var v = g.hooks(Probe);
        v.check = null;
        break :blk v;
    };

    pub fn kernel(p: *Probe) g.Kernel {
        return .{
            .spec = spec(p.kind, p.control),
            .ctx = p,
            .vt = &vt,
            .control = if (p.control) |m| m.gate() else null,
        };
    }

    pub fn count(p: *const Probe) usize {
        return p.shape.rows * p.shape.n;
    }

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0) return false;
        switch (p.kind) {
            .embed_norm => if (shape.n != hidden or shape.k != hidden) return false,
            .dense => if (shape.k % 64 != 0) return false,
            .logits => if (shape.k != hidden) return false,
        }
        p.clear();
        p.shape = shape;
        const aa = p.arena.allocator();
        if (p.kind == .embed_norm) {
            const vocab: usize = 32;
            var prng = std.Random.DefaultPrng.init(seed);
            const rng = prng.random();
            p.ids = try aa.alloc(u32, shape.rows);
            for (p.ids, 0..) |*id, i| id.* = @intCast(if (fill == .nonfinite) i % vocab else (i * 7 + 3) % vocab);
            p.words = try aa.alloc(u32, vocab * words_hidden);
            p.scales = try aa.alloc(u16, vocab * groups_hidden);
            p.biases = try aa.alloc(u16, vocab * groups_hidden);
            p.norm_bf = try aa.alloc(u16, hidden);
            p.norm_w = try aa.alloc(f32, hidden);
            p.embed_rows = try aa.alloc(f32, vocab * hidden);
            for (p.words) |*w| w.* = rng.int(u32);
            for (p.scales) |*s| s.* = c.toBf16(rng.float(f32) * 0.04 + 0.01);
            for (p.biases) |*b| b.* = c.toBf16((rng.float(f32) - 0.5) * 0.8);
            for (p.norm_bf, p.norm_w) |*nb, *nf| {
                nb.* = c.toBf16(rng.float(f32) + 0.5);
                nf.* = c.fromBf16(nb.*);
            }
            if (fill == .nonfinite) p.scales[0] = c.toBf16(std.math.nan(f32));
            for (0..vocab) |tok| {
                for (0..hidden) |d| {
                    const word = p.words[tok * words_hidden + d / 8];
                    const q: f32 = @floatFromInt((word >> @intCast((d % 8) * 4)) & 0xf);
                    const sc = c.fromBf16(p.scales[tok * groups_hidden + d / 64]);
                    const bi = c.fromBf16(p.biases[tok * groups_hidden + d / 64]);
                    p.embed_rows[tok * hidden + d] = c.fromBf16(c.toBf16(q * sc + bi));
                }
            }
            p.refs[0] = try p.rig.input(std.mem.sliceAsBytes(p.ids));
            p.refs[1] = try p.rig.input(std.mem.sliceAsBytes(p.words));
            p.refs[2] = try p.rig.input(std.mem.sliceAsBytes(p.scales));
            p.refs[3] = try p.rig.input(std.mem.sliceAsBytes(p.biases));
            p.refs[4] = try p.rig.input(std.mem.sliceAsBytes(p.norm_bf));
            p.out = try p.rig.output(p.count() * 2);
            return true;
        }
        var aff = try Affine.init(p.a, shape, 4, 64);
        errdefer aff.deinit();
        aff.form = .dequantized;
        try aff.fill(fill, seed);
        if (p.kind == .dense and fill == .random) plantDenseOrderProbes(&aff);
        if (p.kind == .logits and fill == .random) {
            // Ensure v=0 saturates softcap30 (~30) while other columns exercise bf16 rounding before tanh.
            aff.scales[0] = c.toBf16(1.0);
            aff.biases[0] = c.toBf16(1.0);
            for (0..64) |d| aff.x[d] = c.toBf16(1.0);
        }
        p.words = try aa.alloc(u32, shape.n * (shape.k / 8));
        aff.words(p.words);
        p.scales = aff.scales;
        p.biases = aff.biases;
        p.aff = aff;
        p.refs[0] = try p.rig.input(std.mem.sliceAsBytes(aff.x));
        p.refs[1] = try p.rig.input(std.mem.sliceAsBytes(p.words));
        p.refs[2] = try p.rig.input(std.mem.sliceAsBytes(aff.scales));
        p.refs[3] = try p.rig.input(std.mem.sliceAsBytes(aff.biases));
        p.out = try p.rig.output(p.count() * @as(usize, if (p.kind == .logits) 4 else 2));
        return true;
    }

    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        switch (p.kind) {
            .embed_norm => host.embedNormRefs(&p.pipes, e, hr(p.refs[0]), hr(p.refs[1]), hr(p.refs[2]), hr(p.refs[3]), hr(p.refs[4]), hr(p.out), @intCast(p.shape.rows)),
            .dense => host.denseRefs(&p.pipes, e, hr(p.refs[0]), hr(p.refs[1]), hr(p.refs[2]), hr(p.refs[3]), hr(p.out), .{
                .rows = @intCast(p.shape.rows),
                .n = @intCast(p.shape.n),
                .k = @intCast(p.shape.k),
                .ldy = @intCast(p.shape.n),
            }),
            .logits => host.logitsRefs(&p.pipes, e, hr(p.refs[0]), hr(p.refs[1]), hr(p.refs[2]), hr(p.refs[3]), hr(p.out), @intCast(p.shape.rows), @intCast(p.shape.n)),
        }
        try p.rig.finish(pass);
    }

    /// Named 0-ULP scalar baseline bytes; also cross-checks the baseline against the FP64 reference as evidence.
    pub fn replaced(p: *Probe) ![]const u8 {
        const aa = p.arena.allocator();
        const bytes: []const u8 = switch (p.kind) {
            .embed_norm => blk: {
                const out = try aa.alloc(u16, p.count());
                host.ref.embedNormBytes(p.ids, std.mem.sliceAsBytes(p.words), std.mem.sliceAsBytes(p.scales), std.mem.sliceAsBytes(p.biases), std.mem.sliceAsBytes(p.norm_bf), out);
                break :blk std.mem.sliceAsBytes(out);
            },
            .dense => blk: {
                const out = try aa.alloc(u16, p.count());
                host.ref.denseBytes(p.aff.?.x, std.mem.sliceAsBytes(p.words), std.mem.sliceAsBytes(p.scales), std.mem.sliceAsBytes(p.biases), out, p.shape.rows, p.shape.n, p.shape.k, p.shape.n);
                break :blk std.mem.sliceAsBytes(out);
            },
            .logits => blk: {
                const out = try aa.alloc(f32, p.count());
                host.ref.logitsBytes(p.aff.?.x, std.mem.sliceAsBytes(p.words), std.mem.sliceAsBytes(p.scales), std.mem.sliceAsBytes(p.biases), out, p.shape.rows, p.shape.n);
                break :blk std.mem.sliceAsBytes(out);
            },
        };
        try p.verifyFp64Evidence(bytes);
        return bytes;
    }

    fn verifyFp64Evidence(p: *Probe, baseline_bytes: []const u8) !void {
        const n = p.count();
        const want = try p.a.alloc(f64, n);
        defer p.a.free(want);
        const mag = try p.a.alloc(f64, n);
        defer p.a.free(mag);
        try p.reference(want, mag);
        const sp = spec(p.kind, null);
        for (want, mag, 0..) |w, m, i| {
            const got = c.read(sp.out, baseline_bytes, i);
            const limit = c.bound(sp, p.shape, w, m);
            const q = if (got == w) 0 else @abs(got - w) / limit;
            if (!(q <= 1)) return error.Fp64EvidenceExceeded;
        }
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        switch (p.kind) {
            .embed_norm => {
                for (0..p.shape.rows) |row| {
                    const x = p.embed_rows[p.ids[row] * hidden ..][0..hidden];
                    for (0..hidden) |d| {
                        const w = glue_plan.embedValue(x, p.norm_w, d);
                        want[row * hidden + d] = w;
                        mag[row * hidden + d] = 10 * @abs(w) + 1e-5;
                    }
                }
            },
            .dense => try p.aff.?.reference(want, mag),
            .logits => {
                const aff = p.aff.?;
                for (0..p.shape.rows) |r| for (0..p.shape.n) |v| {
                    const o = aff.one(r, v);
                    want[r * p.shape.n + v] = 30.0 * std.math.tanh(o[0] / 30.0);
                    mag[r * p.shape.n + v] = o[1];
                };
            },
        }
    }

    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        if (p.kind == .dense) return p.aff.?.simulate(acc, order, out);
        const mag = try p.a.alloc(f64, out.len);
        defer p.a.free(mag);
        return p.reference(out, mag);
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    probes: std.ArrayList(*Probe) = .empty,

    pub fn deinit(s: *Suite) void {
        for (s.probes.items) |probe| {
            probe.deinit();
            s.a.destroy(probe);
        }
        s.probes.deinit(s.a);
    }

    fn add(s: *Suite, only: ?[]const u8, kind: Kind, control: ?controls.Mutation) !void {
        if (only) |needle| if (std.mem.indexOf(u8, label(kind, control), needle) == null) return;
        const probe = try s.a.create(Probe);
        errdefer s.a.destroy(probe);
        probe.* = try Probe.init(s.a, s.device, s.queue, kind, control);
        errdefer probe.deinit();
        try contract.register(s.a, &s.probes, s.kernels, probe, probe.kernel());
    }

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for ([_]Kind{ .embed_norm, .dense, .logits }) |kind| try s.add(only, kind, null);
        for (std.enums.values(controls.Mutation)) |m| {
            if (m == .none) continue;
            try s.add(only, m.kind(), m);
        }
    }
};
