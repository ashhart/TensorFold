//! Inkling 2 elementwise glue via the graph's operators (softcap, residualAttn, residualMlp), against float64 formulas.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const u = @import("inkling2_util.zig");
const math = @import("inkling2_math_norm.zig");
const rig_ = @import("rig.zig");
const c = u.c;
const g = u.g;
const ops = u.ops;

pub const Kind = enum { softcap_pitched, softcap_tight, softcap_split, attn_block, attn_noblock, attn_split, mlp, mlp_split };

const cap: f32 = 30;
/// Words before and after each input tensor in its buffer (a view offset of lead elements).
const lead = 8;

fn isSoftcap(k: Kind) bool {
    return k == .softcap_pitched or k == .softcap_tight or k == .softcap_split;
}
fn isMlp(k: Kind) bool {
    return k == .mlp or k == .mlp_split;
}
fn hasBlock(k: Kind) bool {
    return k == .attn_block or k == .attn_split;
}
fn pitched(k: Kind) bool {
    return k == .softcap_pitched or k == .softcap_split;
}
fn split(k: Kind) bool {
    return k == .softcap_split or k == .attn_split or k == .mlp_split;
}

pub fn name(k: Kind) []const u8 {
    return switch (k) {
        .softcap_pitched => "metal inkling2 softcap, ldy past n",
        .softcap_tight => "metal inkling2 softcap, ldy equal to n",
        .softcap_split => "metal inkling2 softcap, ldy past n, rows split over two calls",
        .attn_block => "metal inkling2 residual attn, with a block",
        .attn_noblock => "metal inkling2 residual attn, without a block",
        .attn_split => "metal inkling2 residual attn, with a block, rows split over two calls",
        .mlp => "metal inkling2 residual mlp",
        .mlp_split => "metal inkling2 residual mlp, rows split over two calls",
    };
}

fn noSums(_: c.Shape) usize {
    return 0;
}

// rows, columns, extra softcap pitch. A split needs two rows at least; the one-row shapes are for whole calls.
const plain_shared = [_]c.Shape{
    .{ .rows = 6, .n = 128, .k = 0 },  .{ .rows = 3, .n = 4608, .k = 0 },  .{ .rows = 17, .n = 1000, .k = 0 },
    .{ .rows = 40, .n = 256, .k = 0 }, .{ .rows = 33, .n = 4608, .k = 0 }, .{ .rows = 31, .n = 257, .k = 0 },
    .{ .rows = 16, .n = 255, .k = 0 }, .{ .rows = 2, .n = 32768, .k = 0 },
};
const plain_single = plain_shared ++ [_]c.Shape{.{ .rows = 1, .n = 4608, .k = 0 }};
const pitch_shared = [_]c.Shape{
    .{ .rows = 6, .n = 128, .k = 3 },  .{ .rows = 3, .n = 4608, .k = 8 },  .{ .rows = 17, .n = 1000, .k = 24 },
    .{ .rows = 40, .n = 256, .k = 1 }, .{ .rows = 33, .n = 4608, .k = 7 }, .{ .rows = 2, .n = 100000, .k = 9 },
    .{ .rows = 32, .n = 255, .k = 5 },
};
const pitch_single = pitch_shared ++ [_]c.Shape{.{ .rows = 1, .n = 4608, .k = 16 }};

/// A tensor's view `first` elements past its leading words.
fn tensor(r: rig_.Ref, comptime esize: usize, first: usize) u.View {
    return .{ .buffer = r.buf, .offset = r.off + (lead + first) * esize };
}

fn padded(comptime T: type, a: std.mem.Allocator, data: []const T, fill: T) ![]T {
    const out = try a.alloc(T, lead + data.len + lead);
    @memset(out, fill);
    @memcpy(out[lead..][0..data.len], data);
    return out;
}

pub const Probe = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    rig: rig_.Rig,
    o: *const ops.Ops,
    kind: Kind,
    label: []const u8,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    /// The softcap's logits (rows `ldy` apart) or the residuals' conv output, bf16 words.
    ins: []u16 = &.{},
    /// The attention residual's block, bf16 words.
    block: []u16 = &.{},
    /// The layer output's fp32 partial state.
    h: []f32 = &.{},
    r_a: rig_.Ref = undefined,
    r_b: rig_.Ref = undefined,
    r_h: rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    alt: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, o: *const ops.Ops, kind: Kind, label: []const u8) Probe {
        return .{ .a = a, .arena = .init(a), .rig = .{ .a = a, .device = device, .queue = queue }, .o = o, .kind = kind, .label = label };
    }

    pub fn deinit(p: *Probe) void {
        p.arena.deinit();
        p.rig.deinit();
    }

    const vt = g.hooks(Probe);

    pub fn kernel(p: *Probe) g.Kernel {
        const sp = split(p.kind);
        const served: []const c.Shape = if (pitched(p.kind))
            (if (sp) &pitch_shared else &pitch_single)
        else
            (if (sp) &plain_shared else &plain_single);
        return .{
            .ctx = p,
            .vt = &vt,
            .spec = .{
                .name = p.label,
                .accum = .none,
                .depth = noSums,
                // tanh and its two roundings, in fp32 (precise::tanh is a few ulps); an fp32 add rounds once.
                .ops = if (isSoftcap(p.kind)) 12 else 1,
                .out = if (isMlp(p.kind)) .bf16 else .f32,
                .replaces = if (sp) "the same rows in one call" else null,
                .tiles = .{ .{ .size = 16 }, .{ .size = 256, .step = 1, .min = 1 }, .{} },
                .served = served,
            },
        };
    }

    fn ld(p: *const Probe) usize {
        return if (pitched(p.kind)) p.shape.n + p.shape.k else p.shape.n;
    }

    fn outBytes(p: *const Probe) usize {
        return p.shape.rows * p.shape.n * (if (isMlp(p.kind)) @as(usize, 2) else 4);
    }

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const rows = shape.rows;
        const cols = shape.n;
        if (rows == 0 or cols == 0 or (split(p.kind) and rows < 2)) return false;
        const a = u.reset(&p.arena, &p.rig);
        p.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const stride = p.ld();
        p.ins = try a.alloc(u16, rows * stride);
        @memset(p.ins, u.nan_bits);
        for (0..rows) |r| {
            // Logits spread over the cap's linear zone, its knee and far into saturation; residuals over +-4.
            const scale: f32 = if (isSoftcap(p.kind)) (switch (r % 3) {
                0 => 120,
                1 => 600,
                else => 3,
            }) else 4;
            for (0..cols) |j| p.ins[r * stride + j] = u.uniform(rng, scale);
        }
        p.block = try a.alloc(u16, if (hasBlock(p.kind)) rows * cols else 0);
        for (p.block) |*v| v.* = u.uniform(rng, 4);
        p.h = try a.alloc(f32, if (isMlp(p.kind)) rows * cols else 0);
        for (p.h, 0..) |*v, i| {
            v.* = (rng.float(f32) * 2 - 1) * 4;
            if (i % 5 == 0) {
                // Cancelling with the conv: h + conv is a few 2^-18, exact in fp32.
                const k: f32 = @floatFromInt(rng.intRangeAtMost(i32, -64, 64));
                v.* = -c.fromBf16(p.ins[(i / cols) * stride + i % cols]) + k * 0x1p-18;
            }
        }
        if (fill == .nonfinite) p.plant(rows, cols, stride);
        p.r_a = try p.rig.input(std.mem.sliceAsBytes(try padded(u16, a, p.ins, u.nan_bits)));
        if (hasBlock(p.kind)) p.r_b = try p.rig.input(std.mem.sliceAsBytes(try padded(u16, a, p.block, u.nan_bits)));
        if (isMlp(p.kind)) p.r_h = try p.rig.input(std.mem.sliceAsBytes(try padded(f32, a, p.h, std.math.nan(f32))));
        p.out = try p.rig.output(p.outBytes());
        p.alt = try p.rig.scratch(p.outBytes());
        return true;
    }

    /// NaN, +inf, -inf each in a row of its own and both infinities in row 3 (the block and h take their own).
    fn plant(p: *Probe, rows: usize, cols: usize, stride: usize) void {
        const at = [_][2]usize{ .{ 0, cols / 3 }, .{ 1, cols / 2 }, .{ 2, cols - 1 }, .{ 3, 0 }, .{ 3, cols - 1 } };
        const put = [_]u16{ u.nan_bits, u.pinf_bits, u.ninf_bits, u.pinf_bits, u.ninf_bits };
        for (at, put) |q, v| if (q[0] < rows) {
            p.ins[q[0] * stride + q[1]] = v;
        };
        if (hasBlock(p.kind)) {
            const extra = [_][3]usize{ .{ 3, 0, u.ninf_bits }, .{ 4, 1, u.pinf_bits }, .{ 5, 2, u.ninf_bits } };
            for (extra) |q| if (q[0] < rows) {
                p.block[q[0] * cols + q[1]] = @intCast(q[2]);
            };
        }
        if (isMlp(p.kind)) {
            const extra = [_][3]usize{ .{ 3, 0, u.ninf_bits }, .{ 4, 1, u.pinf_bits }, .{ 5, 2, u.nan_bits } };
            for (extra) |q| if (q[0] < rows) {
                p.h[q[0] * cols + q[1]] = c.fromBf16(@intCast(q[2]));
            };
        }
    }

    fn piece(p: *Probe, e: mtl.ComputeEncoder, dst: rig_.Ref, first: usize, n: usize) !void {
        const cols: u32 = @intCast(p.shape.n);
        const rows: u32 = @intCast(n);
        const esize: usize = if (isMlp(p.kind)) 2 else 4;
        const ov = u.View{ .buffer = dst.buf, .offset = dst.off + first * cols * esize };
        switch (p.kind) {
            .softcap_pitched, .softcap_tight, .softcap_split => try p.o.softcap(e, .{ .y = tensor(p.r_a, 2, first * p.ld()), .out = ov, .rows = rows, .n = cols, .ldy = @intCast(p.ld()), .cap = cap }),
            .attn_block, .attn_noblock, .attn_split => try p.o.residualAttn(e, .{
                .conv = tensor(p.r_a, 2, first * cols),
                .block = if (hasBlock(p.kind)) tensor(p.r_b, 2, first * cols) else null,
                .out = ov,
                .rows = rows,
                .cols = cols,
            }),
            .mlp, .mlp_split => try p.o.residualMlp(e, .{ .h = tensor(p.r_h, 4, first * cols), .conv = tensor(p.r_a, 2, first * cols), .out = ov, .rows = rows, .cols = cols }),
        }
    }

    fn compute(p: *Probe, dst: rig_.Ref, cut: bool) !void {
        const rows = p.shape.rows;
        const pass = p.rig.begin();
        const m = if (cut) rows / 2 else rows;
        try p.piece(pass.e, dst, 0, m);
        if (cut) try p.piece(pass.e, dst, m, rows - m);
        try p.rig.finish(pass);
    }

    pub fn run(p: *Probe) !void {
        try p.compute(p.out, split(p.kind));
    }

    pub fn replaced(p: *Probe) ![]const u8 {
        try p.compute(p.alt, false);
        return p.rig.bytes(p.alt, p.outBytes());
    }

    pub fn count(p: *Probe) usize {
        return p.shape.rows * p.shape.n;
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        const rows = p.shape.rows;
        const cols = p.shape.n;
        const stride = p.ld();
        for (0..rows) |r| for (0..cols) |j| {
            const conv = u.val(p.ins[r * stride + j]);
            const at = r * cols + j;
            want[at] = switch (p.kind) {
                .softcap_pitched, .softcap_tight, .softcap_split => math.softcap(conv, cap),
                .attn_block, .attn_split => u.val(p.block[at]) + conv,
                .attn_noblock => conv,
                .mlp, .mlp_split => @as(f64, p.h[at]) + conv,
            };
            mag[at] = @abs(want[at]);
        };
    }

    /// Softcap of an infinity is exactly +-cap; elsewhere a floor of 8 * 2^-24 * cap absorbs fp32 tanh's near-0 error.
    pub fn bounds(p: *Probe, _: []const f64, _: []const f64, allow: []f64) !void {
        if (!isSoftcap(p.kind)) return;
        const cols = p.shape.n;
        for (allow, 0..) |*l, i| {
            if (std.math.isInf(u.val(p.ins[(i / cols) * p.ld() + i % cols]))) {
                l.* = 0;
            } else {
                l.* = @max(l.*, 8 * 0x1p-24 * cap);
            }
        }
    }
};

/// One planted fault in the glue sources; its shapes gate must go red on it.
pub const Mutation = struct { kind: Kind, label: []const u8, from: []const u8, to: []const u8 };

pub const mutations = [_]Mutation{
    .{ .kind = .softcap_pitched, .label = "control: inkling2 softcap, rows read at the output's pitch", .from = "float(y[ulong(at.y) * a.in_ld + at.x])", .to = "float(y[ulong(at.y) * a.out_ld + at.x])" },
    .{ .kind = .softcap_tight, .label = "control: inkling2 softcap, the cap not multiplied back", .from = "/ a.cap) * a.cap;", .to = "/ a.cap);" },
    .{ .kind = .softcap_split, .label = "control: inkling2 softcap, the last row of each call skipped", .from = "if (at.x >= a.cols || at.y >= a.rows) return;\n  out[ulong(at.y) * a.out_ld + at.x]", .to = "if (at.x >= a.cols || at.y + 1 >= a.rows) return;\n  out[ulong(at.y) * a.out_ld + at.x]" },
    .{ .kind = .attn_block, .label = "control: inkling2 residual attn, the block not added", .from = "out[i] = a.has_block != 0 ? float(block[i]) + c : c;", .to = "out[i] = c;" },
    .{ .kind = .attn_noblock, .label = "control: inkling2 residual attn, a block added when there is none", .from = "out[i] = a.has_block != 0 ? float(block[i]) + c : c;", .to = "out[i] = float(block[i]) + c;" },
    .{ .kind = .mlp, .label = "control: inkling2 residual mlp, h rounded to bf16 before the add", .from = "out[i] = bfloat(h[i] + float(conv[i]));", .to = "out[i] = bfloat(float(bfloat(h[i])) + float(conv[i]));" },
};

pub fn controlSource(a: std.mem.Allocator, m: Mutation) ![]u8 {
    return u.mutate(a, sources.inkling2_ops, m.from, m.to);
}

pub fn controlKernel(p: *Probe) g.Kernel {
    var k = p.kernel();
    k.control = .shapes;
    return k;
}
