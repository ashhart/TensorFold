//! Inkling 2's block attention residual fused with the norm that consumes it (Ops.attnRes), against a float64 formula
//! written from Inkling2BlockAttentionResidual + Inkling2RMSNorm (inkling2_math_norm.zig attnResRow): the stack of
//! finished blocks and the partial state, each RMS-normed without a weight, a learned query attending over them with
//! scale hidden^-0.5, the mix normed again with the query's weight.
//!
//! The kinds cover one query and three (a full-attention layer's q, k, v inputs), a bf16 and an fp32 partial state.
//! Blocks sit `block_stride` apart with a NaN gap after each (a stride past rows * hidden on odd row counts), the
//! partial rows `partial_ld` apart (hidden + 24 unless the row count divides by 3, where the encoder's default 0
//! applies), the queries' outputs `out_stride` apart with a gap; the gaps are poisoned NaN in the inputs and primed to
//! bf16 1.0 in the output (an element the kernel must not touch), every tensor sits 16 bytes or more into its buffer
//! between NaN words, and the weights views of queries 1 and 2 hold NaN when there is one query. Source scales vary
//! (1, 2^-10, 4 by source; 2^-9 more on every fourth row) so the 1e-6 of the unweighted norm matters.
//!
//! Gate 3 allowance: no reduction here can be stressed (a softmax and two norms follow every sum), so the kind declares
//! no fp32 sums and gives its own bound: first-order fp32 sensitivities of the whole chain, from fp32 error alone
//! (inkling2_math_norm.zig attnResRow, `err`), then the bf16 output's own rounding. A bf16 accumulator misses it.
//!
//! Non-finite rule: float64 arithmetic. A non-finite value in a source makes that row's every output NaN (the source's
//! 1 / rms is 0 or NaN, its normed value NaN, its score NaN, the softmax NaN); a NaN query element makes that query's
//! output NaN in every row; a NaN weight makes its column NaN; other rows stay finite.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const u = @import("inkling2_util.zig");
const math = @import("inkling2_math_norm.zig");
const rig_ = @import("rig.zig");
const c = u.c;
const g = u.g;
const ops = u.ops;

pub const Kind = enum { q1_f32, q1_bf16, q3_f32, q3_bf16, q1_split, q3_split };

const eps: f32 = 1e-6;
/// Elements before and after each tensor in its buffer (the view offset is lead elements in).
const lead = 8;
/// An output element the kernel must not write: bf16 1.0, finite so float64's value for it is comparable.
const sentinel: u16 = 0x3f80;

fn queryCount(k: Kind) usize {
    return switch (k) {
        .q1_f32, .q1_bf16, .q1_split => 1,
        else => 3,
    };
}
/// q1_split runs a bf16 partial and q3_split an fp32 one: both seams are covered.
fn partialF32(k: Kind) bool {
    return k == .q1_f32 or k == .q3_f32 or k == .q3_split;
}
fn split(k: Kind) bool {
    return k == .q1_split or k == .q3_split;
}

pub fn name(k: Kind) []const u8 {
    return switch (k) {
        .q1_f32 => "metal inkling2 attention residual, one query, fp32 partial",
        .q1_bf16 => "metal inkling2 attention residual, one query, bf16 partial",
        .q3_f32 => "metal inkling2 attention residual, three queries, fp32 partial",
        .q3_bf16 => "metal inkling2 attention residual, three queries, bf16 partial",
        .q1_split => "metal inkling2 attention residual, one query, rows split over two calls",
        .q3_split => "metal inkling2 attention residual, three queries, rows split over two calls",
    };
}

fn noSums(_: c.Shape) usize {
    return 0;
}

/// Scale of source `s` (blocks, then the partial state) on row `r`.
fn sourceScale(s: usize, r: usize) f32 {
    const base: f32 = switch (s % 3) {
        0 => 1,
        1 => 0x1p-10,
        else => 4,
    };
    return if (r % 4 == 3) base * 0x1p-9 else base;
}

fn padded(comptime T: type, a: std.mem.Allocator, data: []const T, fill: T) ![]T {
    const out = try a.alloc(T, lead + data.len + lead);
    @memset(out, fill);
    @memcpy(out[lead..][0..data.len], data);
    return out;
}

/// The tensor of `r` at element `first` of its data (past the leading words).
fn tensor(r: rig_.Ref, comptime esize: usize, first: usize) u.View {
    return .{ .buffer = r.buf, .offset = r.off + (lead + first) * esize };
}

/// Shapes served: rows, hidden, finished blocks (up to 16: 17 sources). A split needs two rows at least.
const served_shared = [_]c.Shape{
    .{ .rows = 17, .n = 2304, .k = 2 },
    .{ .rows = 128, .n = 2304, .k = 4 },
    .{ .rows = 6, .n = 128, .k = 4 },   .{ .rows = 3, .n = 4608, .k = 1 },  .{ .rows = 16, .n = 4608, .k = 4 },
    .{ .rows = 40, .n = 128, .k = 8 },  .{ .rows = 17, .n = 1000, .k = 4 }, .{ .rows = 33, .n = 128, .k = 1 },
    .{ .rows = 2, .n = 4608, .k = 8 },  .{ .rows = 5, .n = 256, .k = 16 },  .{ .rows = 4, .n = 4608, .k = 16 },
    .{ .rows = 31, .n = 128, .k = 0 },  .{ .rows = 32, .n = 128, .k = 4 },  .{ .rows = 2, .n = 6144, .k = 4 },
    .{ .rows = 15, .n = 1000, .k = 8 }, .{ .rows = 2, .n = 4608, .k = 0 },
};
const served_single = served_shared ++ [_]c.Shape{.{ .rows = 1, .n = 2304, .k = 4 }};

pub const Probe = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    rig: rig_.Rig,
    o: *const ops.Ops,
    kind: Kind,
    label: []const u8,
    /// rows, n = hidden, k = finished blocks.
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    blk: []u16 = &.{},
    part32: []f32 = &.{},
    part16: []u16 = &.{},
    queries: []f32 = &.{},
    norms: []u16 = &.{},
    err: []f64 = &.{},
    block_stride: usize = 0,
    ld: usize = 0,
    out_stride: usize = 0,
    r_blk: rig_.Ref = undefined,
    r_part: rig_.Ref = undefined,
    r_q: rig_.Ref = undefined,
    r_w: [ops.max_queries]rig_.Ref = undefined,
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
        const served: []const c.Shape = if (split(p.kind)) &served_shared else &served_single;
        return .{ .ctx = p, .vt = &vt, .spec = .{
            .name = p.label,
            .accum = .none,
            .depth = noSums,
            .out = .bf16,
            .replaces = if (split(p.kind)) "the same rows in one call" else null,
            .tiles = .{ .{ .size = 16 }, .{}, .{ .size = 4, .step = 1, .min = 0 } },
            .served = served,
        } };
    }

    fn ldEff(p: *const Probe) usize {
        return if (p.ld == 0) p.shape.n else p.ld;
    }

    fn outLen(p: *const Probe) usize {
        return (queryCount(p.kind) - 1) * p.out_stride + p.shape.rows * p.shape.n;
    }

    /// Source `s` (a block, or the partial state when s == nb) of row `r`, element `i`, set to a bf16 word.
    fn put(p: *Probe, r: usize, s: usize, i: usize, v: u16) void {
        if (r >= p.shape.rows) return;
        const H = p.shape.n;
        if (s < p.shape.k) {
            p.blk[s * p.block_stride + r * H + i] = v;
        } else if (partialF32(p.kind)) {
            p.part32[r * p.ldEff() + i] = c.fromBf16(v);
        } else {
            p.part16[r * p.ldEff() + i] = v;
        }
    }

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const rows = shape.rows;
        const H = shape.n;
        const nb = shape.k;
        if (rows == 0 or H == 0 or H > ops.max_hidden or nb + 1 > ops.max_sources or (split(p.kind) and rows < 2)) return false;
        const a = u.reset(&p.arena, &p.rig);
        p.shape = shape;
        const nq = queryCount(p.kind);
        p.block_stride = rows * H + (if (rows % 2 == 0) @as(usize, 0) else 40);
        p.ld = if (rows % 3 == 0) 0 else H + 24;
        p.out_stride = rows * H + (if (rows % 5 == 0) @as(usize, 0) else 56);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        p.blk = try a.alloc(u16, nb * p.block_stride);
        @memset(p.blk, u.nan_bits);
        for (0..nb) |s| for (0..rows) |r| for (0..H) |i| {
            p.blk[s * p.block_stride + r * H + i] = u.uniform(rng, sourceScale(s, r));
        };
        p.part32 = try a.alloc(f32, if (partialF32(p.kind)) rows * p.ldEff() else 0);
        p.part16 = try a.alloc(u16, if (partialF32(p.kind)) 0 else rows * p.ldEff());
        @memset(p.part32, std.math.nan(f32));
        @memset(p.part16, u.nan_bits);
        for (0..rows) |r| for (0..H) |i| {
            if (partialF32(p.kind)) {
                p.part32[r * p.ldEff() + i] = (rng.float(f32) * 2 - 1) * sourceScale(nb, r);
            } else {
                p.part16[r * p.ldEff() + i] = u.uniform(rng, sourceScale(nb, r));
            }
        };
        p.queries = try a.alloc(f32, nq * H);
        for (p.queries) |*v| v.* = (rng.float(f32) * 2 - 1) * 2;
        p.norms = try a.alloc(u16, nq * H);
        for (p.norms) |*v| v.* = c.toBf16(0.5 + rng.float(f32));
        if (fill == .nonfinite) {
            const last = if (nb > 0) nb - 1 else 0;
            p.put(0, 0, H / 3, u.nan_bits);
            p.put(1, nb, H / 2, u.pinf_bits);
            p.put(2, last, H - 1, u.ninf_bits);
            p.put(3, 0, 0, u.pinf_bits);
            p.put(3, nb, H - 1, u.ninf_bits);
            if (nq == 3 and H > 3) p.queries[H + 3] = std.math.nan(f32);
            if (H > 7) p.norms[(nq - 1) * H + 7] = u.nan_bits;
        }
        p.r_blk = try p.rig.input(std.mem.sliceAsBytes(try padded(u16, a, p.blk, u.nan_bits)));
        p.r_part = if (partialF32(p.kind))
            try p.rig.input(std.mem.sliceAsBytes(try padded(f32, a, p.part32, std.math.nan(f32))))
        else
            try p.rig.input(std.mem.sliceAsBytes(try padded(u16, a, p.part16, u.nan_bits)));
        p.r_q = try p.rig.input(std.mem.sliceAsBytes(try padded(f32, a, p.queries, std.math.nan(f32))));
        const nan_row = try a.alloc(u16, H);
        @memset(nan_row, u.nan_bits);
        for (&p.r_w, 0..) |*ref, q| {
            const words = if (q < nq) p.norms[q * H ..][0..H] else nan_row;
            ref.* = try p.rig.input(std.mem.sliceAsBytes(try padded(u16, a, words, u.nan_bits)));
        }
        p.out = try p.rig.output(p.outLen() * 2);
        p.alt = try p.rig.scratch(p.outLen() * 2);
        return true;
    }

    fn piece(p: *Probe, e: mtl.ComputeEncoder, dst: rig_.Ref, first: usize, n: usize) !void {
        const H = p.shape.n;
        var norms: [ops.max_queries]u.View = undefined;
        for (&norms, p.r_w) |*v, ref| v.* = tensor(ref, 2, 0);
        const partial = if (partialF32(p.kind)) tensor(p.r_part, 4, first * p.ldEff()) else tensor(p.r_part, 2, first * p.ldEff());
        try p.o.attnRes(e, .{
            .blocks = tensor(p.r_blk, 2, first * H),
            .block_stride = @intCast(p.block_stride),
            .nb = @intCast(p.shape.k),
            .partial = partial,
            .partial_f32 = partialF32(p.kind),
            .partial_ld = @intCast(p.ld),
            .queries = tensor(p.r_q, 4, 0),
            .nq = @intCast(queryCount(p.kind)),
            .norms = norms,
            .out = .{ .buffer = dst.buf, .offset = dst.off + first * H * 2 },
            .out_stride = @intCast(p.out_stride),
            .rows = @intCast(n),
            .hidden = @intCast(H),
            .eps = eps,
        });
    }

    /// Every real output element NaN, every gap element the finite sentinel.
    fn prime(p: *Probe, dst: rig_.Ref) void {
        const total = p.outLen();
        const bytes = p.rig.bytes(dst, total * 2);
        const real = p.shape.rows * p.shape.n;
        for (0..total) |i| {
            std.mem.writeInt(u16, bytes[2 * i ..][0..2], if (i % p.out_stride < real) 0xffff else sentinel, .little);
        }
    }

    fn compute(p: *Probe, dst: rig_.Ref, cut: bool) !void {
        p.prime(dst);
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
        return p.rig.bytes(p.alt, p.outLen() * 2);
    }

    pub fn count(p: *Probe) usize {
        return p.outLen();
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        const a = p.arena.allocator();
        const rows = p.shape.rows;
        const H = p.shape.n;
        const nb = p.shape.k;
        const nq = queryCount(p.kind);
        const q = try a.alloc(f64, nq * H);
        for (q, p.queries) |*o, v| o.* = v;
        const w = try u.decode(a, p.norms);
        const src = try a.alloc(f64, (nb + 1) * H);
        const row_out = try a.alloc(f64, nq * H);
        const row_err = try a.alloc(f64, nq * H);
        p.err = try a.alloc(f64, p.outLen());
        @memset(p.err, 0);
        @memset(want[0..p.outLen()], c.fromBf16(sentinel));
        @memset(mag[0..p.outLen()], 1);
        for (0..rows) |r| {
            for (0..nb) |s| for (0..H) |i| {
                src[s * H + i] = u.val(p.blk[s * p.block_stride + r * H + i]);
            };
            for (0..H) |i| {
                src[nb * H + i] = if (partialF32(p.kind)) p.part32[r * p.ldEff() + i] else u.val(p.part16[r * p.ldEff() + i]);
            }
            try math.attnResRow(p.a, .{ .hidden = H, .sources = nb + 1, .queries = nq, .eps_u = eps, .eps_n = eps }, src, q, w, row_out, row_err);
            for (0..nq) |qi| for (0..H) |i| {
                const at = qi * p.out_stride + r * H + i;
                want[at] = row_out[qi * H + i];
                mag[at] = @abs(want[at]);
                p.err[at] = row_err[qi * H + i];
            };
        }
    }

    /// fp32's own error from attnResRow (before the output's rounding), then bf16's half ulp of the result; gaps exact.
    pub fn bounds(p: *Probe, want: []const f64, _: []const f64, allow: []f64) !void {
        const real = p.shape.rows * p.shape.n;
        for (allow, want, p.err, 0..) |*l, w, e, i| {
            l.* = if (i % p.out_stride < real) e + 0x1p-8 * (@abs(w) + e) + 0x1p-133 else 0;
        }
    }
};

/// One planted fault in the attention residual source; its shapes gate must go red on it.
pub const Mutation = struct { kind: Kind, label: []const u8, from: []const u8, to: []const u8 };

pub const mutations = [_]Mutation{
    .{ .kind = .q1_f32, .label = "control: inkling2 attention residual, scores not scaled by hidden^-0.5", .from = "score[s] = (t * inv_tg[s]) * a.scale;", .to = "score[s] = (t * inv_tg[s]);" },
    .{ .kind = .q3_f32, .label = "control: inkling2 attention residual, every query normed with the first weight", .from = "const device bfloat* w = q == 0 ? w0 : (q == 1 ? w1 : w2);", .to = "const device bfloat* w = w0;" },
    .{ .kind = .q3_f32, .label = "control: inkling2 attention residual, queries' outputs written hidden apart", .from = "device bfloat* dst = out + ulong(q) * a.out_stride + ulong(row) * H;", .to = "device bfloat* dst = out + ulong(q) * H + ulong(row) * H;" },
    .{ .kind = .q1_bf16, .label = "control: inkling2 attention residual, bf16 partial rows read hidden apart", .from = "return float(((const device bfloat*)partial)[ulong(row) * a.partial_pitch + i]);", .to = "return float(((const device bfloat*)partial)[ulong(row) * a.hidden + i]);" },
    .{ .kind = .q1_f32, .label = "control: inkling2 attention residual, fp32 partial rows read hidden apart", .from = "return ((const device float*)partial)[ulong(row) * a.partial_pitch + i];", .to = "return ((const device float*)partial)[ulong(row) * a.hidden + i];" },
    .{ .kind = .q1_f32, .label = "control: inkling2 attention residual, blocks one row-width apart", .from = "blocks[ulong(s) * a.block_stride + ulong(row) * a.hidden + i]", .to = "blocks[ulong(s) * a.hidden + ulong(row) * a.hidden + i]" },
    .{ .kind = .q1_f32, .label = "control: inkling2 attention residual, eps left out of the sources' norm", .from = "inv_tg[tid] = i2_rsqrt_mean(t, H, a.eps_unweighted);", .to = "inv_tg[tid] = i2_rsqrt_mean(t, H, 0.0f);" },
};

pub fn controlSource(a: std.mem.Allocator, m: Mutation) ![]u8 {
    return u.mutate(a, sources.inkling2_ops, m.from, m.to);
}

pub fn controlKernel(p: *Probe) g.Kernel {
    var k = p.kernel();
    k.control = .shapes;
    return k;
}
