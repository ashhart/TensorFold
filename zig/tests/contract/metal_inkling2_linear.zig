//! Inkling 2's dense projections through `ops.linear` (the shared bf16 matvec for fewer than 16 rows, its bit-equal MMA
//! tile from 16 rows), at the shapes the backbone and the tiny checkpoint call it, against float64 dot products.
//!
//! The bound is the fp32 chain's: gamma(depth) of the terms' magnitude (depth = a lane's FMAs, the butterfly's adds and
//! the output's rounding) and the output's rounding; no oracle of the kernel's order is replayed. Gate 4 plants integer
//! weights and rows whose partial sums reach 2^20 and cancel (the second half of each row mirrors the first), so fp32
//! sums are exact in any order and fp16 / bf16 sums miss by far more than 100 times the exact bound.
//!
//! Row independence, the seam a prompt chunk, a verify window and a decode step cross: the `alone` kinds run all rows
//! in one call and must equal every row as its own one-row call (the matvec tier) byte for byte; the `split` kind cuts
//! the call at 1, 3, 16, 17 and 32 rows through views at non-zero offsets and must equal the whole call. Their gate 1
//! runs 16 rows more than the harness asks (a full MMA tile and a partial one), so the tile meets the non-finite
//! plants too. The x input has NaN rows around the call's rows. NaN and both infinities are planted in x rows and in
//! three weight rows: float64 and the kernel agree on every class whatever the order (the terms' set decides it).
//!
//! The planted controls are the matvec's and tile's own (their kernels carry `TF_CTL` edits, as the dsv41 probes use):
//! the probe swaps them into a private copy of the Ops, so they run through the same `ops.linear`.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const contract = @import("contract");
const u = @import("inkling2_util.zig");
const rig_ = @import("rig.zig");
const c = u.c;
const g = u.g;
const ops = u.ops;
const mv = tf.dsv41_metal.mv;
const mv_mma = tf.dsv41_metal.mv_mma;
const reference_threads = contract.reference_threads;

pub const Kind = enum {
    bf16,
    f32,
    alone_bf16,
    alone_f32,
    split_bf16,
    head,
    /// Planted controls: a plain bf16 probe over a matvec and tile with one source fault (see `controls`).
    ctl_tail,
    ctl_half,
    ctl_flush,
};

/// The kinds the suite registers on the production operators.
pub const kinds = [_]Kind{ .bf16, .f32, .alone_bf16, .alone_f32, .split_bf16, .head };

fn f32Out(k: Kind) bool {
    return k == .f32 or k == .alone_f32;
}
/// Kinds whose gate 1 runs 16 more rows than the harness asks: an MMA tile and a partial one.
fn bumped(k: Kind) bool {
    return k == .alone_bf16 or k == .alone_f32 or k == .split_bf16;
}

pub fn name(k: Kind) []const u8 {
    return switch (k) {
        .bf16, .ctl_tail, .ctl_half, .ctl_flush => "metal inkling2 linear",
        .f32 => "metal inkling2 linear, f32 out",
        .alone_bf16 => "metal inkling2 linear, each row its own one-row call",
        .alone_f32 => "metal inkling2 linear, f32 out, each row its own one-row call",
        .split_bf16 => "metal inkling2 linear, rows cut over views at offsets",
        .head => "metal inkling2 linear, lm_head (n 200058)",
    };
}

/// Longest chain of roundings: a lane's FMAs, the butterfly's five adds and the output's own.
fn depth(s: c.Shape) usize {
    return (s.k + 31) / 32 + 5 + 1;
}

const max_k = 8192;
const pad_rows = 2;

const sweep_rows = [_]usize{ 1, 2, 3, 5, 9, 15, 16, 17, 31, 32, 33, 40 };
fn sweep() [sweep_rows.len]c.Shape {
    var out: [sweep_rows.len]c.Shape = undefined;
    for (&out, sweep_rows) |*s, r| s.* = .{ .rows = r, .n = 96, .k = 4608 };
    return out;
}

// Rows x outputs x K. Hidden 4608 (6144 into the output projection), heads 48/8 x 128, r 16 x 48, the tiny
// checkpoint's hidden 128, and odd edges. The first shape with the largest K (6144) is gate 4's.
const plain_served = [_]c.Shape{
    .{ .rows = 1, .n = 3072, .k = 2304 },
    .{ .rows = 16, .n = 2304, .k = 3072 },
    .{ .rows = 17, .n = 1024, .k = 2304 },
    .{ .rows = 16, .n = 384, .k = 2304 },
    .{ .rows = 1, .n = 32, .k = 2304 },
    .{ .rows = 8, .n = 1024, .k = 4608 },
    .{ .rows = 1, .n = 1024, .k = 4608 },
    .{ .rows = 2, .n = 6144, .k = 4608 },
    .{ .rows = 4, .n = 768, .k = 6144 },
    .{ .rows = 15, .n = 768, .k = 4608 },
    .{ .rows = 16, .n = 48, .k = 4608 },
    .{ .rows = 17, .n = 4608, .k = 6144 },
    .{ .rows = 31, .n = 1024, .k = 4608 },
    .{ .rows = 32, .n = 6144, .k = 4608 },
    .{ .rows = 33, .n = 768, .k = 4608 },
    .{ .rows = 40, .n = 1024, .k = 4608 },
    .{ .rows = 40, .n = 48, .k = 6144 },
    .{ .rows = 1, .n = 128, .k = 128 },
    .{ .rows = 9, .n = 64, .k = 128 },
    .{ .rows = 17, .n = 16, .k = 128 },
    .{ .rows = 40, .n = 128, .k = 128 },
    .{ .rows = 5, .n = 33, .k = 97 },
    .{ .rows = 17, .n = 33, .k = 289 },
    .{ .rows = 3, .n = 7, .k = 31 },
} ++ sweep();

// The seam kinds compare bytes, so their shapes run no float64: the same envelope with smaller outputs.
const seam_served = [_]c.Shape{
    .{ .rows = 17, .n = 3072, .k = 2304 },
    .{ .rows = 16, .n = 2304, .k = 3072 },
    .{ .rows = 8, .n = 1024, .k = 4608 },
    .{ .rows = 1, .n = 1024, .k = 4608 },
    .{ .rows = 40, .n = 96, .k = 6144 },
    .{ .rows = 16, .n = 48, .k = 4608 },
    .{ .rows = 17, .n = 1024, .k = 4608 },
    .{ .rows = 15, .n = 768, .k = 4608 },
    .{ .rows = 31, .n = 1024, .k = 4608 },
    .{ .rows = 32, .n = 48, .k = 4608 },
    .{ .rows = 33, .n = 768, .k = 4608 },
    .{ .rows = 40, .n = 1024, .k = 4608 },
    .{ .rows = 17, .n = 33, .k = 289 },
    .{ .rows = 5, .n = 33, .k = 97 },
    .{ .rows = 16, .n = 16, .k = 128 },
    .{ .rows = 33, .n = 128, .k = 128 },
} ++ sweep();

// The lm_head's vocabulary width (200058 outputs: grid dimensions past 65535 groups) at K 1024 so the float64
// copy of the weights stays at 1.6 GB; K 4608 and 6144 are served by the plain kinds.
const head_served = [_]c.Shape{
    .{ .rows = 5, .n = 200058, .k = 1024 },
    .{ .rows = 1, .n = 200058, .k = 1024 },
    .{ .rows = 2, .n = 200058, .k = 1024 },
    .{ .rows = 17, .n = 200058, .k = 1024 },
};

fn servedOf(k: Kind) []const c.Shape {
    return switch (k) {
        .alone_bf16, .alone_f32, .split_bf16 => &seam_served,
        .head => &head_served,
        else => &plain_served,
    };
}

/// bf16 words with random sign and mantissa and an exponent in [lo, lo + span): never zero, never non-finite.
fn fillBf16(rng: std.Random, out: []u16, lo: u16, span: u16) void {
    var i: usize = 0;
    while (i < out.len) {
        var r = rng.int(u64);
        var l: usize = 0;
        while (l < 4 and i < out.len) : (l += 1) {
            const w: u16 = @truncate(r);
            r >>= 16;
            const e: u16 = lo + ((w >> 7) & 0xff) % span;
            out[i] = (w & 0x807f) | (e << 7);
            i += 1;
        }
    }
}

const RefJob = struct {
    x: []const f64,
    w: []const u16,
    rows: usize,
    n: usize,
    k: usize,
    j0: usize,
    j1: usize,
    want: []f64,
    mag: []f64,

    pub fn run(job: *RefJob) void {
        var col: [max_k]f64 = undefined;
        const k = job.k;
        for (job.j0..job.j1) |j| {
            for (0..k) |i| col[i] = u.val(job.w[j * k + i]);
            for (0..job.rows) |r| {
                const xr = job.x[r * k ..][0..k];
                var s: [4]f64 = @splat(0);
                var m: [4]f64 = @splat(0);
                var i: usize = 0;
                while (i + 4 <= k) : (i += 4) {
                    inline for (0..4) |l| {
                        const t = xr[i + l] * col[i + l];
                        s[l] += t;
                        m[l] += @abs(t);
                    }
                }
                while (i < k) : (i += 1) {
                    const t = xr[i] * col[i];
                    s[0] += t;
                    m[0] += @abs(t);
                }
                job.want[r * job.n + j] = (s[0] + s[1]) + (s[2] + s[3]);
                job.mag[r * job.n + j] = (m[0] + m[1]) + (m[2] + m[3]);
            }
        }
    }
};

const How = enum { whole, alone, cuts };

pub const Probe = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    rig: rig_.Rig,
    device: mtl.Device,
    o: *const ops.Ops,
    /// The Ops of a control: the production one with the matvec and tile swapped for faulted ones.
    owned: ?ops.Ops = null,
    kind: Kind,
    label: []const u8,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    /// Rows of the call: the shape's, plus 16 for a bumped kind's non-finite gate.
    rows: usize = 0,
    xs: []u16 = &.{},
    wts: []u16 = &.{},
    x: rig_.Ref = undefined,
    w: rig_.Ref = undefined,
    y: rig_.Ref = undefined,
    alt: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, o: *const ops.Ops, kind: Kind, label: []const u8) Probe {
        return .{ .a = a, .arena = .init(a), .rig = .{ .a = a, .device = device, .queue = queue }, .device = device, .o = o, .kind = kind, .label = label };
    }

    pub fn deinit(p: *Probe) void {
        if (p.owned) |*o| {
            o.dense.deinit();
            o.mma.deinit();
        }
        p.arena.deinit();
        p.rig.deinit();
    }

    const vt = g.hooks(Probe);

    pub fn kernel(p: *Probe) g.Kernel {
        const wide = p.kind == .head;
        return .{ .ctx = p, .vt = &vt, .spec = .{
            .name = p.label,
            // The head's gate 4 would replay 4 x 200058 chains nine times; the plain kinds prove the same kernels'
            // sums at K 6144, so the head declares none and `bounds` gives gate 3 the fp32 chain's allowance itself.
            .accum = if (wide) .none else .f32,
            .depth = depth,
            .out = if (f32Out(p.kind)) .f32 else .bf16,
            .replaces = switch (p.kind) {
                .alone_bf16, .alone_f32 => "each row as its own one-row call",
                .split_bf16 => "the same rows in one call",
                else => null,
            },
            .served = servedOf(p.kind),
            .tiles = if (wide) .{ .{}, .{}, .{} } else .{ .{ .size = 16 }, .{ .size = 16 }, .{ .size = 32 } },
        } };
    }

    fn opsOf(p: *const Probe) *const ops.Ops {
        return if (p.owned) |*o| o else p.o;
    }

    /// A control's matvec and tile: the dsv41 sources with their own TF_CTL fault.
    fn buildControl(p: *Probe) !void {
        if (p.owned != null) return;
        const ctl: u8 = switch (p.kind) {
            .ctl_tail => 2,
            .ctl_half => 3,
            .ctl_flush => 1,
            else => return,
        };
        const dense_text = try std.fmt.allocPrint(p.a, "#define TF_CTL {d}\n{s}", .{ ctl, sources.dsv41.mv });
        defer p.a.free(dense_text);
        const mma_text = try std.fmt.allocPrint(p.a, "#define TF_CTL {d}\n{s}", .{ ctl, sources.dsv41.mv_mma });
        defer p.a.free(mma_text);
        var o = p.o.*;
        o.dense = try mv.Pipelines.loadSource(p.device, dense_text);
        errdefer o.dense.deinit();
        o.mma = try mv_mma.Pipelines.loadSource(p.device, mma_text);
        p.owned = o;
    }

    fn es(p: *const Probe) usize {
        return if (f32Out(p.kind)) 4 else 2;
    }

    pub fn count(p: *const Probe) usize {
        return p.rows * p.shape.n;
    }

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k < 2 or shape.k > max_k) return false;
        try p.buildControl();
        const a = u.reset(&p.arena, &p.rig);
        p.shape = shape;
        p.rows = shape.rows + (if (fill == .nonfinite and bumped(p.kind)) @as(usize, 16) else 0);
        const rows = p.rows;
        const k = shape.k;
        const n = shape.n;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const xfull = try a.alloc(u16, (rows + 2 * pad_rows) * k);
        @memset(xfull, u.nan_bits);
        p.xs = xfull[pad_rows * k ..][0 .. rows * k];
        p.wts = try a.alloc(u16, n * k);
        fillBf16(rng, p.xs, 125, 5); // |x| in [1/4, 8)
        fillBf16(rng, p.wts, 120, 7); // |w| in [2^-7, 1)
        if (fill == .stress) {
            // Weights 1..8 and their negations, rows 1..255 mirrored (the second half one larger in the first 8):
            // every partial sum is an integer below 2^24 and the result is minus the first 8 weights.
            const h = k / 2;
            const xmax: u32 = @min(255, ((1 << 24) - 1) / (8 * @as(u32, @intCast(k))));
            for (0..n) |j| {
                for (0..h) |i| {
                    const q: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 8));
                    p.wts[j * k + i] = c.toBf16(q);
                    p.wts[j * k + h + i] = c.toBf16(-q);
                }
                if (k % 2 == 1) p.wts[j * k + k - 1] = 0;
            }
            for (0..rows) |r| {
                const xr = p.xs[r * k ..][0..k];
                for (0..h) |i| {
                    const v = 1 + rng.uintLessThan(u32, xmax);
                    xr[i] = c.toBf16(@floatFromInt(v));
                    xr[h + i] = c.toBf16(@floatFromInt(v + @intFromBool(i < 8)));
                }
                if (k % 2 == 1) xr[k - 1] = 0;
            }
        }
        if (fill == .nonfinite) {
            p.xs[k / 3] = u.nan_bits;
            if (rows > 1) p.xs[k + k / 2] = u.pinf_bits;
            if (rows > 2) p.xs[2 * k + k - 1] = u.ninf_bits;
            if (rows > 3) {
                p.xs[3 * k] = u.pinf_bits;
                p.xs[3 * k + k - 1] = u.ninf_bits;
            }
            if (n > 3) {
                p.wts[k + k / 5] = u.nan_bits;
                p.wts[2 * k + k / 7] = u.pinf_bits;
                p.wts[3 * k + k / 3 + 1] = u.ninf_bits;
            }
        }
        p.x = try p.rig.input(std.mem.sliceAsBytes(xfull));
        p.w = try p.rig.input(std.mem.sliceAsBytes(p.wts));
        p.y = try p.rig.output(p.count() * p.es());
        p.alt = try p.rig.scratch(p.count() * p.es());
        return true;
    }

    /// Rows r0..r0+rows of the prepared call as one `ops.linear`, over views at their offsets.
    fn piece(p: *Probe, e: mtl.ComputeEncoder, r0: usize, rows: usize, dst: rig_.Ref) !void {
        const k = p.shape.k;
        const n = p.shape.n;
        try p.opsOf().linear(e, .{
            .x = .{ .buffer = p.x.buf, .offset = p.x.off + (pad_rows + r0) * k * 2 },
            .w = u.view(p.w),
            .y = .{ .buffer = dst.buf, .offset = dst.off + r0 * n * p.es() },
            .rows = @intCast(rows),
            .n = @intCast(n),
            .k = @intCast(k),
            .f32_out = f32Out(p.kind),
        });
    }

    fn compute(p: *Probe, dst: rig_.Ref, how: How) !void {
        const pass = p.rig.begin();
        switch (how) {
            .whole => try p.piece(pass.e, 0, p.rows, dst),
            .alone => {
                for (0..p.rows) |r| try p.piece(pass.e, r, 1, dst);
            },
            .cuts => {
                var r0: usize = 0;
                for ([_]usize{ 1, 3, 16, 17, 32 }) |cut| {
                    if (cut <= r0 or cut >= p.rows) continue;
                    try p.piece(pass.e, r0, cut - r0, dst);
                    r0 = cut;
                }
                try p.piece(pass.e, r0, p.rows - r0, dst);
            },
        }
        try p.rig.finish(pass);
    }

    pub fn run(p: *Probe) !void {
        try p.compute(p.y, if (p.kind == .split_bf16) .cuts else .whole);
    }

    pub fn replaced(p: *Probe) ![]const u8 {
        try p.compute(p.alt, if (p.kind == .split_bf16) .whole else .alone);
        return p.rig.bytes(p.alt, p.count() * p.es());
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        const x = try u.decode(p.arena.allocator(), p.xs);
        const n = p.shape.n;
        const workers = reference_threads.count(reference_threads.maximum, n);
        var jobs: [reference_threads.maximum]RefJob = undefined;
        for (jobs[0..workers], 0..) |*job, i| job.* = .{
            .x = x,
            .w = p.wts,
            .rows = p.rows,
            .n = n,
            .k = p.shape.k,
            .j0 = i * n / workers,
            .j1 = (i + 1) * n / workers,
            .want = want,
            .mag = mag,
        };
        reference_threads.run(RefJob, jobs[0..workers], .normal);
    }

    /// The head's allowance, derived here as it declares no accumulator: gamma(depth, 2^-24) of the terms' magnitude
    /// (the fp32 chain), then the bf16 output's rounding of the value and of that error. The other kinds keep the
    /// generic bound, which is the same formula.
    pub fn bounds(p: *Probe, want: []const f64, mag: []const f64, allow: []f64) !void {
        if (p.kind != .head) return;
        const gm = c.gamma(@floatFromInt(depth(p.shape)), 0x1p-24);
        for (allow, want, mag) |*l, w, m| {
            const work = gm * m;
            l.* = work + 0x1p-8 * (@abs(w) + work) + 0x1p-133;
        }
    }

    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = p.arena.allocator();
        const k = p.shape.k;
        const x = try u.decode(a, p.xs);
        const t = try a.alloc(f64, k);
        for (0..p.rows) |r| for (0..p.shape.n) |j| {
            for (0..k) |i| t[i] = x[r * k + i] * u.val(p.wts[j * k + i]);
            out[r * p.shape.n + j] = contract.rows.sumIn(acc, order, t);
        };
    }
};

/// A planted fault in the matvec and the tile (their sources' TF_CTL edits); `controlKernel` names the gate.
pub const Control = struct { kind: Kind, label: []const u8 };

pub const controls = [_]Control{
    .{ .kind = .ctl_tail, .label = "control: inkling2 linear, a partial last K step dropped" },
    .{ .kind = .ctl_half, .label = "control: inkling2 linear, sums rounded to fp16" },
    .{ .kind = .ctl_flush, .label = "control: inkling2 linear, non-finite outputs flushed" },
};

pub fn controlKernel(p: *Probe) g.Kernel {
    var k = p.kernel();
    k.control = switch (p.kind) {
        .ctl_tail => .shapes,
        .ctl_half => .sums,
        .ctl_flush => .nonfinite,
        else => unreachable,
    };
    return k;
}
