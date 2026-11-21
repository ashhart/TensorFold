//! Inkling 2's short convolutions through the operators the model graph calls (convStatic / convDynamic then
//! convStore), against float64 formulas over the whole position axis.
//!
//! The history ring is poisoned with NaN everywhere the convolution must not read (positions before 0, positions more
//! than three back), so a kernel that reads a negative position or a stale slot goes NaN where float64 is finite. The
//! outputs end with the ring rows the store wrote, read back from the device and held exact. The `split` kinds run
//! the same window as two calls with a store between them and must equal the one call byte for byte: the seam a
//! drafted window, a prompt chunk and a decode step all cross.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const u = @import("inkling2_util.zig");
const math = @import("inkling2_math.zig");
const rig_ = @import("rig.zig");
const c = u.c;
const g = u.g;
const ops = u.ops;

pub const Kind = enum { static_whole, static_split, dynamic_whole, dynamic_split };
pub const kv_heads = 8;
const ring: usize = ops.conv_ring;

fn dynamic(k: Kind) bool {
    return k == .dynamic_whole or k == .dynamic_split;
}
fn split(k: Kind) bool {
    return k == .static_split or k == .dynamic_split;
}

pub fn name(k: Kind) []const u8 {
    return switch (k) {
        .static_whole => "metal inkling2 conv static",
        .static_split => "metal inkling2 conv static, window split over a store",
        .dynamic_whole => "metal inkling2 conv dynamic",
        .dynamic_split => "metal inkling2 conv dynamic, window split over a store",
    };
}

fn depth(_: c.Shape) usize {
    return 8;
}

pub const Probe = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    rig: rig_.Rig,
    o: *const ops.Ops,
    kind: Kind,
    label: []const u8,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    stream: []u16 = &.{},
    wts: []u16 = &.{},
    dyn: []u16 = &.{},
    x: rig_.Ref = undefined,
    hist: rig_.Ref = undefined,
    w: rig_.Ref = undefined,
    dynr: rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    work: rig_.Ref = undefined,
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
        const served: []const c.Shape = &.{
            .{ .rows = 6, .n = 1024, .k = 7 },
            .{ .rows = 2, .n = 2304, .k = 0 },
            .{ .rows = 3, .n = 1024, .k = 3 },
            .{ .rows = 40, .n = 1024, .k = 31 },
            .{ .rows = 17, .n = 2304, .k = 100 },
            .{ .rows = 33, .n = 1024, .k = 2 },
            .{ .rows = 128, .n = 2304, .k = 31 },
        };
        return .{ .ctx = p, .vt = &vt, .spec = .{
            .name = p.label,
            .accum = if (split(p.kind)) .none else .f32,
            .depth = depth,
            .ops = 1,
            .out = .bf16,
            .replaces = if (split(p.kind)) "the same window in one call" else null,
            .served = served,
            .tiles = .{ .{ .size = 32, .min = 2 }, .{}, .{ .size = 4, .min = 0 } },
        } };
    }

    fn ng(p: *const Probe) usize {
        return @min(p.shape.rows, ring);
    }

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const rows = shape.rows;
        const ch = shape.n;
        const pos0 = shape.k;
        if (rows == 0 or ch == 0 or (split(p.kind) and rows < 2)) return false;
        if (dynamic(p.kind) and ch % kv_heads != 0) return false;
        const a = u.reset(&p.arena, &p.rig);
        p.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const total = pos0 + rows;
        const stream = try a.alloc(u16, total * ch);
        u.fillUniform(rng, stream, 1);
        const wts = try a.alloc(u16, 4 * ch);
        u.fillUniform(rng, wts, if (dynamic(p.kind)) 0.5 else 1);
        const dyn = try a.alloc(u16, if (dynamic(p.kind)) rows * kv_heads * 4 else 0);
        u.fillUniform(rng, dyn, 0.5);
        if (fill == .stress and rows >= 4) {
            // Every window of every row sums to integers below 2^24 (exact in fp32 in any order); row 3's cancels to
            // zero through 65536 and 255 - 255: fp16 overflows on 65536 and bf16 loses the 255s.
            // Static taps run oldest first; dynamic taps run newest first. Both put 65536, 255, -255, 0 in that
            // accumulation order, then cancel the exact sum with row 3's -65536 residual.
            const dyn_kind = dynamic(p.kind);
            @memset(stream, 0);
            @memset(dyn, 0);
            const pattern: [4]f32 = if (dyn_kind) .{ 0, -255, 255, -65536 } else .{ 65536, 255, -255, -65536 };
            const weigh: [4]f32 = if (dyn_kind) .{ 0, 1, 1, -1 } else .{ 1, 1, 1, 0 }; // oldest first
            for (0..ch) |i| {
                for (0..4) |j| {
                    stream[(pos0 + j) * ch + i] = c.toBf16(pattern[j]);
                    if (dynamic(p.kind)) wts[(3 - j) * ch + i] = c.toBf16(weigh[j]) else wts[i * 4 + j] = c.toBf16(weigh[j]);
                }
            }
        }
        if (fill == .nonfinite) {
            const r1 = @min(rows - 1, 1);
            const r2 = @min(rows - 1, 2);
            const r3 = @min(rows - 1, 3);
            stream[pos0 * ch + 0] = u.nan_bits;
            stream[(pos0 + r1) * ch + 1] = u.pinf_bits;
            stream[(pos0 + r2) * ch + 2] = u.ninf_bits;
            stream[(pos0 + r1) * ch + 3] = u.pinf_bits;
            stream[(pos0 + r3) * ch + 3] = u.ninf_bits;
            if (pos0 > 0) stream[(pos0 - 1) * ch + 4] = u.pinf_bits;
            if (dynamic(p.kind)) {
                wts[1 * ch + 5] = u.nan_bits;
                dyn[0 * kv_heads * 4 + 1 * 4 + 0] = u.nan_bits;
            } else {
                wts[5 * 4 + 1] = u.nan_bits;
            }
        }
        // History: the three positions before the window, nothing else: every other slot is NaN.
        const hist = try a.alloc(u16, ring * ch);
        @memset(hist, u.nan_bits);
        var pos = pos0 -| 3;
        while (pos < pos0) : (pos += 1) @memcpy(hist[(pos % ring) * ch ..][0..ch], stream[pos * ch ..][0..ch]);
        p.stream = stream;
        p.wts = wts;
        p.dyn = dyn;
        p.x = try p.rig.input(std.mem.sliceAsBytes(stream[pos0 * ch ..]));
        p.hist = try p.rig.input(std.mem.sliceAsBytes(hist));
        p.w = try p.rig.input(std.mem.sliceAsBytes(wts));
        if (dynamic(p.kind)) p.dynr = try p.rig.input(std.mem.sliceAsBytes(dyn));
        const out_bytes = (rows + p.ng()) * ch * 2;
        p.out = try p.rig.output(out_bytes);
        p.work = try p.rig.scratch(ring * ch * 2);
        p.alt = try p.rig.scratch(out_bytes);
        return true;
    }

    fn piece(p: *Probe, e: mtl.ComputeEncoder, dst: rig_.Ref, first: usize, len: usize) !void {
        const ch: u32 = @intCast(p.shape.n);
        const xv = u.View{ .buffer = p.x.buf, .offset = p.x.off + first * ch * 2 };
        const ov = u.View{ .buffer = dst.buf, .offset = dst.off + first * ch * 2 };
        const pos: u32 = @intCast(p.shape.k + first);
        const n: u32 = @intCast(len);
        if (dynamic(p.kind)) {
            const dv = u.View{ .buffer = p.dynr.buf, .offset = p.dynr.off + first * kv_heads * 4 * 2 };
            try p.o.convDynamic(e, .{ .x = xv, .dyn = dv, .static_w = u.view(p.w), .hist = u.view(p.work), .out = ov, .rows = n, .channels = ch, .kv_heads = kv_heads, .pos0 = pos });
        } else {
            try p.o.convStatic(e, .{ .x = xv, .hist = u.view(p.work), .w = u.view(p.w), .out = ov, .rows = n, .channels = ch, .pos0 = pos });
        }
        try p.o.convStore(e, .{ .x = xv, .hist = u.view(p.work), .rows = n, .channels = ch, .pos0 = pos });
    }

    /// The window through the operators, into `dst`: the convolution's rows, then the ring rows the stores wrote.
    fn compute(p: *Probe, dst: rig_.Ref, cut: bool) !void {
        const ch = p.shape.n;
        const rows = p.shape.rows;
        const ring_bytes = ring * ch * 2;
        @memcpy(p.rig.bytes(p.work, ring_bytes), p.rig.bytes(p.hist, ring_bytes));
        const pass = p.rig.begin();
        const m = if (cut) rows / 2 else rows;
        try p.piece(pass.e, dst, 0, m);
        if (cut) try p.piece(pass.e, dst, m, rows - m);
        try p.rig.finish(pass);
        const n = p.ng();
        const all = p.rig.bytes(dst, (rows + n) * ch * 2);
        const held = p.rig.bytes(p.work, ring_bytes);
        for (0..n) |q| {
            const position = p.shape.k + rows - n + q;
            @memcpy(all[(rows + q) * ch * 2 ..][0 .. ch * 2], held[(position % ring) * ch * 2 ..][0 .. ch * 2]);
        }
    }

    pub fn run(p: *Probe) !void {
        try p.compute(p.out, split(p.kind));
    }

    pub fn replaced(p: *Probe) ![]const u8 {
        try p.compute(p.alt, false);
        return p.rig.bytes(p.alt, (p.shape.rows + p.ng()) * p.shape.n * 2);
    }

    pub fn count(p: *Probe) usize {
        return (p.shape.rows + p.ng()) * p.shape.n;
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        const a = p.arena.allocator();
        const ch = p.shape.n;
        const rows = p.shape.rows;
        const pos0 = p.shape.k;
        const stream = try u.decode(a, p.stream);
        const wts = try u.decode(a, p.wts);
        if (dynamic(p.kind)) {
            const dyn = try u.decode(a, p.dyn);
            math.dynamicConv(stream, wts, dyn, ch, kv_heads, pos0, rows, want[0 .. rows * ch], mag[0 .. rows * ch]);
        } else {
            math.staticConv(stream, wts, ch, pos0, rows, want[0 .. rows * ch], mag[0 .. rows * ch]);
        }
        const n = p.ng();
        for (0..n) |q| for (0..ch) |i| {
            const v = stream[(pos0 + rows - n + q) * ch + i];
            want[(rows + q) * ch + i] = v;
            mag[(rows + q) * ch + i] = @abs(v);
        };
    }

    /// The ring rows are copies: held exact.
    pub fn bounds(p: *Probe, _: []const f64, _: []const f64, allow: []f64) !void {
        @memset(allow[p.shape.rows * p.shape.n ..], 0);
    }

    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = p.arena.allocator();
        const ch = p.shape.n;
        const rows = p.shape.rows;
        const pos0 = p.shape.k;
        const stream = try u.decode(a, p.stream);
        const wts = try u.decode(a, p.wts);
        if (dynamic(p.kind)) {
            const dyn = try u.decode(a, p.dyn);
            math.dynamicConvIn(acc, order, stream, wts, dyn, ch, kv_heads, pos0, rows, out[0 .. rows * ch]);
        } else {
            math.staticConvIn(acc, order, stream, wts, ch, pos0, rows, out[0 .. rows * ch]);
        }
        const n = p.ng();
        for (0..n) |q| for (0..ch) |i| {
            out[(rows + q) * ch + i] = stream[(pos0 + rows - n + q) * ch + i];
        };
    }
};

/// One planted fault in the convolution source; the shapes gate must go red on it.
pub const Mutation = struct { kind: Kind, label: []const u8, from: []const u8, to: []const u8 };

pub const mutations = [_]Mutation{
    .{ .kind = .static_whole, .label = "control: inkling2 conv static, taps read in reverse", .from = "float(w[ulong(c) * I2_CONV_TAPS + j])", .to = "float(w[ulong(c) * I2_CONV_TAPS + (I2_CONV_TAPS - 1 - j)])" },
    .{ .kind = .static_whole, .label = "control: inkling2 conv static, a position before 0 read from the ring", .from = "if (position < 0) return 0.0f;", .to = "if (position < 0) return float(hist[(ulong(position) % a.ring) * a.channels + c]);" },
    .{ .kind = .static_split, .label = "control: inkling2 conv store, only the newest row kept", .from = "const uint kept = min(a.rows, a.ring);", .to = "const uint kept = min(a.rows, 1u);" },
    .{ .kind = .dynamic_whole, .label = "control: inkling2 conv dynamic, the lags of the weight projection reversed", .from = "head * I2_CONV_TAPS + lag]", .to = "head * I2_CONV_TAPS + (I2_CONV_TAPS - 1 - lag)]" },
};

pub fn controlSource(a: std.mem.Allocator, m: Mutation) ![]u8 {
    return u.mutate(a, sources.inkling2_ops, m.from, m.to);
}

pub fn controlKernel(p: *Probe) g.Kernel {
    var k = p.kernel();
    k.control = .shapes;
    return k;
}
