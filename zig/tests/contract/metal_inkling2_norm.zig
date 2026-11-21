//! Inkling 2's normed embedding, plain RMS norm and strided row copy through the operators the model graph calls
//! (Ops.embed, Ops.rmsNorm, Ops.copyRows), against float64 formulas written from Inkling2NormedEmbedding and
//! Inkling2RMSNorm (inkling2_math_norm.zig).
//!
//! Every input tensor sits 16 bytes into its buffer (a non-zero view offset) between NaN words, so a kernel that reads
//! outside its tensor goes NaN where float64 is finite. A row's scale varies by row (1, 4, 2^-10) so the 1e-6 in the
//! norm matters in a third of the rows. The `split` kinds run the same rows as two calls over offset views and must
//! equal the one call byte for byte (a row's bytes never depend on the rows around it).
//!
//! Stated rule (embed): an id at or past the vocabulary gives a NaN row and reads nothing past the table; the
//! non-finite fill plants such ids (the vocabulary itself and 0xffffffff) and the reference implements the rule.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const u = @import("inkling2_util.zig");
const math = @import("inkling2_math_norm.zig");
const rig_ = @import("rig.zig");
const c = u.c;
const g = u.g;
const ops = u.ops;

pub const Kind = enum { embed, embed_split, rms, rms_split, copy, copy_split };

const eps: f32 = 1e-6;
/// Elements (bf16) or half-words before and after each input tensor in its buffer: a 16-byte view offset.
const lead = 8;
/// What the copy leaves in its destination gaps (bf16 1.0) and, before each run, what an unwritten element holds (NaN).
const sentinel: u16 = 0x3f80;

fn split(k: Kind) bool {
    return k == .embed_split or k == .rms_split or k == .copy_split;
}
fn isEmbed(k: Kind) bool {
    return k == .embed or k == .embed_split;
}
fn isCopy(k: Kind) bool {
    return k == .copy or k == .copy_split;
}

pub fn name(k: Kind) []const u8 {
    return switch (k) {
        .embed => "metal inkling2 embed norm",
        .embed_split => "metal inkling2 embed norm, rows split over two calls",
        .rms => "metal inkling2 rms norm",
        .rms_split => "metal inkling2 rms norm, rows split over two calls",
        .copy => "metal inkling2 copy rows",
        .copy_split => "metal inkling2 copy rows, rows split over two calls",
    };
}

fn depth(s: c.Shape) usize {
    return math.rowSumDepth(s.k);
}
fn noSums(_: c.Shape) usize {
    return 0;
}

const norm_why = "a bf16 output follows the sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps";
const embed_rule = "an id at or past the vocabulary gives a NaN row; every other row is the RMS norm of its table row";

/// Row r of `x` ([rows][k]) scaled 1, 4 or 2^-10 by r % 3; or integers whose squares sum to about 2^20 (gate 4).
fn fillRows(rng: std.Random, x: []u16, rows: usize, k: usize, fill: g.Fill) void {
    for (0..rows) |r| {
        const row = x[r * k ..][0..k];
        if (fill == .stress) {
            // Integers in [-32, 32]: the squares' partial sums reach ~2^20 and stay exact in fp32 in any order.
            const h = k / 2;
            for (0..h) |i| {
                const v: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 64));
                const m: f32 = @floatFromInt(rng.uintLessThan(u32, 65));
                row[i] = c.toBf16(v - 32);
                row[h + i] = c.toBf16(m - 32);
            }
            if (k % 2 == 1) row[k - 1] = c.toBf16(0);
            continue;
        }
        const scale: f32 = switch (r % 3) {
            0 => 1,
            1 => 4,
            else => 0x1p-10,
        };
        for (row) |*v| v.* = u.uniform(rng, scale);
    }
}

/// NaN in row 0, +inf in row 1, -inf in row 2, both infinities in row 3.
fn plant(x: []u16, rows: usize, k: usize) void {
    const at = [_][2]usize{ .{ 0, k / 3 }, .{ 1, k / 2 }, .{ 2, k - 1 }, .{ 3, 0 }, .{ 3, k - 1 } };
    const put = [_]u16{ u.nan_bits, u.pinf_bits, u.ninf_bits, u.pinf_bits, u.ninf_bits };
    for (at, put) |q, v| if (q[0] < rows) {
        x[q[0] * k + q[1]] = v;
    };
}

/// `data` between `lead` NaN words on each side.
fn padded(a: std.mem.Allocator, data: []const u16) ![]u16 {
    const out = try a.alloc(u16, lead + data.len + lead);
    @memset(out, u.nan_bits);
    @memcpy(out[lead..][0..data.len], data);
    return out;
}

// Served shapes. A split needs two rows at least, so the one-row shapes are appended for the whole-call kinds only.
// embed: rows tokens, vocabulary, hidden. rms: rows, hidden, hidden. copy: rows, columns, extra source pitch.
const embed_shared = [_]c.Shape{
    .{ .rows = 7, .n = 61, .k = 128 },   .{ .rows = 16, .n = 97, .k = 4608 }, .{ .rows = 40, .n = 61, .k = 1000 },
    .{ .rows = 33, .n = 61, .k = 128 },  .{ .rows = 31, .n = 61, .k = 4608 }, .{ .rows = 32, .n = 61, .k = 128 },
    .{ .rows = 17, .n = 53, .k = 4608 }, .{ .rows = 15, .n = 61, .k = 1000 }, .{ .rows = 2, .n = 61, .k = 4608 },
};
const embed_single = embed_shared ++ [_]c.Shape{.{ .rows = 1, .n = 61, .k = 4608 }};
const rms_shared = [_]c.Shape{
    .{ .rows = 6, .n = 128, .k = 128 },    .{ .rows = 16, .n = 4608, .k = 4608 }, .{ .rows = 40, .n = 1000, .k = 1000 },
    .{ .rows = 33, .n = 128, .k = 128 },   .{ .rows = 31, .n = 4608, .k = 4608 }, .{ .rows = 32, .n = 128, .k = 128 },
    .{ .rows = 17, .n = 4608, .k = 4608 }, .{ .rows = 15, .n = 1000, .k = 1000 }, .{ .rows = 2, .n = 4608, .k = 4608 },
};
const rms_single = rms_shared ++ [_]c.Shape{.{ .rows = 1, .n = 4608, .k = 4608 }};
const copy_shared = [_]c.Shape{
    .{ .rows = 6, .n = 128, .k = 0 },     .{ .rows = 3, .n = 4608, .k = 7 }, .{ .rows = 17, .n = 1000, .k = 3 },
    .{ .rows = 40, .n = 4608, .k = 100 }, .{ .rows = 33, .n = 128, .k = 1 }, .{ .rows = 2, .n = 257, .k = 5 },
    .{ .rows = 16, .n = 255, .k = 9 },    .{ .rows = 31, .n = 256, .k = 0 },
};
const copy_single = copy_shared ++ [_]c.Shape{.{ .rows = 1, .n = 4608, .k = 0 }};

pub const Probe = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    rig: rig_.Rig,
    o: *const ops.Ops,
    kind: Kind,
    label: []const u8,
    /// embed: rows tokens, n vocabulary, k hidden. rms: rows, n = k = hidden. copy: rows, n columns, k extra pitch.
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    ids: []u32 = &.{},
    /// The embedding table, the RMS norm's input rows or the copy's source rows (with its pitch gaps), as bf16 words.
    data: []u16 = &.{},
    w: []u16 = &.{},
    r_ids: rig_.Ref = undefined,
    r_x: rig_.Ref = undefined,
    r_w: rig_.Ref = undefined,
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
        const rows16: c.Tile = .{ .size = 16 };
        const hidden: c.Tile = .{ .size = 256, .step = 1, .min = 1 };
        const sp = split(p.kind);
        const embed_served: []const c.Shape = if (sp) &embed_shared else &embed_single;
        const rms_served: []const c.Shape = if (sp) &rms_shared else &rms_single;
        const copy_served: []const c.Shape = if (sp) &copy_shared else &copy_single;
        const s: c.Spec = switch (p.kind) {
            .embed, .embed_split => .{
                .name = p.label,
                .accum = if (sp) .none else .f32,
                .depth = depth,
                .ops = 8,
                .out = .bf16,
                .replaces = if (sp) "the same rows in one call" else null,
                .nan = .{ .stated = embed_rule },
                .tiles = .{ rows16, .{}, hidden },
                .served = embed_served,
                .sums = .{ .bf16 = 0, .why = norm_why },
            },
            .rms, .rms_split => .{
                .name = p.label,
                .accum = if (sp) .none else .f32,
                .depth = depth,
                .ops = 8,
                .out = .bf16,
                .replaces = if (sp) "the same rows in one call" else null,
                .tiles = .{ rows16, .{}, hidden },
                .served = rms_served,
                .sums = .{ .bf16 = 0, .why = norm_why },
            },
            .copy, .copy_split => .{
                .name = p.label,
                .accum = .none,
                .depth = noSums,
                .out = .bf16,
                .replaces = if (sp) "the same rows in one call" else null,
                .tiles = .{ rows16, .{ .size = 256, .step = 1, .min = 1 }, .{} },
                .served = copy_served,
            },
        };
        return .{ .ctx = p, .vt = &vt, .spec = s };
    }

    fn srcLd(p: *const Probe) usize {
        return p.shape.n + p.shape.k;
    }
    fn dstLd(p: *const Probe) usize {
        return p.shape.n + 2 * p.shape.k;
    }

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const rows = shape.rows;
        if (rows == 0 or shape.n == 0 or (split(p.kind) and rows < 2)) return false;
        const embed = isEmbed(p.kind);
        const hidden = if (isCopy(p.kind)) shape.n else shape.k;
        if (hidden == 0 or hidden > ops.max_hidden or (embed and shape.n < 8)) return false;
        const a = u.reset(&p.arena, &p.rig);
        p.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (isCopy(p.kind)) return p.prepareCopy(a, rng, fill);
        const vocab = shape.n;
        const nrows = if (embed) vocab else rows;
        const data = try a.alloc(u16, nrows * hidden);
        fillRows(rng, data, nrows, hidden, fill);
        const w = try a.alloc(u16, hidden);
        for (w) |*v| v.* = c.toBf16(0.5 + rng.float(f32));
        const ids = try a.alloc(u32, if (embed) lead / 2 + rows else 0);
        if (embed) {
            @memset(ids[0 .. lead / 2], 0xffffffff);
            for (ids[lead / 2 ..], 0..) |*id, r| id.* = switch (r) {
                0 => @intCast(vocab - 1),
                1 => 0,
                else => rng.uintLessThan(u32, @intCast(vocab)),
            };
        }
        if (fill == .nonfinite) {
            plant(data, nrows, hidden);
            if (hidden > 7) w[7] = u.pinf_bits;
            if (embed) {
                const tok = ids[lead / 2 ..];
                for (tok, 0..) |*id, r| id.* = switch (r) {
                    0, 1, 2, 3 => @intCast(r),
                    4 => @intCast(vocab),
                    5 => 0xffffffff,
                    else => 4 + rng.uintLessThan(u32, @intCast(vocab - 4)),
                };
            }
        }
        p.ids = ids;
        p.data = data;
        p.w = w;
        if (embed) p.r_ids = try p.rig.input(std.mem.sliceAsBytes(ids));
        p.r_x = try p.rig.input(std.mem.sliceAsBytes(try padded(a, data)));
        p.r_w = try p.rig.input(std.mem.sliceAsBytes(try padded(a, w)));
        p.out = try p.rig.output(rows * hidden * 2);
        p.alt = try p.rig.scratch(rows * hidden * 2);
        return true;
    }

    fn prepareCopy(p: *Probe, a: std.mem.Allocator, rng: std.Random, fill: g.Fill) !bool {
        const rows = p.shape.rows;
        const cols = p.shape.n;
        const src_ld = p.srcLd();
        // Source rows `src_ld` apart; the gap after each row's `cols` elements is NaN.
        const src = try a.alloc(u16, rows * src_ld);
        @memset(src, u.nan_bits);
        for (0..rows) |r| for (0..cols) |j| {
            src[r * src_ld + j] = u.uniform(rng, 4);
        };
        if (fill == .nonfinite) {
            const at = [_][2]usize{ .{ 0, 0 }, .{ 1, cols / 2 }, .{ 2, cols - 1 }, .{ 3, 0 }, .{ 3, cols - 1 } };
            const put = [_]u16{ u.nan_bits, u.pinf_bits, u.ninf_bits, u.pinf_bits, u.ninf_bits };
            for (at, put) |q, v| if (q[0] < rows) {
                src[q[0] * src_ld + q[1]] = v;
            };
        }
        p.data = src;
        p.r_x = try p.rig.input(std.mem.sliceAsBytes(try padded(a, src)));
        const out_bytes = (lead + rows * p.dstLd()) * 2;
        p.out = try p.rig.output(out_bytes);
        p.alt = try p.rig.scratch(out_bytes);
        return true;
    }

    /// The tensor's view: past the leading words of its buffer.
    fn tensor(r: rig_.Ref, first_bytes: usize) u.View {
        return .{ .buffer = r.buf, .offset = r.off + lead * 2 + first_bytes };
    }

    fn piece(p: *Probe, e: mtl.ComputeEncoder, dst: rig_.Ref, first: usize, n: usize) !void {
        const rows: u32 = @intCast(n);
        if (isCopy(p.kind)) {
            const cols: u32 = @intCast(p.shape.n);
            try p.o.copyRows(e, .{
                .src = tensor(p.r_x, first * p.srcLd() * 2),
                .src_ld = @intCast(p.srcLd()),
                .dst = tensor(dst, first * p.dstLd() * 2),
                .dst_ld = @intCast(p.dstLd()),
                .rows = rows,
                .cols = cols,
            });
            return;
        }
        const hidden: u32 = @intCast(p.shape.k);
        const ov = u.View{ .buffer = dst.buf, .offset = dst.off + first * hidden * 2 };
        if (isEmbed(p.kind)) {
            // The ids sit lead / 2 words into their buffer (also 16 bytes).
            const iv = u.View{ .buffer = p.r_ids.buf, .offset = p.r_ids.off + lead * 2 + first * 4 };
            try p.o.embed(e, .{ .ids = iv, .table = tensor(p.r_x, 0), .norm = tensor(p.r_w, 0), .out = ov, .rows = rows, .vocab = @intCast(p.shape.n), .hidden = hidden, .eps = eps });
        } else {
            try p.o.rmsNorm(e, .{ .x = tensor(p.r_x, first * hidden * 2), .w = tensor(p.r_w, 0), .out = ov, .rows = rows, .hidden = hidden, .eps = eps });
        }
    }

    fn outLen(p: *const Probe) usize {
        return if (isCopy(p.kind)) lead + p.shape.rows * p.dstLd() else p.shape.rows * p.shape.k;
    }

    /// An unwritten element reads NaN; a copy's gaps and lead read the finite sentinel.
    fn prime(p: *Probe, dst: rig_.Ref) void {
        const total = p.outLen();
        const bytes = p.rig.bytes(dst, total * 2);
        const ld = p.dstLd();
        for (0..total) |i| {
            const real = i >= lead and (i - lead) % ld < p.shape.n;
            std.mem.writeInt(u16, bytes[2 * i ..][0..2], if (real) 0xffff else sentinel, .little);
        }
    }

    fn compute(p: *Probe, dst: rig_.Ref, cut: bool) !void {
        if (isCopy(p.kind)) p.prime(dst);
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
        if (isCopy(p.kind)) {
            const cols = p.shape.n;
            @memset(want[0..p.outLen()], c.fromBf16(sentinel));
            @memset(mag[0..p.outLen()], 1);
            for (0..rows) |r| for (0..cols) |j| {
                const v = u.val(p.data[r * p.srcLd() + j]);
                want[lead + r * p.dstLd() + j] = v;
                mag[lead + r * p.dstLd() + j] = @abs(v);
            };
            return;
        }
        const k = p.shape.k;
        const x = try u.decode(a, p.data);
        const w = try u.decode(a, p.w);
        for (0..rows) |r| {
            const o = want[r * k ..][0..k];
            const m = mag[r * k ..][0..k];
            if (isEmbed(p.kind)) {
                math.embedRow(x, p.ids[lead / 2 + r], p.shape.n, k, w, eps, o, m);
            } else {
                math.rmsRow(x[r * k ..][0..k], w, eps, o, m);
            }
        }
    }

    /// A copy moves words: exact.
    pub fn bounds(p: *Probe, _: []const f64, _: []const f64, allow: []f64) !void {
        if (isCopy(p.kind)) @memset(allow, 0);
    }

    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        if (isCopy(p.kind)) return error.NoSums;
        const a = p.arena.allocator();
        const k = p.shape.k;
        const x = try u.decode(a, p.data);
        const w = try u.decode(a, p.w);
        const terms = try a.alloc(f64, k);
        for (0..p.shape.rows) |r| {
            const o = out[r * k ..][0..k];
            if (isEmbed(p.kind)) {
                math.embedRowIn(acc, order, x, p.ids[lead / 2 + r], p.shape.n, k, w, eps, terms, o);
            } else {
                math.rmsRowIn(acc, order, x[r * k ..][0..k], w, eps, terms, o);
            }
        }
    }
};

/// One planted fault in the norm sources; its `gate` must go red on it.
pub const Mutation = struct { kind: Kind, gate: c.Gate, label: []const u8, from: []const u8, to: []const u8 };

pub const mutations = [_]Mutation{
    .{ .kind = .embed, .gate = .nonfinite, .label = "control: inkling2 embed, an id past the vocabulary gives zeros, not NaN", .from = "dst[i] = bfloat(NAN);", .to = "dst[i] = bfloat(0.0f);" },
    .{ .kind = .rms, .gate = .shapes, .label = "control: inkling2 rms norm, the weight not applied", .from = "out[i] = bfloat(float(w[i]) * (float(x[i]) * inv));", .to = "out[i] = bfloat(float(x[i]) * inv);" },
    .{ .kind = .rms, .gate = .shapes, .label = "control: inkling2 rms norm, hidden rounded down to whole threads", .from = "for (uint i = tid; i < hidden; i += I2_TG) out[i] = bfloat(float(w[i]) * (float(x[i]) * inv));", .to = "for (uint i = tid; i < (hidden / I2_TG) * I2_TG; i += I2_TG) out[i] = bfloat(float(w[i]) * (float(x[i]) * inv));" },
    .{ .kind = .rms, .gate = .shapes, .label = "control: inkling2 rms norm, eps left out of the scale", .from = "sum / float(n) + eps", .to = "sum / float(n)" },
    .{ .kind = .copy, .gate = .shapes, .label = "control: inkling2 copy rows, the source read at a pitch of its columns", .from = "dst[ulong(at.y) * a.dst_ld + at.x] = src[ulong(at.y) * a.src_ld + at.x];", .to = "dst[ulong(at.y) * a.dst_ld + at.x] = src[ulong(at.y) * a.cols + at.x];" },
    .{ .kind = .copy_split, .gate = .shapes, .label = "control: inkling2 copy rows, the last row of each call skipped", .from = "if (at.x >= a.cols || at.y >= a.rows) return;\n  dst[ulong(at.y) * a.dst_ld + at.x]", .to = "if (at.x >= a.cols || at.y + 1 >= a.rows) return;\n  dst[ulong(at.y) * a.dst_ld + at.x]" },
};

pub fn controlSource(a: std.mem.Allocator, m: Mutation) ![]u8 {
    return u.mutate(a, sources.inkling2_ops, m.from, m.to);
}

pub fn controlKernel(p: *Probe, gate: c.Gate) g.Kernel {
    var k = p.kernel();
    k.control = gate;
    return k;
}
