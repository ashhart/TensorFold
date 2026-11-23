//! core affine_mm's expert gathers under the contract: rows sorted by expert, each through its own expert's W^T.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");

const c = contract.spec;
const g = contract.gates;
const Affine = contract.affine.Affine;
const mm = tf.affine_mm;
const identity = @import("affine_identity.zig");

/// The experts a test spreads its rows over: some take none, so a tile walk must skip empty experts.
const weights = [_]usize{ 5, 0, 3, 1, 7, 2, 0, 4 };
const experts = weights.len;

/// A planted fault in affine_mm.metal's text; each must turn its one gate red.
pub const Mutation = enum {
    none,
    tail,
    expert,
    scatter,

    const Edit = struct { from: []const u8, to: []const u8 };

    fn edit(m: Mutation) Edit {
        return switch (m) {
            .none => unreachable,
            .tail => .{ .from = "rows = min(BM, simd_shuffle(last, ushort(owner)) - row);", .to = "rows = min(BM, simd_shuffle(last, ushort(owner)) - row - 1);" },
            .expert => .{ .from = "long(expert) * a.n + col", .to = "long(expert > 0 ? expert - 1 : 1) * a.n + col" },
            .scatter => .{ .from = "y + long(dst[r]) * N", .to = "y + long(r) * N" },
        };
    }
};

/// One registered gather: the weights' bits (groups of 64), whether each output goes back to its row's own place.
pub const Variant = struct { bits: u8, scatter: bool, glm_m3_fast: bool = false, identity: bool = false, control: bool = false, mutation: Mutation = .none };

/// The widths qwen27 admits for its routed experts on tensor units, each plain and scattering, and controls.
pub const registered = [_]Variant{
    .{ .bits = 4, .scatter = false },
    .{ .bits = 4, .scatter = true },
    .{ .bits = 8, .scatter = false },
    .{ .bits = 8, .scatter = true },
    .{ .bits = 4, .scatter = false, .glm_m3_fast = true },
    .{ .bits = 8, .scatter = true, .glm_m3_fast = true },
    .{ .bits = 4, .scatter = false, .glm_m3_fast = true, .identity = true },
    .{ .bits = 4, .scatter = false, .glm_m3_fast = true, .identity = true, .control = true },
    .{ .bits = 4, .scatter = false, .mutation = .tail },
    .{ .bits = 4, .scatter = false, .glm_m3_fast = true, .mutation = .tail },
    .{ .bits = 8, .scatter = false, .mutation = .expert },
    .{ .bits = 4, .scatter = true, .mutation = .scatter },
};

pub fn name(v: Variant) []const u8 {
    if (v.identity) return if (v.control) "control: metal affine_mm glm m3 q4 g64 gather finite identity, one BF16 rounding" else "metal affine_mm glm m3 q4 g64 gather finite identity";
    return switch (v.mutation) {
        .none => switch (v.bits) {
            4 => if (v.glm_m3_fast) (if (v.scatter) "metal affine_mm glm m3 q4 g64 gather scatter" else "metal affine_mm glm m3 q4 g64 gather") else if (v.scatter) "metal affine_mm q4 g64 gather scatter" else "metal affine_mm q4 g64 gather",
            else => if (v.glm_m3_fast) (if (v.scatter) "metal affine_mm glm m3 q8 g64 gather scatter" else "metal affine_mm glm m3 q8 g64 gather") else if (v.scatter) "metal affine_mm q8 g64 gather scatter" else "metal affine_mm q8 g64 gather",
        },
        .tail => if (v.glm_m3_fast) "control: metal affine_mm glm m3 gather q4 g64, each expert's last row dropped" else "control: metal affine_mm gather q4 g64, each expert's last row dropped",
        .expert => "control: metal affine_mm gather q8 g64, a tile reads its neighbour expert's weights",
        .scatter => "control: metal affine_mm gather scatter q4 g64, outputs left in sorted order",
    };
}

/// A group's products in the tile op, its scale and bias, the groups in turn (affine_mm's own bound).
fn groupDepth64(s: c.Shape) usize {
    return 64 + s.k / 32 + 8;
}

/// affine_mm's 4-bit codes enter as q + 128 and its bias as b - 128 s, so an infinite input meets opposite signs.
const offset_inf = c.Known{ .gate = .nonfinite, .finding = "an infinite input gives NaN where MLX's s*sum(q x) + b*sum(x) keeps the infinity (the +128 code offset moves the bias to b - 128 s)" };

pub fn spec(v: Variant) c.Spec {
    const known = if (v.bits == 4 and v.mutation == .none and !v.glm_m3_fast) offset_inf else null;
    // Ornith-1.5-122B: gate and up 1024 x 3072, down 3072 x 1024; 32 rows an expert (32-row tiles), past 64 (64-row)
    const served = &[_]c.Shape{ .{ .rows = 256, .n = 1024, .k = 3072 }, .{ .rows = 520, .n = 3072, .k = 1024 } };
    return .{ .name = name(v), .accum = .f32, .depth = groupDepth64, .out = .bf16, .known = known, .replaces = if (v.identity) "the frozen affine_mm before exact-nonfinite groups" else null, .tiles = .{ .{ .size = 32 }, .{ .size = 64, .step = 64, .min = 64 }, .{ .size = 128, .step = 64, .min = 64 } }, .served = served };
}

pub const Gather = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    base_pipes: ?mm.Pipes = null,
    v: Variant,
    pipes: mm.Pipes,
    parts: [experts]?Affine = @splat(null),
    counts: [experts]usize = undefined,
    dst: []i32 = &.{},
    shape: c.Shape = undefined,
    x: rig_.Ref = undefined,
    w: rig_.Ref = undefined,
    s: rig_.Ref = undefined,
    b: rig_.Ref = undefined,
    sums: rig_.Ref = undefined,
    base_sums: rig_.Ref = undefined,
    base: rig_.Ref = undefined,
    offsets: rig_.Ref = undefined,
    order: rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, v: Variant) !Gather {
        const format: mm.Format = .{ .bits = v.bits, .group = 64, .glm_m3_fast = v.glm_m3_fast, .exact_nonfinite = v.bits == 4 and v.glm_m3_fast };
        var pipes: mm.Pipes = undefined;
        var base_pipes: ?mm.Pipes = null;
        if (v.identity) {
            const src = try identity.source(device, a, format, if (v.control) .rounding_control else .current);
            defer a.free(src);
            pipes = try identity.compile(device, src);
            const old = try identity.source(device, a, format, .baseline);
            defer a.free(old);
            base_pipes = try identity.compile(device, old);
        } else {
            const plain = try mm.source(device, a, format);
            defer a.free(plain);
            const src = if (v.mutation == .none) try a.dupe(u8, plain) else blk: {
                const edit = v.mutation.edit();
                if (std.mem.indexOf(u8, plain, edit.from) == null) return error.ControlEditMissing;
                break :blk try std.mem.replaceOwned(u8, a, plain, edit.from, edit.to);
            };
            defer a.free(src);
            pipes = try identity.compile(device, src);
        }
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .v = v, .pipes = pipes, .base_pipes = base_pipes };
    }

    pub fn deinit(p: *Gather) void {
        p.drop();
        if (p.base_pipes) |pipes| for (pipes) |x| x.deinit();
        for (p.pipes) |x| x.deinit();
        p.rig.deinit();
    }

    fn drop(p: *Gather) void {
        for (&p.parts) |*f| {
            if (f.*) |x| x.deinit();
            f.* = null;
        }
        if (p.dst.len > 0) p.a.free(p.dst);
        p.dst = &.{};
        p.rig.reset();
    }

    pub fn kernel(p: *Gather) g.Kernel {
        return .{ .spec = spec(p.v), .ctx = p, .vt = &vt, .control = if (p.v.mutation != .none or p.v.control) .shapes else null };
    }

    const vt = g.hooks(Gather);

    /// Each expert's rows: `rows` shared by `weights`, the remainder to the first experts that take any.
    fn spread(rows: usize, out: *[experts]usize) void {
        var total: usize = 0;
        for (weights) |w| total += w;
        var given: usize = 0;
        for (weights, out) |w, *n| {
            n.* = rows * w / total;
            given += n.*;
        }
        var e: usize = 0;
        while (given < rows) : (e = (e + 1) % experts) if (weights[e] > 0) {
            out[e] += 1;
            given += 1;
        };
    }

    pub fn prepare(p: *Gather, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        p.drop();
        if (shape.rows == 0 or shape.n % 64 != 0 or shape.k % 64 != 0 or shape.n == 0 or shape.k == 0) return false;
        p.shape = shape;
        spread(shape.rows, &p.counts);
        const a = p.a;
        const kg = shape.k / 64;
        const words_each = shape.n * shape.k * p.v.bits / 32;
        const x = try a.alloc(u16, shape.rows * shape.k);
        defer a.free(x);
        const w = try a.alloc(u32, experts * words_each);
        defer a.free(w);
        const s = try a.alloc(u16, experts * shape.n * kg);
        defer a.free(s);
        const b = try a.alloc(u16, s.len);
        defer a.free(b);
        var offsets: [experts]i32 = undefined;
        var at: usize = 0;
        var planted = false;
        for (&p.parts, p.counts, 0..) |*slot, n, e| {
            var f = try Affine.init(a, .{ .rows = n, .n = shape.n, .k = shape.k }, p.v.bits, 64);
            if (p.v.bits == 4) f.off = 128;
            slot.* = f;
            // gate 1's NaN and infinities in the first expert with rows only, so the others' outputs stay finite
            try f.fill(if (fill == .nonfinite and (planted or n == 0)) .random else fill, seed +% e);
            if (p.v.identity and fill == .random) identity.plant(f);
            if (n > 0) planted = true;
            offsets[e] = @intCast(at);
            @memcpy(x[at * shape.k ..][0 .. n * shape.k], f.x);
            f.words(w[e * words_each ..][0..words_each]);
            @memcpy(s[e * shape.n * kg ..][0 .. shape.n * kg], f.scales);
            @memcpy(b[e * shape.n * kg ..][0 .. shape.n * kg], f.biases);
            at += n;
        }
        p.x = try p.rig.input(std.mem.sliceAsBytes(x));
        p.w = try p.rig.input(std.mem.sliceAsBytes(w));
        p.s = try p.rig.input(std.mem.sliceAsBytes(s));
        p.b = try p.rig.input(std.mem.sliceAsBytes(b));
        p.offsets = try p.rig.input(std.mem.sliceAsBytes(&offsets));
        if (p.v.scatter) {
            // a permutation that moves every row but one-row shapes: i * step + 3, step prime to the rows
            p.dst = try a.alloc(i32, shape.rows);
            var step: usize = 7;
            while (std.math.gcd(step, shape.rows) != 1) step += 2;
            for (p.dst, 0..) |*d, i| d.* = @intCast((i * step + 3) % shape.rows);
            p.order = try p.rig.input(std.mem.sliceAsBytes(p.dst));
        }
        p.sums = try p.rig.scratch(shape.rows * kg * 4);
        if (p.v.identity) {
            p.base_sums = try p.rig.scratch(shape.rows * kg * 4);
            p.base = try p.rig.scratch(shape.rows * shape.n * 2);
        }
        p.out = try p.rig.output(shape.rows * shape.n * 2);
        return true;
    }

    pub fn run(p: *Gather) !void {
        const pass = p.rig.begin();
        p.encodeGather(pass.e, p.pipes, p.sums, p.out);
        try p.rig.finish(pass);
    }

    fn encodeGather(p: *Gather, e: mtl.ComputeEncoder, pipes: mm.Pipes, sums: rig_.Ref, y: rig_.Ref) void {
        const shape = p.shape;
        const x = .{ .buf = p.x.buf, .off = p.x.off };
        const xs = .{ .buf = sums.buf, .off = sums.off };
        const offs = .{ .buf = p.offsets.buf, .off = p.offsets.off };
        const q = .{ .w = .{ .buf = p.w.buf, .off = p.w.off }, .s = .{ .buf = p.s.buf, .off = p.s.off }, .b = .{ .buf = p.b.buf, .off = p.b.off }, .n = @as(u32, @intCast(shape.n)), .k = @as(u32, @intCast(shape.k)) };
        mm.rowSums(e, pipes, 64, x, xs, @intCast(shape.rows), @intCast(shape.k));
        e.barrier();
        if (p.v.scatter) {
            mm.gatherTo(e, pipes, x, xs, q, offs, .{ .buf = y.buf, .off = y.off }, .{ .buf = p.order.buf, .off = p.order.off }, @intCast(shape.rows), experts);
        } else {
            mm.gather(e, pipes, x, xs, q, offs, .{ .buf = y.buf, .off = y.off }, @intCast(shape.rows), experts);
        }
    }

    pub fn count(p: *Gather) usize {
        return p.shape.rows * p.shape.n;
    }

    /// Sorted row i's output row: its own place for a scattering gather, else i.
    fn place(p: *const Gather, i: usize) usize {
        return if (p.v.scatter) @intCast(p.dst[i]) else i;
    }

    /// Each expert's outputs in sorted order through `each`, then moved to their places.
    fn gatherAll(p: *Gather, out: []f64, out2: ?[]f64, comptime each: anytype, args: anytype) !void {
        const n = p.shape.n;
        var at: usize = 0;
        for (p.parts, p.counts) |f, rows| {
            if (rows == 0) continue;
            const t = try p.a.alloc(f64, rows * n);
            defer p.a.free(t);
            const t2 = try p.a.alloc(f64, if (out2 != null) rows * n else 0);
            defer p.a.free(t2);
            try each(f.?, t, t2, args);
            for (0..rows) |r| {
                const to = p.place(at + r) * n;
                @memcpy(out[to..][0..n], t[r * n ..][0..n]);
                if (out2) |o2| @memcpy(o2[to..][0..n], t2[r * n ..][0..n]);
            }
            at += rows;
        }
    }

    fn refOne(f: Affine, want: []f64, mag: []f64, _: void) !void {
        return f.reference(want, mag);
    }

    pub fn reference(p: *Gather, want: []f64, mag: []f64) !void {
        return p.gatherAll(want, mag, refOne, {});
    }

    const Sim = struct { acc: c.Accum, order: g.Order };

    fn simOne(f: Affine, out: []f64, _: []f64, s: Sim) !void {
        return f.simulate(s.acc, s.order, out);
    }

    pub fn simulate(p: *Gather, acc: c.Accum, order: g.Order, out: []f64) !void {
        return p.gatherAll(out, null, simOne, Sim{ .acc = acc, .order = order });
    }

    pub fn replaced(p: *Gather) ![]const u8 {
        if (!p.v.identity) return error.NotAReplacement;
        const len = p.shape.rows * p.shape.n * 2;
        @memset(p.rig.bytes(p.base, len), 0xff);
        const pass = p.rig.begin();
        p.encodeGather(pass.e, p.base_pipes.?, p.base_sums, p.base);
        try p.rig.finish(pass);
        return p.rig.bytes(p.base, len);
    }
};

test "the rows' spread: every row placed, empty experts kept empty, the remainder to experts that take rows" {
    for ([_]usize{ 1, 7, 22, 256, 520 }) |rows| {
        var n: [experts]usize = undefined;
        Gather.spread(rows, &n);
        var total: usize = 0;
        for (n, weights) |x, w| {
            total += x;
            if (w == 0) try std.testing.expectEqual(@as(usize, 0), x);
        }
        try std.testing.expectEqual(rows, total);
    }
}
