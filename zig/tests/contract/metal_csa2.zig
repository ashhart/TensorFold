//! DeepSeek V4.1's chunk-only sparse attention: no ring and no compressed keys, every key is a chunk row.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");

const c = contract.spec;
const g = contract.gates;

pub const heads = 4;
pub const width = 64;
pub const rows = 4;
pub const window = 128;
/// The chunk's rows at the prepared call: the gates shrink a served shape's rows, never its n or k.
pub var live: usize = rows;
const chunk_stride = width + width / 32;
const q_stride = heads * width;
const scale: f32 = 0.04419417382415922;

/// One query element a lane, then the compensated lane reduction and the online denominator.
fn onlineDepth(s: c.Shape) usize {
    return (s.k + 31) / 32 + 6;
}

const why = "a bf16 output of a causal softmax over at most 512 keys: each probability is rounded to bf16 before its product, whose error stays under an output ulp a hundred times over";

pub fn spec() c.Spec {
    const served = &[_]c.Shape{.{ .rows = rows, .n = heads, .k = rows }};
    return .{
        .name = "deepseek_v41 sparse attention dsv41_sparse_attention",
        .accum = .f32,
        .depth = onlineDepth,
        .ops = 512,
        .out = .bf16,
        .tiles = .{ .{}, .{}, .{} },
        .served = served,
        .nan = .{ .stated = "a NaN query makes every column of its head NaN; the packed key cache cannot hold one" },
        .sums = .{ .f16 = 1, .bf16 = 1, .why = why },
    };
}

const Params = extern struct {
    rows: u32,
    heads: u32,
    width: u32,
    q_stride: u32,
    position: u32,
    chunk_rows: u32,
    window: u32,
    ring_stride: u32,
    chunk_stride: u32,
    main_stride: u32,
    main_block: u32,
    ratio: u32,
    compressed_rows: u32,
    topk: u32,
    index_stride: u32,
    scale: f32,
};

fn params() Params {
    return .{
        .rows = @intCast(live),
        .heads = heads,
        .width = width,
        .q_stride = q_stride,
        .position = 0,
        .chunk_rows = @intCast(live),
        .window = window,
        .ring_stride = chunk_stride,
        .chunk_stride = chunk_stride,
        .main_stride = width / 2 + width / 16,
        .main_block = 16,
        .ratio = 0,
        .compressed_rows = 0,
        .topk = 0,
        .index_stride = 512,
        .scale = scale,
    };
}

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    q: rig_.Ref = undefined,
    chunk: rig_.Ref = undefined,
    sink: rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    n: usize = 0,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue) !Probe {
        // kv.metal owns the exact packed-cache decoders this source is compiled after.
        const source = try std.mem.concat(a, u8, &.{ sources.deepseek_v41_kv, sources.deepseek_v41_csa2 });
        defer a.free(source);
        return .{
            .a = a,
            .rig = .{ .a = a, .device = device, .queue = queue },
            .pipe = try rig_.pipeline(device, source, "dsv41_sparse_attention"),
        };
    }

    pub fn deinit(self: *Probe) void {
        self.rig.deinit();
        self.pipe.deinit();
    }

    pub fn kernel(self: *Probe) g.Kernel {
        return .{ .spec = spec(), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Probe);

    pub fn prepare(self: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != heads or shape.rows == 0 or shape.rows > window or shape.k < 1 or shape.k > shape.rows) return false;
        live = shape.rows;
        self.rig.reset();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const q = try self.a.alloc(u16, live * q_stride);
        defer self.a.free(q);
        // The chunk is the engine's packed FP4 cache: one byte a value, then a shared power-of-two scale a block.
        const cache = try self.a.alloc(u8, live * chunk_stride);
        defer self.a.free(cache);
        const keys = try self.a.alloc(f32, live * width);
        defer self.a.free(keys);
        for (q) |*v| v.* = c.toBf16(rng.float(f32) * 2 - 1);
        for (keys) |*v| v.* = rng.float(f32) * 2 - 1;
        if (fill == .nonfinite) {
            // The keys cannot hold a NaN: a value is an e4 code and a block scale is a power of two.
            q[0] = c.toBf16(std.math.nan(f32));
            q[q_stride + 1] = c.toBf16(std.math.inf(f32));
            q[2 * q_stride + 2] = c.toBf16(-std.math.inf(f32));
        }
        if (fill == .stress) {
            // Large integer products that cancel down the column, so a narrow accumulator loses the result.
            for (q, 0..) |*v, i| v.* = c.toBf16(if (i % 2 == 0) @as(f32, 3.0e4) else @as(f32, 1.0));
            for (keys, 0..) |*v, i| v.* = if (i % 4 < 2) 448 else -448;
        }
        for (0..live) |row| packRow(cache[row * chunk_stride ..][0..chunk_stride], keys[row * width ..][0..width]);
        var sink: [heads]f32 = undefined;
        for (&sink) |*s| s.* = rng.float(f32) - 0.5;
        self.q = try self.rig.input(std.mem.sliceAsBytes(q));
        self.chunk = try self.rig.input(std.mem.sliceAsBytes(cache));
        self.sink = try self.rig.input(std.mem.sliceAsBytes(&sink));
        self.out = try self.rig.output(live * q_stride * 2);
        self.n = live * heads * width;
        return true;
    }

    pub fn run(self: *Probe) !void {
        const p = self.rig.begin();
        const e = p.e;
        e.setPipeline(self.pipe);
        // Buffers 1, 3 and 4 are the ring, compressed keys and index list: unused here, bound to a valid view.
        e.setBuffer(self.q.buf, self.q.off, 0);
        e.setBuffer(self.q.buf, self.q.off, 1);
        e.setBuffer(self.chunk.buf, self.chunk.off, 2);
        e.setBuffer(self.q.buf, self.q.off, 3);
        e.setBuffer(self.q.buf, self.q.off, 4);
        e.setBuffer(self.sink.buf, self.sink.off, 5);
        e.setBuffer(self.out.buf, self.out.off, 6);
        e.setValue(params(), 7);
        e.dispatchGroups(mtl.Size.of(heads, @intCast(live), 1), mtl.Size.of(32, 1, 1));
        try self.rig.finish(p);
    }

    /// Gate 1's own check: the kernel is trusted for its NaN rule, the planted inputs must both classes show.
    pub fn check(self: *Probe, at: g.Gate) !?[]const u8 {
        if (at != .nonfinite) return null;
        const want = try self.a.alloc(f64, self.n);
        defer self.a.free(want);
        const mag = try self.a.alloc(f64, self.n);
        defer self.a.free(mag);
        try self.reference(want, mag);
        const out = try self.rig.outputBytes();
        var bad: usize = 0;
        var first: usize = self.n;
        for (want, 0..) |w, i| {
            if (c.classOf(c.read(.bf16, out, i)) == c.classOf(w)) continue;
            bad += 1;
            if (first == self.n) first = i;
        }
        if (bad == 0) return null;
        return try std.fmt.allocPrint(self.a, "{d} of {d} outputs differ in class at the prepared call; first at {d}: {t} where the reference gives {t}", .{ bad, self.n, first, c.classOf(c.read(.bf16, out, first)), c.classOf(want[first]) });
    }

    pub fn count(self: *Probe) usize {
        return self.n;
    }

    /// The same products and bf16-rounded probabilities, with the per-column accumulation in `acc` and `order`.
    pub fn simulate(self: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        const q: [*]const u16 = @ptrCast(@alignCast(self.rig.bytes(self.q, live * q_stride * 2)));
        const k: [*]const u8 = @ptrCast(@alignCast(self.rig.bytes(self.chunk, live * chunk_stride)));
        const sink: [*]const f32 = @ptrCast(@alignCast(self.rig.bytes(self.sink, heads * 4)));
        const scores = try self.a.alloc(f64, window);
        defer self.a.free(scores);
        const terms = try self.a.alloc(f64, window + 1);
        defer self.a.free(terms);
        for (0..live) |row| {
            const scanned: usize = if (row == 0) 1 else window;
            for (0..heads) |head| {
                var mx: f64 = -std.math.inf(f64);
                for (0..scanned) |j| {
                    if (j > row) {
                        scores[j] = -std.math.inf(f64);
                        continue;
                    }
                    // The width-long reduction is where a narrow accumulator loses the cancellations.
                    for (0..width) |d| terms[d] = bf(q[row * q_stride + head * width + d]) * unpack(k + j * chunk_stride, d);
                    scores[j] = contract.rows.sumIn(acc, order, terms[0..width]) * @as(f64, scale);
                    mx = @max(mx, scores[j]);
                }
                // The denominator is the kernel's own sum, so it accumulates in `acc` and `order` too.
                for (scores[0..scanned], 0..) |s, j| terms[j] = @exp(s - mx);
                terms[scanned] = @exp(@as(f64, sink[head]) - mx);
                const denom = contract.rows.sumIn(acc, order, terms[0 .. scanned + 1]);
                for (0..width) |d| {
                    const products = try self.a.alloc(f64, scanned);
                    defer self.a.free(products);
                    for (scores[0..scanned], 0..) |s, j| products[j] = roundBf(@exp(s - mx)) * unpack(k + j * chunk_stride, d);
                    out[row * q_stride + head * width + d] = contract.rows.sumIn(acc, order, products[0..scanned]) / denom;
                }
            }
        }
    }

    /// The same bf16 products and bf16-rounded probabilities, with every sum in FP64.
    pub fn reference(self: *Probe, want: []f64, mag: []f64) !void {
        const q: [*]const u16 = @ptrCast(@alignCast(self.rig.bytes(self.q, live * q_stride * 2)));
        const k: [*]const u8 = @ptrCast(@alignCast(self.rig.bytes(self.chunk, live * chunk_stride)));
        const sink: [*]const f32 = @ptrCast(@alignCast(self.rig.bytes(self.sink, heads * 4)));
        const scores = try self.a.alloc(f64, window);
        defer self.a.free(scores);
        for (0..live) |row| {
            // The kernel always scans a whole window; slots past the query position are skipped (a null key).
            const scanned: usize = if (row == 0) 1 else window;
            for (0..heads) |head| {
                var mx: f64 = -std.math.inf(f64);
                var columns: usize = 0;
                for (0..scanned) |j| {
                    // position 0: slot 0 is chunk row 0, and no slot past the query is visible.
                    if (j > row) {
                        scores[j] = -std.math.inf(f64);
                        continue;
                    }
                    var dot: f64 = 0;
                    for (0..width) |d| {
                        dot += bf(q[row * q_stride + head * width + d]) * unpack(k + j * chunk_stride, d);
                    }
                    scores[j] = dot * @as(f64, scale);
                    // Metal's max drops a NaN operand, and every term feeding the dot is a rounded bf16.
                    if (std.math.isNan(scores[j])) scores[j] = -std.math.inf(f64);
                    mx = @max(mx, scores[j]);
                    columns += 1;
                }
                // The denominator adds the unrounded probabilities; only the product rounds.
                var sum: f64 = 0;
                for (scores[0..scanned]) |s| sum += @exp(s - mx);
                const denom = sum + @exp(@as(f64, sink[head]) - mx);
                for (0..width) |d| {
                    var acc: f64 = 0;
                    for (scores[0..scanned], 0..) |s, j| {
                        const probability = roundBf(@exp(s - mx));
                        acc += probability * unpack(k + j * chunk_stride, d);
                    }
                    const i = row * q_stride + head * width + d;
                    want[i] = acc / denom;
                    mag[i] = @floatFromInt(columns);
                }
            }
        }
    }
};

/// A bf16 stored value as its exact float64.
fn bf(v: u16) f64 {
    return @floatCast(@as(f32, @bitCast(@as(u32, v) << 16)));
}

/// The kernel's `dsv41_e4`: a sign, a 4-bit exponent and a 3-bit mantissa.
fn e4(code: u8) f32 {
    const e = (code >> 3) & 15;
    const m = code & 7;
    const v: f32 = if (e == 0) @as(f32, @floatFromInt(m)) * 0x1p-9 else std.math.ldexp(1.0 + @as(f32, @floatFromInt(m)) / 8.0, @as(i32, e) - 7);
    return if (code & 128 != 0) -v else v;
}

/// The kernel's `dsv41_e4code` for a value already reduced to one block's scale.
fn e4code(x: f32) u8 {
    const a = @min(@abs(x), 448.0);
    if (a == 0) return if (std.math.signbit(x)) 128 else 0;
    const e = std.math.clamp(@as(i32, @intCast((@as(u32, @bitCast(a)) >> 23) & 255)) - 127, -6, 7);
    const q: i32 = @intFromFloat(@round(std.math.ldexp(a, 3 - e)));
    const bits: u32 = if (e == -6 and q < 8) @intCast(q) else @intCast((e + 7) * 8 + q - 8);
    return @intCast(@min(bits, 126) | (if (std.math.signbit(x)) @as(u32, 128) else 0));
}

/// The kernel's `dsv41_unpack` for kind 0: an e4 value times a power-of-two block scale, then to bf16.
fn unpack(row: [*]const u8, d: usize) f64 {
    const s: f32 = @bitCast(@as(u32, row[width + d / 32]) << 23);
    return roundBf(@as(f64, e4(row[d])) * @as(f64, s));
}

/// Pack one chunk row the way `dsv41_pack` does, at a block scale of one so every code is exact.
fn packRow(out: []u8, x: []const f32) void {
    for (x, 0..) |v, d| {
        out[width + d / 32] = 127;
        out[d] = e4code(v);
    }
}

/// The kernel's `dsv41_round`: to the nearest bf16, even on a tie.
fn roundBf(x: f64) f64 {
    const f: f32 = @floatCast(x);
    const u: u32 = @bitCast(f);
    return @floatCast(@as(f32, @bitCast((u + 0x7fff + ((u >> 16) & 1)) & 0xffff_0000)));
}
