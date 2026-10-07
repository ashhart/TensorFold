//! The core row projection on synthetic 2-, 4-, 6- and 8-bit matrices: every width equals one row, a CPU reference, indexed experts.
const std = @import("std");
const mtl = @import("metal");
const row = @import("row_projection");

const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
const a = std.heap.page_allocator;

fn toBf(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}
fn fromBf(v: u16) f32 {
    return @bitCast(@as(u32, v) << 16);
}

const Case = struct { n: usize, k: usize, experts: usize, rows: usize, repeat: usize, bits: u8 = 4, sum: row.Sum = .f32 };

/// Code `i` of a row's bit stream (value i at bits i*bits and up, bytes little-endian; 6-bit: four values in three bytes).
fn code(bytes: []const u8, bits: u8, i: usize) u32 {
    const off = i * @as(usize, bits);
    const lo: u32 = bytes[off / 8];
    const hi: u32 = if (off / 8 + 1 < bytes.len) bytes[off / 8 + 1] else 0;
    return ((lo | (hi << 8)) >> @intCast(off % 8)) & ((@as(u32, 1) << @intCast(bits)) - 1);
}

/// Random words, scales, biases and inputs for `c`, in the row layout; the CPU product in f64.
const Synth = struct {
    w: mtl.Buffer,
    s: mtl.Buffer,
    b: mtl.Buffer,
    x: mtl.Buffer,
    out: mtl.Buffer,
    one: mtl.Buffer,
    ids: mtl.Buffer,

    fn init(device: mtl.Device, c: Case, seed: u64) !Synth {
        var prng = std.Random.DefaultPrng.init(seed);
        const r = prng.random();
        const groups = c.k / 64;
        const words = c.experts * c.n * c.k * c.bits / 32;
        const self = Synth{
            .w = try device.buffer(words * 4, opts),
            .s = try device.buffer(c.experts * c.n * groups * 2, opts),
            .b = try device.buffer(c.experts * c.n * groups * 2, opts),
            .x = try device.buffer(c.rows * c.k * 2, opts),
            .out = try device.buffer(c.rows * c.n * 2 * 4, opts),
            .one = try device.buffer(c.rows * c.n * 2 * 4, opts),
            .ids = try device.buffer(@max(c.rows, 8) * 4, opts),
        };
        for (self.w.slice(u32, words)) |*v| v.* = r.int(u32);
        for (self.s.slice(u16, c.experts * c.n * groups)) |*v| v.* = toBf(r.float(f32) * 0.02 + 0.001);
        for (self.b.slice(u16, c.experts * c.n * groups)) |*v| v.* = toBf((r.float(f32) - 0.5) * 0.1);
        for (self.x.slice(u16, c.rows * c.k)) |*v| v.* = toBf((r.float(f32) - 0.5) * 2);
        return self;
    }

    fn deinit(self: Synth) void {
        for ([_]mtl.Buffer{ self.w, self.s, self.b, self.x, self.out, self.one, self.ids }) |x| x.deinit();
    }

    /// Row m of x against expert e's column n in f64, and the sum of its terms' magnitudes (the error's scale).
    fn reference(self: Synth, c: Case, e: usize, m: usize, n: usize) [2]f64 {
        const groups = c.k / 64;
        const bytes = self.w.slice(u8, c.experts * c.n * c.k * c.bits / 8);
        const sc = self.s.slice(u16, c.experts * c.n * groups);
        const bi = self.b.slice(u16, c.experts * c.n * groups);
        const x = self.x.slice(u16, c.rows * c.k);
        var total: f64 = 0;
        var scale: f64 = 0;
        for (0..groups) |g| {
            var dot: f64 = 0;
            var sum: f64 = 0;
            for (0..64) |i| {
                const kk = g * 64 + i;
                const q: f64 = @floatFromInt(code(bytes[(e * c.n + n) * (c.k * c.bits / 8) ..][0 .. c.k * c.bits / 8], c.bits, kk));
                const xv: f64 = fromBf(x[m * c.k + kk]);
                dot += xv * q;
                sum += xv;
            }
            const sd = @as(f64, fromBf(sc[(e * c.n + n) * groups + g])) * dot;
            const bs = @as(f64, fromBf(bi[(e * c.n + n) * groups + g])) * sum;
            total += sd + bs;
            scale += @abs(sd) + @abs(bs);
        }
        return .{ total, scale };
    }
};

fn run(queue: mtl.Queue, body: anytype) !void {
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    try body.encode(e);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |m| {
        std.debug.print("GPU failed: {s}\n", .{m});
        return error.GpuFailed;
    }
}

const Dense = struct {
    p: *const row.Pipelines,
    w: row.Weights,
    x: mtl.Buffer,
    rows: usize,
    out: mtl.Buffer,
    relu2: bool,
    fn encode(d: @This(), e: mtl.ComputeEncoder) !void {
        const c = try row.call(d.p, d.w, d.rows, d.relu2);
        e.setPipeline(c.pipeline);
        e.setBuffer(d.x, 0, 0);
        e.setBuffer(d.w.w, d.w.w_off, 1);
        e.setBuffer(d.w.scales, d.w.s_off, 2);
        e.setBuffer(d.w.biases, d.w.b_off, 3);
        e.setValue(c.dims, 4);
        e.setBuffer(d.out, 0, 5);
        e.dispatchGroups(mtl.Size.of(c.groups, 1, 1), mtl.Size.of(c.threads, 1, 1));
    }
};

const Indexed = struct {
    p: *const row.Pipelines,
    x: row.Experts,
    in: mtl.Buffer,
    slots: usize,
    ids: mtl.Buffer,
    out: mtl.Buffer,
    relu2: bool,
    fn encode(d: @This(), e: mtl.ComputeEncoder) !void {
        const c = try row.callIndexed(d.p, d.x, d.slots, d.relu2);
        e.setPipeline(c.pipeline);
        e.setBuffer(d.in, 0, 0);
        e.setBuffer(d.x.one.w, 0, 1);
        e.setBuffer(d.x.one.scales, 0, 2);
        e.setBuffer(d.x.one.biases, 0, 3);
        e.setValue(c.dims, 4);
        e.setBuffer(d.out, 0, 5);
        e.setBuffer(d.ids, 0, 6);
        e.dispatchGroups(mtl.Size.of(c.groups, c.slots, 1), mtl.Size.of(c.threads, 1, 1));
    }
};

pub fn main(_: std.process.Init) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var p = try row.Pipelines.load(device);
    defer p.deinit();
    var checked: usize = 0;
    var bits: usize = 0;
    for ([_]Case{ .{ .n = 64, .k = 128, .experts = 3, .rows = 9, .repeat = 3 }, .{ .n = 128, .k = 2688, .experts = 2, .rows = 8, .repeat = 2 }, .{ .n = 16, .k = 576, .experts = 4, .rows = 7, .repeat = 1 }, .{ .n = 64, .k = 128, .experts = 3, .rows = 9, .repeat = 3, .bits = 8 }, .{ .n = 128, .k = 2688, .experts = 2, .rows = 8, .repeat = 2, .bits = 8 }, .{ .n = 64, .k = 128, .experts = 3, .rows = 9, .repeat = 3, .sum = .bf16 }, .{ .n = 128, .k = 2688, .experts = 2, .rows = 8, .repeat = 2, .bits = 8, .sum = .bf16 }, .{ .n = 64, .k = 128, .experts = 3, .rows = 9, .repeat = 3, .bits = 2 }, .{ .n = 128, .k = 2688, .experts = 2, .rows = 8, .repeat = 2, .bits = 2 }, .{ .n = 64, .k = 128, .experts = 3, .rows = 9, .repeat = 3, .bits = 6 }, .{ .n = 128, .k = 2688, .experts = 2, .rows = 8, .repeat = 2, .bits = 6 }, .{ .n = 64, .k = 576, .experts = 2, .rows = 5, .repeat = 1, .bits = 6, .sum = .bf16 } }) |c| {
        const s = try Synth.init(device, c, 7 + c.n);
        defer s.deinit();
        const groups = c.k / 64;
        const stride_w = c.n * c.k * c.bits / 32 * 4;
        const stride_sb = c.n * groups * 2;
        const w0 = row.Weights{ .w = s.w, .scales = s.s, .biases = s.b, .n = c.n, .k = c.k, .bits = c.bits, .sum = c.sum };
        // every width of expert 0 against one row at a time, the CPU reference within rounding, relu2 as relu squared of the plain output
        try run(queue, Dense{ .p = &p, .w = w0, .x = s.x, .rows = c.rows, .out = s.out, .relu2 = false });
        const wide = try a.dupe(u16, s.out.slice(u16, c.rows * c.n));
        defer a.free(wide);
        try run(queue, Dense{ .p = &p, .w = w0, .x = s.x, .rows = c.rows, .out = s.out, .relu2 = true });
        const wide2 = try a.dupe(u16, s.out.slice(u16, c.rows * c.n));
        defer a.free(wide2);
        for (0..c.rows) |m| {
            const xm = try device.buffer(c.k * 2, opts);
            defer xm.deinit();
            @memcpy(xm.slice(u16, c.k), s.x.slice(u16, c.rows * c.k)[m * c.k ..][0..c.k]);
            try run(queue, Dense{ .p = &p, .w = w0, .x = xm, .rows = 1, .out = s.one, .relu2 = false });
            const one = s.one.slice(u16, c.n);
            for (0..c.n) |n| {
                if (one[n] != wide[m * c.n + n]) return error.WidthChangesBits;
                const h = @max(fromBf(one[n]), 0);
                if (wide2[m * c.n + n] != toBf(h * h)) return error.Relu2DiffersFromPlain;
                bits += 2;
                const ref = s.reference(c, 0, m, n);
                const got: f64 = fromBf(one[n]);
                if (@abs(got - ref[0]) > 0.01 * ref[1] + 0.002) {
                    std.debug.print("n {d} k {d} row {d} col {d}: GPU {d} CPU {d} (terms {d})\n", .{ c.n, c.k, m, n, got, ref[0], ref[1] });
                    return error.ReferenceMismatch;
                }
                checked += 1;
            }
        }
        // indexed slots: each equals the dense kernel on its expert, an id past the count is a zero row
        for ([_]bool{ false, true }) |relu2| {
            const ids = s.ids.slice(u32, c.rows);
            for (ids, 0..) |*id, i| id.* = @intCast(if (i == c.rows - 1) c.experts else i % c.experts);
            const x = row.Experts{ .one = w0, .experts = c.experts, .repeat = c.repeat };
            const slots = c.rows / c.repeat * c.repeat;
            try run(queue, Indexed{ .p = &p, .x = x, .in = s.x, .slots = slots, .ids = s.ids, .out = s.out, .relu2 = relu2 });
            const got = try a.dupe(u16, s.out.slice(u16, slots * c.n));
            defer a.free(got);
            for (0..slots) |slot| {
                const e = ids[slot];
                const xm = try device.buffer(c.k * 2, opts);
                defer xm.deinit();
                @memcpy(xm.slice(u16, c.k), s.x.slice(u16, c.rows * c.k)[(slot / c.repeat) * c.k ..][0..c.k]);
                if (e >= c.experts) {
                    for (got[slot * c.n ..][0..c.n]) |v| if (v != 0) return error.MissingExpertNotZero;
                    continue;
                }
                const we = row.Weights{ .w = s.w, .w_off = e * stride_w, .scales = s.s, .s_off = e * stride_sb, .biases = s.b, .b_off = e * stride_sb, .n = c.n, .k = c.k, .bits = c.bits, .sum = c.sum };
                try run(queue, Dense{ .p = &p, .w = we, .x = xm, .rows = 1, .out = s.one, .relu2 = relu2 });
                for (s.one.slice(u16, c.n), got[slot * c.n ..][0..c.n]) |d, g| if (d != g) return error.IndexedDiffersFromDense;
                bits += c.n;
            }
            std.debug.print("PASS q{d} {s} sums n {d} k {d} experts {d} rows {d} repeat {d} relu2 {}\n", .{ c.bits, @tagName(c.sum), c.n, c.k, c.experts, c.rows, c.repeat, relu2 });
        }
    }
    std.debug.print("PASS {d} values against the CPU reference, {d} bit-equal pairs across widths and experts\n", .{ checked, bits });
}
