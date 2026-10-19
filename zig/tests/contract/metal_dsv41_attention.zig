//! The parallel sparse schedule: predecessor bytes, packed-cache FP64 formula, and uniform-score FP32 stress.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const c = contract.spec;
const g = contract.gates;
const csa = tf.deepseek_v41.csa2;
const View = tf.deepseek_v41_view.View;
const width = 512;
const ring_stride = 528;
const main_stride = 288;

fn depth(s: c.Shape) usize {
    return s.k + 32;
}
pub fn spec() c.Spec {
    return .{ .name = "deepseek sparse dsv41_sparse_attention_parallel", .replaces = "dsv41_sparse_attention", .accum = .f32, .depth = depth, .ops = 32, .mids = 1, .out = .bf16, .tiles = .{ .{ .size = 16 }, .{}, .{ .size = 192, .min = 128 } }, .served = &.{ .{ .rows = 8, .n = width, .k = 640 }, .{ .rows = 256, .n = width, .k = 128 }, .{ .rows = 1, .n = width, .k = 191 }, .{ .rows = 1, .n = width, .k = 192 }, .{ .rows = 1, .n = width, .k = 193 } } };
}

pub const Probe = struct {
    const vt = g.hooks(Probe);
    a: std.mem.Allocator,
    rig: rig_.Rig,
    p: csa.Pipelines,
    refs: [8]rig_.Ref = undefined,
    shape: c.Shape = undefined,
    params: csa.Params = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue) !Probe {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .p = try csa.Pipelines.load(device) };
    }
    pub fn deinit(self: *Probe) void {
        self.p.deinit();
        self.rig.deinit();
    }
    pub fn kernel(self: *Probe) g.Kernel {
        return .{ .spec = spec(), .ctx = self, .vt = &vt };
    }
    fn bytes(self: *Probe, i: usize) []u8 {
        const sizes = [_]usize{ self.shape.rows * width * 2, 128 * ring_stride, self.shape.rows * ring_stride, 513 * main_stride, self.shape.rows * @max(1, self.shape.k - 128) * 4, 4, self.count() * 2, self.count() * 2 };
        return self.rig.bytes(self.refs[i], sizes[i]);
    }
    pub fn prepare(self: *Probe, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (s.n != width or s.k < 128 or s.k > 640 or s.rows == 0 or s.rows > 256) return false;
        self.rig.reset();
        self.shape = s;
        var arena = std.heap.ArenaAllocator.init(self.a);
        defer arena.deinit();
        const a = arena.allocator();
        const sizes = [_]usize{ s.rows * width * 2, 128 * ring_stride, s.rows * ring_stride, 513 * main_stride, s.rows * @max(1, s.k - 128) * 4, 4 };
        var rng = std.Random.DefaultPrng.init(seed);
        for (sizes, 0..) |size, i| {
            const b = try a.alloc(u8, size);
            @memset(b, 0);
            if (i == 0) {
                for (0..s.rows * width) |j| std.mem.writeInt(u16, b[j * 2 ..][0..2], c.toBf16(if (fill == .stress) 0 else (rng.random().float(f32) * 2 - 1)), .little);
                if (fill == .nonfinite) {
                    const vals = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), std.math.inf(f32) };
                    for (vals, 0..) |v, row| std.mem.writeInt(u16, b[(row * width + 7) * 2 ..][0..2], c.toBf16(v), .little);
                    std.mem.writeInt(u16, b[(3 * width + 9) * 2 ..][0..2], c.toBf16(-std.math.inf(f32)), .little);
                }
            } else if (i == 1 or i == 2) {
                const count_ = if (i == 1) 128 else s.rows;
                for (0..count_) |row| {
                    const physical = if (i == 1) row else (8191 + row) % 128;
                    const code: u8 = if (fill == .stress) (if (physical < 56) @as(u8, @intCast(0x40 + physical % 8)) else if (physical < 64) 0x38 else if (physical < 120) @as(u8, @intCast(0xc0 + (physical - 64) % 8)) else 0) else @as(u8, @intCast(0x30 + rng.random().uintLessThan(u8, 16)));
                    @memset(b[row * ring_stride ..][0..width], code);
                    const exponent: u8 = if (fill == .stress and (physical < 56 or (physical >= 64 and physical < 120))) 139 else 127;
                    @memset(b[row * ring_stride + width ..][0..16], exponent);
                }
            } else if (i == 3 and fill != .stress) {
                for (0..513) |row| {
                    for (0..256) |j| b[row * main_stride + j] = rng.random().int(u8);
                    @memset(b[row * main_stride + 256 ..][0..32], 0x38);
                }
            } else if (i == 4) {
                for (0..size / 4) |j| std.mem.writeInt(i32, b[j * 4 ..][0..4], if (j % 7 == 0) -1 else @as(i32, @intCast((j * 73 + 13) % 513)), .little);
            }
            self.refs[i] = try self.rig.input(b);
        }
        self.refs[6] = try self.rig.output(self.count() * 2);
        self.refs[7] = try self.rig.scratch(self.count() * 2);
        self.params = .{ .rows = @intCast(s.rows), .heads = 1, .width = width, .q_stride = width, .position = 8191, .chunk_rows = @intCast(s.rows), .ratio = 2, .compressed_rows = 513, .topk = @intCast(s.k - 128), .index_stride = @intCast(@max(1, s.k - 128)) };
        return true;
    }
    fn launch(self: *Probe, parallel: bool, out: usize) !void {
        self.p.parallel_enabled = parallel;
        const pass = self.rig.begin();
        var v: [7]View = undefined;
        for (0..6) |i| v[i] = .{ .buffer = self.refs[i].buf, .offset = self.refs[i].off };
        v[6] = .{ .buffer = self.refs[out].buf, .offset = self.refs[out].off };
        try self.p.encode(pass.e, v[0], v[1], v[2], v[3], v[4], v[5], v[6], self.params);
        try self.rig.finish(pass);
    }
    pub fn run(self: *Probe) !void {
        try self.launch(true, 6);
    }
    pub fn replaced(self: *Probe) ![]const u8 {
        try self.launch(false, 7);
        return self.bytes(7);
    }
    pub fn count(self: *Probe) usize {
        return self.shape.rows * width;
    }

    fn fp8(code: u8) f64 {
        const sign: f64 = if (code & 128 != 0) -1 else 1;
        const e = (code >> 3) & 15;
        const m = code & 7;
        if (e == 0) return sign * @as(f64, @floatFromInt(m)) * 0x1p-9;
        if (e == 15 and m == 7) return std.math.nan(f64);
        return sign * (1 + @as(f64, @floatFromInt(m)) / 8) * std.math.pow(f64, 2, @as(f64, @floatFromInt(e)) - 7);
    }
    fn key(self: *Probe, row: usize, slot: usize, dim: usize) ?f64 {
        if (slot < 128) {
            const position = 8191 + row + 1 - 128 + slot;
            const which: usize = if (position >= 8191) 2 else 1;
            const at = (if (which == 2) position - 8191 else position % 128) * ring_stride;
            const b = self.bytes(which);
            return fp8(b[at + dim]) * std.math.pow(f64, 2, @as(f64, @floatFromInt(b[at + width + dim / 32])) - 127);
        }
        const b = self.bytes(4);
        const off = (row * @max(1, self.shape.k - 128) + slot - 128) * 4;
        const index = std.mem.readInt(i32, b[off..][0..4], .little);
        if (index < 0 or index >= 513) return null;
        const at = @as(usize, @intCast(index)) * main_stride;
        const main = self.bytes(3);
        const code = (main[at + dim / 2] >> @as(u3, @intCast((dim % 2) * 4))) & 15;
        const lut = [_]f64{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };
        return (if (code & 8 != 0) -lut[code & 7] else lut[code & 7]) * fp8(main[at + 256 + dim / 16]);
    }

    pub fn reference(self: *Probe, want: []f64, mag: []f64) !void {
        @setFloatMode(.strict);
        const query = self.bytes(0);
        for (0..self.shape.rows) |row| {
            var acc: [width]f64 = @splat(0);
            var magnitude: [width]f64 = @splat(0);
            var high: f64 = -1e30;
            var denominator: f64 = 0;
            var base: usize = 0;
            while (base < self.shape.k) : (base += 64) {
                var scores: [64]f64 = @splat(-std.math.inf(f64));
                var next = high;
                for (0..@min(64, self.shape.k - base)) |j| {
                    if (self.key(row, base + j, 0) == null) continue;
                    var dot: f64 = 0;
                    for (0..width) |d| {
                        const q = c.fromBf16(std.mem.readInt(u16, query[(row * width + d) * 2 ..][0..2], .little));
                        dot += @as(f64, q) * self.key(row, base + j, d).?;
                    }
                    scores[j] = dot / @sqrt(@as(f64, width));
                    if (!std.math.isNan(scores[j])) next = @max(next, scores[j]);
                }
                const correction = @exp(high - next);
                for (&acc, &magnitude) |*v, *m| {
                    v.* *= correction;
                    m.* *= correction;
                }
                denominator *= correction;
                for (scores, 0..) |score, j| {
                    const probability = @exp(score - next);
                    denominator += probability;
                    if (base + j >= self.shape.k or self.key(row, base + j, 0) == null) continue;
                    const rounded: f64 = c.fromBf16(c.toBf16(@floatCast(probability)));
                    for (0..width) |d| {
                        const term = rounded * self.key(row, base + j, d).?;
                        acc[d] += term;
                        magnitude[d] += @abs(term);
                    }
                }
                high = next;
            }
            denominator += @exp(-high);
            for (0..width) |d| {
                want[row * width + d] = acc[d] / denominator;
                mag[row * width + d] = magnitude[d] / denominator;
            }
        }
    }
    pub fn simulate(self: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        var terms: [640]f64 = undefined;
        for (0..self.shape.rows) |row| for (0..width) |d| {
            var live: usize = 0;
            for (terms[0..self.shape.k], 0..) |*v, j| {
                const value = self.key(row, j, d);
                v.* = value orelse 0;
                live += @intFromBool(value != null);
            }
            out[row * width + d] = contract.rows.sumIn(acc, order, terms[0..self.shape.k]) / @as(f64, @floatFromInt(live + 1));
        };
    }
};
