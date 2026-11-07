//! Production attention dots and FP32 numerator taps, with exact integer cancellation witnesses.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const ct = @import("contract");
const c = ct.spec;
const g = ct.gates;
const rig_ = @import("rig.zig");
pub const Kind = enum { dot, full_values, sliding_values, full_merge, sliding_merge };
const selected = 4;

pub fn name(kind: Kind, bad: bool) []const u8 {
    return switch (kind) {
        inline else => |k| if (bad) "control: LFM3 attention sums " ++ @tagName(k) ++ " fp16" else "metal LFM3 attention sums " ++ @tagName(k),
    };
}
fn vision(_: Kind) bool {
    return false;
}
fn dot(kind: Kind) bool {
    return kind == .dot;
}
fn merge(kind: Kind) bool {
    return kind == .full_merge or kind == .sliding_merge;
}
fn sliding(kind: Kind) bool {
    return kind == .sliding_values or kind == .sliding_merge;
}
fn width(kind: Kind) usize {
    return if (vision(kind)) 64 else 128;
}
fn reduction(kind: Kind) usize {
    return if (merge(kind)) 12288 else if (dot(kind)) width(kind) else if (vision(kind)) 256 else 4095;
}
fn dotDepth(shape: c.Shape) usize {
    return (shape.k + 31) / 32 + 5;
}
fn valuesDepth(shape: c.Shape) usize {
    return 2 * @min(shape.k, 1024) + 16;
}
fn mergeDepth(shape: c.Shape) usize {
    return 2 * ((shape.k + shape.rows + 4095) / 4096) + 8;
}
fn hash(i: usize) usize {
    var v: u32 = @as(u32, @intCast(i + 1)) *% 0x9e3779b9;
    v = (v ^ (v >> 16)) *% 0x85ebca6b;
    return v ^ (v >> 13);
}

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    kind: Kind,
    bad: bool,
    stress: bool = false,
    pipeline: mtl.Pipeline,
    served: [2]c.Shape,
    shape: c.Shape = .{ .rows = 0, .n = 1, .k = 0 },
    data: []u16 = &.{},
    other: []u16 = &.{},
    bytes: []u8 = &.{},
    merge_values: []f32 = &.{},
    merge_totals: []f32 = &.{},
    input: rig_.Ref = undefined,
    keys: rig_.Ref = undefined,
    values: rig_.Ref = undefined,
    raw: rig_.Ref = undefined,
    maxima: rig_.Ref = undefined,
    totals: rig_.Ref = undefined,
    risk: rig_.Ref = undefined,
    const vt = blk: {
        var hooks = g.hooks(Probe);
        hooks.output = output;
        break :blk hooks;
    };

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kind: Kind, bad: bool) !Probe {
        const source = try std.mem.concat(a, u8, &.{ if (bad) "#define L3_ATTN_ACC half\n" else "", sources.lfm3_attention });
        defer a.free(source);
        const symbol = switch (kind) {
            .dot => "lfm3_attention_dot_sums",
            .full_values, .sliding_values => "lfm3_attn_partial",
            .full_merge, .sliding_merge => "lfm3_attention_merge_sums",
        };
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .kind = kind, .bad = bad, .pipeline = try rig_.pipeline(device, source, symbol), .served = .{ .{ .rows = 4, .n = if (dot(kind)) 1 else selected + 1, .k = reduction(kind) }, .{ .rows = 17, .n = if (dot(kind)) 1 else selected + 1, .k = reduction(kind) } } };
    }
    pub fn deinit(p: *Probe) void {
        p.a.free(p.data);
        p.a.free(p.other);
        p.a.free(p.bytes);
        p.a.free(p.merge_values);
        p.a.free(p.merge_totals);
        p.rig.deinit();
        p.pipeline.deinit();
    }
    pub fn kernel(p: *Probe) g.Kernel {
        return .{ .ctx = p, .vt = &vt, .control = if (p.bad) .sums else null, .spec = .{ .name = name(p.kind, p.bad), .accum = .f32, .depth = if (dot(p.kind)) dotDepth else if (merge(p.kind)) mergeDepth else valuesDepth, .out = .f32, .served = &p.served, .tiles = .{ .{ .size = 8 }, .{}, .{} } } };
    }
    fn cap(p: *const Probe) usize {
        return if (vision(p.kind)) p.shape.k else p.shape.k + p.shape.rows;
    }
    fn chunks(p: *const Probe) usize {
        return if (vision(p.kind)) 1 else (p.cap() + 4095) / 4096;
    }
    fn index(p: *const Probe, key: usize, ch: usize, head: usize) usize {
        return if (vision(p.kind)) (key * 8 + head) * 64 + ch else (head * p.cap() + key) * 128 + ch;
    }
    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 17 or shape.k != reduction(p.kind) or shape.n != (if (dot(p.kind)) @as(usize, 1) else selected + 1)) return false;
        p.rig.reset();
        p.a.free(p.data);
        p.a.free(p.other);
        p.a.free(p.bytes);
        p.data = &.{};
        p.other = &.{};
        p.bytes = &.{};
        p.shape = shape;
        p.stress = fill == .stress;
        if (merge(p.kind)) return p.prepareMerge(fill, seed);
        const dim = width(p.kind);
        const kv: usize = if (vision(p.kind)) 8 else 4;
        p.data = try p.a.alloc(u16, if (dot(p.kind)) shape.rows * dim else p.cap() * kv * dim);
        var prng = std.Random.DefaultPrng.init(seed);
        const random = prng.random();
        for (p.data) |*v| v.* = c.toBf16(random.float(f32) - 0.5);
        if (dot(p.kind)) {
            p.other = try p.a.alloc(u16, p.data.len);
            for (p.other) |*v| v.* = c.toBf16(random.float(f32) - 0.5);
            if (fill == .stress) for (0..shape.rows) |r| {
                for (0..dim) |i| {
                    const first = i % (dim / 2);
                    const x: f32 = @floatFromInt(64 * (1 + hash(first) % 128) + (if (i >= dim / 2 and first < 8) @as(usize, 64) else 0));
                    const w: f32 = @floatFromInt(1 + hash(first + 999) % 32);
                    p.data[r * dim + i] = c.toBf16(x);
                    p.other[r * dim + i] = c.toBf16(if (i < dim / 2) w else -w);
                }
            };
            if (fill == .nonfinite) {
                for ([_]u16{ 0x7fc0, 0x7f80, 0xff80, 0x7f80 }, 0..) |v, r| if (r < shape.rows) {
                    p.data[r * dim] = v;
                };
                if (shape.rows > 3) p.data[3 * dim + 1] = 0xff80;
            }
            p.input = try p.rig.input(std.mem.sliceAsBytes(p.data));
            p.keys = try p.rig.input(std.mem.sliceAsBytes(p.other));
            p.raw = try p.rig.output(shape.rows * 4);
            return true;
        }
        if (fill == .stress) {
            @memset(p.data, 0);
            const length: usize = if (p.kind == .full_values) 4096 else 256;
            const first_key: usize = if (p.kind == .sliding_values) 3840 else 0;
            for (0..length) |i| {
                const first = i % (length / 2);
                const scale: usize = if (length == 4096) 64 else 512;
                const value: f32 = @floatFromInt(scale * (1 + (if (length == 4096) (first * 7 + 3) % 32 else hash(first) % 128)));
                const nudge: f32 = if (first < 8) @floatFromInt(scale) else 0;
                for (0..kv) |h| for (0..dim) |ch| {
                    p.data[p.index(first_key + i, ch, h)] = c.toBf16(if (i < length / 2) value else -value + nudge);
                };
            }
        }
        if (fill == .nonfinite) {
            const first: usize = if (p.kind == .sliding_values) shape.k + 1 - 256 else 0;
            for (0..kv) |h| {
                for ([_]u16{ 0x7fc0, 0x7f80, 0xff80, 0x7f80 }, 0..) |v, ch| p.data[p.index(first, ch, h)] = v;
                p.data[p.index(first + 1, 3, h)] = 0xff80;
            }
        }
        const zero = try p.a.alloc(u16, @max(shape.rows * 16 * dim, p.data.len));
        defer p.a.free(zero);
        @memset(zero, 0);
        p.input = try p.rig.input(std.mem.sliceAsBytes(zero[0 .. shape.rows * 16 * dim]));
        p.keys = try p.rig.input(std.mem.sliceAsBytes(zero[0..p.data.len]));
        p.values = try p.rig.input(std.mem.sliceAsBytes(p.data));
        if (vision(p.kind)) {
            p.raw = try p.rig.output(shape.rows * 16 * 65 * 4);
        } else {
            p.risk = try p.rig.scratch(shape.rows * 16 * 64 * 4);
            p.maxima = try p.rig.scratch(shape.rows * 16 * 64 * 4);
            p.totals = try p.rig.scratch(shape.rows * 16 * 64 * 4);
            p.raw = try p.rig.output(shape.rows * 16 * 64 * 128 * 4);
        }
        p.bytes = try p.a.alloc(u8, p.count() * 4);
        return true;
    }
    fn prepareMerge(p: *Probe, fill: g.Fill, seed: u64) !bool {
        p.a.free(p.merge_values);
        p.a.free(p.merge_totals);
        p.merge_values = &.{};
        p.merge_totals = &.{};
        p.merge_values = try p.a.alloc(f32, p.shape.rows * 16 * 64 * 128);
        p.merge_totals = try p.a.alloc(f32, p.shape.rows * 16 * 64);
        @memset(p.merge_values, 0);
        @memset(p.merge_totals, 0);
        const maxima = try p.a.alloc(f32, p.merge_totals.len);
        defer p.a.free(maxima);
        @memset(maxima, 0);
        var prng = std.Random.DefaultPrng.init(seed);
        const random = prng.random();
        const first: usize = if (sliding(p.kind)) 2 else 0;
        for (0..p.shape.rows * 16) |row| for (first..4) |chunk| {
            const at = row * 64 + chunk;
            p.merge_totals[at] = if (chunk == first + 1) 2 else 1;
            for (0..128) |ch| {
                p.merge_values[at * 128 + ch] = if (fill == .stress)
                    (if (sliding(p.kind)) (if (chunk == 2) @as(f32, 65536) else -65535) else ([_]f32{ 65536, -65790, 254, 1 })[chunk])
                else
                    random.float(f32) - 0.5;
            }
            if (fill == .nonfinite and chunk == first) {
                for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), std.math.inf(f32) }, 0..) |v, ch| p.merge_values[at * 128 + ch] = v;
            }
            if (fill == .nonfinite and chunk == first + 1) p.merge_values[at * 128 + 3] = -std.math.inf(f32);
        };
        p.input = try p.rig.input(std.mem.sliceAsBytes(maxima));
        p.keys = try p.rig.input(std.mem.sliceAsBytes(p.merge_totals));
        p.values = try p.rig.input(std.mem.sliceAsBytes(p.merge_values));
        p.raw = try p.rig.output(p.shape.rows * 16 * 129 * 4);
        p.bytes = try p.a.alloc(u8, p.count() * 4);
        return true;
    }
    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.pipeline);
        e.setBuffer(p.input.buf, p.input.off, 0);
        e.setBuffer(p.keys.buf, p.keys.off, 1);
        if (dot(p.kind)) {
            e.setBuffer(p.raw.buf, p.raw.off, 2);
            e.dispatchGroups(.of(p.shape.rows, 1, 1), .of(32, 1, 1));
        } else {
            e.setBuffer(p.values.buf, p.values.off, 2);
            if (vision(p.kind)) {
                e.setBuffer(p.raw.buf, p.raw.off, 3);
                e.setValue(@as(u32, @intCast(p.cap())), 4);
                e.dispatchGroups(.of(16, p.shape.rows, 1), .of(32, 1, 1));
            } else {
                const args: @import("tensorfold").lfm3.kernels.AttentionArgs = .{ .len = @intCast(p.shape.k), .cap = @intCast(p.cap()), .scale = 1 / @sqrt(@as(f32, 128)), .window = if (sliding(p.kind)) 256 else 0 };
                e.setValue(args, 3);
                if (merge(p.kind)) {
                    e.setBuffer(p.raw.buf, p.raw.off, 4);
                    e.dispatchGroups(.of(16, p.shape.rows, 1), .of(32, 1, 1));
                } else {
                    e.setBuffer(p.maxima.buf, p.maxima.off, 4);
                    e.setBuffer(p.totals.buf, p.totals.off, 5);
                    e.setBuffer(p.raw.buf, p.raw.off, 6);
                    e.setBuffer(p.risk.buf, p.risk.off, 7);
                    e.dispatchGroups(.of(16, p.shape.rows, p.chunks()), .of(128, 1, 1));
                }
            }
        }
        try p.rig.finish(pass);
        if (p.stress and !p.bad) {
            const actual = try output(p);
            const wanted = try p.a.alloc(f64, p.count());
            defer p.a.free(wanted);
            const mag = try p.a.alloc(f64, p.count());
            defer p.a.free(mag);
            try p.reference(wanted, mag);
            var worst: f64 = 0;
            for (wanted, 0..) |want, i| worst = @max(worst, @abs(c.read(.f32, actual, i) - want));
            std.debug.print("FP64 {s}: stress max_abs={d}\n", .{ name(p.kind, false), worst });
        }
    }
    fn output(ctx: *anyopaque) ![]const u8 {
        const p: *Probe = @ptrCast(@alignCast(ctx));
        if (dot(p.kind)) return p.rig.outputBytes();
        const flat = vision(p.kind) or merge(p.kind);
        const dim = width(p.kind);
        const raw = p.rig.bytes(p.raw, if (flat) p.shape.rows * 16 * (dim + 1) * 4 else p.shape.rows * 16 * 64 * 128 * 4);
        const totals = if (flat) raw else p.rig.bytes(p.totals, p.shape.rows * 16 * 64 * 4);
        var at: usize = 0;
        for (0..p.shape.rows) |row| for (0..if (merge(p.kind)) @as(usize, 1) else p.chunks()) |chunk| {
            for (0..selected + 1) |ch| {
                const source = if (flat) ((row * 16) * (dim + 1) + (if (ch == selected) dim else ch)) * 4 else if (ch == selected) (row * 16 * 64 + chunk) * 4 else ((row * 16 * 64 + chunk) * 128 + ch) * 4;
                const bytes = if (ch == selected and !flat) totals else raw;
                @memcpy(p.bytes[at..][0..4], bytes[source..][0..4]);
                at += 4;
            }
        };
        return p.bytes;
    }
    pub fn count(p: *Probe) usize {
        return if (dot(p.kind)) p.shape.rows else p.shape.rows * (if (merge(p.kind)) @as(usize, 1) else p.chunks()) * (selected + 1);
    }
    fn interval(p: *const Probe, row: usize, chunk: usize) [2]usize {
        if (vision(p.kind)) return .{ 0, p.cap() };
        const count_ = p.shape.k + row + 1;
        const cutoff = if (p.kind == .sliding_values) count_ -| 256 else 0;
        const first = @max(chunk * 4096, cutoff);
        const end = @min((chunk + 1) * 4096, count_);
        return .{ @min(first, end), end };
    }
    fn evaluate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64, mag: ?[]f64) !void {
        const terms = try p.a.alloc(f64, @max(p.cap(), width(p.kind)));
        defer p.a.free(terms);
        for (out, 0..) |*value, i| {
            var n: usize = 0;
            if (merge(p.kind)) {
                const ch = i % (selected + 1);
                const row = i / (selected + 1);
                const first: usize = if (sliding(p.kind)) 2 else 0;
                n = 4 - first;
                for (terms[0..n], first..) |*term, chunk| {
                    const at = row * 16 * 64 + chunk;
                    term.* = if (ch == selected) p.merge_totals[at] else p.merge_values[at * 128 + ch];
                }
            } else if (dot(p.kind)) {
                n = width(p.kind);
                for (terms[0..n], p.data[i * n ..][0..n], p.other[i * n ..][0..n]) |*term, x, w| term.* = @as(f64, c.fromBf16(x)) * c.fromBf16(w);
            } else {
                const ch = i % (selected + 1);
                const item = i / (selected + 1);
                const row = item / p.chunks();
                const span = p.interval(row, item % p.chunks());
                n = span[1] - span[0];
                for (terms[0..n], span[0]..) |*term, key| term.* = if (ch == selected) 1 else c.fromBf16(p.data[p.index(key, ch, 0)]);
            }
            var sum: f64 = 0;
            var magnitude: f64 = 0;
            for (terms[0..n]) |term| {
                sum += term;
                magnitude += @abs(term);
            }
            value.* = if (acc == .none) sum else ct.rows.sumIn(acc, order, terms[0..n]);
            if (mag) |m| m[i] = magnitude;
        }
    }
    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        try p.evaluate(.none, .sequential, want, mag);
    }
    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        try p.evaluate(acc, order, out, null);
    }
};
