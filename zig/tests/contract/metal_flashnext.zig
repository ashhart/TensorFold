//! Guarded mask and image/text M-RoPE probes using the production source and an independent scalar reference.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const f = tf.flashnext_vision_contract;
const c = contract.spec;
const g = contract.gates;

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    which: f.Which,
    factor: f64 = 0,
    shape: c.Shape = undefined,
    host: []u16 = &.{},
    positions: []i32 = &.{},
    bits: []u32 = &.{},
    weights: [3][256]f32 = .{ @splat(1), @splat(0.75), @splat(1.25) },
    input: rig_.Ref = undefined,
    pos: rig_.Ref = undefined,
    ws: [3]rig_.Ref = undefined,
    eps: rig_.Ref = undefined,
    base: rig_.Ref = undefined,
    out: [3]rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, which: f.Which) !Probe {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try rig_.pipeline(device, f.source(which), f.symbol(which)), .which = which };
    }
    pub fn deinit(p: *Probe) void {
        p.a.free(p.host);
        p.a.free(p.positions);
        p.a.free(p.bits);
        p.rig.deinit();
        p.pipe.deinit();
    }
    pub fn kernel(p: *Probe) g.Kernel {
        return .{ .ctx = p, .spec = f.specScaled(c, p.which, p.factor > 1), .vt = if (p.which == .mask) &mask_vt else &mrope_vt };
    }
    const mask_vt: g.VTable = blk: {
        var v = g.hooks(Probe);
        v.bounds = null;
        break :blk v;
    };
    const mrope_vt: g.VTable = blk: {
        var v = g.hooks(Probe);
        v.check = null;
        break :blk v;
    };
    pub fn check(p: *Probe, _: g.Gate) !?[]const u8 {
        const actual = p.rig.bytes(p.out[0], p.host.len * 2);
        const words = (p.shape.n + 31) / 32;
        for (p.host, 0..) |h, at| {
            const r = at / p.shape.n;
            const i = at % p.shape.n;
            const expected: u16 = if (((p.bits[r * words + i / 32] >> @intCast(i % 32)) & 1) != 0) h else 0xff80;
            if (std.mem.readInt(u16, actual[at * 2 ..][0..2], .little) != expected) return "mask changed allowed bits or retained a forbidden token";
        }
        return null;
    }
    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 128) return false;
        if (p.which != .mask and (shape.n != f.output_width or shape.k != 256)) return false;
        if (p.which == .mask and (shape.n == 0 or shape.n > 248320 or shape.k != 1)) return false;
        p.rig.reset();
        p.a.free(p.host);
        p.host = &.{};
        p.a.free(p.positions);
        p.positions = &.{};
        p.a.free(p.bits);
        p.bits = &.{};
        p.shape = shape;
        const width = if (p.which == .mask) shape.n else f.packed_width;
        p.host = try p.a.alloc(u16, shape.rows * width);
        for (p.host, 0..) |*h, i| h.* = c.toBf16(@as(f32, @floatFromInt((i +% seed) % 17)) / 8 - 1);
        if (fill == .nonfinite) for (0..shape.rows) |r| {
            p.host[r * width] = switch (r % 3) {
                0 => 0x7fc0,
                1 => 0x7f80,
                else => 0xff80,
            };
        };
        p.input = try p.rig.input(std.mem.sliceAsBytes(p.host));
        if (p.which == .mask) {
            const words = (shape.n + 31) / 32;
            p.bits = try p.a.alloc(u32, shape.rows * words);
            @memset(p.bits, 0);
            for (0..shape.rows) |r| for (0..shape.n) |i| if ((r + i) % 3 != 0) {
                p.bits[r * words + i / 32] |= @as(u32, 1) << @intCast(i % 32);
            };
            p.pos = try p.rig.input(std.mem.sliceAsBytes(p.bits));
            p.out[0] = try p.rig.output(p.host.len * 2);
            @memcpy(p.rig.bytes(p.out[0], p.host.len * 2), std.mem.sliceAsBytes(p.host));
        } else {
            const axes: usize = if (p.which == .mrope) 3 else 1;
            p.positions = try p.a.alloc(i32, shape.rows * axes);
            for (0..shape.rows) |r| for (0..axes) |axis| {
                p.positions[r * axes + axis] = @intCast((if (p.factor > 1) @as(usize, 262140) else 0) + r * 17 + (if (r % 2 == 0) @as(usize, 0) else axis * 3));
            };
            p.pos = try p.rig.input(std.mem.sliceAsBytes(p.positions));
            for (&p.ws, &p.weights, 0..) |*w, values, i| w.* = try p.rig.input(std.mem.sliceAsBytes(values[0..if (i == 2) 128 else 256]));
            const epsilon = [_]f32{1e-5};
            const base = f.table(23.253496170043945, p.factor);
            p.eps = try p.rig.input(std.mem.asBytes(&epsilon));
            p.base = try p.rig.input(std.mem.asBytes(&base));
            for (&p.out, [_]usize{ 24 * 256, 2 * 256, 4 * 128 }) |*o, n| o.* = try p.rig.output(shape.rows * n * 2);
        }
        return true;
    }
    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.pipe);
        if (p.which == .mask) {
            e.setBuffer(p.out[0].buf, p.out[0].off, 0);
            e.setBuffer(p.pos.buf, p.pos.off, 1);
            e.setValue([2]u32{ @intCast(p.shape.n), @intCast((p.shape.n + 31) / 32) }, 2);
            e.dispatchThreads(mtl.Size.of(p.shape.n, p.shape.rows, 1), mtl.Size.of(256, 1, 1));
        } else {
            e.setBuffer(p.input.buf, p.input.off, 0);
            e.setBuffer(p.pos.buf, p.pos.off, 1);
            for (p.ws, 2..) |w, i| e.setBuffer(w.buf, w.off, i);
            e.setBuffer(p.eps.buf, p.eps.off, 5);
            e.setBuffer(p.base.buf, p.base.off, 6);
            for (p.out, 7..) |o, i| e.setBuffer(o.buf, o.off, i);
            e.dispatchGroups(mtl.Size.of(1, 30, p.shape.rows), mtl.Size.of(256, 1, 1));
        }
        try p.rig.finish(pass);
    }
    pub fn rowCountInvariant(p: *Probe) !void {
        if (p.which == .mask or f.isPool(p.which)) return;
        var wide = try Probe.init(p.a, p.rig.device, p.rig.queue, p.which);
        defer wide.deinit();
        wide.factor = p.factor;
        var solo = try Probe.init(p.a, p.rig.device, p.rig.queue, p.which);
        defer solo.deinit();
        solo.factor = p.factor;
        for ([_]usize{ 7, if (p.which == .mtp) 16 else 128 }) |rows| {
            _ = try wide.prepare(.{ .rows = rows, .n = f.output_width, .k = 256 }, .random, 23);
            try wide.run();
            const batched = try p.a.dupe(u8, try wide.rig.outputBytes());
            defer p.a.free(batched);
            for (0..rows) |r| {
                _ = try solo.prepare(.{ .rows = 1, .n = f.output_width, .k = 256 }, .random, 23);
                @memcpy(solo.rig.bytes(solo.input, f.packed_width * 2), std.mem.sliceAsBytes(wide.host[r * f.packed_width ..][0..f.packed_width]));
                @memcpy(solo.rig.bytes(solo.pos, (if (p.which == .mrope) @as(usize, 3) else 1) * 4), std.mem.sliceAsBytes(wide.positions[r * (if (p.which == .mrope) @as(usize, 3) else 1) ..][0..if (p.which == .mrope) 3 else 1]));
                try solo.run();
                const one = try solo.rig.outputBytes();
                var block: usize = 0;
                var at: usize = 0;
                for ([_]usize{ 24 * 256, 2 * 256, 4 * 128 }) |n| {
                    if (!std.mem.eql(u8, one[at..][0 .. n * 2], batched[block + r * n * 2 ..][0 .. n * 2])) return error.RotaryRowCountChangesBits;
                    block += rows * n * 2;
                    at += n * 2;
                }
            }
        }
        std.debug.print("PASS shapes {s}: rows 1, 7 and {d} keep each row's bits\n", .{ f.name(p.which, p.factor > 1), if (p.which == .mtp) @as(usize, 16) else 128 });
    }
    pub fn count(p: *Probe) usize {
        return p.shape.rows * p.shape.n;
    }
    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        try p.compute(.f64, .sequential, want, mag);
    }
    pub fn bounds(p: *Probe, _: []const f64, _: []const f64, allow: []f64) !void {
        const rotary = f.table(23.253496170043945, p.factor);
        for (0..p.shape.rows) |r| {
            var row: [f.output_width]f64 = undefined;
            const pos = if (p.which == .mrope) p.positions[r * 3 ..][0..3].* else @as([3]i32, @splat(p.positions[r]));
            for (0..30) |head| {
                const kind: usize = if (head < 24) 0 else if (head < 26) 1 else 2;
                const width: usize = if (kind == 2) 128 else 256;
                const src: usize = if (kind == 0) head * 512 else if (kind == 1) 24 * 512 + (head - 24) * 256 else 24 * 512 + 4 * 256 + (head - 26) * 128;
                const dst: usize = if (head < 26) head * 256 else 26 * 256 + (head - 26) * 128;
                try f.normalizedBounds(p.host[r * f.packed_width + src ..][0..width], pos, p.weights[kind][0..width], 1e-5, &rotary, row[dst..][0..width]);
            }
            var src: usize = 0;
            var block: usize = 0;
            for ([_]usize{ 24 * 256, 2 * 256, 4 * 128 }) |n| {
                @memcpy(allow[block + r * n ..][0..n], row[src..][0..n]);
                src += n;
                block += p.shape.rows * n;
            }
        }
    }
    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        const mag = try p.a.alloc(f64, out.len);
        defer p.a.free(mag);
        try p.compute(switch (acc) {
            .f32 => .f32,
            .f16 => .f16,
            .bf16 => .bf16,
            .none => return error.NoSums,
        }, switch (order) {
            .sequential => .sequential,
            .pairwise => .pairwise,
            .strided => .strided,
        }, out, mag);
    }
    fn compute(p: *Probe, acc: f.Accum, order: f.Order, want: []f64, mag: []f64) !void {
        if (p.which == .mask) {
            const words = (p.shape.n + 31) / 32;
            for (want, mag, p.host, 0..) |*w, *m, h, at| {
                const r = at / p.shape.n;
                const i = at % p.shape.n;
                w.* = if (((p.bits[r * words + i / 32] >> @intCast(i % 32)) & 1) != 0) c.fromBf16(h) else -std.math.inf(f64);
                m.* = @abs(w.*);
            }
        } else {
            var row: [f.output_width]u16 = undefined;
            var row_mag: [f.output_width]f64 = undefined;
            for (0..p.shape.rows) |r| {
                const rotary = f.table(23.253496170043945, p.factor);
                f.referenceTableWith(p.host[r * f.packed_width ..][0..f.packed_width], if (p.which == .mrope) p.positions[r * 3 ..][0..3].* else @as([3]i32, @splat(p.positions[r])), .{ &p.weights[0], &p.weights[1], p.weights[2][0..128] }, 1e-5, &rotary, acc, order, &row, &row_mag);
                var src: usize = 0;
                var block: usize = 0;
                for ([_]usize{ 24 * 256, 2 * 256, 4 * 128 }) |n| {
                    for (0..n) |i| {
                        const at = block + r * n + i;
                        want[at] = c.fromBf16(row[src + i]);
                        mag[at] = row_mag[src + i];
                    }
                    src += n;
                    block += p.shape.rows * n;
                }
            }
        }
    }
};
