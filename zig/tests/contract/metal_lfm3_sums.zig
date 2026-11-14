//! FP32 taps of the production LFM3 square reductions, before their nonlinear BF16 outputs.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const ct = @import("contract");
const c = ct.spec;
const g = ct.gates;
const rig_ = @import("rig.zig");
pub const Kind = enum { norm, qk };

pub fn name(kind: Kind, bad: bool) []const u8 {
    return switch (kind) {
        inline else => |k| if (bad) "control: LFM3 sums " ++ @tagName(k) ++ " fp16" else "metal LFM3 sums " ++ @tagName(k),
    };
}
fn width(kind: Kind) usize {
    return switch (kind) {
        .norm => 2048,
        .qk => 128,
    };
}
fn threads(kind: Kind) usize {
    return if (kind == .norm) 256 else 32;
}
fn depth(_: c.Shape) usize {
    return 32;
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
    input: rig_.Ref = undefined,
    output: rig_.Ref = undefined,
    const vt = g.hooks(Probe);

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kind: Kind, bad: bool) !Probe {
        const body = sources.lfm3;
        const source = try std.mem.concat(a, u8, &.{ if (bad) "#define L3_SQUARE_ACC half\n" else "", body });
        defer a.free(source);
        const symbol = switch (kind) {
            .norm => "lfm3_norm_sums",
            .qk => "lfm3_qk_sums",
        };
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .kind = kind, .bad = bad, .pipeline = try rig_.pipeline(device, source, symbol), .served = .{ .{ .rows = 4, .n = 1, .k = width(kind) }, .{ .rows = 17, .n = 1, .k = width(kind) } } };
    }
    pub fn deinit(p: *Probe) void {
        p.a.free(p.data);
        p.rig.deinit();
        p.pipeline.deinit();
    }
    pub fn kernel(p: *Probe) g.Kernel {
        return .{ .ctx = p, .vt = &vt, .control = if (p.bad) .sums else null, .spec = .{ .name = name(p.kind, p.bad), .accum = .f32, .depth = depth, .out = .f32, .served = &p.served, .tiles = .{ .{ .size = 8 }, .{}, .{} } } };
    }
    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 17 or shape.k != width(p.kind) or shape.n != 1) return false;
        p.rig.reset();
        p.a.free(p.data);
        p.data = &.{};
        p.data = try p.a.alloc(u16, shape.rows * shape.k);
        p.shape = shape;
        p.stress = fill == .stress;
        var prng = std.Random.DefaultPrng.init(seed);
        const random = prng.random();
        for (p.data, 0..) |*value, i| {
            const col = i % shape.k;
            const magnitude: f32 = @floatFromInt(1 + (col * 37 + 11) % 63);
            value.* = c.toBf16(if (fill == .stress) (if (col % 2 == 0) -magnitude else magnitude) else random.float(f32) - 0.5);
        }
        if (fill == .nonfinite) {
            for ([_]u16{ 0x7fc0, 0x7f80, 0xff80, 0x7f80 }, 0..) |value, row| {
                if (row < shape.rows) p.data[row * shape.k] = value;
            }
            if (shape.rows > 3) p.data[3 * shape.k + 1] = 0xff80;
        }
        p.input = try p.rig.input(std.mem.sliceAsBytes(p.data));
        p.output = try p.rig.output(shape.rows * 4);
        return true;
    }
    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        pass.e.setPipeline(p.pipeline);
        pass.e.setBuffer(p.input.buf, p.input.off, 0);
        pass.e.setBuffer(p.output.buf, p.output.off, 1);
        pass.e.dispatchGroups(.of(p.shape.rows, 1, 1), .of(threads(p.kind), 1, 1));
        try p.rig.finish(pass);
        if (p.stress and !p.bad) {
            const bytes = try p.rig.outputBytes();
            var worst: f64 = 0;
            for (0..p.shape.rows) |row| {
                var want: f64 = 0;
                for (p.data[row * p.shape.k ..][0..p.shape.k]) |bits| {
                    const x: f64 = c.fromBf16(bits);
                    want += x * x;
                }
                worst = @max(worst, @abs(c.read(.f32, bytes, row) - want));
            }
            std.debug.print("FP64 {s}: stress square sums max_abs={d}\n", .{ name(p.kind, false), worst });
        }
    }
    pub fn count(p: *Probe) usize {
        return p.shape.rows;
    }
    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        for (want, mag, 0..) |*out, *magnitude, row| {
            out.* = 0;
            for (p.data[row * p.shape.k ..][0..p.shape.k]) |bits| {
                const x: f64 = c.fromBf16(bits);
                out.* += x * x;
            }
            magnitude.* = @abs(out.*);
        }
    }
    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        const terms = try p.a.alloc(f64, p.shape.k);
        defer p.a.free(terms);
        for (out, 0..) |*value, row| {
            for (terms, p.data[row * p.shape.k ..][0..p.shape.k]) |*term, bits| {
                const x: f64 = c.fromBf16(bits);
                term.* = x * x;
            }
            value.* = ct.rows.sumIn(acc, order, terms);
        }
    }
};
