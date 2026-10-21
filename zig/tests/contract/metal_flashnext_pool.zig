//! Both indexer pooling paths bind the full rotary table, with a nonzero absolute block and guarded outputs.
const std = @import("std");
const mtl = @import("metal");
const f = @import("tensorfold").flashnext_vision_contract;
const contract = @import("contract");
const c = contract.spec;
const g = contract.gates;
const rig_ = @import("rig.zig");

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    which: f.Which,
    factor: f64 = 0,
    long_positions: bool = false,
    calibration: bool = false,
    accuracy: ?*const f.rotary_accuracy.Receipt = null,
    before: ?struct { ctx: *anyopaque, call: *const fn (*anyopaque, mtl.ComputeEncoder) void } = null,
    shape: c.Shape = undefined,
    host: []u16 = &.{},
    raw: rig_.Ref = undefined,
    select: rig_.Ref = undefined,
    weights: rig_.Ref = undefined,
    eps: rig_.Ref = undefined,
    base: rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, which: f.Which) !Probe {
        std.debug.assert(f.isPool(which));
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try rig_.pipeline(device, f.source(which), f.symbol(which)), .which = which };
    }
    pub fn deinit(p: *Probe) void {
        p.a.free(p.host);
        p.rig.deinit();
        p.pipe.deinit();
    }
    fn prefix(p: *const Probe) usize {
        return if (p.which == .absolute) p.firstBlock() * 128 * 2 else 0;
    }
    pub fn firstBlock(p: *const Probe) usize {
        if (p.calibration) return 0;
        return if (p.long_positions) @import("flashnext_positions.zig").firstBlock(p.shape.rows) else 7;
    }
    pub fn kernel(p: *Probe) g.Kernel {
        return .{ .ctx = p, .spec = f.specScaled(c, p.which, p.factor > 1), .vt = &vt };
    }
    const vt: g.VTable = blk: {
        var v = g.hooks(Probe);
        v.output = output;
        v.intact = intact;
        break :blk v;
    };
    fn output(ctx: *anyopaque) ![]const u8 {
        const p: *Probe = @ptrCast(@alignCast(ctx));
        return (try p.rig.outputBytes())[p.prefix()..];
    }
    fn intact(ctx: *anyopaque) !bool {
        const p: *Probe = @ptrCast(@alignCast(ctx));
        if (!try p.rig.intact()) return false;
        for (p.rig.bytes(p.out, p.prefix())) |byte| if (byte != 0xff) return false;
        return true;
    }
    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 128 or shape.n != 128 or shape.k != 128) return false;
        p.shape = shape;
        const start = p.firstBlock();
        p.rig.reset();
        p.a.free(p.host);
        p.host = &.{};
        p.host = try p.a.alloc(u16, (start + shape.rows) * 4 * 128);
        if (p.long_positions) @memset(p.host, 0);
        const first = if (p.long_positions) start * 4 * 128 else 0;
        for (p.host[first..], first..) |*h, i| h.* = c.toBf16(@as(f32, @floatFromInt((i +% seed) % 17)) / 8 - 1);
        if (fill == .nonfinite) for (0..shape.rows) |r| {
            p.host[(start + r) * 4 * 128] = switch (r % 3) {
                0 => 0x7fc0,
                1 => 0x7f80,
                else => 0xff80,
            };
        };
        p.raw = try p.rig.input(std.mem.sliceAsBytes(p.host));
        var selection: [128]i32 = @splat(0);
        selection[0] = @intCast(start);
        selection[64] = @intCast(start);
        selection[65] = @intCast(shape.rows);
        p.select = try p.rig.input(std.mem.asBytes(&selection));
        const weights: [128]f32 = @splat(1.25);
        const eps = [_]f32{1e-5};
        const table = f.table(23.253496170043945, p.factor);
        p.weights = try p.rig.input(std.mem.asBytes(&weights));
        p.eps = try p.rig.input(std.mem.asBytes(&eps));
        p.base = try p.rig.input(std.mem.asBytes(&table));
        p.out = try p.rig.output(p.prefix() + shape.rows * 128 * 2);
        return true;
    }
    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        if (p.before) |hook| hook.call(hook.ctx, e);
        e.setPipeline(p.pipe);
        for ([_]rig_.Ref{ p.raw, p.select, p.weights, p.eps, p.base, p.out }, 0..) |b, i| e.setBuffer(b.buf, b.off, i);
        e.dispatchGroups(mtl.Size.of(1, p.shape.rows, 1), mtl.Size.of(128, 1, 1));
        try p.rig.finish(pass);
    }
    pub fn count(p: *Probe) usize {
        return p.shape.rows * 128;
    }
    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        try p.compute(.f64, .sequential, want, mag);
    }
    pub fn bounds(p: *Probe, want: []const f64, _: []const f64, allow: []f64) !void {
        const high = try p.a.alloc(f64, allow.len);
        defer p.a.free(high);
        try p.intervals(allow, high);
        for (allow, high, want) |*low, h, ref| low.* = @max(@abs(low.* - ref), @abs(h - ref));
    }
    pub fn intervals(p: *Probe, low: []f64, high: []f64) !void {
        const receipt = p.accuracy orelse return error.UnrecordedRotaryCompiler;
        const rotary = f.table(23.253496170043945, p.factor);
        if (!std.mem.eql(u8, p.rig.bytes(p.base, f.table_words * 4), std.mem.asBytes(&rotary))) return error.UnrecordedRotaryTable;
        const record = try receipt.find(if (p.which == .pooled) .pooled else .absolute, &rotary);
        const weights: [128]f32 = @splat(1.25);
        for (0..p.shape.rows) |r| {
            var row: [128]f.rotary_interval.Interval = undefined;
            try f.rotary_interval.pooled(record, p.host[(p.firstBlock() + r) * 512 ..][0..512], @intCast(4 * (p.firstBlock() + r)), &weights, 1e-5, &rotary, &row);
            for (row, 0..) |x, i| {
                low[r * 128 + i] = x.low;
                high[r * 128 + i] = x.high;
            }
        }
    }
    pub fn conditionalBounds(p: *Probe, _: []const f64, _: []const f64, allow: []f64) !void {
        const start = p.firstBlock();
        const weights: [128]f32 = @splat(1.25);
        const rotary = f.table(23.253496170043945, p.factor);
        for (0..p.shape.rows) |r| try f.pooledBounds(p.host[(start + r) * 4 * 128 ..][0 .. 4 * 128], @intCast(4 * (start + r)), &weights, 1e-5, &rotary, allow[r * 128 ..][0..128]);
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
        const start = p.firstBlock();
        const weights: [128]f32 = @splat(1.25);
        const rotary = f.table(23.253496170043945, p.factor);
        for (0..p.shape.rows) |r| {
            var mean: [128]u16 = undefined;
            for (&mean, 0..) |*h, d| {
                var total: f64 = c.fromBf16(p.host[(start + r) * 4 * 128 + d]);
                for (1..4) |k| total = f.round(total + c.fromBf16(p.host[((start + r) * 4 + k) * 128 + d]), acc);
                h.* = c.toBf16(@floatCast(total * 0.25));
            }
            var row: [128]u16 = undefined;
            f.normalizedRotation(&mean, @splat(@intCast(4 * (start + r))), &weights, 1e-5, &rotary, acc, order, &row, mag[r * 128 ..][0..128]);
            for (row, want[r * 128 ..][0..128]) |h, *value| value.* = c.fromBf16(h);
        }
    }
};
