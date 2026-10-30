//! The MLX stream tile against its full-group factored float64 formula, all packed widths and served group sizes.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const Kernels = @FieldType(hip.Launcher, "affine");
const Arg = @typeInfo(@TypeOf(Kernels.streamGo)).@"fn".param_types[2].?;

pub const cases = blk: {
    const Case = struct { bits: u8, group: usize, name: []const u8 };
    var out: [18]Case = undefined;
    for ([_]u8{ 2, 3, 4, 5, 6, 8 }, 0..) |bits, b| {
        for ([_]usize{ 32, 64, 128 }, 0..) |group, i| {
            out[3 * b + i] = .{ .bits = bits, .group = group, .name = std.fmt.comptimePrint("hip affine_dot2_stream ({d}-bit, group {d}, fp32 out)", .{ bits, group }) };
        }
    }
    break :blk out;
};

fn depth(s: c.Shape) usize {
    return s.k / 2 + s.k / 32 + 8;
}

pub const Stream = struct {
    rig: rig_.Rig,
    l: *const hip.Launcher,
    arena: std.heap.ArenaAllocator,
    index: usize,
    fp16: bool,
    f: ?contract.affine.Affine = null,
    arg: Arg = undefined,

    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher, i: usize) Stream {
        const caps = hip.Caps.of(hip.kernels.arch);
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .l = l, .arena = .init(a), .index = i, .fp16 = if (caps) |v| v.act != .bf16 else false };
    }

    pub fn deinit(self: *Stream) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Stream) g.Kernel {
        const cfg = cases[self.index];
        return .{ .spec = .{
            .name = cfg.name,
            .accum = .f32,
            .depth = depth,
            .out = .f32,
            .served = if (cfg.bits == 4 and cfg.group == 64) &.{
                .{ .rows = 1, .n = 1856, .k = 2688 },
                .{ .rows = 2, .n = 3712, .k = 2688 },
            } else if (cfg.bits == 2) &.{
                .{ .rows = 1, .n = 32, .k = 384 },
                .{ .rows = 2, .n = 64, .k = 512 },
                .{ .rows = 1, .n = 4096, .k = 2048 },
            } else &.{
                .{ .rows = 1, .n = 32, .k = 384 },
                .{ .rows = 2, .n = 64, .k = 512 },
            },
            .tiles = .{ .{ .size = 4 }, .{ .size = 32 }, .{ .size = 1024, .step = cfg.group, .min = cfg.group } },
        }, .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Stream);

    pub fn prepare(self: *Stream, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const cfg = cases[self.index];
        if (shape.rows == 0 or shape.n == 0 or shape.k % cfg.group != 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        const f = try contract.affine.Affine.init(a, shape, cfg.bits, cfg.group);
        try f.fill(fill, seed);
        // Two-bit codes are 3/1 in the mirrored halves: fixed 255/256 inputs lose eight unit nudges in low precision.
        if (fill == .stress and cfg.bits == 2) for (0..shape.rows) |r| {
            for (0..shape.k) |i| {
                const nudge = i >= shape.k / 2 and i < shape.k / 2 + 8;
                f.x[r * shape.k + i] = c.toBf16(if (nudge) 256 else 255);
            }
        };
        // Table plants supplement every original x plant and leave ordinary columns finite in the last row.
        if (fill == .nonfinite) {
            const groups = shape.k / cfg.group;
            f.scales[3 * groups] = c.toBf16(std.math.nan(f32));
            f.biases[5 * groups + 1] = c.toBf16(std.math.inf(f32));
            f.scales[7 * groups + 2] = c.toBf16(-std.math.inf(f32));
        }
        if (self.fp16) for (f.x) |*v| {
            if (@abs(c.fromBf16(v.*)) < 0x1p-14) v.* = 0;
        };
        self.f = f;
        const xb = try a.alloc(u16, f.x.len);
        for (f.x, xb) |v, *out| {
            const value = c.fromBf16(v);
            out.* = if (self.fp16) @bitCast(@as(f16, @floatCast(value))) else v;
        }
        const words = try a.alloc(u32, shape.n * shape.k * cfg.bits / 32);
        f.words(words);
        self.arg = .{
            .x = try self.rig.input(std.mem.sliceAsBytes(xb)),
            .words = try self.rig.input(std.mem.sliceAsBytes(words)),
            .scale = .{ .p = try self.rig.input(std.mem.sliceAsBytes(f.scales)), .kind = 1 },
            .bias = .{ .p = try self.rig.input(std.mem.sliceAsBytes(f.biases)), .kind = 1 },
            .out = try self.rig.output(f.outputs() * 4),
            .m = @intCast(shape.rows),
            .n = @intCast(shape.n),
            .k = @intCast(shape.k),
            .bits = cfg.bits,
            .group = @intCast(cfg.group),
            .fp16 = @intFromBool(self.fp16),
        };
        return true;
    }

    pub fn run(self: *Stream) !void {
        try self.l.affine.streamGo(self.l.r, self.arg, .{}, self.arg.n, 1, self.rig.s.handle);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Stream) usize {
        return self.f.?.outputs();
    }

    pub fn reference(self: *Stream, want: []f64, mag: []f64) !void {
        try self.f.?.reference(want, mag);
    }

    pub fn simulate(self: *Stream, acc: c.Accum, order: g.Order, out: []f64) !void {
        try self.f.?.simulate(acc, order, out);
    }
};
