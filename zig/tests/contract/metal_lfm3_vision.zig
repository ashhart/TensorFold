//! Image-tower operators and placeholder copies exercise their production kernels with guarded BF16 buffers.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const ct = @import("contract");
const c = ct.spec;
const g = ct.gates;
const rig_ = @import("rig.zig");
const math = @import("lfm3_vision_math.zig");
pub const Kind = math.Kind;
fn depth(s: c.Shape) usize {
    return (s.k + 31) / 32 + 8;
}
fn attentionDepth(s: c.Shape) usize {
    return 5 * s.rows + 80;
}
pub fn name(kind: Kind) []const u8 {
    return switch (kind) {
        inline else => |k| "metal LFM3 vision " ++ @tagName(k),
    };
}
fn symbol(kind: Kind) []const u8 {
    return switch (kind) {
        .image_rows => "lfm3_image_rows",
        inline else => |k| "l3v_" ++ @tagName(k),
    };
}
fn dense(kind: Kind) bool {
    return kind == .dense or kind == .project;
}
fn inplace(kind: Kind) bool {
    return kind == .add or kind == .qk or kind == .gelu;
}

pub const Probe = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    rig: rig_.Rig,
    kind: Kind,
    pipeline: mtl.Pipeline,
    data: math.Data = undefined,
    args: [4]rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    key_out: rig_.Ref = undefined,
    const vt = g.hooks(Probe);
    const copy_vt = blk: {
        var hooks = vt;
        hooks.check = copyCheck;
        break :blk hooks;
    };
    fn copyCheck(ctx: *anyopaque, _: g.Gate) !?[]const u8 {
        const p: *Probe = @ptrCast(@alignCast(ctx));
        const actual = try p.rig.outputBytes();
        if (!std.mem.eql(u8, actual, std.mem.sliceAsBytes(p.data.x))) return "image rows changed BF16 bits";
        return null;
    }
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kind: Kind) !Probe {
        return .{ .a = a, .arena = .init(a), .rig = .{ .a = a, .device = device, .queue = queue }, .kind = kind, .pipeline = try rig_.pipeline(device, if (kind == .image_rows) sources.lfm3_image_rows else sources.lfm3_vision, symbol(kind)) };
    }
    pub fn deinit(p: *Probe) void {
        p.arena.deinit();
        p.rig.deinit();
        p.pipeline.deinit();
    }
    pub fn kernel(p: *Probe) g.Kernel {
        const shapes: []const c.Shape = switch (p.kind) {
            .dense => &.{ .{ .rows = 8, .n = 1024, .k = 1536 }, .{ .rows = 17, .n = 3072, .k = 1024 }, .{ .rows = 4, .n = 2560, .k = 1024 }, .{ .rows = 4, .n = 1024, .k = 2560 }, .{ .rows = 4, .n = 512, .k = 1024 }, .{ .rows = 4, .n = 1024, .k = 1024 } },
            .project => &.{ .{ .rows = 8, .n = 4096, .k = 4096 }, .{ .rows = 17, .n = 2048, .k = 4096 } },
            .swiglu => &.{ .{ .rows = 8, .n = 2560, .k = 2560 }, .{ .rows = 17, .n = 2560, .k = 2560 } },
            .conv, .qk, .attention => &.{ .{ .rows = 8, .n = 1024, .k = 64 }, .{ .rows = 16, .n = 1024, .k = 64 }, .{ .rows = 256, .n = 1024, .k = 64 } },
            .gelu => &.{ .{ .rows = 8, .n = 4096, .k = 4096 }, .{ .rows = 17, .n = 4096, .k = 4096 } },
            .image_rows => &.{ .{ .rows = 8, .n = 2048, .k = 2048 }, .{ .rows = 17, .n = 2048, .k = 2048 } },
            else => &.{ .{ .rows = 8, .n = 1024, .k = 1024 }, .{ .rows = 17, .n = 1024, .k = 1024 } },
        };
        const reduction = dense(p.kind) or p.kind == .rms or p.kind == .layer_norm or p.kind == .conv or p.kind == .qk or p.kind == .attention;
        return .{ .ctx = p, .vt = if (p.kind == .image_rows) &copy_vt else &vt, .spec = .{ .name = name(p.kind), .accum = if (reduction) .f32 else .none, .depth = if (p.kind == .attention) attentionDepth else depth, .ops = 64, .mids = if (p.kind == .rms or p.kind == .swiglu) 1 else if (p.kind == .conv or p.kind == .qk) 2 else 0, .out = .bf16, .in_place = inplace(p.kind), .served = shapes, .tiles = .{ .{ .size = 4, .step = if (p.kind == .conv or p.kind == .qk) 4 else 1, .min = if (p.kind == .conv or p.kind == .qk) 4 else 1 }, .{}, .{} }, .sums = if (dense(p.kind)) .{} else if (p.kind == .conv or p.kind == .layer_norm) .{ .exact = false } else .{ .f16 = 0, .bf16 = 0, .exact = false, .why = "nonlinear BF16 image nodes do not guarantee a 100x accumulator witness" } } };
    }
    fn values(a: std.mem.Allocator, count_: usize, seed: u64) ![]u16 {
        const data = try a.alloc(u16, count_);
        var prng = std.Random.DefaultPrng.init(seed);
        const random = prng.random();
        for (data) |*v| v.* = c.toBf16(random.float(f32) - 0.5);
        return data;
    }
    pub fn prepare(p: *Probe, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (s.rows == 0 or s.rows > 256 or ((p.kind == .conv or p.kind == .qk) and s.rows % 4 != 0)) return false;
        p.rig.reset();
        _ = p.arena.reset(.retain_capacity);
        const a = p.arena.allocator();
        const xlen = if (dense(p.kind)) s.rows * s.k else if (p.kind == .conv) s.rows * 3072 else s.rows * s.n;
        const wlen = if (dense(p.kind)) s.n * s.k else if (p.kind == .conv) 1024 * 9 else if (p.kind == .qk or p.kind == .attention) s.rows * 512 else s.rows * s.n;
        const zlen = if (dense(p.kind)) s.n else if (p.kind == .attention) s.rows * 512 else s.rows * s.n;
        const d: math.Data = .{ .kind = p.kind, .shape = s, .x = try values(a, xlen, seed), .w = try values(a, wlen, seed +% 1), .z = try values(a, zlen, seed +% 2), .gain = try values(a, 1024, seed +% 3) };
        for (d.gain) |*v| v.* = c.toBf16(1 + c.fromBf16(v.*) / 4);
        if (fill == .nonfinite) {
            const stride = if (dense(p.kind)) s.k else if (p.kind == .conv) @as(usize, 3072) else s.n;
            for ([_]u16{ 0x7fc0, 0x7f80, 0xff80, 0x7f80 }, 0..) |value, row| if (row < s.rows) {
                d.x[row * stride] = value;
            };
            if (s.rows > 3) d.x[3 * stride + 1] = 0xff80;
            if (p.kind == .image_rows) {
                d.x[5] = 0x7fa1;
                d.x[6] = 0xffc1;
                d.x[7] = 0x8000;
            }
        }
        if (fill == .stress) {
            if (dense(p.kind)) {
                var prng = std.Random.DefaultPrng.init(seed);
                const random = prng.random();
                for (0..s.rows) |r| for (0..s.k / 2) |i| {
                    const value: f32 = 64 * @as(f32, @floatFromInt(1 + random.uintLessThan(u32, 32)));
                    d.x[r * s.k + i] = c.toBf16(value);
                    d.x[r * s.k + s.k / 2 + i] = c.toBf16(value + if (i < 8) @as(f32, 64) else 0);
                };
                for (0..s.n) |col| for (0..s.k) |i| {
                    const value: f32 = @floatFromInt(1 + ((i % (s.k / 2)) * 7 + col) % 6);
                    d.w[col * s.k + i] = c.toBf16(if (i < s.k / 2) value else -value);
                };
                @memset(d.z, 0);
            } else if (p.kind == .layer_norm) {
                @memset(d.gain, c.toBf16(1));
                var mean: f64 = 0;
                for (0..1024) |i| {
                    const value: f32 = @floatFromInt(1 + (i * 37 + 11) % 63);
                    const x = if (i % 2 == 0) -value else value;
                    mean += x;
                    for (0..s.rows) |r| d.x[r * 1024 + i] = c.toBf16(x);
                }
                mean /= 1024;
                var squares: f64 = 0;
                for (d.x[0..1024]) |x| {
                    const centered = @as(f64, c.fromBf16(x)) - mean;
                    squares += centered * centered;
                }
                const inverse = 1 / @sqrt(squares / 1024 + 1e-6);
                for (d.x[0..1024], d.z[0..1024]) |x, *z| z.* = c.toBf16(@floatCast(-(@as(f64, c.fromBf16(x)) - mean) * inverse));
            } else if (p.kind == .conv) {
                @memset(d.x, 0);
                @memset(d.w, 0);
                for (0..s.rows) |row| for (0..1024) |ch| {
                    d.x[row * 3072 + 1024 + ch] = c.toBf16(1);
                    d.x[row * 3072 + 2048 + ch] = c.toBf16(1);
                };
                for (0..1024) |ch| {
                    for ([_]f32{ 65536, 255, 254, 1 }, [_]usize{ 4, 5, 7, 8 }, 0..) |bx_value, tap, row| {
                        d.x[row * 3072 + ch] = c.toBf16(bx_value);
                        d.w[ch * 9 + tap] = c.toBf16(if (row == 1) @as(f32, -258) else 1);
                    }
                }
            } else {
                for (d.x, 0..) |*v, i| v.* = c.toBf16(if (i % 4 < 2) 32 else -32);
                for (d.w, 0..) |*v, i| v.* = c.toBf16(if (i % 4 < 2) 2 else -2);
            }
        }
        p.data = d;
        for ([_][]const u16{ d.x, d.w, d.z, d.gain }, 0..) |data, i| p.args[i] = try p.rig.input(std.mem.sliceAsBytes(data));
        p.out = try p.rig.output(s.rows * s.n * 2);
        if (p.kind == .qk) p.key_out = try p.rig.output(s.rows * 512 * 2);
        return true;
    }
    fn bind(e: mtl.ComputeEncoder, refs: []const rig_.Ref) void {
        for (refs, 0..) |ref, i| e.setBuffer(ref.buf, ref.off, i);
    }
    pub fn run(p: *Probe) !void {
        const s = p.data.shape;
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.pipeline);
        switch (p.kind) {
            .dense, .project => {
                const args: extern struct { m: u32, n: u32, k: u32 } = .{ .m = @intCast(s.rows), .n = @intCast(s.n), .k = @intCast(s.k) };
                if (p.kind == .project) {
                    bind(e, &.{ p.args[0], p.args[1], p.args[2], p.out });
                    e.setValue(args, 4);
                } else {
                    bind(e, &.{ p.args[0], p.args[1], p.out });
                    e.setValue(args, 3);
                }
                e.dispatchGroups(.of((s.n + 3) / 4, (s.rows + 3) / 4, 1), .of(32, 1, 1));
            },
            .rms => {
                bind(e, &.{ p.args[0], p.args[3], p.out });
                e.dispatchGroups(.of(s.rows, 1, 1), .of(256, 1, 1));
            },
            .layer_norm => {
                bind(e, &.{ p.args[0], p.args[3], p.args[2], p.out });
                e.dispatchGroups(.of(s.rows, 1, 1), .of(256, 1, 1));
            },
            .conv => {
                bind(e, &.{ p.args[0], p.args[1], p.out });
                e.setValue(extern struct { gh: u32, gw: u32 }{ .gh = 2, .gw = @intCast(s.rows / 2) }, 3);
                e.dispatchThreads(.of(1024, s.rows, 1), .of(256, 1, 1));
            },
            .qk => {
                @memcpy(p.rig.bytes(p.out, p.data.x.len * 2), std.mem.sliceAsBytes(p.data.x));
                @memcpy(p.rig.bytes(p.key_out, p.data.w.len * 2), std.mem.sliceAsBytes(p.data.w));
                bind(e, &.{ p.out, p.key_out, p.args[3], p.args[3] });
                e.setValue(@as(u32, @intCast(s.rows / 2)), 4);
                e.dispatchGroups(.of(24, s.rows, 1), .of(32, 1, 1));
            },
            .attention => {
                bind(e, &.{ p.args[0], p.args[1], p.args[2], p.out });
                e.setValue(@as(u32, @intCast(s.rows)), 4);
                e.dispatchGroups(.of(16, s.rows, 1), .of(32, 1, 1));
            },
            .swiglu => {
                bind(e, &.{ p.args[0], p.args[1], p.out });
                e.setValue(@as(u32, @intCast(s.rows * s.n)), 3);
                e.dispatchThreads(.of(s.rows * s.n, 1, 1), .of(256, 1, 1));
            },
            .add => {
                @memcpy(p.rig.bytes(p.out, p.data.z.len * 2), std.mem.sliceAsBytes(p.data.z));
                bind(e, &.{ p.args[0], p.out });
                e.setValue(@as(u32, @intCast(s.rows * s.n)), 2);
                e.dispatchThreads(.of(s.rows * s.n, 1, 1), .of(256, 1, 1));
            },
            .gelu => {
                @memcpy(p.rig.bytes(p.out, p.data.x.len * 2), std.mem.sliceAsBytes(p.data.x));
                e.setBuffer(p.out.buf, p.out.off, 0);
                e.setValue(@as(u32, @intCast(s.rows * s.n)), 1);
                e.dispatchThreads(.of(s.rows * s.n, 1, 1), .of(256, 1, 1));
            },
            .image_rows => {
                bind(e, &.{ p.args[0], p.out });
                e.setValue(@as(u32, @intCast(s.rows * s.n)), 2);
                e.dispatchThreads(.of(s.rows * s.n, 1, 1), .of(256, 1, 1));
            },
        }
        try p.rig.finish(pass);
    }
    pub fn count(p: *Probe) usize {
        return p.data.shape.rows * (p.data.shape.n + if (p.kind == .qk) @as(usize, 512) else 0);
    }
    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        try math.evaluate(p.a, p.data, .none, .sequential, want, mag);
    }
    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        const mag = try p.a.alloc(f64, out.len);
        defer p.a.free(mag);
        try math.evaluate(p.a, p.data, acc, order, out, mag);
    }
};
