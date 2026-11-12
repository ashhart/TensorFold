//! Focused shared HIP operator, position and optional checkpoint upload checks for Parse.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const a = std.testing.allocator;
const bf = core.bf16_round;
const Tensor = hip.ops.Tensor;

const Rig = struct {
    r: hip.Runtime,
    ctx: hip.Context,
    s: hip.Stream,
    module: hip.Module,
    k: hip.vision.Kernels,

    fn init(t: *Rig) !void {
        t.r = try hip.Runtime.open();
        errdefer t.r.close();
        t.ctx = try hip.Context.init(&t.r, 0);
        errdefer t.ctx.deinit();
        t.s = try hip.Stream.init(&t.r);
        errdefer t.s.deinit();
        t.module = try hip.Module.load(&t.r, @import("parse_gpu").bytes);
        errdefer t.module.unload();
        t.k = try hip.vision.Kernels.load(t.module);
    }
    fn deinit(t: *Rig) void {
        t.s.synchronize() catch {};
        t.module.unload();
        t.s.deinit();
        t.ctx.deinit();
        t.r.close();
    }
};

const Buffer = struct {
    dev: hip.DeviceBuffer,
    host: []u16,
    count: usize,

    fn init(r: *const hip.Runtime, count: usize, seed: usize) !Buffer {
        const host = try a.alloc(u16, count + 64);
        errdefer a.free(host);
        @memset(host, 0xa5a5);
        for (host[32..][0..count], 0..) |*v, i| {
            const x: i32 = @intCast((i * 7 + seed) % 37);
            v.* = bf.bits(@as(f32, @floatFromInt(x - 18)) / 16);
        }
        return .{ .dev = try hip.DeviceBuffer.fromHost(r, std.mem.sliceAsBytes(host)), .host = host, .count = count };
    }
    fn free(b: *Buffer) void {
        b.dev.free();
        a.free(b.host);
    }
    fn tensor(b: Buffer) Tensor {
        return .{ .ptr = b.dev.base() + 64, .kind = .bf16 };
    }
    fn values(b: Buffer) []u16 {
        return b.host[32..][0..b.count];
    }
    fn read(b: Buffer) ![]u16 {
        const got = try a.alloc(u16, b.host.len);
        errdefer a.free(got);
        try b.dev.download(0, std.mem.sliceAsBytes(got));
        try std.testing.expectEqualSlices(u16, b.host[0..32], got[0..32]);
        try std.testing.expectEqualSlices(u16, b.host[b.count + 32 ..], got[b.count + 32 ..]);
        return got;
    }
    fn unchanged(b: Buffer) !void {
        const got = try b.read();
        defer a.free(got);
        try std.testing.expectEqualSlices(u16, b.host, got);
    }
};

test "Parse HIP dense projections match FP64 at tails and served widths, with five repeats and guards" {
    var t: Rig = undefined;
    try t.init();
    defer t.deinit();
    for ([_][3]usize{ .{ 1, 1, 1 }, .{ 15, 63, 19 }, .{ 16, 64, 16 }, .{ 17, 65, 1152 }, .{ 2, 1024, 4304 }, .{ 1, 4304, 1152 } }) |shape| {
        const m, const n, const k = .{ shape[0], shape[1], shape[2] };
        var x = try Buffer.init(&t.r, m * k, 3);
        defer x.free();
        var w = try Buffer.init(&t.r, n * k, 5);
        defer w.free();
        var bias = try Buffer.init(&t.r, n, 7);
        defer bias.free();
        var out = try Buffer.init(&t.r, m * n, 0);
        defer out.free();
        const want = try a.alloc(u16, m * n);
        defer a.free(want);
        for (0..m) |r| for (0..n) |col| {
            var sum: f64 = bf.value(bias.values()[col]);
            for (0..k) |j| sum += @as(f64, bf.value(x.values()[r * k + j])) * bf.value(w.values()[col * k + j]);
            want[r * n + col] = bf.bits(@floatCast(sum)); // dyadic fixture sums are exactly representable in FP32
        };
        for (0..5) |_| {
            try t.k.linear(t.s, x.tensor(), w.tensor().ptr, bias.tensor().ptr, out.tensor(), m, n, k);
            try t.s.synchronize();
            const got = try out.read();
            defer a.free(got);
            try std.testing.expectEqualSlices(u16, want, got[32..][0 .. m * n]);
        }
        try x.unchanged();
        try w.unchanged();
        try bias.unchanged();
    }
}

test "Parse HIP LayerNorm matches FP64 with BF16 output and propagates non-finite rows" {
    var t: Rig = undefined;
    try t.init();
    defer t.deinit();
    for ([_]usize{ 1, 255, 256, 257, 1024, 1152 }) |width| {
        var x = try Buffer.init(&t.r, 4 * width, 3);
        defer x.free();
        var w = try Buffer.init(&t.r, width, 6);
        defer w.free();
        var b = try Buffer.init(&t.r, width, 9);
        defer b.free();
        var out = try Buffer.init(&t.r, 4 * width, 0);
        defer out.free();
        for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }, 1..) |v, row|
            x.values()[row * width] = bf.bits(v);
        try x.dev.upload(0, std.mem.sliceAsBytes(x.host));
        try t.k.norm(t.s, x.tensor(), w.tensor().ptr, b.tensor().ptr, out.tensor(), 4, width, 1e-6);
        try t.s.synchronize();
        const got = try out.read();
        defer a.free(got);
        var mean: f64 = 0;
        for (x.values()[0..width]) |v| mean += bf.value(v);
        mean /= @floatFromInt(width);
        var variance: f64 = 0;
        for (x.values()[0..width]) |v| variance += std.math.pow(f64, @as(f64, bf.value(v)) - mean, 2);
        variance /= @floatFromInt(width);
        for (0..width) |j| {
            const want = (@as(f64, bf.value(x.values()[j])) - mean) / @sqrt(variance + 1e-6) * bf.value(w.values()[j]) + bf.value(b.values()[j]);
            const actual: f64 = bf.value(got[32 + j]);
            try std.testing.expect(@abs(actual - want) <= @abs(want) / 256 + 2e-6);
        }
        for (got[32 + width ..][0 .. 3 * width]) |v| try std.testing.expect(std.math.isNan(bf.value(v)));
        try x.unchanged();
        try w.unchanged();
        try b.unchanged();
    }
}

test "Parse HIP horizontal neck and attention layouts preserve every element" {
    var t: Rig = undefined;
    try t.init();
    defer t.deinit();
    const rows = 12;
    const heads = 3;
    const dim = 7;
    var x = try Buffer.init(&t.r, rows * heads * dim, 3);
    defer x.free();
    var out = try Buffer.init(&t.r, rows * heads * dim, 0);
    defer out.free();
    for ([_]hip.vision.Kernels.Layout{ .heads_to_rows, .horizontal_four }) |mode| {
        try t.k.layout(t.s, x.tensor(), out.tensor(), rows, if (mode == .heads_to_rows) heads else 1, if (mode == .heads_to_rows) dim else heads * dim, mode);
        try t.s.synchronize();
        const got = try out.read();
        defer a.free(got);
        for (0..x.count) |i| {
            const from = if (mode == .heads_to_rows)
                ((i / dim % heads) * rows + i / (heads * dim)) * dim + i % dim
            else
                (i / (4 * heads * dim) * 4 + i % 4) * heads * dim + i / 4 % (heads * dim);
            try std.testing.expectEqual(x.values()[from], got[32 + i]);
        }
    }
    try x.unchanged();
}

extern "c" fn erf(x: f64) f64;

test "Parse HIP GELU, residual and embedding scale retain guards and repeat bytes" {
    var t: Rig = undefined;
    try t.init();
    defer t.deinit();
    const count = 1027;
    var x = try Buffer.init(&t.r, count, 3);
    defer x.free();
    var y = try Buffer.init(&t.r, count, 9);
    defer y.free();
    var out = try Buffer.init(&t.r, count, 0);
    defer out.free();
    for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }, 0..) |v, i|
        x.values()[i] = bf.bits(v);
    try x.dev.upload(0, std.mem.sliceAsBytes(x.host));
    for ([_]hip.vision.Kernels.Point{ .gelu, .add, .scale }) |mode| {
        var first: ?[]u16 = null;
        defer if (first) |v| a.free(v);
        for (0..5) |_| {
            try t.k.point(t.s, x.tensor(), y.tensor(), out.tensor(), count, mode, 32);
            try t.s.synchronize();
            const got = try out.read();
            defer a.free(got);
            if (first) |v| try std.testing.expectEqualSlices(u16, v, got) else first = try a.dupe(u16, got);
            for (x.values(), y.values(), got[32..][0..count]) |xv, yv, actual| {
                const v: f64 = bf.value(xv);
                const want = switch (mode) {
                    .gelu => (0.5 * v) * (1 + erf(v / @sqrt(@as(f64, 2)))),
                    .add => v + bf.value(yv),
                    .scale => v * 32,
                };
                const value: f64 = bf.value(actual);
                if (std.math.isNan(want)) {
                    try std.testing.expect(std.math.isNan(value));
                } else if (std.math.isInf(want)) {
                    try std.testing.expectEqual(want, value);
                } else {
                    try std.testing.expect(@abs(value - want) <= @abs(want) / 256 + 2e-7);
                }
            }
        }
    }
    try x.unchanged();
    try y.unchanged();
}

test "Parse HIP uploads the actual FP32 checkpoint as owned BF16 weights (TF_PARSE_DIR)" {
    const dir = @import("sys").getenv("TF_PARSE_DIR") orelse return error.SkipZigTest;
    var t: Rig = undefined;
    try t.init();
    defer t.deinit();
    const before = hip.usage(false).device;
    var w = try @import("weights.zig").Weights.load(a, &t.r, std.testing.io, dir);
    var live = true;
    defer if (live) w.deinit();
    try std.testing.expectEqual(@as(usize, 509), w.names.count());
    try std.testing.expectEqual(@as(usize, 1261601004), w.bytes);
    var ck = try @import("checkpoint.zig").Checkpoint.open(a, std.testing.io, dir);
    defer ck.close();
    const table = try ck.get("decoder.embed_tokens.weight");
    var got: [32]u16 = undefined;
    try w.names.get("decoder.embed_tokens.weight").?.download(0, std.mem.sliceAsBytes(&got));
    for (got, 0..) |value, i| {
        const bits = std.mem.readInt(u32, table.bytes[i * 4 ..][0..4], .little);
        try std.testing.expectEqual(bf.bits(@bitCast(bits)), value);
    }
    try std.testing.expectEqual(before + w.bytes, hip.usage(false).device);
    {
        const ids = [_]i32{ 0, 1, 2, 72511 };
        var dev_ids = try hip.DeviceBuffer.fromHost(&t.r, std.mem.sliceAsBytes(&ids));
        defer dev_ids.free();
        var out = try hip.DeviceBuffer.alloc(&t.r, ids.len * 1024 * 2);
        defer out.free();
        try t.k.embed(t.s, try w.get("decoder.embed_tokens.weight"), dev_ids.base(), .{ .ptr = out.base(), .kind = .bf16 }, ids.len, 1024, 32);
        try t.s.synchronize();
        var values: [4 * 1024]u16 = undefined;
        try out.download(0, std.mem.sliceAsBytes(&values));
        for (ids, 0..) |id, row| for (0..1024) |j| {
            const at = (@as(usize, @intCast(id)) * 1024 + j) * 4;
            const bits = std.mem.readInt(u32, table.bytes[at..][0..4], .little);
            const rounded = bf.round32(@bitCast(bits));
            try std.testing.expectEqual(bf.bits(rounded * 32), values[row * 1024 + j]);
        };
    }

    w.deinit();
    live = false;
    try std.testing.expectEqual(before, hip.usage(false).device);
}

test {
    _ = @import("hip_position_test.zig");
}

test "Parse decoder refuses invalid contexts and suffix growth before GPU access" {
    const decoder = @import("decoder.zig");
    const r: hip.Runtime = undefined;
    try std.testing.expectError(error.BadContext, decoder.State.init(&r, 0));
    try std.testing.expectError(error.BadContext, decoder.State.init(&r, 16385));
    var state: decoder.State = .{ .layers = undefined, .hidden = undefined, .total = 128, .len = 65 };
    try state.keep(63);
    try std.testing.expectEqual(@as(usize, 63), state.len);
    try std.testing.expectError(error.BadKeep, state.keep(64));
    try std.testing.expectError(error.BadPosition, state.hiddenAt(63));
    try std.testing.expectError(error.ContextExceeded, decoder.rowsOf(&.{}));
    try std.testing.expectError(error.DuplicateState, decoder.rowsOf(&.{ .{ .state = &state, .rows = 1 }, .{ .state = &state, .rows = 1 } }));
    const compute: @import("compute.zig").Compute = undefined;
    try std.testing.expectError(error.ContextExceeded, decoder.forward(compute, &state, undefined, 66));
    try std.testing.expectError(error.ContextExceeded, decoder.forward(compute, &state, undefined, 0));
}

test "Parse HIP linear matches FP64 classes for non-finite products and preserves finite dots" {
    var t: Rig = undefined;
    try t.init();
    defer t.deinit();
    const rows = 5;
    const n = 65;
    const width = 1025;
    var x = try Buffer.init(&t.r, rows * width, 0);
    defer x.free();
    var w = try Buffer.init(&t.r, n * width, 3);
    defer w.free();
    var bias = try Buffer.init(&t.r, n, 7);
    defer bias.free();
    var out = try Buffer.init(&t.r, rows * n, 0);
    defer out.free();
    var rng: std.Random.DefaultPrng = .init(42);
    const random = rng.random();
    for (x.values()) |*v| v.* = bf.bits(std.math.ldexp(random.float(f32) * 2 - 1, @as(i32, @intCast(random.uintLessThan(u8, 17))) - 8));
    for (w.values()) |*v| v.* = bf.bits(std.math.ldexp(random.float(f32) * 2 - 1, @as(i32, @intCast(random.uintLessThan(u8, 17))) - 8));
    x.values()[width + 3] = bf.bits(std.math.nan(f32));
    x.values()[2 * width + 5] = bf.bits(std.math.inf(f32));
    x.values()[3 * width + 7] = bf.bits(-std.math.inf(f32));
    x.values()[4 * width + 9] = bf.bits(std.math.inf(f32));
    x.values()[4 * width + 11] = bf.bits(-std.math.inf(f32));
    try x.dev.upload(0, std.mem.sliceAsBytes(x.host));
    try w.dev.upload(0, std.mem.sliceAsBytes(w.host));
    try t.k.linear(t.s, x.tensor(), w.tensor().ptr, bias.tensor().ptr, out.tensor(), rows, n, width);
    try t.s.synchronize();
    const got = try out.read();
    defer a.free(got);
    const contract = core.contract.spec;
    const spec: contract.Spec = .{ .name = "linear probe", .accum = .f32, .depth = struct {
        fn f(shape: contract.Shape) usize {
            return shape.k * 4 + 1;
        }
    }.f, .out = .bf16, .served = &.{} };
    for (0..rows) |row| for (0..n) |col| {
        var sum: f64 = bf.value(bias.values()[col]);
        var magnitude = @abs(sum);
        for (0..width) |j| {
            const term = @as(f64, bf.value(x.values()[row * width + j])) * bf.value(w.values()[col * width + j]);
            sum += term;
            magnitude += @abs(term);
        }
        const actual: f64 = bf.value(got[32 + row * n + col]);
        try std.testing.expectEqual(contract.classOf(sum), contract.classOf(actual));
        if (std.math.isFinite(sum)) try std.testing.expect(@abs(actual - sum) <=
            contract.bound(spec, .{ .rows = rows, .n = n, .k = width }, sum, magnitude));
    };
    try x.unchanged();
    try w.unchanged();
    try bias.unchanged();
}
