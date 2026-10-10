//! Compare complete GPU top-k bytes with generated CPU oracle rows.
const std = @import("std");
const mtl = @import("metal");
const cpu = @import("bf16_topk");
const gpu = @import("bf16_topk_gpu");
const Guarded = struct {
    buf: mtl.Buffer,
    off: usize,
    bytes: usize,
    fn init(device: mtl.Device, bytes: usize, off: usize) !Guarded {
        const buf = try device.buffer(off + bytes + 16, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        @memset(buf.contents()[0 .. off + bytes + 16], 0xa5);
        return .{ .buf = buf, .off = off, .bytes = bytes };
    }
    fn ref(b: Guarded) gpu.Ref {
        return .{ .buf = b.buf, .off = b.off };
    }
    fn data(b: Guarded) []u8 {
        return b.buf.contents()[b.off .. b.off + b.bytes];
    }
    fn check(b: Guarded) !void {
        for (b.buf.contents()[0..b.off]) |byte| if (byte != 0xa5) return error.BufferGuardChanged;
        for (b.buf.contents()[b.off + b.bytes ..][0..16]) |byte| if (byte != 0xa5) return error.BufferGuardChanged;
    }
};
const Runner = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    ops: gpu.Ops,
    cases: usize = 0,
    values: usize = 0,
    entries: usize = 0,
    rejected: usize = 0,
    fn finish(cb: mtl.CommandBuffer, e: mtl.ComputeEncoder) !void {
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.GpuFailure;
    }
    fn guards(r: *Runner) !void {
        const input = try Guarded.init(r.device, 256, 16);
        defer input.buf.deinit();
        const output = try Guarded.init(r.device, @sizeOf(cpu.Row), 16);
        defer output.buf.deinit();
        const cb = r.queue.commandBuffer();
        const e = cb.compute(.serial);
        var encoding = true;
        defer if (encoding) e.end();
        for ([_]cpu.Params{
            .{ .rows = 0, .vocab = 16, .k = 1, .stride = 16 },
            .{ .rows = 17, .vocab = 16, .k = 1, .stride = 16 },
            .{ .rows = 1, .vocab = 16, .k = 17, .stride = 16 },
            .{ .rows = 1, .vocab = 1, .k = 2, .stride = 1 },
            .{ .rows = 1, .vocab = 16, .k = 1, .stride = 15 },
        }) |p| try std.testing.expectError(error.BadTopKShape, r.ops.encode(e, input.ref(), output.ref(), p));
        const p = cpu.Params{ .rows = 1, .vocab = 16, .k = 1, .stride = 16 };
        try std.testing.expectError(error.BadTopKBuffer, r.ops.encode(e, .{ .buf = input.buf, .off = 1 }, output.ref(), p));
        try std.testing.expectError(error.BadTopKBuffer, r.ops.encode(e, input.ref(), .{ .buf = output.buf, .off = 2 }, p));
        try std.testing.expectError(error.BadTopKBuffer, r.ops.encode(e, .{ .buf = input.buf, .off = 258 }, output.ref(), p));
        try std.testing.expectError(error.BadTopKBuffer, r.ops.encode(e, input.ref(), .{ .buf = output.buf, .off = 36 }, p));
        try std.testing.expectError(error.TopKAlias, r.ops.encode(e, input.ref(), input.ref(), p));
        encoding = false;
        try finish(cb, e);
        try input.check();
        try output.check();
    }
    fn run(r: *Runner, rows: u32, vocab: u32, k: u32, seed: u32) !void {
        const p = cpu.Params{ .rows = rows, .vocab = vocab, .k = k, .stride = vocab + seed % 11 };
        const words = try r.a.alloc(u16, rows * p.stride);
        defer r.a.free(words);
        @memset(words, 0x7fc1);
        var expected: [16]cpu.Row = undefined;
        for (0..rows) |row| {
            const x = words[row * p.stride ..][0..vocab];
            const mode = (seed + row) % 12;
            for (x, 0..) |*word, token| {
                const mix = @as(u32, @truncate(token)) *% 1664525 +% seed *% 1013904223 +% @as(u32, @intCast(row)) *% 747796405;
                var raw: u16 = @truncate(mix >> 8);
                if (raw & 0x7f80 == 0x7f80) raw ^= 0x80;
                word.* = switch (mode) {
                    1 => 0x3f80,
                    2 => if (token % 2 == 0) 0x8000 else 0,
                    3 => @as(u16, @intCast(token % 128)) | (if (token % 3 == 0) @as(u16, 0x8000) else 0),
                    4 => @intCast(@as(u32, @bitCast(@as(f32, @floatFromInt(token)))) >> 16),
                    5 => if (token % 2 == 0) 0xff7f else 0x7f7f,
                    10 => ([_]u16{ 0x0080, 0x007f, 0x0001, 0x8001, 0x807f, 0x8080, 0, 0x8000 })[token % 8],
                    11 => blk: {
                        const bits: u16 = @truncate(token);
                        break :blk if (bits & 0x7f80 == 0x7f80) 0 else bits;
                    },
                    else => raw,
                };
            }
            if (mode >= 6 and mode <= 8) x[vocab - 1] = switch (mode) {
                6 => 0x7fc1,
                7 => 0x7f80,
                else => 0xff80,
            };
            if (mode == 9) {
                x[0] = 0x7f81;
                if (vocab > 1) x[1] = 0x7f80;
                if (vocab > 2) x[2] = 0xff80;
            }
            expected[row] = try cpu.oracle(x, k);
            if (expected[row].nonfinite != 0) r.rejected += 1 else r.entries += expected[row].count;
        }
        const input = try Guarded.init(r.device, words.len * 2, 2 * (1 + seed % 7));
        defer input.buf.deinit();
        @memcpy(input.data(), std.mem.sliceAsBytes(words));
        const output = try Guarded.init(r.device, rows * @sizeOf(cpu.Row), 4 * (1 + seed % 9));
        defer output.buf.deinit();
        const isolated = try Guarded.init(r.device, rows * @sizeOf(cpu.Row), 4);
        defer isolated.buf.deinit();
        const cb = r.queue.commandBuffer();
        const e = cb.compute(.serial);
        try r.ops.encode(e, input.ref(), output.ref(), p);
        const separate = rows > 1 and seed % 13 == 0;
        if (separate) {
            var single = p;
            single.rows = 1;
            for (0..rows) |row| try r.ops.encode(e, .{ .buf = input.buf, .off = input.off + row * p.stride * 2 }, .{ .buf = isolated.buf, .off = isolated.off + row * @sizeOf(cpu.Row) }, single);
        }
        try finish(cb, e);
        if (!std.mem.eql(u8, output.data(), std.mem.sliceAsBytes(expected[0..rows]))) {
            std.debug.print("top-k mismatch rows={d} vocab={d} k={d} seed={d}\n", .{ rows, vocab, k, seed });
            return error.TopKBytesDiffer;
        }
        if (separate and !std.mem.eql(u8, output.data(), isolated.data())) return error.TopKRowInvariant;
        const actual: [*]const cpu.Row = @ptrCast(@alignCast(output.data().ptr));
        for (actual[0..rows]) |*row| {
            if (row.nonfinite == 0) _ = try cpu.decode(row, p) else try std.testing.expectError(error.NonfiniteTopK, cpu.decode(row, p));
        }
        if (!std.mem.eql(u8, input.data(), std.mem.sliceAsBytes(words))) return error.TopKInputChanged;
        for ([_]Guarded{ input, output, isolated }) |buffer| try buffer.check();
        r.cases += 1;
        r.values += rows * vocab;
    }
};
pub fn main(init: std.process.Init) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const ops = try gpu.Ops.init(device);
    defer ops.deinit();
    var r = Runner{ .a = init.arena.allocator(), .device = device, .queue = queue, .ops = ops };
    try r.guards();
    for ([_]u32{ 1, 2, 3, 15, 16, 17, 31, 32, 255, 256, 257, 511, 513, 1025 }) |vocab| {
        for (1..17) |rows| for (1..@min(vocab, 16) + 1) |k| try r.run(@intCast(rows), vocab, @intCast(k), @intCast(rows * 19 + k * 23 + vocab));
    }
    for ([_]u32{ 65536, 248320 }) |vocab| for ([_]u32{ 1, 8, 16 }) |rows| for (1..17) |k| try r.run(rows, vocab, @intCast(k), @intCast(k * 13));
    std.debug.print("BF16 top-k: {d} cases, {d} values, {d} valid entries, {d} rejected rows; exact CPU bytes, row invariance and guards pass\n", .{ r.cases, r.values, r.entries, r.rejected });
}
