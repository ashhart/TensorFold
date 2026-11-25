//! GLM TFQ3 decode on Metal: the fused KDA input and dense GLU give a row the same bytes in 1-, 8- and 16-row windows.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const affine = tf.glm.affine;
const wide = tf.glm.wide;
const tfq3 = tf.glm.tfq3;
const testing = std.testing;

const rows: u32 = 16;
/// Window sizes a decode step runs: one 16-row window, two of 8, sixteen of 1.
const windows = [_]u32{ 16, 8, 1 };

fn shared(device: mtl.Device, bytes: usize) !mtl.Buffer {
    return device.buffer(@max(bytes, 16), mtl.ResourceOptions.shared);
}

/// Words a seeded hash, half scales 0.25, and every table the kernels read finite (as glm_tfq3_test's fillTables).
fn fill(w: mtl.Buffer, words: usize, s: mtl.Buffer, halfs: usize, seed: u32) void {
    for (w.slice(u32, words), 0..) |*x, i| x.* = @truncate((i *% 2654435761) ^ seed);
    for (s.slice(u16, halfs)) |*x| x.* = @bitCast(@as(f16, 0.25));
}

fn tables(device: mtl.Device) !mtl.Buffer {
    const b = try shared(device, 16384);
    const t = b.slice(u8, 16384);
    @memset(t, 0);
    std.mem.writeInt(u32, t[0..4], tfq3.magic_v3_tables, .little);
    std.mem.writeInt(u32, t[4..8], tfq3.format_version, .little);
    for ([_][2]usize{ .{ 64, 512 }, .{ 1088, 2048 }, .{ 5184, 2048 } }) |lut| for (0..lut[1]) |i| {
        const v: f16 = @as(f16, @floatFromInt(@as(i32, @intCast(i % 15)) - 7)) * 0.125;
        std.mem.writeInt(u16, t[lut[0] + 2 * i ..][0..2], @bitCast(v), .little);
    };
    for ([_]usize{ 9280, 9792, 10304 }) |off| for (0..64) |i| {
        const q = [4]f16{ if (i & 1 != 0) -1.0 else 1.0, if (i & 2 != 0) -1.0 else 1.0, 0.01, -0.01 };
        for (q, 0..) |v, j| std.mem.writeInt(u16, t[off + 8 * i + 2 * j ..][0..2], @bitCast(v), .little);
    };
    return b;
}

fn input(device: mtl.Device, k: u32) !mtl.Buffer {
    const x = try shared(device, @as(usize, rows) * k * 2);
    for (x.slice(u16, @as(usize, rows) * k), 0..) |*v, i| {
        const f: f32 = @as(f32, @floatFromInt(@as(i32, @intCast((i * 11) % 19)) - 9)) * 0.1;
        v.* = @intCast(@as(u32, @bitCast(f)) >> 16);
    }
    return x;
}

/// Every window size's output against the 16-row one: the first differing row and output named.
fn same(what: []const u8, bits: u8, ys: []const mtl.Buffer, width: usize) !void {
    const want = ys[0].slice(u16, rows * width);
    for (ys[1..], windows[1..]) |y, win| {
        const i = std.mem.indexOfDiff(u16, want, y.slice(u16, rows * width)) orelse continue;
        std.debug.print("TFQ3 {s}, {d}-bit: {d}-row windows differ from one 16-row window at row {d}, output {d}\n", .{ what, bits, win, i / width, i % width });
        return error.TestExpectedEqual;
    }
}

fn run(queue: anytype) !struct { cb: @TypeOf(queue.commandBuffer()), enc: mtl.ComputeEncoder } {
    const cb = queue.commandBuffer();
    return .{ .cb = cb, .enc = cb.compute(.serial) };
}

fn finish(r: anytype) !void {
    r.enc.end();
    r.cb.commit();
    r.cb.wait();
    try testing.expect(r.cb.failure() == null);
}

test "TFQ3 decode rows: the fused KDA input gives each row the same bytes in 1-, 8- and 16-row windows" {
    if (comptime !@hasDecl(mtl.Device, "init")) return error.SkipZigTest;
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    var pipes = try tfq3.Pipes.initUnits(testing.allocator, device, device.tensorUnits(), wide.glue_math);
    defer pipes.deinit();
    const n0: u32 = 24576;
    const k: u32 = 4096;
    const n1: u32 = 320; // the 8-bit/64 gate rows tryKdaIn fuses with the input rotation
    const tb = try tables(device);
    defer tb.deinit();
    const x = try input(device, k);
    defer x.deinit();
    const w1 = try shared(device, @as(usize, n1) * k);
    defer w1.deinit();
    const s1 = try shared(device, @as(usize, n1) * (k / 64) * 2);
    defer s1.deinit();
    const b1 = try shared(device, @as(usize, n1) * (k / 64) * 2);
    defer b1.deinit();
    fill(w1, @as(usize, n1) * k / 4, s1, @as(usize, n1) * (k / 64), 0x5A5A1234);
    for (b1.slice(u16, @as(usize, n1) * (k / 64))) |*v| v.* = @bitCast(@as(f16, -0.05));
    for ([_]u8{ 4, 5 }) |bits| {
        const words = @as(usize, n0) * k * bits / 32;
        const halfs = (n0 / 128) * (k / 128);
        const w0 = try shared(device, words * 4);
        defer w0.deinit();
        const s0 = try shared(device, halfs * 2);
        defer s0.deinit();
        fill(w0, words, s0, halfs, 0xA55A1234);
        var q: affine.Affine = .{ .w = .{ .buf = w0 }, .s = .{ .buf = s0 }, .b = .{ .buf = tb }, .n = n0 + n1, .k = k, .bits = bits, .group = 512, .more = 1 };
        q.runs[0] = .{ .w = .{ .buf = w1 }, .s = .{ .buf = s1 }, .b = .{ .buf = b1 }, .row = n0, .n = n1, .bits = 8, .group = 64 };
        var ys: [windows.len]mtl.Buffer = undefined;
        for (&ys) |*y| y.* = try shared(device, @as(usize, rows) * (n0 + n1) * 2);
        defer for (ys) |y| y.deinit();
        for (ys, windows) |y, win| {
            const r = try run(queue);
            var r0: u32 = 0;
            while (r0 < rows) : (r0 += win) {
                const in: affine.Ref = .{ .buf = x, .off = @as(usize, r0) * k * 2 };
                const out: affine.Ref = .{ .buf = y, .off = @as(usize, r0) * (n0 + n1) * 2 };
                try testing.expect(tfq3.dispatch.tryKdaIn(&pipes, r.enc, q, in, out, win, k, n0 + n1));
            }
            try finish(r);
        }
        try same("KDA input", bits, &ys, n0 + n1);
    }
}

test "TFQ3 decode rows: dense GLU gives each row the same bytes in 1-, 8- and 16-row windows" {
    if (comptime !@hasDecl(mtl.Device, "init")) return error.SkipZigTest;
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    var pipes = try tfq3.Pipes.initUnits(testing.allocator, device, device.tensorUnits(), wide.glue_math);
    defer pipes.deinit();
    const tb = try tables(device);
    defer tb.deinit();
    // gate and up of n rows each over k: the split-K kernel (2n < 8192) and the no-split one
    for ([_][2]u32{ .{ 128, 1024 }, .{ 6144, 4096 } }) |nk| for ([_]u8{ 4, 5 }) |bits| {
        const n = nk[0];
        const k = nk[1];
        const words = @as(usize, 2 * n) * k * bits / 32;
        const halfs = (2 * n / 128) * (k / 128);
        const w = try shared(device, words * 4);
        defer w.deinit();
        const s = try shared(device, halfs * 2);
        defer s.deinit();
        fill(w, words, s, halfs, 0x1234A55A);
        const x = try input(device, k);
        defer x.deinit();
        var ys: [windows.len]mtl.Buffer = undefined;
        for (&ys) |*y| y.* = try shared(device, @as(usize, rows) * n * 2);
        defer for (ys) |y| y.deinit();
        for (ys, windows) |y, win| {
            const r = try run(queue);
            var r0: u32 = 0;
            while (r0 < rows) : (r0 += win) {
                const in: affine.Ref = .{ .buf = x, .off = @as(usize, r0) * k * 2 };
                const out: affine.Ref = .{ .buf = y, .off = @as(usize, r0) * n * 2 };
                tfq3.dispatch.denseGlu(&pipes, r.enc, .{ .buf = w }, .{ .buf = s }, .{ .buf = tb }, bits, in, out, n, k, win, 7.0);
            }
            try finish(r);
        }
        try same("dense GLU", bits, &ys, n);
    };
}
