//! Generated GPU tree rounds compare exactly with an independent CPU oracle.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("tree_round");
const gpu = @import("tree_round_gpu");
const Guarded = struct {
    buf: mtl.Buffer,
    bytes: usize,
    fn init(device: mtl.Device, payload: []const u8) !Guarded {
        const buf = try device.buffer(payload.len + 32, mtl.ResourceOptions.shared);
        @memset(buf.contents()[0 .. payload.len + 32], 0xa5);
        @memcpy(buf.contents()[16 .. 16 + payload.len], payload);
        return .{ .buf = buf, .bytes = payload.len };
    }
    fn empty(device: mtl.Device, bytes: usize) !Guarded {
        const buf = try device.buffer(bytes + 32, mtl.ResourceOptions.shared);
        @memset(buf.contents()[0 .. bytes + 32], 0xa5);
        return .{ .buf = buf, .bytes = bytes };
    }
    fn ref(b: Guarded) gpu.Ref {
        return .{ .buf = b.buf, .off = 16 };
    }
    fn data(b: Guarded) []u8 {
        return b.buf.contents()[16 .. 16 + b.bytes];
    }
    fn check(b: Guarded) !void {
        for (b.buf.contents()[0..16]) |byte| if (byte != 0xa5) return error.BufferGuardChanged;
        for (b.buf.contents()[16 + b.bytes ..][0..16]) |byte| if (byte != 0xa5) return error.BufferGuardChanged;
    }
};
const Runner = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    ops: gpu.Ops,
    picks: usize = 0,
    rounds: usize = 0,
    values: usize = 0,
    fn guards(r: *Runner) !void {
        const input = try Guarded.empty(r.device, 128);
        defer input.buf.deinit();
        const output = try Guarded.empty(r.device, @sizeOf(contract.Result));
        defer output.buf.deinit();
        const cb = r.queue.commandBuffer();
        const e = cb.compute(.serial);
        try std.testing.expectError(error.BadRoundShape, r.ops.argmax(e, input.ref(), output.ref(), .{ .rows = 17, .vocab = 1, .stride = 1 }));
        try std.testing.expectError(error.BadRoundBuffer, r.ops.argmax(e, .{ .buf = input.buf, .off = 17 }, output.ref(), .{ .rows = 1, .vocab = 16, .stride = 16 }));
        try std.testing.expectError(error.BadRoundBuffer, r.ops.argmax(e, input.ref(), .{ .buf = output.buf, .off = 18 }, .{ .rows = 1, .vocab = 16, .stride = 16 }));
        try std.testing.expectError(error.BadRoundBuffer, r.ops.argmax(e, input.ref(), output.ref(), .{ .rows = 1, .vocab = 128, .stride = 128 }));
        try std.testing.expectError(error.RoundAlias, r.ops.argmax(e, input.ref(), input.ref(), .{ .rows = 1, .vocab = 16, .stride = 16 }));
        try std.testing.expectError(error.BadRoundShape, r.ops.match(e, input.ref(), input.ref(), input.ref(), null, output.ref(), .{ .rows = 1, .vocab = 16, .budget = 1, .eos_count = 1 }));
        try std.testing.expectError(error.BadRoundBuffer, r.ops.match(e, input.ref(), input.ref(), input.ref(), null, input.ref(), .{ .rows = 1, .vocab = 16, .budget = 1 }));
        try std.testing.expectError(error.RoundAlias, r.ops.match(e, output.ref(), input.ref(), input.ref(), null, output.ref(), .{ .rows = 1, .vocab = 16, .budget = 1 }));
        try finish(cb, e);
        try input.check();
        try output.check();
    }
    fn finish(cb: mtl.CommandBuffer, e: mtl.ComputeEncoder) !void {
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.GpuFailure;
    }
    fn argmax(r: *Runner, rows: u32, vocab: u32, mode: u32) !void {
        const stride = vocab + 7;
        const words = try r.a.alloc(u16, rows * stride);
        defer r.a.free(words);
        @memset(words, 0x7fc1);
        var expected: [16]contract.Pick = undefined;
        for (0..rows) |row| {
            const x = words[row * stride ..][0..vocab];
            for (x, 0..) |*word, token| {
                const value: f32 = @floatFromInt(@as(i32, @intCast((token * 17 + row * 13) % 65)) - 32);
                word.* = @intCast(@as(u32, @bitCast(value)) >> 16);
            }
            const first = @min(vocab - 1, (row * 31) % vocab);
            x[first] = 0x42c8;
            x[vocab - 1] = 0x42c8;
            if (mode == 1) {
                @memset(x, 0x8000);
                x[vocab - 1] = 0;
            } else if (mode == 6) {
                x[first] = 0xc040;
                x[vocab - 1] = 0x42c8;
            } else if (mode == 7) {
                @memset(x, 0xff7f);
                x[first] = 0x7f7f;
                x[vocab - 1] = 0x7f7f;
            } else if (mode > 1) {
                const bad: u16 = switch (mode) {
                    2 => 0x7fc1,
                    3 => 0x7f80,
                    4 => 0xff80,
                    else => 0x7f81,
                };
                x[(row * 37) % vocab] = bad;
                if (mode == 5 and vocab >= 3) {
                    x[0] = 0x7fc1;
                    x[1] = 0x7f80;
                    x[2] = 0xff80;
                }
            }
            expected[row] = contract.rowArgmax(x);
        }
        const input = try Guarded.init(r.device, std.mem.sliceAsBytes(words));
        defer input.buf.deinit();
        const output = try Guarded.empty(r.device, rows * @sizeOf(contract.Pick));
        defer output.buf.deinit();
        const cb = r.queue.commandBuffer();
        const e = cb.compute(.serial);
        try r.ops.argmax(e, input.ref(), output.ref(), .{ .rows = rows, .vocab = vocab, .stride = stride });
        try finish(cb, e);
        if (!std.mem.eql(u8, output.data(), std.mem.sliceAsBytes(expected[0..rows]))) return error.ArgmaxMismatch;
        try input.check();
        try output.check();
        r.picks += rows;
        r.values += rows * vocab;
    }
    fn round(r: *Runner, tokens: []const u32, parents: []const i32, wants: []const u32, budget: u32, eos: []const u32, mode: u32) !void {
        const vocab: u32 = 64;
        const rows: u32 = @intCast(tokens.len);
        const p = contract.Match{ .rows = rows, .vocab = vocab, .budget = budget, .eos_count = @intCast(eos.len) };
        var words: [16 * 67]u16 = @splat(0xc080);
        var picks: [16]contract.Pick = undefined;
        for (0..rows) |row| {
            const x = words[row * 67 ..][0..vocab];
            x[wants[row]] = 0x4080;
            if (mode >= 1 and mode <= 3 and row == rows - 1) x[63] = switch (mode) {
                1 => 0x7fc1,
                2 => 0x7f80,
                else => 0xff80,
            };
            picks[row] = contract.rowArgmax(x);
        }
        if (mode == 4) picks[rows - 1].token = vocab;
        if (mode == 5) picks[rows - 1].nonfinite = 8;
        const expected = try contract.oracle(p, tokens, parents, picks[0..rows], eos);
        _ = try contract.decode(&expected, p);
        const input = try Guarded.init(r.device, std.mem.sliceAsBytes(words[0 .. rows * 67]));
        defer input.buf.deinit();
        const ids = try Guarded.init(r.device, std.mem.sliceAsBytes(tokens));
        defer ids.buf.deinit();
        const pars = try Guarded.init(r.device, std.mem.sliceAsBytes(parents));
        defer pars.buf.deinit();
        const picked = try Guarded.init(r.device, std.mem.sliceAsBytes(picks[0..rows]));
        defer picked.buf.deinit();
        const stops = try Guarded.init(r.device, std.mem.sliceAsBytes(eos));
        defer stops.buf.deinit();
        const output = try Guarded.empty(r.device, @sizeOf(contract.Result));
        defer output.buf.deinit();
        const cb = r.queue.commandBuffer();
        const e = cb.compute(.serial);
        if (mode < 4) try r.ops.argmax(e, input.ref(), picked.ref(), .{ .rows = rows, .vocab = vocab, .stride = 67 });
        try r.ops.match(e, ids.ref(), pars.ref(), picked.ref(), if (eos.len == 0) null else stops.ref(), output.ref(), p);
        try finish(cb, e);
        if (!std.mem.eql(u8, output.data(), std.mem.asBytes(&expected))) {
            const actual: *const contract.Result = @ptrCast(@alignCast(output.data().ptr));
            std.debug.print("round mismatch rows={d} budget={d} mode={d}: expected={any} actual={any}\n", .{ rows, budget, mode, expected, actual.* });
            return error.RoundMismatch;
        }
        const actual: *const contract.Result = @ptrCast(@alignCast(output.data().ptr));
        _ = try contract.decode(actual, p);
        for ([_]Guarded{ input, ids, pars, picked, stops, output }) |buffer| try buffer.check();
        r.rounds += 1;
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
    for ([_]u32{ 1, 2, 31, 32, 255, 256, 257, 513, 1025 }) |vocab| {
        for (1..17) |rows| for (0..8) |mode| try r.argmax(@intCast(rows), vocab, @intCast(mode));
    }
    for (0..8) |mode| try r.argmax(16, 248320, @intCast(mode));
    const Case = struct { tokens: []const u32, parents: []const i32, wants: []const u32 };
    const cases = [_]Case{
        .{ .tokens = &.{1}, .parents = &.{-1}, .wants = &.{9} },
        .{ .tokens = &.{ 1, 2, 3, 4 }, .parents = &.{ -1, 0, 1, 2 }, .wants = &.{ 9, 3, 4, 5 } },
        .{ .tokens = &.{ 1, 2, 3, 4 }, .parents = &.{ -1, 0, 1, 2 }, .wants = &.{ 2, 3, 4, 5 } },
        .{ .tokens = &.{ 1, 2, 3, 4, 5, 6 }, .parents = &.{ -1, 0, 0, 1, 2, 4 }, .wants = &.{ 3, 4, 5, 7, 6, 8 } },
        .{ .tokens = &.{ 1, 2, 2, 3, 4 }, .parents = &.{ -1, 0, 0, 1, 2 }, .wants = &.{ 2, 3, 4, 5, 6 } },
    };
    for (cases) |c| {
        for (0..c.tokens.len + 2) |budget| {
            try r.round(c.tokens, c.parents, c.wants, @intCast(budget), &.{}, 0);
            for (c.wants) |eos| try r.round(c.tokens, c.parents, c.wants, @intCast(budget), &.{eos}, 0);
            try r.round(c.tokens, c.parents, c.wants, @intCast(budget), &.{ 61, 62, 63 }, 0);
        }
        for (1..6) |mode| try r.round(c.tokens, c.parents, c.wants, 20, &.{}, @intCast(mode));
    }
    for (0..96) |seed| {
        const rows = seed % 16 + 1;
        var tokens: [16]u32 = @splat(0);
        var parents: [16]i32 = @splat(-1);
        var wants: [16]u32 = @splat(0);
        for (0..rows) |row| {
            tokens[row] = @intCast((row * 7 + seed * 3) % 19 + 1);
            wants[row] = @intCast((row * 5 + seed * 11) % 19 + 1);
            if (row > 0) parents[row] = @intCast((row * 13 + seed) % row);
        }
        for (1..rows) |row| if (seed % 2 == 0) {
            wants[@intCast(parents[row])] = tokens[row];
        };
        for (0..rows + 2) |budget| try r.round(tokens[0..rows], parents[0..rows], wants[0..rows], @intCast(budget), if (seed % 3 == 0) &.{7} else &.{}, 0);
    }
    try r.round(&.{ 1, 2 }, &.{ -1, 1 }, &.{ 2, 3 }, 20, &.{}, 0);
    try r.round(&.{ 1, 2 }, &.{ 0, 0 }, &.{ 2, 3 }, 20, &.{}, 0);
    try r.round(&.{ 1, 2 }, &.{ -1, -1 }, &.{ 2, 3 }, 20, &.{}, 0);
    try r.round(&.{ 1, 64 }, &.{ -1, 0 }, &.{ 2, 3 }, 20, &.{}, 0);
    try r.round(&.{1}, &.{-1}, &.{2}, 20, &.{64}, 0);
    std.debug.print("tree round: {d} BF16 values, {d} picks, {d} rounds; exact CPU agreement and buffer guards pass\n", .{ r.values, r.picks, r.rounds });
}
