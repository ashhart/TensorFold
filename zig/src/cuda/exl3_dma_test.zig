const std = @import("std");
const dma = @import("exl3_dma.zig");
const pool = @import("exl3_pool.zig");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Stream = @import("stream.zig").Stream;
const Event = @import("stream.zig").Event;

const Fake = struct {
    const Call = enum { query, wait, copy, table, event };
    const Copy = struct { dst: u64, src: usize, len: usize, stream: abi.Stream };
    const Word = struct { dst: u64, value: u32 };
    var calls: [32]Call = undefined;
    var count: usize = 0;
    var copies: [3]Copy = undefined;
    var copied: usize = 0;
    var fail_copy: usize = 0;
    var words: [18]Word = undefined;
    var written: usize = 0;

    fn note(call: Call) void {
        calls[count] = call;
        count += 1;
    }

    fn copy(dst: u64, src: ?*const anyopaque, len: usize, stream: abi.Stream) callconv(.c) abi.Result {
        note(.copy);
        copies[copied] = .{ .dst = dst, .src = @intFromPtr(src), .len = len, .stream = stream };
        copied += 1;
        return if (copied == fail_copy) abi.error_not_ready else abi.success;
    }

    fn table(dst: u64, value: c_uint, len: usize, stream: abi.Stream) callconv(.c) abi.Result {
        std.debug.assert(len == 1 and stream == @as(abi.Stream, @ptrFromInt(7)));
        note(.table);
        words[written] = .{ .dst = dst, .value = value };
        written += 1;
        return abi.success;
    }

    fn event(_: abi.Event, _: abi.Stream) callconv(.c) abi.Result {
        note(.event);
        return abi.success;
    }

    fn query(_: abi.Event) callconv(.c) abi.Result {
        note(.query);
        return abi.error_not_ready;
    }

    fn wait(_: abi.Stream, _: abi.Event, _: c_uint) callconv(.c) abi.Result {
        note(.wait);
        return abi.success;
    }

    fn driver() Driver {
        count = 0;
        copied = 0;
        fail_copy = 0;
        written = 0;
        var result: Driver = undefined;
        result.api.cuMemcpyHtoDAsync_v2 = copy;
        result.api.cuMemsetD32Async = table;
        result.api.cuEventRecord = event;
        result.api.cuEventQuery = query;
        result.api.cuStreamWaitEvent = wait;
        return result;
    }
};

fn run(src: [3][]const u8, offsets: [3]usize, bank: ?[]const u8, lengths: []const usize) !void {
    var driver = Fake.driver();
    const stream: Stream = .{ .d = &driver, .handle = @ptrFromInt(7) };
    const moved = try dma.directLines(stream, 4096, 32, offsets, src, bank);
    try std.testing.expectEqual(src[0].len + src[1].len + src[2].len, moved);
    try std.testing.expectEqual(lengths.len, Fake.copied);
    var projection: usize = 0;
    for (Fake.copies[0..Fake.copied], lengths) |copy, len| {
        try std.testing.expectEqual(len, copy.len);
        try std.testing.expectEqual(4096 + offsets[projection], copy.dst);
        try std.testing.expectEqual(@intFromPtr(src[projection].ptr), copy.src);
        try std.testing.expectEqual(stream.handle, copy.stream);
        var consumed: usize = 0;
        while (projection < 3) {
            const bytes = src[projection];
            const uploaded: [*]const u8 = @ptrFromInt(copy.src);
            try std.testing.expectEqualSlices(u8, bytes, uploaded[consumed..][0..bytes.len]);
            consumed += bytes.len;
            projection += 1;
            if (consumed == len) break;
        }
        try std.testing.expectEqual(len, consumed);
    }
    try std.testing.expectEqual(@as(usize, 3), projection);
}

test "DMA coalescing: exact unequal widths merge three projections or either pair" {
    const bank = [_]u8{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    const src: [3][]const u8 = .{ bank[0..2], bank[2..5], bank[5..9] };
    try run(src, .{ 0, 2, 5 }, &bank, &.{9});
    try run(src, .{ 0, 2, 8 }, &bank, &.{ 5, 4 });
    try run(src, .{ 0, 4, 7 }, &bank, &.{ 2, 7 });
    try run(.{ bank[0..2], bank[2..5], bank[6..10] }, .{ 0, 2, 5 }, &bank, &.{ 5, 4 });
    try run(.{ bank[0..2], bank[3..6], bank[6..10] }, .{ 0, 2, 5 }, &bank, &.{ 2, 7 });
}

test "DMA coalescing: absent provenance and source or destination gaps retain exact fallback copies" {
    const bank: [32]u8 = @splat(1);
    const src: [3][]const u8 = .{ bank[0..2], bank[2..5], bank[5..9] };
    try run(src, .{ 0, 2, 5 }, null, &.{ 2, 3, 4 });
    try run(src, .{ 0, 4, 9 }, &bank, &.{ 2, 3, 4 });
    try run(.{ bank[0..2], bank[4..7], bank[9..13] }, .{ 0, 2, 5 }, &bank, &.{ 2, 3, 4 });
    try run(src, .{ 0, 2, 5 }, bank[10..20], &.{ 2, 3, 4 });
    try run(src, .{ 0, 2, 5 }, bank[0..4], &.{ 2, 3, 4 });
    try run(src, .{ 0, 2, 5 }, bank[2..9], &.{ 2, 7 });
    try run(src, .{ 0, 2, 5 }, bank[0..0], &.{ 2, 3, 4 });
    const foreign: [9]u8 = @splat(3);
    try run(.{ foreign[0..2], foreign[2..5], foreign[5..9] }, .{ 0, 2, 5 }, &bank, &.{ 2, 3, 4 });
}

test "DMA coalescing: zero lengths preserve fallback and never bridge a gap" {
    const bank: [9]u8 = @splat(2);
    try run(.{ bank[0..0], bank[0..3], bank[3..9] }, .{ 0, 0, 3 }, &bank, &.{ 0, 9 });
    try run(.{ bank[0..2], bank[2..2], bank[2..9] }, .{ 0, 2, 2 }, &bank, &.{ 2, 0, 7 });
    try run(.{ bank[0..2], bank[2..2], bank[3..9] }, .{ 0, 2, 3 }, &bank, &.{ 2, 0, 6 });
    try run(.{ bank[0..0], bank[0..0], bank[0..0] }, .{ 0, 0, 0 }, &bank, &.{ 0, 0, 0 });
}

test "DMA coalescing: bounds and overflowing source bank or destination fail before copying" {
    var driver = Fake.driver();
    const stream: Stream = .{ .d = &driver, .handle = null };
    const bank: [9]u8 = @splat(1);
    const src: [3][]const u8 = .{ bank[0..2], bank[2..5], bank[5..9] };
    try std.testing.expectError(error.Invalid, dma.directLines(stream, 4096, 8, .{ 0, 2, 5 }, src, &bank));
    try std.testing.expectError(error.Invalid, dma.directLines(stream, 4096, 9, .{ 0, 10, 5 }, src, &bank));
    try std.testing.expectError(error.Invalid, dma.directLines(stream, 4096, 9, .{ 0, 2, std.math.maxInt(usize) }, src, &bank));
    try std.testing.expectError(error.Invalid, dma.directLines(stream, std.math.maxInt(u64) - 8, 9, .{ 0, 2, 5 }, src, &bank));
    const high: [*]const u8 = @ptrFromInt(std.math.maxInt(usize) - 1);
    try std.testing.expectError(error.Invalid, dma.directLines(stream, 4096, 9, .{ 0, 2, 5 }, .{ src[0], src[1], high[0..4] }, null));
    try std.testing.expectError(error.Invalid, dma.directLines(stream, 4096, 9, .{ 0, 2, 5 }, src, high[0..4]));
    try std.testing.expectEqual(@as(usize, 0), Fake.count);
}

test "DMA coalescing: production three-bit sizes issue one exact transfer" {
    const width = 614400;
    const bank = try std.testing.allocator.alloc(u8, 3 * width);
    defer std.testing.allocator.free(bank);
    @memset(bank, 1);
    var driver = Fake.driver();
    const stream: Stream = .{ .d = &driver, .handle = null };
    const layout = pool.Layout.init(.{ width, width, width });
    const src: [3][]const u8 = .{ bank[0..width], bank[width .. 2 * width], bank[2 * width ..] };
    try std.testing.expectEqual(@as(usize, 3 * width), try dma.directLines(stream, 4096, layout.line_bytes, layout.off, src, bank));
    try std.testing.expectEqual(@as(usize, 1), Fake.copied);
    try std.testing.expectEqual(@as(usize, 3 * width), Fake.copies[0].len);
}

test "DMA coalescing: pool preserves reader waits table writes event and expert byte statistics" {
    const bank: [1536]u8 = @splat(1);
    const src: [3][]const u8 = .{ bank[0..256], bank[256..768], bank[768..1536] };
    for ([_]?[]const u8{ null, &bank, bank[256..] }, [_]usize{ 3, 1, 2 }) |provenance, copies| {
        for (0..copies + 1) |failure| {
            var driver = Fake.driver();
            Fake.fail_copy = failure;
            var epochs = [_]u64{1};
            var pending = [_]bool{false};
            var events = [_]Event{.{ .d = &driver, .handle = @ptrFromInt(2) }};
            var lines: pool.Lines = undefined;
            lines.d = &driver;
            lines.spec = .{ .experts = 2, .bytes = .{ 256, 512, 768 }, .lines = 1, .staging = 0 };
            lines.layout = pool.Layout.init(lines.spec.bytes);
            lines.tlay = pool.Tables.init(2);
            lines.mem = .{ .d = &driver, .ptr = 4096, .len = 1536 };
            lines.tables = .{ .d = &driver, .ptr = 8192, .len = lines.tlay.bytes };
            lines.staging = null;
            lines.pinned_bank = provenance;
            lines.copy = .{ .d = &driver, .handle = @ptrFromInt(7) };
            lines.line_ev = &events;
            lines.line_pending = &pending;
            lines.line_epoch = &epochs;
            lines.rel_ev[1] = .{ .d = &driver, .handle = @ptrFromInt(3) };
            lines.done_epoch = 0;
            lines.stats = .{};
            const result = lines.load(0, 1, src, .{ 3, 6, 9 }, 0, true, undefined);
            try std.testing.expectEqualSlices(Fake.Call, &.{ .query, .wait }, Fake.calls[0..2]);
            if (failure > 0) {
                try std.testing.expectError(error.NotReady, result);
                try std.testing.expectEqual(2 + failure, Fake.count);
                try std.testing.expectEqual(failure, Fake.copied);
                try std.testing.expect(!pending[0]);
                try std.testing.expectEqual(@as(u64, 0), lines.stats.copies);
                try std.testing.expectEqual(@as(u64, 0), lines.stats.bytes);
            } else {
                try result;
                try std.testing.expectEqual(2 + copies + 18 + 1, Fake.count);
                for (Fake.calls[2 + copies .. Fake.count - 1]) |call| try std.testing.expectEqual(Fake.Call.table, call);
                try std.testing.expectEqual(Fake.Call.event, Fake.calls[Fake.count - 1]);
                try std.testing.expect(pending[0]);
                try std.testing.expectEqual(@as(u64, 1), lines.stats.copies);
                try std.testing.expectEqual(@as(u64, bank.len), lines.stats.bytes);
                try std.testing.expectEqual(@as(u64, 1), lines.stats.release_waits);
                for (0..3) |which| {
                    const old = Fake.words[3 * which ..][0..3];
                    const installed = Fake.words[9 + 3 * which ..][0..3];
                    try std.testing.expectEqualDeep(Fake.Word{ .dst = lines.table(which), .value = 0 }, old[0]);
                    try std.testing.expectEqualDeep(Fake.Word{ .dst = lines.table(which) + 4, .value = 0 }, old[1]);
                    try std.testing.expectEqualDeep(Fake.Word{ .dst = lines.widths(which), .value = 0 }, old[2]);
                    try std.testing.expectEqualDeep(Fake.Word{ .dst = lines.table(which) + 8, .value = @intCast(4096 + lines.layout.off[which]) }, installed[0]);
                    try std.testing.expectEqualDeep(Fake.Word{ .dst = lines.table(which) + 12, .value = 0 }, installed[1]);
                    try std.testing.expectEqualDeep(Fake.Word{ .dst = lines.widths(which) + 4, .value = @intCast(3 * (which + 1)) }, installed[2]);
                }
            }
            for (Fake.calls[2 .. 2 + Fake.copied]) |call| try std.testing.expectEqual(Fake.Call.copy, call);
        }
    }
}
