const std = @import("std");
const Stream = @import("stream.zig").Stream;

pub fn directLines(stream: Stream, dst: u64, capacity: usize, offsets: [3]usize, src: [3][]const u8, pinned_bank: ?[]const u8) !usize {
    const Span = struct { dst: u64, bytes: []const u8, banked: bool };
    var spans: [3]Span = undefined;
    var count: usize = 0;
    var moved: usize = 0;
    _ = std.math.add(u64, dst, capacity) catch return error.Invalid;
    const bank_start = if (pinned_bank) |bank| @intFromPtr(bank.ptr) else 0;
    const bank_end = if (pinned_bank) |bank| std.math.add(usize, bank_start, bank.len) catch return error.Invalid else 0;
    for (src, offsets) |bytes, offset| {
        if (offset > capacity or bytes.len > capacity - offset) return error.Invalid;
        const start = @intFromPtr(bytes.ptr);
        const end = std.math.add(usize, start, bytes.len) catch return error.Invalid;
        moved = std.math.add(usize, moved, bytes.len) catch return error.Invalid;
        const address = dst + offset;
        const banked = pinned_bank != null and start >= bank_start and end <= bank_end;
        if (count > 0 and bytes.len > 0 and banked) {
            const previous = &spans[count - 1];
            if (previous.banked and previous.bytes.len > 0 and
                @intFromPtr(previous.bytes.ptr) + previous.bytes.len == start and
                previous.dst + previous.bytes.len == address)
            {
                previous.bytes = pinned_bank.?[@intFromPtr(previous.bytes.ptr) - bank_start ..][0 .. previous.bytes.len + bytes.len];
                continue;
            }
        }
        spans[count] = .{ .dst = address, .bytes = bytes, .banked = banked };
        count += 1;
    }
    for (spans[0..count]) |span| try stream.d.check(stream.d.api.cuMemcpyHtoDAsync_v2(span.dst, span.bytes.ptr, span.bytes.len, stream.handle), "cuMemcpyHtoDAsync");
    return moved;
}

test {
    _ = @import("exl3_dma_test.zig");
}
