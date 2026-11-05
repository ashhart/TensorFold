//! Each prompt chunk chooses its width from the streams that can decode beside it.
const std = @import("std");
pub fn choose(b: anytype, current: anytype) usize {
    for (b.owners[0..b.runner.slots]) |owner| if (owner) |stream| {
        if (stream != current and !stream.finished and !stream.paused and !b.fills.contains(stream)) {
            return @min(b.chunk, 512);
        }
    };
    return b.chunk;
}

/// Only completed target prompt passes count; an observation failure never fails inference or reports a partial list.
pub fn completed(gpa: std.mem.Allocator, stream: anytype, rows: usize) void {
    if (stream.prefill_widths_failed) return;
    stream.prefill_widths.append(gpa, @intCast(rows)) catch {
        stream.prefill_widths.clearRetainingCapacity();
        stream.prefill_widths_failed = true;
    };
}
