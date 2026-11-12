//! Optional request-owned observations of completed target prompt passes, independent of backend arithmetic.
const std = @import("std");

/// Observation failure never fails inference or reports a partial list.
pub fn completed(gpa: std.mem.Allocator, stream: anytype, rows: usize) void {
    if (stream.prefill_widths_failed) return;
    stream.prefill_widths.append(gpa, @intCast(rows)) catch {
        stream.prefill_widths.clearRetainingCapacity();
        stream.prefill_widths_failed = true;
    };
}
