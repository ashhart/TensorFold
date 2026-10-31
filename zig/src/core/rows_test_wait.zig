//! Completion-driven waits for host cache tests; the ceiling diagnoses hangs rather than setting a write budget.
const std = @import("std");
const rows = @import("rows_store.zig");

pub fn deadline() i128 {
    return rows.write.nowNs() + 60 * std.time.ns_per_s;
}

/// Engine turns stage queued saves and publish worker completions before a test inspects their metadata or files.
pub fn published(r: *rows.Rows) !void {
    const end = deadline();
    while (rows.write.pending(r)) {
        _ = rows.write.step(r);
        if (!rows.write.pending(r)) break;
        if (rows.write.nowNs() >= end) return error.WriterTimedOut;
        rows.write.io.nap();
    }
}
