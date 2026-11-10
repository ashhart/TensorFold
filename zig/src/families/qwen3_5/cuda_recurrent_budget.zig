//! Recurrent scratch bounds preserve every FP32 rollback state and power-of-two verification log.
const std = @import("std");
const common = @import("core").recurrent_workspace;
pub const max_total = 128;

pub fn perWindow(state: usize, row: usize, rows: usize, maximum: usize) !usize {
    const count = try common.rowClass(rows, maximum);
    return std.math.add(usize, try std.math.mul(usize, count, row), if (rows == 1) state else 0);
}

/// This is one frame only; old unkept leases and unfinished physical disposal need their own lifetime guarantee.
pub fn frame(state: usize, row: usize, streams: usize, total: usize, maximum: usize) !usize {
    if (streams == 0) return 0;
    if (total == 0 or total > max_total or maximum == 0 or maximum > total) return error.InvalidRecurrentBudget;
    var dp: [max_total + 1]?usize = @splat(null);
    dp[0] = 0;
    var best: usize = 0;
    for (0..@min(streams, total)) |_| {
        var next: [max_total + 1]?usize = @splat(null);
        for (dp[0..total], 0..) |value, used| if (value) |prior| {
            for (1..@min(maximum, total - used) + 1) |rows| {
                const bytes = try std.math.add(usize, prior, try perWindow(state, row, rows, maximum));
                const at = used + rows;
                if (next[at] == null or next[at].? < bytes) next[at] = bytes;
                best = @max(best, bytes);
            }
        };
        dp = next;
    }
    return best;
}
