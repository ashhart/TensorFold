//! How a failure on one rank stops both: the stop inside a call both ranks run, and the parity tool's planted faults.
const std = @import("std");
const cuda = @import("cuda");

/// A failure the parity tool plants on one rank (its fault mode); a serve never sets one.
pub const Fault = struct {
    rank: ?usize = null,
    at: Stage = .start,

    /// Before NCCL connects, after this rank's weights load (before the hello), or in the first call's rows.
    pub const Stage = enum { start, load, call };

    pub fn hit(f: Fault, rank: usize, at: Stage) !void {
        if (f.rank == rank and f.at == at) return error.PlantedFault;
    }
};

/// A failure in a call both ranks run strands the other rank in it: both stop (no lines: `err` goes to the caller).
pub fn fatal(rank: usize, err: anyerror) anyerror {
    if (!cuda.tp_watch.open()) return err;
    std.log.err("ricursive rank {d}: {s} inside a call both ranks run; both ranks stop", .{ rank, @errorName(err) });
    cuda.tp_watch.fatal(@errorName(err));
    return err;
}

test "a planted fault fires on its own rank and stage only" {
    const f: Fault = .{ .rank = 1, .at = .load };
    try std.testing.expectError(error.PlantedFault, f.hit(1, .load));
    try f.hit(0, .load);
    try f.hit(1, .start);
    try f.hit(1, .call);
    const none: Fault = .{};
    for ([_]Fault.Stage{ .start, .load, .call }) |at| for (0..2) |r| try none.hit(r, at);
}
