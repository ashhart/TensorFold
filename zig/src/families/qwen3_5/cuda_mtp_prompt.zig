const std = @import("std");

pub const Mode = enum { normal, mtp, mtp_direct, none };

pub const State = struct {
    mode: Mode = .normal,
    from: ?usize = null,
    epoch: u64 = 0,

    pub const TapConfig = struct { mode: Mode, from: ?usize };

    pub fn setForPrompt(s: *State, from: ?usize, end: usize, direct: bool, taps: bool) TapConfig {
        const previous: TapConfig = .{ .mode = s.mode, .from = s.from };
        s.mode = if (from) |boundary|
            (if (end <= boundary) .none else if (direct) .mtp_direct else .mtp)
        else if (taps) (if (direct) .mtp_direct else .mtp) else .normal;
        s.from = from;
        return previous;
    }

    pub fn restoreTapConfig(s: *State, previous: TapConfig) void {
        s.mode = previous.mode;
        s.from = previous.from;
    }

    pub fn tapFrom(s: State, start: usize, rows: usize) ?usize {
        const boundary = s.from orelse return null;
        if (boundary <= start) return null;
        const offset = boundary - start;
        return if (offset < rows) offset else null;
    }
};

test "MTP tap boundary is local only when it crosses a prompt chunk" {
    const state: State = .{ .from = 4096 };
    try std.testing.expectEqual(@as(?usize, null), state.tapFrom(0, 4096));
    try std.testing.expectEqual(@as(?usize, null), state.tapFrom(4094, 2));
    try std.testing.expectEqual(@as(?usize, 2), state.tapFrom(4094, 3));
    try std.testing.expectEqual(@as(?usize, null), state.tapFrom(4096, 8));
    try std.testing.expectEqual(@as(?usize, null), state.tapFrom(4094, 0));
}

test "prompt tap scope restores mode and boundary without rewinding target epoch" {
    var state: State = .{ .epoch = 7 };
    const previous = state.setForPrompt(4096, 4096, false, true);
    try std.testing.expectEqual(Mode.none, state.mode);
    state.epoch += 1;
    state.restoreTapConfig(previous);
    try std.testing.expectEqual(Mode.normal, state.mode);
    try std.testing.expectEqual(@as(?usize, null), state.from);
    try std.testing.expectEqual(@as(u64, 8), state.epoch);
}
