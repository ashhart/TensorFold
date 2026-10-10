const std = @import("std");
const words = @import("fabric").words;

pub const State = struct {
    failed: *const std.atomic.Value(bool),
    quitting: *const std.atomic.Value(bool),
    stop: *const std.atomic.Value(bool),

    fn check(s: State) error{ TpPeerFailed, TpStopping }!void {
        if (s.failed.load(.acquire)) return error.TpPeerFailed;
        if (s.quitting.load(.acquire) or s.stop.load(.acquire)) return error.TpStopping;
    }
};

pub fn wait(word: *const u64, value: u64, state: State, timeout_ns: u64) !void {
    const began = try words.checkedNowNs();
    while (true) {
        try state.check();
        const got: u32 = @truncate(@atomicLoad(u64, word, .acquire));
        const want: u32 = @truncate(value);
        if (@as(i32, @bitCast(got -% want)) >= 0) {
            try state.check();
            return;
        }
        if ((try words.checkedNowNs()) -| began >= timeout_ns) return error.TpPeerSilent;
        std.atomic.spinLoopHint();
    }
}

test "completed words preserve the wrapping call counter" {
    var failed: std.atomic.Value(bool) = .init(false);
    var quitting: std.atomic.Value(bool) = .init(false);
    var stop: std.atomic.Value(bool) = .init(false);
    const state: State = .{ .failed = &failed, .quitting = &quitting, .stop = &stop };
    for ([_][2]u64{ .{ 0, 0 }, .{ 1, 1 }, .{ 0, 0xffff_ffff }, .{ 1, 0 }, .{ 0xffff_ffff, 0xffff_fffe } }) |pair| {
        var word = pair[0];
        try wait(&word, pair[1], state, 0);
    }
    var word: u64 = 0xffff_ffff;
    try std.testing.expectError(error.TpPeerSilent, wait(&word, 0, state, 0));
    word = 0;
    try std.testing.expectError(error.TpPeerSilent, wait(&word, 1, state, 0));
}

test "failure and shutdown stop an absent handoff without claiming it arrived" {
    var failed: std.atomic.Value(bool) = .init(false);
    var quitting: std.atomic.Value(bool) = .init(false);
    var stop: std.atomic.Value(bool) = .init(false);
    const state: State = .{ .failed = &failed, .quitting = &quitting, .stop = &stop };
    var word: u64 = 0;
    failed.store(true, .release);
    try std.testing.expectError(error.TpPeerFailed, wait(&word, 1, state, std.time.ns_per_s));
    failed.store(false, .release);
    quitting.store(true, .release);
    try std.testing.expectError(error.TpStopping, wait(&word, 1, state, std.time.ns_per_s));
    quitting.store(false, .release);
    stop.store(true, .release);
    try std.testing.expectError(error.TpStopping, wait(&word, 1, state, std.time.ns_per_s));
    try std.testing.expectEqual(@as(u64, 0), word);
    word = 1;
    try std.testing.expectError(error.TpStopping, wait(&word, 1, state, std.time.ns_per_s));
}
