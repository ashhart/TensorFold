//! Native writes have no cancellable ABI: join a retired relay, otherwise leave its resources alive until exit.
const std = @import("std");
const words = @import("fabric").words;

/// Completion, cancellation and the deadline are checked even when a flag was already satisfied.
pub fn await(done: *const std.atomic.Value(u64), seq: u64, stop: *const std.atomic.Value(bool), failed: *const std.atomic.Value(bool), until: u64) !void {
    while (true) {
        if (stop.load(.acquire)) return error.PairStopped;
        if (failed.load(.acquire)) return error.PairLinkFailed;
        if (try words.checkedNowNs() >= until) return error.PairControlTimeout;
        if (done.load(.acquire) >= seq) return;
        words.pause(20_000);
    }
}

pub fn join(thread: std.Thread, done: *const std.atomic.Value(bool), budget_ns: u64) bool {
    const start = words.checkedNowNs() catch {
        thread.detach();
        return false;
    };
    while (!done.load(.acquire)) {
        const now = words.checkedNowNs() catch {
            thread.detach();
            return false;
        };
        if (now -| start >= budget_ns) {
            thread.detach();
            return false;
        }
        words.pause(20_000);
    }
    thread.join();
    return true;
}

const Fixture = struct {
    release: std.atomic.Value(bool) = .init(false),
    done: std.atomic.Value(bool) = .init(false),

    fn run(f: *Fixture) void {
        while (!f.release.load(.acquire)) words.pause(20_000);
        f.done.store(true, .release);
    }
};

test "pair shutdown: a blocked native call does not hold close and its owner stays alive until retirement" {
    var f: Fixture = .{};
    const thread = try std.Thread.spawn(.{}, Fixture.run, .{&f});
    try std.testing.expect(!join(thread, &f.done, 0));
    try std.testing.expect(!f.done.load(.acquire));
    f.release.store(true, .release);
    while (!f.done.load(.acquire)) words.pause(20_000);
}

test "pair shutdown: a retired relay joins normally" {
    var f: Fixture = .{};
    f.release.store(true, .release);
    const thread = try std.Thread.spawn(.{}, Fixture.run, .{&f});
    try std.testing.expect(join(thread, &f.done, std.time.ns_per_s));
}

test "pair shutdown: control callers time out or cancel even if a native call cannot retire" {
    var done: std.atomic.Value(u64) = .init(0);
    var stop: std.atomic.Value(bool) = .init(false);
    var failed: std.atomic.Value(bool) = .init(false);
    try std.testing.expectError(error.PairControlTimeout, await(&done, 1, &stop, &failed, 0));
    const until = (try words.checkedNowNs()) + std.time.ns_per_s;
    done.store(1, .release);
    try await(&done, 1, &stop, &failed, until);
    try std.testing.expectError(error.PairControlTimeout, await(&done, 1, &stop, &failed, 0));
    failed.store(true, .release);
    try std.testing.expectError(error.PairLinkFailed, await(&done, 1, &stop, &failed, until));
    stop.store(true, .release);
    try std.testing.expectError(error.PairStopped, await(&done, 1, &stop, &failed, until));
}
