//! One proxy owns every ring QP; default mesh posts and the two-rank wire do not enter here.
const std = @import("std");
const relay = @import("roce_relay.zig");
const sock = @import("sock.zig");
pub fn agreed(b: anytype, enabled: bool) !bool {
    var blobs: [8][]const u8 = undefined;
    var buf: [8]u8 = undefined;
    for (try b.allGather(&.{@intFromBool(enabled)}, &buf, &blobs)) |value|
        if (value.len != 1 or value[0] != @intFromBool(enabled)) return false;
    return true;
}
fn Adapter(comptime Wire: type) type {
    return struct {
        wire: *Wire,
        pub fn segment(a: *@This(), s: relay.Segment) !void {
            switch (a.wire.*) {
                inline else => |*w| try w.segment(s),
            }
        }
        pub fn reap(a: *@This()) !void {
            try a.wire.reap();
        }
        pub fn idle(a: *const @This()) bool {
            return switch (a.wire.*) {
                inline else => |*w| w.idle(),
            };
        }
    };
}
pub fn run(r: anytype) void {
    var state = relay.Relay.init(r.me, r.layout, r.region, r.timeout_ns) catch |e|
        return r.fail("relay rank {d} sequence {d}: {t}", .{ r.me, r.last_seq, e });
    var wire: Adapter(@TypeOf(r.wire)) = .{ .wire = &r.wire };
    while (r.running.load(.monotonic)) {
        if (r.word(12).load(.monotonic) != 0) return;
        const doorbell = r.word(0).load(.acquire);
        const padded = r.word(4 + ((state.last +% 1) & 1)).load(.acquire);
        _ = state.progress(&wire, doorbell, r.word(8).load(.acquire), padded, sock.nowNs()) catch |e| {
            var line: [96]u8 = undefined;
            return r.fail("relay rank {d}: {s}: {t}", .{ r.me, state.fault.describe(&line), e });
        };
        r.last_seq = state.last;
        std.atomic.spinLoopHint();
    }
}
