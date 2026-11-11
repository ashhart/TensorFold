//! NCCL's root for a family over hosts: --master and --master-port become NCCL_COMM_ID when it is unset.
const std = @import("std");
const env = @import("sys").env;

/// NCCL's root from --master and --master-port when NCCL_COMM_ID is unset; first in `open`, before the liveness line.
pub fn fromMaster(o: anytype, lookup: anytype, out: anytype) !void {
    if (o.tp <= 1) return;
    const master = o.master orelse return; // without --master, NCCL_COMM_ID stays required as today (NoCommId)
    var buf: [320]u8 = undefined;
    const want = std.fmt.bufPrintSentinel(&buf, "{s}:{d}", .{ master, o.master_port }, 0) catch return error.RendezvousDisagree;
    if (lookup.get("NCCL_COMM_ID")) |have| {
        if (!std.mem.eql(u8, have, want)) return error.RendezvousDisagree;
        return;
    }
    out.put("NCCL_COMM_ID", want);
}

/// The process environment, for `rendezvous` outside tests: NCCL's own variable, the only one it reads.
pub const Process = struct {
    pub fn get(_: Process, _: [:0]const u8) ?[]const u8 {
        return if (std.c.getenv("NCCL_COMM_ID")) |v| std.mem.span(v) else null;
    }
    pub fn put(_: Process, name: [:0]const u8, value: [:0]const u8) void {
        _ = env.setenv(name, value, 1);
    }
};

test "the rendezvous sets NCCL_COMM_ID from --master once, keeps an equal one, refuses a different one" {
    const Fake = struct {
        have: ?[]const u8 = null,
        set: ?[]const u8 = null,
        buf: [64]u8 = undefined,
        pub fn get(f: *@This(), _: [:0]const u8) ?[]const u8 {
            return f.have;
        }
        pub fn put(f: *@This(), _: [:0]const u8, v: [:0]const u8) void {
            @memcpy(f.buf[0..v.len], v);
            f.set = f.buf[0..v.len];
        }
    };
    const O = struct { tp: u32, master: ?[]const u8, master_port: u16 };
    var f: Fake = .{};
    try fromMaster(O{ .tp = 2, .master = "10.0.0.7", .master_port = 29583 }, &f, &f);
    try std.testing.expectEqualStrings("10.0.0.7:29583", f.set.?);
    f = .{ .have = "10.0.0.7:29583" };
    try fromMaster(O{ .tp = 2, .master = "10.0.0.7", .master_port = 29583 }, &f, &f);
    try std.testing.expect(f.set == null);
    f = .{ .have = "10.0.0.8:29583" };
    try std.testing.expectError(error.RendezvousDisagree, fromMaster(O{ .tp = 2, .master = "10.0.0.7", .master_port = 29583 }, &f, &f));
    f = .{};
    try fromMaster(O{ .tp = 1, .master = "10.0.0.7", .master_port = 29583 }, &f, &f);
    try fromMaster(O{ .tp = 2, .master = null, .master_port = 29583 }, &f, &f);
    try std.testing.expect(f.set == null);
}
