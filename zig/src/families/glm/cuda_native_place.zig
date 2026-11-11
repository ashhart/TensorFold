//! A GLM rank's place from its resolved launch (native/cuda_launch.zig), its weight share and NCCL's rendezvous.
const std = @import("std");
const config = @import("config.zig");
const shard = @import("cuda_shard.zig");
const ranks = @import("cuda_ranks.zig");
const env = @import("sys").env;

pub const Place = struct { rank: usize, world: usize };

/// This rank's place: `o.rank` of `o.tp`, as the launch resolved them (an empty `launch_from`: none was given).
pub fn place(o: anytype) !Place {
    if (o.launch_from.len == 0) return error.NoRank;
    return checked(.{ .rank = o.rank, .world = o.tp });
}

/// A place the family serves: a rank inside its world, and at most `ranks.max_world` ranks.
pub fn checked(p: Place) !Place {
    if (p.world < 1 or p.rank >= p.world) return error.NoRank;
    if (p.world > ranks.max_world) return error.TooManyRanks;
    return p;
}

/// Bytes this rank's weights take at most: the checkpoint (TF_GLM_LAYERS: the tensors read) times its largest share.
pub fn weightShare(a: std.mem.Allocator, io: std.Io, dir: []const u8, total: u64, drafts: bool, rank: u32, world: u32) u64 {
    const p = checked(.{ .rank = rank, .world = world }) catch return total;
    const path = std.fs.path.join(a, &.{ dir, "config.json" }) catch return total;
    const text = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 24)) catch return total;
    const c = config.parse(a, text) catch return total;
    const f = (shard.Share.of(&c, p.rank, p.world) catch return total).part(&c);
    const whole = @import("cuda_thin_bytes.zig").count(a, io, dir, total, ranks.environment("TF_GLM_LAYERS"), c.layers, drafts and c.mtp > 0);
    return @intCast(@as(u128, whole) * f[0] / f[1]);
}

/// NCCL's root from --master and --master-port when NCCL_COMM_ID is unset; first in `open`, before the liveness line.
pub fn rendezvous(o: anytype, lookup: anytype, set: anytype) !void {
    if (o.tp <= 1) return;
    const master = o.master orelse return; // without --master, NCCL_COMM_ID stays required as today (NoCommId)
    var buf: [320]u8 = undefined;
    const want = std.fmt.bufPrintSentinel(&buf, "{s}:{d}", .{ master, o.master_port }, 0) catch return error.RendezvousDisagree;
    if (lookup.get("NCCL_COMM_ID")) |have| {
        if (!std.mem.eql(u8, have, want)) return error.RendezvousDisagree;
        return;
    }
    set.put("NCCL_COMM_ID", want);
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

test "a place inside its world of at most 16 ranks serves; a larger world is refused by name" {
    try std.testing.expectEqual(Place{ .rank = 15, .world = 16 }, try checked(.{ .rank = 15, .world = 16 }));
    try std.testing.expectError(error.TooManyRanks, checked(.{ .rank = 0, .world = 17 }));
    try std.testing.expectError(error.NoRank, checked(.{ .rank = 2, .world = 2 }));
    try std.testing.expectEqual(Place{ .rank = 1, .world = 3 }, try place(.{ .rank = @as(u32, 1), .tp = @as(u32, 3), .launch_from = "--tp" }));
    try std.testing.expectError(error.NoRank, place(.{ .rank = @as(u32, 0), .tp = @as(u32, 1), .launch_from = "" }));
}

test "rendezvous sets NCCL_COMM_ID from --master once, keeps an equal one, refuses a different one" {
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
    try rendezvous(O{ .tp = 2, .master = "10.0.0.7", .master_port = 29583 }, &f, &f);
    try std.testing.expectEqualStrings("10.0.0.7:29583", f.set.?);
    f = .{ .have = "10.0.0.7:29583" };
    try rendezvous(O{ .tp = 2, .master = "10.0.0.7", .master_port = 29583 }, &f, &f);
    try std.testing.expect(f.set == null);
    f = .{ .have = "10.0.0.8:29583" };
    try std.testing.expectError(error.RendezvousDisagree, rendezvous(O{ .tp = 2, .master = "10.0.0.7", .master_port = 29583 }, &f, &f));
    f = .{};
    try rendezvous(O{ .tp = 1, .master = "10.0.0.7", .master_port = 29583 }, &f, &f);
    try rendezvous(O{ .tp = 2, .master = null, .master_port = 29583 }, &f, &f);
    try std.testing.expect(f.set == null);
}
