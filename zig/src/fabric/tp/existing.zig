//! The fabric over the released two-rank Group and its existing liveness channel, with one settings agreement.
const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const Nccl = @import("nccl.zig").Nccl;
const Collective = @import("collective.zig").Collective;

pub const Existing = struct {
    group: *cuda.tp_group.Group,
    transport: Nccl,
    cfg: Config,

    /// The caller owns Group and starts tp_watch before opening it, as the released CUDA families do.
    pub fn init(g: *cuda.tp_group.Group, d: *const cuda.Driver, requested: Config) !Existing {
        const cfg = requested.resolved();
        if (cfg.backend != .nccl or cfg.world != 2 or cfg.rank != g.rank or g.world != 2) return error.Invalid;
        var transport = try Nccl.borrow(g, d);
        transport.on_failure = cuda.tp_watch.fail;
        return .{ .group = g, .transport = transport, .cfg = cfg };
    }

    pub fn deinit(e: *Existing) void {
        e.transport.deinit();
    }
    pub fn comm(e: *Existing) Collective {
        return e.transport.iface();
    }

    /// Checkpoint bytes and effective graph/cache settings meet before any graph-dependent collectives.
    pub fn agree(e: *Existing, d: *const cuda.Driver, s: cuda.Stream, checkpoint: [32]u8) !void {
        var mine: [64]u8 = undefined;
        @memcpy(mine[0..32], &checkpoint);
        @memcpy(mine[32..], &e.cfg.digest());
        var all: [2][64]u8 = undefined;
        e.group.gather([64]u8, d, s, mine, &all) catch |err| {
            e.group.abort();
            cuda.tp_watch.fail(@errorName(err));
            return err;
        };
        check(&all) catch |err| {
            e.group.abort();
            cuda.tp_watch.fail(@errorName(err));
            return err;
        };
    }
};

pub fn check(records: []const [64]u8) !void {
    if (records.len != 2) return error.Invalid;
    if (!std.mem.eql(u8, &records[0], &records[1])) return error.Mismatch;
}

test "existing two-rank transport refuses asymmetric effective caches and checkpoint bytes" {
    var records: [2][64]u8 = @splat(@splat(0));
    const on: Config = .{ .graph_cache = .{ .max = 8 } };
    const off: Config = .{ .graph_cache = .{ .max = 8 }, .graph_cache_env = "off" };
    @memcpy(records[0][32..], &on.digest());
    @memcpy(records[1][32..], &off.digest());
    try std.testing.expectError(error.Mismatch, check(&records));
    records[1] = records[0];
    try check(&records);
    records[1][0] = 1;
    try std.testing.expectError(error.Mismatch, check(&records));
}
