//! KV_SPLIT hand-offs: each rank exports the pool pages it stores, on its own link, and the Mac pulls every shard.
//! Rank 0 exports shard 0 (rings, carries, drafter rows and its pages) as it does for a replicated pool, and sends the
//! followers `op_export` at the cut; a follower's `Follower` snapshots its pages there and serves them under the same id.
const std = @import("std");
const api = @import("engine_api");
const model = @import("cuda_model.zig");
const sg = @import("sessions_gpu.zig");
const fwd = @import("forward.zig");
const ff = @import("forward_follow.zig");
const gpu = @import("cuda_export_gpu.zig");
const serve = @import("cuda_export_serve.zig");
const log = std.log.scoped(.dsv41);

/// The follower's link: TF_DSV41_PRODUCE_LINK, the mcdma-rpcd listen link its shard is exported on (null: unset).
pub const link_env: [:0]const u8 = "TF_DSV41_PRODUCE_LINK";

/// The configured link name of a follower, or null when unset; a malformed one sets `problem`.
pub fn linkName(problem: *[]const u8) !?[]const u8 {
    const env = @import("sys").environment;
    const name = env.path(link_env, (env.Process{}).get(link_env)) catch {
        problem.* = "TF_DSV41_PRODUCE_LINK is malformed: it must be an mcdma-rpcd link name without control characters";
        return error.InvalidEnvironment;
    };
    const n = name orelse return null;
    if (n.len == 0 or n.len > 20) {
        problem.* = "TF_DSV41_PRODUCE_LINK is malformed: an mcdma-rpcd link name has 1 to 20 characters";
        return error.InvalidEnvironment;
    }
    for (n) |c| if (!std.ascii.isAlphanumeric(c) and c != '_' and c != '-') {
        problem.* = "TF_DSV41_PRODUCE_LINK is malformed: an mcdma-rpcd link name is made of letters, digits, _ and -";
        return error.InvalidEnvironment;
    };
    return n;
}

/// Whether `m`'s pool is split over several ranks: its hand-offs are exported by shard.
pub fn splitOf(m: *const model.Model) bool {
    const ss = m.sessions orelse return false;
    return ss.kx.layout.opts.split and ss.kx.world > 1;
}

/// A follower rank's producer: serves its shard of every hand-off the leader announces (`op_export`).
pub const Follower = struct {
    gpa: std.mem.Allocator,
    source: gpu.Source,
    produce: *serve.Produce,
    ss: *sg.Sessions,

    /// Opens the producer on `link` (`problem` is set when it cannot): the pool must be split and this rank not 0.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, m: *model.Model, link: []const u8, checkpoint: []const u8, problem: *[]const u8) !*Follower {
        if (!splitOf(m) or m.rank() == 0) {
            problem.* = "TF_DSV41_PRODUCE_LINK names a follower's link: set it on ranks above 0 of a split KV pool (TF_DSV41_KV_SPLIT=1)";
            return error.ExportRefused;
        }
        const cap = @import("sys").environment.read(u64, "TF_PRODUCE_HELD_GIB", 32) catch {
            problem.* = "TF_PRODUCE_HELD_GIB is malformed; the line above names its forms";
            return error.InvalidEnvironment;
        };
        const x = try gpa.create(Follower);
        errdefer gpa.destroy(x);
        x.* = .{ .gpa = gpa, .source = .{ .ss = m.sessions.? }, .produce = undefined, .ss = m.sessions.? };
        const opts: serve.Options = .{ .link = link, .model = checkpoint, .held_bytes = cap << 30, .shard = m.rank(), .shards = x.ss.kx.world };
        x.produce = try serve.Produce.open(gpa, io, x.source.source(), opts, problem);
        log.info("dsv41 handoff export: shard {d} of {d} on link {s}", .{ opts.shard, opts.shards, link });
        return x;
    }

    pub fn stop(x: *Follower) void {
        x.produce.stop();
        x.gpa.destroy(x);
    }

    pub fn ext(x: *Follower) fwd.Forward.Ext {
        return .{ .ptr = x, .run = run };
    }

    /// The leader's op_export / op_export_fail. Nothing here may fail the follower's loop: a shard that cannot be
    /// exported is recorded as failed under its id, which the Mac's pull of this link is told.
    fn run(ptr: *anyopaque, msg: []const i64) anyerror!void {
        const x: *Follower = @ptrCast(@alignCast(ptr));
        if (msg.len < 3) return error.BadPlan;
        const id = ff.idOf(msg[1..3]);
        if (msg[0] == ff.op_export_fail) return x.produce.failed(id, "rank 0 could not export this hand-off");
        if (msg.len != 4 or msg[3] <= 0) return error.BadPlan;
        x.exportShard(id, @intCast(msg[3]));
    }

    fn exportShard(x: *Follower, id: [16]u8, at: u64) void {
        const saved = x.produce.save(at) catch |err| return x.produce.failed(id, @errorName(err));
        const hist = x.ss.hist[x.ss.cur()].items;
        const tokens = x.gpa.alloc(u32, hist.len) catch {
            x.produce.drop(saved);
            return x.produce.failed(id, "OutOfMemory");
        };
        defer x.gpa.free(tokens);
        for (hist, tokens) |h, *t| t.* = @intCast(h);
        x.produce.exported(id, tokens, saved);
    }
};

test "an op_export message carries the hand-off id in two words and back" {
    const id: [16]u8 = .{ 1, 2, 3, 4, 5, 6, 7, 8, 0xff, 0xee, 0xdd, 0xcc, 0xbb, 0xaa, 0x99, 0x88 };
    const w = ff.idWords(id);
    try std.testing.expectEqualSlices(u8, &id, &ff.idOf(&w));
}
