//! The tensor-parallel group of one test process: its options, the link to the other ranks and the RCCL bootstrap.

const std = @import("std");
const hip = @import("hip");
const qwen35 = @import("qwen3_5");

/// One rank of `world`, joined at `master:port`; every rank runs the same command.
pub const Group = struct {
    rank: usize = 0,
    world: usize = 1,
    master: []const u8 = "127.0.0.1",
    port: u16 = 29551,
    /// The policy flags, `key=value,...`.
    policy: []const u8 = "",

    /// Takes the option at `args[i.*]` if it is one of the group's, moving `i` to its value.
    pub fn option(g: *Group, args: []const [:0]const u8, i: *usize) !bool {
        const name = args[i.*];
        const known = [_][]const u8{ "--tp", "--rank", "--master", "--port", "--policy" };
        const index = for (known, 0..) |k, n| {
            if (std.mem.eql(u8, name, k)) break n;
        } else return false;
        i.* += 1;
        if (i.* >= args.len) return error.MissingArgument;
        const v = args[i.*];
        switch (index) {
            0 => g.world = try std.fmt.parseInt(usize, v, 10),
            1 => g.rank = try std.fmt.parseInt(usize, v, 10),
            2 => g.master = v,
            3 => g.port = try std.fmt.parseInt(u16, v, 10),
            else => g.policy = v,
        }
        return true;
    }

    /// What a group of more than one rank holds besides the engine: close it after the engine.
    pub const Joined = struct {
        rccl: ?hip.rccl.Rccl = null,
        link: hip.link.Link = undefined,
        id: ?hip.rccl.UniqueId = null,
        world: usize = 1,
        /// What every rank runs under: rank 0's.
        policy: hip.Policy = .{},

        pub fn close(j: *Joined) void {
            if (j.world > 1) j.link.close();
            if (j.rccl) |*r| r.close();
        }
    };

    /// The unique id starts RCCL's bootstrap thread: the library stays loaded until the engine is gone.
    pub fn join(g: Group, io: std.Io) !Joined {
        const found = try resolve(g.policy, @intCast(g.rank));
        var j: Joined = .{ .world = g.world, .policy = found.policy };
        if (g.world < 2) return j;
        j.rccl = try hip.rccl.Rccl.open(j.policy.rccl_lib.slice());
        errdefer j.rccl.?.close();
        const pair = try hip.link.Link.open(io, g.rank, g.world, g.master, g.port, if (g.rank == 0) try j.rccl.?.uniqueId() else undefined, .{ .caps = hip.Device.groupId(found.caps), .policy = j.policy.words() });
        j.link, j.id, const hello = pair;
        if (g.rank != 0) j.policy = hip.Policy.fromWords(hello.policy, j.policy);
        return j;
    }

    /// The engine on the device of this rank's number, with its share of the model.
    pub fn engine(g: Group, gpa: std.mem.Allocator, io: std.Io, dir: []const u8, j: Joined, o: qwen35.engine.Options) !*qwen35.engine.Engine {
        var own = o;
        own.device = @intCast(g.rank);
        own.rank = g.rank;
        own.world = g.world;
        own.id = j.id;
        own.policy = j.policy;
        return qwen35.engine.Engine.open(gpa, io, dir, own);
    }
};

/// The policy of a test run on `device` and its caps: the GPU's defaults, `flags`, the variables and TF_POLICY, logged.
pub fn resolve(flags: []const u8, device: c_int) !struct { policy: hip.Policy, caps: hip.Caps } {
    var d = try hip.Runtime.open();
    defer d.close();
    var ctx = try hip.Context.init(&d, device);
    defer ctx.deinit();
    const caps = try hip.Device.kernelCaps(&d, device);
    var notes: hip.Policy.Notes = .{};
    const policy = try hip.Policy.resolve(flags, .current, &notes);
    std.debug.print("policy {f} {s}\n", .{ policy, notes.text() });
    return .{ .policy = policy, .caps = caps };
}

/// A rank above 0 runs what rank 0 sends until it says stop.
pub fn follow(gpa: std.mem.Allocator, e: *qwen35.engine.Engine, j: *const Group.Joined) !void {
    var w = try qwen35.worker.Worker.init(gpa, e);
    defer w.deinit();
    return w.follow(&j.link);
}
