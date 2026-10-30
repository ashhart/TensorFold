//! The tensor-parallel group of one test process: --tp, --rank, --master and --port, joined through hip.joinGroup.
const std = @import("std");
const hip = @import("hip");
const nem = @import("nemotron");

/// One rank of `world`, joined at `master:port`; every rank runs the same command on its own GPU.
pub const Group = struct {
    rank: usize = 0,
    world: usize = 1,
    master: []const u8 = "127.0.0.1",
    port: u16 = 29561,

    /// Takes the option at `args[i.*]` if it is one of the group's, moving `i` to its value.
    pub fn option(g: *Group, args: []const [:0]const u8, i: *usize) !bool {
        const name = args[i.*];
        const known = [_][]const u8{ "--tp", "--rank", "--master", "--port" };
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
            else => g.port = try std.fmt.parseInt(u16, v, 10),
        }
        return true;
    }

    /// What a group of more than one rank holds besides the engine: close it after the engine.
    pub const Joined = struct {
        gpa: std.mem.Allocator,
        group: ?hip.collective.Group = null,
        policy: hip.Policy = .{},

        pub fn close(j: *Joined) void {
            if (j.group) |g| g.close(j.gpa);
        }
    };

    /// Joins the other ranks running the checkpoint in `dir` (the group stays open until the engine is gone).
    pub fn join(g: Group, gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Joined {
        var j: Joined = .{ .gpa = gpa };
        // TF_POLICY as the server takes it (graphs=off: every round eager), rank 0's for the group
        if (std.c.getenv("TF_POLICY")) |text| j.policy.apply(std.mem.span(text)) catch return error.BadPolicy;
        if (g.world < 2) return j;
        const device: c_int = @intCast(g.rank);
        var d = try hip.Runtime.open();
        defer d.close();
        var ctx = try hip.Context.init(&d, device);
        defer ctx.deinit();
        const caps = try hip.Device.kernelCaps(&d, device);
        j.group = try hip.joinRun(gpa, io, @intCast(g.rank), @intCast(g.world), g.master, g.port, caps, &j.policy, dir);
        return j;
    }

    /// The engine on the device of this rank's number, with its share of the model.
    pub fn engine(
        g: Group,
        gpa: std.mem.Allocator,
        io: std.Io,
        dir: []const u8,
        j: Joined,
        o: nem.engine.Options,
    ) !*nem.engine.Engine {
        var own = o;
        own.device = @intCast(g.rank);
        own.rank = g.rank;
        own.world = g.world;
        own.group = j.group;
        own.policy = j.policy;
        return nem.engine.Engine.open(gpa, io, dir, own);
    }

    /// `path` on rank 0, `path.rN` on rank N: every rank writes what it saw.
    pub fn outPath(g: Group, a: std.mem.Allocator, path: []const u8) ![]const u8 {
        return if (g.rank == 0) path else std.fmt.allocPrint(a, "{s}.r{d}", .{ path, g.rank });
    }
};
