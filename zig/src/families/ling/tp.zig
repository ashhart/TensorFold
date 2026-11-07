//! Ling at TP=2 over hosts as GLM runs it: rank 0's engine sends each step first (GLM's Link), rank 1 follows.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const engine = @import("engine.zig");

pub const Link = glm.follow.Link;
pub const Header = glm.follow.Header;

/// The steps rank 1 replays, in rank 0's order (Header.kind).
pub const Kind = enum(u32) { begin = 1, restore, chunk, verify, shared, keep, keep_state, drop, release, idle, end };

/// This rank's place: TF_LING_RANK of TF_LING_WORLD (both unset: one GPU).
pub const Place = struct { rank: u32, world: u32 };

/// The place the environment gives; worlds other than 1 and 2 are refused (TP=2 is what was validated).
pub fn place(get: *const fn ([*:0]const u8) ?[*:0]const u8) !Place {
    const r = get("TF_LING_RANK");
    const w = get("TF_LING_WORLD");
    if (r == null and w == null) return .{ .rank = 0, .world = 1 };
    const p: Place = .{
        .rank = std.fmt.parseInt(u32, std.mem.span(r orelse return error.NoRank), 10) catch return error.NoRank,
        .world = std.fmt.parseInt(u32, std.mem.span(w orelse return error.NoRank), 10) catch return error.NoRank,
    };
    if (p.world != 1 and p.world != 2) return error.TpWorld;
    if (p.rank >= p.world) return error.NoRank;
    return p;
}

pub fn env(name: [*:0]const u8) ?[*:0]const u8 {
    return std.c.getenv(name);
}

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

/// The ranks' NCCL group as GLM opens it: rank 0's unique id over the TCP line (tp_watch.start ran first).
pub fn group(io: std.Io, p: Place) !cuda.tp_group.Group {
    // GLM's defaults: connections registered at init, no cuMem host buffers (GB10's driver pool wore out under them)
    inline for (.{ "NCCL_RUNTIME_CONNECT", "NCCL_CUMEM_HOST_ENABLE", "NCCL_CUMEM_ENABLE" }) |name| if (std.c.getenv(name) == null) {
        _ = setenv(name, "0", 0);
    };
    if (p.rank > 0 and std.c.getenv("NCCL_SOCKET_IFNAME") != null) _ = unsetenv("NCCL_COMM_ID");
    var lib = try cuda.nccl.Library.open();
    errdefer lib.close();
    var id: cuda.nccl.UniqueId = undefined;
    if (p.rank == 0) {
        try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
        cuda.tp_watch.sendId(&id.internal);
    } else id.internal = try cuda.tp_watch.takeId(io);
    var comm: cuda.nccl.Comm = null;
    try lib.check(lib.api.ncclCommInitRank(&comm, @intCast(p.world), id, @intCast(p.rank)), "ncclCommInitRank");
    return .{ .lib = lib, .comm = comm, .rank = p.rank, .world = p.world };
}

/// Main's RoCE gathers up to GLM's TF_ROCE_MAX_KB cap, NCCL past it; both ranks open it or neither (results equal).
pub fn roce(g: *cuda.tp_group.Group, gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, s: cuda.Stream) !void {
    try g.openRoce(gpa, io, d, s, try glm.native.roceBytes(if (std.c.getenv("TF_ROCE_MAX_KB")) |x| std.mem.span(x) else null));
    std.log.info("ling tp: rank {d} gathers {s}", .{ g.rank, if (g.fast != null) "by RoCE writes up to the cap, NCCL past it" else "on NCCL (RoCE did not come up)" });
}

/// Rank 1's loop: each of rank 0's steps on its own engine until rank 0 ends; an idle step sleeps on the TCP line.
pub fn follow(e: *engine.Engine, io: std.Io) !void {
    const l = e.link orelse return error.NoLink;
    var out: [engine.shared_rows]u32 = undefined;
    cuda.tp_watch.sleep(io); // rank 0 starts with no stream: its first step wakes this rank
    while (true) {
        const step = try l.receive();
        const h = step.h;
        if (h.kind < 1 or h.kind > @backingInt(Kind.end)) return error.BadStep;
        const kind: Kind = @fromBackingInt(@intCast(h.kind));
        switch (kind) {
            .drop, .idle, .end, .shared, .release => {},
            else => try e.bind(h.slot),
        }
        switch (kind) {
            .begin => try e.begin(if (h.flags & 1 != 0) .{ .seed = h.seed, .temperature = h.temperature, .top_p = h.top_p, .min_p = h.min_p, .top_k = h.top_k } else null),
            .restore => try e.restoreState(h.a),
            .chunk => _ = try e.chunkRun(step.ids, h.depth, @intCast(h.seed), @intCast(h.a)),
            .verify => {
                if (step.ids.len > out.len) return error.BadStep;
                try e.verify(step.ids, out[0..step.ids.len]);
            },
            .shared => {
                const k = h.depth;
                if (k == 0 or k > engine.max_shared or step.ids.len < 2 * k) return error.BadStep;
                var parts: [engine.max_shared]engine.Part = undefined;
                var at: usize = 2 * k;
                var row: usize = 0;
                for (parts[0..k], 0..) |*p, i| {
                    const n = step.ids[2 * i + 1];
                    if (at + n > step.ids.len or row + n > out.len) return error.BadStep;
                    p.* = .{ .slot = step.ids[2 * i], .ids = step.ids[at..][0..n], .out = out[row..][0..n] };
                    at += n;
                    row += n;
                }
                try e.verifyShared(parts[0..k]);
            },
            .keep => try e.keep(@intCast(h.a)),
            .keep_state => if (try e.keepState() != h.a) return error.RanksOutOfStep,
            .drop => for (step.ids) |id| e.dropState(id),
            .release => e.release(h.slot),
            .idle => cuda.tp_watch.sleep(io),
            .end => return cuda.tp_watch.ended(), // rank 0 ends on purpose: its line closing from here is expected
        }
    }
}

test "TF_LING_RANK and TF_LING_WORLD: unset is one GPU; worlds 1 and 2 only; a rank inside its world" {
    const F = struct {
        var r: ?[*:0]const u8 = null;
        var w: ?[*:0]const u8 = null;
        fn get(name: [*:0]const u8) ?[*:0]const u8 {
            return if (std.mem.eql(u8, std.mem.span(name), "TF_LING_RANK")) r else w;
        }
    };
    try std.testing.expectEqual(Place{ .rank = 0, .world = 1 }, try place(F.get));
    F.r = "1";
    F.w = "2";
    try std.testing.expectEqual(Place{ .rank = 1, .world = 2 }, try place(F.get));
    F.w = "3";
    try std.testing.expectError(error.TpWorld, place(F.get));
    F.r = "2";
    F.w = "2";
    try std.testing.expectError(error.NoRank, place(F.get));
    F.r = null;
    try std.testing.expectError(error.NoRank, place(F.get));
}
