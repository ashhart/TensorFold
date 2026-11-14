//! Frozen shared-system workloads with stable agents, replaced agent cohorts, chats and unique prompts.
const std = @import("std");
const common = @import("cache_replay_trace.zig");
pub const Shape = enum { agent_loops, interleaved_chats, one_off, agent_cohorts };
pub const rounds = 48;
pub const budgets = [_]u64{ 4 << 30, 8 << 30, 16 << 30 };

pub fn make(gpa: std.mem.Allocator, shape: Shape) !common.Trace {
    return makeSeeded(gpa, shape, 7);
}

pub fn makeSeeded(gpa: std.mem.Allocator, shape: Shape, seed: u64) !common.Trace {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();
    var requests: std.ArrayList(common.Request) = .empty;
    var rng = std.Random.DefaultPrng.init(seed);
    var unique: u32 = 1000;
    var turns: [12]u32 = @splat(0);
    for (0..rounds) |r| switch (shape) {
        .agent_loops, .agent_cohorts => {
            for (0..6) |session| {
                const phase: usize = if (shape == .agent_cohorts and r >= rounds / 2) 6 else 0;
                const turn = if (shape == .agent_cohorts) r % (rounds / 2) else r;
                const history: u32 = 16384 + @as(u32, @intCast(session % 3)) * 1024 + @as(u32, @intCast(turn)) * 512;
                try add(a, &requests, .agent, @intCast(phase + session + 1), history, 8192, @intCast(phase + 1), @intCast(r));
                for (0..2) |_| {
                    const sizes = [_]u32{ 1024, 4096, 8192, 12000 };
                    try add(a, &requests, .one_off, unique, sizes[rng.random().uintLessThan(usize, sizes.len)], 0, unique, @intCast(r));
                    unique += 1;
                }
            }
        },
        .interleaved_chats => {
            for (0..4) |j| {
                const phase: usize = if (r < rounds / 2) 0 else 6;
                const session = phase + (r + j * 2) % 6;
                const history: u32 = 4096 + @as(u32, @intCast(session % 4)) * 512 + turns[session] * 256;
                try add(a, &requests, .agent, @intCast(session + 1), history, 2048, @intCast(phase + 2), @intCast(r));
                turns[session] += 1;
            }
        },
        .one_off => for (0..8) |_| {
            const sizes = [_]u32{ 256, 1024, 4096, 8192, 16384 };
            try add(a, &requests, .one_off, unique, sizes[rng.random().uintLessThan(usize, sizes.len)], 0, unique, @intCast(r));
            unique += 1;
        },
    };
    var digest: u64 = 0;
    for (requests.items) |req| digest = digest *% 31 +% req.hash +% req.history;
    return .{ .arena = arena, .requests = try requests.toOwnedSlice(a), .digest = digest };
}
fn add(a: std.mem.Allocator, list: *std.ArrayList(common.Request), kind: common.Kind, session: u32, history: u32, shared: u32, group: u32, round: u32) !void {
    const tokens = try a.alloc(u32, history + 16);
    for (tokens, 0..) |*token, i| {
        token.* = if (i < shared) @intCast(100 + group * 100000 + i) else if (i == shared) 2000000 + session else @intCast(3000000 + i);
    }
    try list.append(a, .{ .kind = kind, .session = session, .history = history, .tokens = tokens, .hash = common.hash(tokens), .round = round, .shared = if (shared > 0) try a.dupe(u32, &.{shared}) else &.{} });
}
