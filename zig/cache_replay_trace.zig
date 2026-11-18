//! Deterministic synthetic conversations and unique one-off prompts; no checkpoint or user trace is required.
const std = @import("std");
pub const Kind = enum { agent, one_off, idle_seed, idle_return };
pub const Shape = enum { steady, churn, idle_gap };
pub const Request = struct { kind: Kind, session: u32, history: u32, tokens: []u32, hash: u64, round: u32 };
pub const Trace = struct {
    arena: std.heap.ArenaAllocator,
    requests: []Request,
    digest: u64,
    pub fn deinit(t: *Trace) void {
        t.arena.deinit();
    }
};
pub fn hash(tokens: []const u32) u64 {
    var h: u64 = 0;
    for (tokens) |v| h = h *% 31 +% v;
    return h;
}
pub fn make(gpa: std.mem.Allocator, shape: Shape, rounds: u32, seed: u64) !Trace {
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();
    var out: std.ArrayList(Request) = .empty;
    var random = std.Random.DefaultPrng.init(seed);
    var next: u32 = 10;
    for (0..rounds) |r| {
        for (0..3) |session| {
            if (shape != .idle_gap or r % (@as(usize, 1) << @intCast(session)) == 0) {
                const history: u32 = @intCast(20000 + session * 5000 + r * (1000 - session * 100));
                try add(a, &out, .agent, @intCast(session + 1), history, @intCast(r));
            }
            const count: usize = switch (shape) {
                .steady => 1,
                .churn => 3,
                .idle_gap => 2,
            };
            for (0..count) |_| {
                const sizes = [_]u32{ 256, 512, 1024, 2048, 4096, 8192 };
                const size = sizes[random.random().uintLessThan(usize, sizes.len)];
                try add(a, &out, .one_off, next, size, @intCast(r));
                next += 1;
            }
        }
        if (r < 2) try add(a, &out, .idle_seed, 4, @intCast(28000 + r * 2000), @intCast(r));
        if (r >= 45 and (r - 45) % 5 == 0) try add(a, &out, .idle_return, 4, @intCast(34000 + (r - 45) / 5 * 2000), @intCast(r));
    }
    var digest: u64 = 0;
    for (out.items) |req| digest = digest *% 31 +% req.hash +% @backingInt(req.kind);
    return .{ .arena = arena, .requests = try out.toOwnedSlice(a), .digest = digest };
}
fn add(a: std.mem.Allocator, out: *std.ArrayList(Request), kind: Kind, session: u32, history: u32, round: u32) !void {
    const tokens = try a.alloc(u32, history + 16);
    for (tokens[0..history], 0..) |*v, i| v.* = if (i < 128) @intCast(i + 1) else if (i == 128) 1000 + session else @intCast(10000 + i);
    for (tokens[history..], 0..) |*v, i| v.* = 248000 + @as(u32, @intCast(i));
    try out.append(a, .{ .kind = kind, .session = session, .history = history, .tokens = tokens, .hash = hash(tokens), .round = round });
}
