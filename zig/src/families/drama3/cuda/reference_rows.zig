//! A CUDA reference cache entry's layout: both recurrent blocks, then each K and V cache's first rows, in that order.
const std = @import("std");
const cfg = @import("../config.zig");

/// One K or V cache position, every KV head's bf16 row; positions are outermost, as qk_rope writes them.
pub const kv_row = cfg.kv_heads * cfg.head_dim * 2;

/// The byte qk_rope writes head `g`'s element `i` at position `pos` to: kc[(pos * kv_heads + g) * head_dim + i].
pub fn kvOffset(pos: usize, g: usize, i: usize) usize {
    return ((pos * cfg.kv_heads + g) * cfg.head_dim + i) * 2;
}

/// The live buffers an entry copies, in entry order.
pub const Live = struct { state: [2]u64, state_bytes: [2]usize, caches: []const u64 };

/// Entry bytes for the state after `rows` prompt rows.
pub fn bytes(live: Live, rows: usize) usize {
    return live.state_bytes[0] + live.state_bytes[1] + live.caches.len * rows * kv_row;
}

/// Copies the state after `rows` rows between `live` and the entry at `saved` through `copier.copy(src, dst, n)`.
pub fn move(copier: anytype, live: Live, saved: u64, rows: usize, load: bool) !usize {
    var at = saved;
    for (live.state, live.state_bytes) |buf, n| at = try moveRun(copier, buf, at, n, load);
    for (live.caches) |cache| at = try moveRun(copier, cache, at, rows * kv_row, load);
    return at - saved;
}

fn moveRun(copier: anytype, live: u64, saved: u64, n: usize, load: bool) !u64 {
    if (load) try copier.copy(saved, live, n) else try copier.copy(live, saved, n);
    return saved + n;
}

const HostCopy = struct {
    fn copy(_: HostCopy, src: u64, dst: u64, n: usize) !void {
        @memcpy(@as([*]u8, @ptrFromInt(dst))[0..n], @as([*]const u8, @ptrFromInt(src))[0..n]);
    }
};

test "drama reference rows: a restored K or V cache holds exactly the saved positions, every head in place" {
    const t = std.testing;
    const ctx = 8;
    const per_cache = ctx * kv_row / 2;
    for ([_]usize{ 1, 5, ctx }) |rows| {
        var caches: [2][per_cache]u16 = undefined;
        var state: [2][24]u8 = undefined;
        for (&caches, 0..) |*c, ci| for (0..ctx) |pos| for (0..cfg.kv_heads) |g| for (0..cfg.head_dim) |i| {
            c[kvOffset(pos, g, i) / 2] = @intCast(ci * 4096 + pos * 512 + g * 256 + i); // distinct per element
        };
        for (&state, 0..) |*s, si| for (s, 0..) |*b, i| {
            b.* = @intCast(si * 100 + i);
        };
        const want_caches = caches;
        const want_state = state;
        const addrs = [2]u64{ @intFromPtr(&caches[0]), @intFromPtr(&caches[1]) };
        const live: Live = .{ .state = .{ @intFromPtr(&state[0]), @intFromPtr(&state[1]) }, .state_bytes = .{ 24, 24 }, .caches = &addrs };
        const entry = try t.allocator.alloc(u8, bytes(live, rows));
        defer t.allocator.free(entry);
        try t.expectEqual(entry.len, try move(HostCopy{}, live, @intFromPtr(entry.ptr), rows, false));
        for (&caches) |*c| @memset(c, 0xa5a5); // another request's rows
        for (&state) |*s| @memset(s, 0x5a);
        try t.expectEqual(entry.len, try move(HostCopy{}, live, @intFromPtr(entry.ptr), rows, true));
        try t.expectEqualSlices(u8, std.mem.asBytes(&want_state), std.mem.asBytes(&state));
        for (caches, want_caches) |got, want| for (0..ctx) |pos| for (0..cfg.kv_heads) |g| for (0..cfg.head_dim) |i| {
            const at = kvOffset(pos, g, i) / 2;
            try t.expectEqual(if (pos < rows) want[at] else 0xa5a5, got[at]);
        };
    }
}
