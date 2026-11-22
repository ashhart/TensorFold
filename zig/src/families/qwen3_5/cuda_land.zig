//! Graph capture for uniform draw landings; host finalization remains in Engine.land.
const std = @import("std");
const cuda = @import("cuda");
const Round = @import("cuda_round.zig");
const lever7 = @import("cuda_lever7.zig");
const argmaxSlices = @import("cuda_scratch.zig").argmaxSlices;

pub const Key = struct {
    logits: u64,
    tables: u64,
    amax: u64,
    picks: u64,
    total: u64,
    sampled: u64,
};

pub const Graphs = std.AutoHashMapUnmanaged(Key, cuda.graph.Exec);

pub fn key(e: anytype, total: usize, sampled: bool) Key {
    return .{ .logits = e.logits, .tables = e.tables.ptr, .amax = e.amax, .picks = e.picks, .total = total, .sampled = @intFromBool(sampled) };
}

pub fn clear(e: anytype) void {
    var it = e.land_graphs.valueIterator();
    while (it.next()) |graph| graph.deinit();
    e.land_graphs.clearRetainingCapacity();
}

pub fn deinit(e: anytype) void {
    clear(e);
    e.land_graphs.deinit(e.gpa);
}

/// Mixed greedy/sampled runs keep the eager run loop. Uniform rows have one exact graph key.
pub fn run(e: anytype, pk: anytype, total: usize) !void {
    if (!e.lever7.capture or !lever7.uniformDraws(pk)) return kernels(e, pk, total);
    const k = key(e, total, pk[0].rule != null);
    const gop = try e.land_graphs.getOrPut(e.gpa, k);
    if (!gop.found_existing) {
        errdefer _ = e.land_graphs.remove(k);
        try cuda.graph.beginCapture(e.stream, .thread_local);
        kernels(e, pk, total) catch |err| {
            var broken = cuda.graph.endCapture(e.stream) catch return err;
            broken.deinit();
            return err;
        };
        var captured = try cuda.graph.endCapture(e.stream);
        defer captured.deinit();
        gop.value_ptr.* = try captured.instantiate();
    }
    try gop.value_ptr.launchOn(e.stream);
}

fn kernels(e: anytype, pk: anytype, total: usize) !void {
    const toks = e.tables.ptr + Round.tokens_at;
    var i: usize = 0;
    while (i < pk.len) {
        const sampled = pk[i].rule != null;
        var n: usize = 1;
        while (i + n < pk.len and (pk[i + n].rule != null) == sampled) n += 1;
        if (sampled) {
            try e.draws.drawRows(e.stream, e.logits + i * e.g.vocab * 2, e.g.vocab, e.tables.ptr + Round.rules_at + i * @sizeOf(cuda.sample.Rule), e.tables.ptr + Round.meta_at + i * 4, toks + i * 4, n, e.picks);
        } else try argmax(e, i, n, toks);
        i += n;
    }
    std.debug.assert(i == total);
}

fn argmax(e: anytype, first: usize, n: usize, toks: u64) !void {
    if (n == 0) return;
    const slices = argmaxSlices(e.g.vocab);
    const pv = e.amax;
    const pi = e.amax + n * slices * 4;
    try e.run(e.fns.argmax_slices, .{ slices, n, 1 }, .{ 256, 1, 1 }, .{ e.logits + first * e.g.vocab * 2, @as(c_int, @intCast(e.g.vocab)), pv, pi });
    try e.run(e.fns.argmax_merge, .{ n, 1, 1 }, .{ 32, 1, 1 }, .{ pv, pi, @as(c_int, @intCast(slices)), toks + first * 4 });
}

test "REV295 land keys separate topology and every captured device address" {
    const Fake = struct { logits: u64, tables: struct { ptr: u64 }, amax: u64, picks: u64 };
    var e: Fake = .{ .logits = 1, .tables = .{ .ptr = 2 }, .amax = 3, .picks = 4 };
    const base = key(&e, 8, false);
    try std.testing.expectEqual(base, key(&e, 8, false));
    try std.testing.expect(!std.meta.eql(base, key(&e, 8, true)));
    try std.testing.expect(!std.meta.eql(base, key(&e, 9, false)));
    inline for (.{ "logits", "amax", "picks" }) |name| {
        @field(e, name) += 1;
        try std.testing.expect(!std.meta.eql(base, key(&e, 8, false)));
        @field(e, name) -= 1;
    }
    e.tables.ptr += 1;
    try std.testing.expect(!std.meta.eql(base, key(&e, 8, false)));
}
