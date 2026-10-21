//! GLM's KV pages on a driver of addresses alone: tables in row order, leases given back, a follower's ids checked.
const std = @import("std");
const cuda = @import("cuda");
const weights = @import("cuda_weights.zig");
const shard = @import("cuda_shard.zig");
const pages_mod = @import("cuda_pages.zig");
const Engine = @import("cuda_engine.zig").Engine;
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;

/// The driver: each allocation a fresh address, the page tables written to it kept, the NaN fills counted.
const F = struct {
    var next: u64 = 1 << 30;
    var live: usize = 0;
    var fills: usize = 0;
    var at: [8]u64 = @splat(0); // a table's address, and its words below
    var words: [8][32]u32 = undefined;
    fn alloc(out: *u64, n: usize) callconv(.c) cuda.abi.Result {
        out.* = next;
        next += std.mem.alignForward(u64, n + 1, 1 << 20);
        live += 1;
        return 0;
    }
    fn free(_: u64) callconv(.c) cuda.abi.Result {
        live -= 1;
        return 0;
    }
    fn clear(_: u64, _: u8, _: usize) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn fill(_: u64, v: c_uint, _: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        fills += @intFromBool(v == 0xFFFF_FFFF);
        return 0;
    }
    fn put(dst: u64, src: ?*const anyopaque, n: usize) callconv(.c) cuda.abi.Result {
        const ids: [*]const u32 = @ptrCast(@alignCast(src.?));
        for (&at, &words) |*a, *w| if (a.* == 0 or (dst >= a.* and dst < a.* + 128)) {
            if (a.* == 0) a.* = dst;
            const from = (dst - a.*) / 4;
            @memcpy(w[from..][0 .. n / 4], ids[0 .. n / 4]);
            return 0;
        };
        return 1;
    }
    fn wait(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return 0;
    }
    /// The first `n` words written to the table at `t`.
    fn table(t: u64, n: usize) []const u32 {
        for (at, &words) |a, *w| if (a == t) return w[0..n];
        return &.{};
    }
};

/// An engine with what the pages read: a driver, TP=2's shapes on a KDA, a DSA and a KDA layer, 4,096 rows.
const Fake = struct {
    d: cuda.Driver,
    layers: [3]weights.Layer,
    e: Engine,

    fn init(f: *Fake, gpa: std.mem.Allocator) !void {
        F.live = 0;
        F.fills = 0;
        F.at = @splat(0);
        f.d = undefined;
        f.d.api.cuMemAlloc_v2 = F.alloc;
        f.d.api.cuMemFree_v2 = F.free;
        f.d.api.cuMemsetD8_v2 = F.clear;
        f.d.api.cuMemsetD32Async = F.fill;
        f.d.api.cuMemcpyHtoD_v2 = F.put;
        f.d.api.cuStreamSynchronize = F.wait;
        for (&f.layers, [_]bool{ false, true, false }) |*l, mla| l.mixer = if (mla) .{ .mla = undefined } else .{ .kda = undefined };
        f.e = undefined;
        f.e.gpa = gpa;
        f.e.d = &f.d;
        f.e.stream = .{ .d = &f.d, .handle = null };
        f.e.capacity = 4096;
        f.e.seqs = .empty;
        f.e.pages = null;
        f.e.w = undefined;
        f.e.w.config = .{};
        f.e.w.share = try shard.Share.of(&f.e.w.config, 0, 2);
        f.e.w.layers = &f.layers;
    }

    /// The pages first, then the sequences they made, as the engine frees every slot.
    fn deinit(f: *Fake, p: *pages_mod.Pages) void {
        p.deinit();
        for (f.e.seqs.items) |q| if (q) |s| {
            s.st.deinit();
            f.e.gpa.destroy(s);
        };
        f.e.seqs.deinit(f.e.gpa);
    }
};

test "a paged stream takes its request's pages, its table holds them in row order, and its end gives them back" {
    var f: Fake = undefined;
    try f.init(std.testing.allocator);
    const l = try pages_mod.layout(std.testing.allocator, &f.e, 0);
    try expectEqual(@as(u64, 2 * 1088), l.token_bytes); // a DSA layer and the MTP layer: 1,024 + 64 bytes each
    const p = try pages_mod.Pages.init(std.testing.allocator, &f.e, l, 2, 8, true);
    defer f.deinit(p);
    try expect(!p.paged(0) and p.paged(1) and p.paged(2) and !p.paged(3));
    try expect(p.fits(600, 100));
    try std.testing.expectEqualSlices(u32, &.{ 0, 1 }, try p.open(1, 600, 100)); // 708 rows: 2 pages
    const st1 = &f.e.seqs.items[1].?.st;
    try std.testing.expectEqualSlices(u32, &.{ 0, 1 }, F.table(st1.table, 2));
    try expectEqual(@as(usize, 1024), st1.capacity);
    try expect(!p.fits(3000, 100)); // 7 pages of the 6 left
    try std.testing.expectError(error.LaneMemory, p.open(2, 3000, 100));
    try expectEqual(@as(usize, 0), f.e.seqs.items[2].?.st.capacity);
    try std.testing.expectEqualSlices(u32, &.{ 2, 3, 4, 5, 6 }, try p.open(2, 2000, 100));
    p.close(1);
    try expectEqual(@as(usize, 0), st1.capacity);
    try std.testing.expectEqualSlices(u32, &.{ 1, 0 }, try p.open(1, 600, 100)); // the last page given back, first
    try std.testing.expectEqualSlices(u32, &.{ 1, 0 }, F.table(st1.table, 2));
    p.close(1);
    p.close(2);
    try expect(p.pool.idle() and p.pool.census().free == 8);
    try std.testing.expectError(error.PromptTooLong, p.open(1, 4096, 0)); // past the window's table
}

test "a follower maps rank 0's ids, and refuses ids past its pool or a slot that is not paged" {
    var f: Fake = undefined;
    try f.init(std.testing.allocator);
    const l = try pages_mod.layout(std.testing.allocator, &f.e, 0);
    const p = try pages_mod.Pages.init(std.testing.allocator, &f.e, l, 2, 8, false);
    defer f.deinit(p);
    try p.map(2, &.{ 5, 3, 7 });
    try std.testing.expectEqualSlices(u32, &.{ 5, 3, 7 }, F.table(f.e.seqs.items[2].?.st.table, 3));
    try expectEqual(@as(usize, 1536), f.e.seqs.items[2].?.st.capacity);
    try std.testing.expectError(error.BadStep, p.map(1, &.{8}));
    try std.testing.expectError(error.BadStep, p.map(3, &.{1}));
    p.close(2);
    try expectEqual(@as(usize, 0), f.e.seqs.items[2].?.st.capacity);
    try expect(p.pool.idle()); // a follower's pool lends nothing
}

test "TF_KV_POISON: every page a stream takes is filled with NaN before its rows are written" {
    var f: Fake = undefined;
    try f.init(std.testing.allocator);
    const l = try pages_mod.layout(std.testing.allocator, &f.e, 0);
    const p = try pages_mod.Pages.init(std.testing.allocator, &f.e, l, 1, 8, true);
    defer f.deinit(p);
    p.poison = true;
    _ = try p.open(1, 1200, 100);
    try expectEqual(@as(usize, 3), F.fills);
}

fn made(gpa: std.mem.Allocator) !void {
    var f: Fake = undefined;
    try f.init(gpa);
    defer f.e.seqs.deinit(gpa);
    const l = try pages_mod.layout(gpa, &f.e, 0);
    const p = try pages_mod.Pages.init(gpa, &f.e, l, 2, 8, true);
    p.deinit();
    for (f.e.seqs.items) |q| if (q) |s| {
        s.st.deinit();
        gpa.destroy(s);
    };
}

test "every host allocation failure while making the pages leaves no device buffer behind" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, struct {
        fn run(gpa: std.mem.Allocator) !void {
            made(gpa) catch |err| {
                try expectEqual(@as(usize, 0), F.live);
                return err;
            };
            try expectEqual(@as(usize, 0), F.live);
        }
    }.run, .{});
}
