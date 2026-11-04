//! The real pool's upload and later fills keep every page leased until completion proof.
const std = @import("std");
const cuda = @import("cuda");
const pages = @import("cuda_kv_lanes.zig");
const E = @import("cuda_prompt.zig").Engine;
const D = cuda.paging_test_device.Device;
const t = std.testing;
const Watch = struct {
    var pool: ?*@import("core").kv_pool.Pool = null;
    fn held() bool {
        return if (pool) |p| p.census().leased == D.expected else false;
    }
};

var queue: cuda.Stream = undefined;
fn engine(d: *cuda.Driver) E {
    var e: E = undefined;
    e.driver = d;
    queue = .{ .d = d, .handle = D.stream };
    e.stream = &queue;
    e.eos = 0;
    e.round_graph = .{};
    return e;
}

test "KV lifetime: Flash Next failed table wait and later fill retain pages until proof" {
    for ([_]bool{ false, true }) |later_fill| {
        var d = D.driver();
        var e = engine(&d);
        defer {
            D.observe = null;
            Watch.pool = null;
            D.fail_wait = false;
            cuda.completion.drain(&d, e.stream.*, "the test driver must drain before it leaves");
        }
        const l = pages.layout(64, 16, null, 0);
        const p = try pages.Pool.init(t.allocator, &e, l, 64, null, 1, l.runPages() + 16);
        defer {
            D.fail_wait = false;
            cuda.completion.drain(&d, e.stream.*, "the test queue must drain before its owners leave");
            p.deinit();
        }
        Watch.pool = &p.pool;
        D.observe = Watch.held;
        D.expected = l.pages(1) + l.runPages();
        D.proofs = 0;
        D.fail_wait = !later_fill;
        D.fail_fill = later_fill;
        try t.expectError(error.NotReady, p.open(1));
        try t.expect(D.proofs > 0 and !D.early and !D.pending);
        try t.expect(p.pool.idle());
        try t.expectEqual(@as(usize, 0), D.freed_pending);
    }
}

test "KV lifetime: Flash Next close and deinit retain pages until context proof" {
    for ([_]bool{ false, true }) |whole_pool| {
        var d = D.driver();
        var e = engine(&d);
        defer {
            D.observe = null;
            Watch.pool = null;
            D.fail_wait = false;
            cuda.completion.drain(&d, e.stream.*, "the test driver must drain before it leaves");
        }
        const l = pages.layout(64, 16, null, 0);
        const p = try pages.Pool.init(t.allocator, &e, l, 64, null, 1, l.runPages() + 16);
        var c = try p.open(1);
        Watch.pool = &p.pool;
        D.observe = Watch.held;
        D.expected = l.pages(1) + l.runPages();
        D.proofs = 0;
        D.pending = true;
        D.fail_wait = true;
        if (whole_pool) p.deinit() else {
            try t.expect(p.close(&c));
            try t.expect(p.pool.idle());
            p.deinit();
        }
        D.observe = null;
        Watch.pool = null;
        try t.expect(D.proofs > 0 and !D.early and !D.pending);
        try t.expectEqual(@as(usize, 0), D.freed_pending);
    }
}

test "KV lanes: two ranks whose pages differ in bytes open and close the same page ids in the same order" {
    var d = D.driver();
    var e = engine(&d);
    defer cuda.completion.drain(&d, e.stream.*, "the test driver must drain before it leaves");
    // rank 0 keeps the MTP head's layer and a bigger run, rank 1 neither: both take the larger run in pages
    const head: pages.Head = .{ .draft_n = 4096, .rows = 64 };
    const a = pages.layout(256, 16, head, 0);
    const b = pages.layout(256, 16, null, 0);
    try t.expect(a.pageBytes() != b.pageBytes());
    const k = @max(a.runPages(), b.runPages());
    const left = pages.withRunPages(a, k);
    const right = pages.withRunPages(b, k);
    try t.expectEqual(left.runPages(), right.runPages());
    const count = 4 * k + 64;
    const p0 = try pages.Pool.init(t.allocator, &e, left, 256, head, 3, count);
    defer p0.deinit();
    const p1 = try pages.Pool.init(t.allocator, &e, right, 256, null, 3, count);
    defer p1.deinit();
    const caps = [_]usize{ 40, 17, 100 };
    var s0: [3]@import("cuda_prompt.zig").Seq = undefined;
    var s1: [3]@import("cuda_prompt.zig").Seq = undefined;
    for (caps, 0..) |cap, i| {
        s0[i] = try p0.open(cap);
        s1[i] = try p1.open(cap);
    }
    for (0..3) |i| {
        try t.expectEqualSlices(u32, p0.lanes[i].lease.ids.items, p1.lanes[i].lease.ids.items);
        try t.expectEqual(p0.lanes[i].run, p1.lanes[i].run);
    }
    try t.expect(p0.close(&s0[1]) and p1.close(&s1[1]));
    s0[1] = try p0.open(60);
    s1[1] = try p1.open(60);
    try t.expectEqualSlices(u32, p0.lanes[1].lease.ids.items, p1.lanes[1].lease.ids.items);
    try t.expectEqual(p0.lanes[1].run, p1.lanes[1].run);
    for (0..3) |i| {
        try t.expect(p0.close(&s0[i]) and p1.close(&s1[i]));
    }
    try t.expect(p0.pool.idle() and p1.pool.idle());
}

test "KV lanes: every pass over a lane's MTP head reads its cache through the lane's pages, across a page boundary" {
    const wt = @import("cuda_prompt_wt.zig");
    const pg = @import("cuda_pages.zig");
    var d = D.driver();
    var e = engine(&d);
    defer cuda.completion.drain(&d, e.stream.*, "the test driver must drain before it leaves");
    const head: pages.Head = .{ .draft_n = 4096, .rows = 64 };
    const l = pages.layout(256, 16, head, 0);
    const p = try pages.Pool.init(t.allocator, &e, l, 256, head, 1, 4 * l.runPages() + 64);
    defer p.deinit();
    var seq = try p.open(100);
    var dr = try p.draft(&seq, 100);
    defer dr.deinit();
    const paged = seq.paged.?;
    // the head's rows 15 and 16 are in two pages: not a contiguous cache's next row
    try t.expect(paged.rowAt(wt.attn_n, .key, 16) != paged.rowAt(wt.attn_n, .key, 15) + pg.rowBytes(.key));
    // the head's pass (a prompt chunk, a window's absorb, a shared round, the EXL3 deep chunk) takes the lane's table
    const hs = dr.headSeq(null, 0);
    try t.expect(hs.pg.pt != 0);
    try t.expectEqual(paged.pg(), hs.pg);
    try t.expectEqual(dr.kc.ptr, hs.kc[0].ptr);
    try t.expect(p.close(&seq));
}
