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
