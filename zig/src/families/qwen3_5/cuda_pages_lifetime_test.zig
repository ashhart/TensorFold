//! The real pool's upload and later fills keep every page leased until completion proof.
const std = @import("std");
const cuda = @import("cuda");
const pages = @import("cuda_pages.zig");
const E = @import("cuda_engine.zig").Engine;
const D = cuda.paging_test_device.Device;
const t = std.testing;
const Watch = struct {
    var pool: ?*@import("core").kv_pool.Pool = null;
    fn held() bool {
        return if (pool) |p| p.census().leased == D.expected else false;
    }
};

fn engine(d: *cuda.Driver) E {
    var e: E = undefined;
    e.d = d;
    e.stream = .{ .d = d, .handle = D.stream };
    e.kv8 = false;
    e.n_taps = 0;
    e.g = .{ .hidden = 256, .intermediate = 256, .layers = 2, .vocab = 8, .k_heads = 1, .v_heads = 1, .q_heads = 1, .kv_heads = 1, .eps = 1e-6, .tied = true, .context = 64, .linear = @splat(false) };
    e.g.linear[0] = true;
    return e;
}

test "KV lifetime: Qwen failed table wait and later fill retain pages until proof" {
    for ([_]bool{ false, true }) |later_fill| {
        var d = D.driver();
        var e = engine(&d);
        defer {
            D.observe = null;
            Watch.pool = null;
            D.fail_wait = false;
            cuda.completion.drain(&d, e.stream, "the test driver must drain before it leaves");
        }
        const l = pages.layout(&e, 64, 16, 0, 0);
        const p = try pages.Pages.init(t.allocator, &e, l, 64, 0, 1, l.need(64, 0, pages.window_rows));
        defer {
            D.fail_wait = false;
            cuda.completion.drain(&d, e.stream, "the test queue must drain before its owners leave");
            p.deinit();
        }
        Watch.pool = &p.pool;
        D.observe = Watch.held;
        D.expected = l.need(1, 0, pages.window_rows);
        D.proofs = 0;
        D.fail_wait = !later_fill;
        D.fail_fill = later_fill;
        try t.expectError(error.NotReady, p.open(1, 0));
        try t.expect(D.proofs > 0 and !D.early and !D.pending);
        try t.expect(p.pool.idle());
        try t.expectEqual(@as(usize, 0), D.freed_pending);
    }
}

test "KV lifetime: Qwen close and deinit retain pages until context proof" {
    for ([_]bool{ false, true }) |whole_pool| {
        var d = D.driver();
        var e = engine(&d);
        defer {
            D.observe = null;
            Watch.pool = null;
            D.fail_wait = false;
            cuda.completion.drain(&d, e.stream, "the test driver must drain before it leaves");
        }
        const l = pages.layout(&e, 64, 16, 0, 0);
        const p = try pages.Pages.init(t.allocator, &e, l, 64, 0, 1, l.need(64, 0, pages.window_rows));
        const c = try p.open(1, 0);
        Watch.pool = &p.pool;
        D.observe = Watch.held;
        D.expected = l.need(1, 0, pages.window_rows);
        D.proofs = 0;
        D.pending = true;
        D.fail_wait = true;
        if (whole_pool) p.deinit() else {
            try t.expect(p.close(c));
            try t.expect(p.pool.idle());
            p.deinit();
        }
        D.observe = null;
        Watch.pool = null;
        try t.expect(D.proofs > 0 and !D.early and !D.pending);
        try t.expectEqual(@as(usize, 0), D.freed_pending);
    }
}
