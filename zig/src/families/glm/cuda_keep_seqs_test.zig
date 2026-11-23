//! TF_GLM_KEEP_SEQS: a released slot keeps its sequence, so its next request starts with no allocation or fill.
const std = @import("std");
const cuda = @import("cuda");
const shard = @import("cuda_shard.zig");
const weights = @import("cuda_weights.zig");
const pages_mod = @import("cuda_pages.zig");
const Engine = @import("cuda_engine.zig").Engine;
const Seq = @import("cuda_engine_types.zig").Seq;
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;

/// The driver: allocations and frees counted, synchronous fills (a fresh arena's zeros) and stream fills by bytes.
const F = struct {
    var next: u64 = 1 << 30;
    var live: usize = 0;
    var allocs: usize = 0;
    var frees: usize = 0;
    var zeroed: usize = 0;
    var stream_bytes: usize = 0;
    fn alloc(out: *u64, n: usize) callconv(.c) cuda.abi.Result {
        out.* = next;
        next += std.mem.alignForward(u64, n + 1, 1 << 20);
        live += 1;
        allocs += 1;
        return 0;
    }
    fn free(_: u64) callconv(.c) cuda.abi.Result {
        live -= 1;
        frees += 1;
        return 0;
    }
    fn clear(_: u64, _: u8, n: usize) callconv(.c) cuda.abi.Result {
        zeroed += n;
        return 0;
    }
    fn clearAsync(_: u64, _: u8, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        stream_bytes += n;
        return 0;
    }
    fn fill(_: u64, _: c_uint, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        stream_bytes += 4 * n;
        return 0;
    }
    fn put(_: u64, _: ?*const anyopaque, _: usize) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn wait(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return 0;
    }
    /// The counts now, for a later difference.
    const Marks = struct { allocs: usize, frees: usize, zeroed: usize, stream_bytes: usize };
    fn mark() Marks {
        return .{ .allocs = allocs, .frees = frees, .zeroed = zeroed, .stream_bytes = stream_bytes };
    }
};

/// An engine with what bind, begin and release read: a driver, TP=2's shapes on a KDA, a DSA and a KDA layer.
const Fake = struct {
    d: cuda.Driver,
    layers: [3]weights.Layer,
    e: Engine,

    /// The engine before its own sequence is made.
    fn setup(f: *Fake, gpa: std.mem.Allocator, kv8: bool) !void {
        F.live = 0;
        f.d = undefined;
        f.d.api.cuMemAlloc_v2 = F.alloc;
        f.d.api.cuMemFree_v2 = F.free;
        f.d.api.cuMemsetD8_v2 = F.clear;
        f.d.api.cuMemsetD8Async = F.clearAsync;
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
        f.e.images = null;
        f.e.drafter = null;
        f.e.settings = .{ .kv8 = kv8 };
        f.e.w = undefined;
        f.e.w.config = .{};
        f.e.w.share = try shard.Share.of(&f.e.w.config, 0, 2);
        f.e.w.layers = &f.layers;
    }

    /// The engine with its own sequence, as Engine.init makes it without pages.
    fn init(f: *Fake, gpa: std.mem.Allocator, kv8: bool, keep: bool) !void {
        try f.setup(gpa, kv8);
        f.e.keep_seqs = keep;
        try f.e.bind(0);
    }

    fn deinit(f: *Fake) void {
        for (f.e.seqs.items) |q| if (q) |s| {
            s.st.deinit();
            s.cuts.deinit(f.e.gpa);
            f.e.gpa.destroy(s);
        };
        f.e.seqs.deinit(f.e.gpa);
    }

    /// One request on `slot`: bound (made on first use), begun (State.reset), then released.
    fn request(f: *Fake, slot: u32) !*Seq {
        try f.e.bind(slot);
        try f.e.begin(null);
        const q = f.e.cur;
        f.e.release(slot);
        return q;
    }
};

test "GLM keep seqs: off, a released slot's sequence is freed and its next request allocates and zeroes a whole one" {
    for ([_]bool{ false, true }) |kv8| {
        var f: Fake = undefined;
        try f.init(std.testing.allocator, kv8, false);
        defer f.deinit();
        const whole = f.e.seqs.items[0].?.st.arena.buf.len;
        _ = try f.request(1);
        try expect(f.e.seqs.items[1] == null and F.live == 1); // slot 1's arena went back, slot 0's stays
        for (0..3) |_| {
            const m = F.mark();
            _ = try f.request(1);
            try expectEqual(m.allocs + 1, F.allocs);
            try expectEqual(m.frees + 1, F.frees);
            try expectEqual(m.zeroed + whole, F.zeroed); // the whole window filled again
        }
        try expect(f.e.cur == f.e.seqs.items[0].? and f.e.slot == 0);
    }
}

test "GLM keep seqs: on, a released slot keeps its sequence and its next request allocates, fills and frees nothing" {
    for ([_]bool{ false, true }) |kv8| {
        var f: Fake = undefined;
        try f.init(std.testing.allocator, kv8, true);
        defer f.deinit();
        const first = try f.request(1);
        try expect(f.e.seqs.items[1].? == first and F.live == 2);
        for (0..3) |_| {
            const m = F.mark();
            const q = try f.request(1);
            try expect(q == first); // the same sequence, its arena untouched but for State.reset
            try expectEqual(m.allocs, F.allocs);
            try expectEqual(m.frees, F.frees);
            try expectEqual(m.zeroed, F.zeroed);
            const st = &q.st;
            const reset_bytes = st.kda_layers * 3 * st.conv_c * 2 + 2 * st.kda_layers * st.rec_layer + 2 * 4;
            try expectEqual(m.stream_bytes + reset_bytes, F.stream_bytes); // conv, rec, pos and MTP length alone
            try expect(st.pos == 0 and st.mtp_len == 0 and st.pending == 0 and st.base == 0 and st.mtp_base == 0);
        }
        try expect(f.e.cur == f.e.seqs.items[0].? and f.e.slot == 0);
    }
}

test "GLM keep seqs: slot 0 keeps its sequence either way, and every slot's arena goes back when the engine closes" {
    for ([_]bool{ false, true }) |keep| {
        var f: Fake = undefined;
        try f.init(std.testing.allocator, true, keep);
        const own = f.e.seqs.items[0].?;
        const m = F.mark();
        try expect(try f.request(0) == own);
        try expect(F.allocs == m.allocs and F.frees == m.frees and F.zeroed == m.zeroed);
        for ([_]u32{ 1, 2, 3 }) |slot| _ = try f.request(slot);
        try expectEqual(@as(usize, if (keep) 4 else 1), F.live);
        try expect(Engine.keepsSequence(&f.e, 0) and Engine.keepsSequence(&f.e, 3) == keep);
        f.deinit();
        try expectEqual(@as(usize, 0), F.live); // every slot's arena given back with the engine
    }
}

test "GLM keep seqs: on, a paged slot still gives its pages back when its stream is released" {
    const gpa = std.testing.allocator;
    var f: Fake = undefined;
    try f.setup(gpa, false);
    f.e.keep_seqs = true;
    const l = try pages_mod.layout(gpa, &f.e, 0);
    try pages_mod.ownSequence(&f.e);
    const p = try pages_mod.Pages.init(gpa, &f.e, l, 2, 8, true);
    f.e.pages = p;
    defer {
        p.deinit();
        f.deinit();
    }
    try std.testing.expectEqualSlices(u32, &.{ 0, 1 }, try p.open(1, 600, 100));
    try f.e.bind(1);
    try expectEqual(@as(usize, 1024), f.e.cur.st.capacity);
    f.e.release(1);
    try expect(f.e.seqs.items[1].?.st.capacity == 0 and p.pool.idle() and p.pool.census().free == 8);
}
