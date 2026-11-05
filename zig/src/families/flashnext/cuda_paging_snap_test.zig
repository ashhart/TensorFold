//! Snapshot copy arguments across shuffled pages, using only initialized borrowed address views.
const std = @import("std");
const cuda = @import("cuda");
const pages = @import("cuda_pages.zig");
const prompt = @import("cuda_prompt.zig");
const wt = @import("cuda_prompt_wt.zig");
const snap = @import("cuda_snap.zig");
const kv = @import("core").kv_pool;
const host_tier = @import("exl3/kv/cuda_exl3_kvhost.zig");
const kvt = @import("exl3/kv/cuda_exl3_kvtier.zig");

const Fake = struct {
    var copies: [512][3]u64 = @splat(.{ 0, 0, 0 });
    var n: usize = 0;
    var pending = false;
    var fail_copy: usize = 0;
    var fail_wait = false;
    var early_free = false;
    fn alloc(out: *u64, _: usize) callconv(.c) c_int {
        out.* = 0x8000_0000;
        return 0;
    }
    fn free(_: u64) callconv(.c) c_int {
        early_free = early_free or pending;
        return 0;
    }
    fn copy(dst: u64, src: u64, size: usize, _: cuda.abi.Stream) callconv(.c) c_int {
        copies[n] = .{ dst, src, size };
        n += 1;
        pending = true;
        return if (n == fail_copy) cuda.abi.error_not_ready else 0;
    }
    fn wait(_: cuda.abi.Stream) callconv(.c) c_int {
        if (fail_wait) return cuda.abi.error_not_ready;
        pending = false;
        return 0;
    }
    fn context() callconv(.c) c_int {
        pending = false;
        return 0;
    }
    fn driver() cuda.Driver {
        n = 0;
        pending = false;
        fail_copy = 0;
        fail_wait = false;
        early_free = false;
        var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
        d.api.cuMemAlloc_v2 = alloc;
        d.api.cuMemFree_v2 = free;
        d.api.cuMemcpyDtoDAsync_v2 = copy;
        d.api.cuStreamSynchronize = wait;
        d.api.cuCtxSynchronize = context;
        return d;
    }
};

const Fixture = struct {
    conv: [wt.linear_n]cuda.DeviceBuffer,
    states: [wt.linear_n]cuda.DeviceBuffer,
    kv: [4][wt.attn_n]cuda.DeviceBuffer,
    ids: [4]u32 = .{ 7, 2, 11, 0 },
    fn init(f: *Fixture, d: *const cuda.Driver) void {
        for (&f.conv, &f.states, 0..) |*cv, *st, i| {
            cv.* = .{ .d = d, .ptr = 0x10000 + i * 0x1000, .len = 0, .borrowed = true };
            st.* = .{ .d = d, .ptr = 0x20000 + i * 0x1000, .len = 0, .borrowed = true };
        }
        const p = f.paged(d);
        for (std.enums.values(pages.Part), 0..) |part, i| for (&f.kv[i], 0..) |*v, layer| {
            v.* = .{ .d = d, .ptr = p.at(layer, part), .len = 0, .borrowed = true };
        };
    }
    fn paged(f: *Fixture, d: *const cuda.Driver) pages.Paged {
        return .{ .base = 0x4000_0000, .table = .{ .d = d, .ptr = 0x5000, .len = 0, .borrowed = true }, .ids = &f.ids, .tokens = 64, .own_ids = false };
    }
    fn seq(f: *Fixture, d: *const cuda.Driver) prompt.Seq {
        return .{ .conv = &f.conv, .state = &f.states, .kc = &f.kv[0], .vc = &f.kv[1], .ikc = &f.kv[2], .pooled = &f.kv[3], .paged = f.paged(d), .tail = .{ .d = d, .ptr = 0x9000_0000, .len = 0, .borrowed = true }, .history = .{ 5, 9 }, .fed = 150, .logits = &.{}, .cap = 200 };
    }
};

test "paged snapshots: save and restore keep contiguous order across partial shuffled pages" {
    var d = Fake.driver();
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    var f: Fixture = undefined;
    f.ids = .{ 7, 2, 11, 0 };
    f.init(&d);
    var seq = f.seq(&d);
    const kept = try snap.save(std.testing.allocator, &d, stream, &seq, null);
    defer snap.drop(std.testing.allocator, stream, kept);
    const saved = Fake.copies;
    const n = Fake.n;
    var off: u64 = 0;
    for (saved[0..n]) |copy| {
        try std.testing.expectEqual(kept.buf.ptr + off, copy[0]);
        off += copy[2];
    }
    try std.testing.expectEqual(@as(u64, snap.bytes(150, false, .full)), off);
    const p = f.paged(&d);
    var i: usize = 2 * wt.linear_n;
    for (std.enums.values(pages.Part)) |part| for (0..wt.attn_n) |layer| {
        const per = pages.rows(part, 64);
        const count: usize = if (part == .pooled) 150 / 4 else 150;
        var r: usize = 0;
        while (r < count) : (i += 1) {
            const take = @min(per, count - r);
            try std.testing.expectEqual(p.rowAt(layer, part, r), saved[i][1]);
            try std.testing.expectEqual(take * pages.rowBytes(part), saved[i][2]);
            r += take;
        }
    };
    try std.testing.expectEqual(n - 1, i);
    Fake.n = 0;
    seq.fed = 0;
    try snap.restore(stream, &seq, null, kept);
    try std.testing.expectEqual(n, Fake.n);
    for (saved[0..n], Fake.copies[0..n]) |a, b| try std.testing.expectEqual([3]u64{ a[1], a[0], a[2] }, b);
    try std.testing.expectEqual(@as(usize, 150), seq.fed);
    var flat = seq;
    flat.paged = null;
    Fake.n = 0;
    try snap.restore(stream, &flat, null, kept);
    try std.testing.expectEqual(@as(usize, 2 * wt.linear_n + 4 * wt.attn_n + 1), Fake.n);
}

test "paged snapshots: a failed copy drains before freeing the snapshot destination" {
    var d = Fake.driver();
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    var f: Fixture = undefined;
    f.ids = .{ 7, 2, 11, 0 };
    f.init(&d);
    var seq = f.seq(&d);
    Fake.fail_copy = 5;
    Fake.fail_wait = true;
    try std.testing.expectError(error.NotReady, snap.save(std.testing.allocator, &d, stream, &seq, null));
    try std.testing.expect(!Fake.pending and !Fake.early_free);
}

test "paged snapshots: drop requires context proof when its stream wait fails" {
    var d = Fake.driver();
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    var f: Fixture = undefined;
    f.ids = .{ 7, 2, 11, 0 };
    f.init(&d);
    var seq = f.seq(&d);
    const kept = try snap.save(std.testing.allocator, &d, stream, &seq, null);
    Fake.pending = true;
    Fake.fail_wait = true;
    snap.drop(std.testing.allocator, stream, kept);
    try std.testing.expect(!Fake.pending and !Fake.early_free);
}

test "paged snapshots: the pooled MTP head occupies the thirteenth layer with the same logical byte order" {
    var d = Fake.driver();
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    var f: Fixture = undefined;
    f.ids = .{ 7, 2, 11, 0 };
    f.init(&d);
    var seq = f.seq(&d);
    var p = f.paged(&d);
    p.layers = wt.attn_n + 1;
    seq.paged = p;
    const Draft = @import("cuda_mtp_draft.zig").Draft;
    var head = try Draft.initIn(std.testing.allocator, &d, 200, 8, 16, &p, 0xa000_0000, Draft.runBytes(8, 16));
    defer head.deinit();
    head.len = seq.fed;
    const kept = try snap.save(std.testing.allocator, &d, stream, &seq, &head);
    defer snap.drop(std.testing.allocator, stream, kept);
    const saved = Fake.copies;
    const n = Fake.n;
    var i: usize = 2 * wt.linear_n;
    for (std.enums.values(pages.Part)) |part| {
        const count: usize = if (part == .pooled) 150 / 4 else 150;
        i += wt.attn_n * ((count + pages.rows(part, 64) - 1) / pages.rows(part, 64));
    }
    i += 1;
    for (std.enums.values(pages.Part)) |part| {
        const per = pages.rows(part, 64);
        const count: usize = if (part == .pooled) 150 / 4 else 150;
        var row: usize = 0;
        while (row < count) : (i += 1) {
            const take = @min(per, count - row);
            try std.testing.expectEqual(p.rowAt(wt.attn_n, part, row), saved[i][1]);
            try std.testing.expectEqual(take * pages.rowBytes(part), saved[i][2]);
            row += take;
        }
    }
    try std.testing.expectEqual(n, i);
    var off: u64 = 0;
    for (saved[0..n]) |copy| {
        try std.testing.expectEqual(kept.buf.ptr + off, copy[0]);
        off += copy[2];
    }
    try std.testing.expectEqual(@as(u64, snap.bytes(150, true, .full)), off);
    Fake.n = 0;
    seq.fed = 0;
    try snap.restore(stream, &seq, &head, kept);
    try std.testing.expectEqual(n, Fake.n);
    for (saved[0..n], Fake.copies[0..n]) |a, b| try std.testing.expectEqual([3]u64{ a[1], a[0], a[2] }, b);
    try std.testing.expectEqual(@as(usize, 150), head.len);
}

test "exl3: a host-tier snapshot refuses another lease's pin before copying and restores after release" {
    const t = std.testing;
    var d = Fake.driver();
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    var backing: [32]u8 = @splat(7);
    var host: host_tier.Host = .{ .tier = undefined, .store = undefined, .kv_page = 8 };
    try host.tier.init(t.allocator, &backing, 2, 8);
    defer host.tier.deinit();
    var own: kv.tier.Lease = .{};
    try host.tier.reserve(&own, 4, 1);
    defer host.tier.finish(&own);
    try host.tier.cover(&own, 4, null);
    var other: kv.tier.Lease = .{};
    try host.tier.reserve(&other, 0, 1);
    defer host.tier.finish(&other);
    const id = own.pages.ids.items[0];
    const first = try host.tier.pin(&own, id);
    _ = try host.tier.pin(&other, id);
    try host.tier.markDirty(&own, id);
    defer host.tier.sync(id);
    var fids: std.ArrayList(u32) = .empty;
    defer fids.deinit(t.allocator);
    try fids.appendSlice(t.allocator, &.{ first.frame, kvt_none, kvt_none, kvt_none });
    // Restore only consumes these CPU ownership fields; no selector GPU table is used here.
    var selector: kvt.LaneSel = undefined;
    selector.host = &host;
    selector.tl = &own;
    selector.fids = &fids;
    selector.reset_pending = false;
    var f: Fixture = undefined;
    @memcpy(&f.ids, own.pages.ids.items);
    f.init(&d);
    var seq = f.seq(&d);
    seq.sel = &selector;
    seq.paged.?.hbase = 0x6000_0000;
    seq.fed = 0;
    seq.history = .{ 1, 2 };
    const kept: snap.Snap = .{ .buf = .{ .d = &d, .ptr = 0x8000_0000, .len = snap.bytes(150, false, seq.heads), .borrowed = true }, .at = 150, .history = .{ 5, 9 }, .head = false, .heads = seq.heads };
    try t.expectError(error.Pinned, snap.restore(stream, &seq, null, &kept));
    try t.expectEqual(@as(usize, 0), Fake.n);
    try t.expectEqual(@as(usize, 0), seq.fed);
    try t.expectEqual([2]i64{ 1, 2 }, seq.history);
    try t.expectEqual(@as(u32, 2), host.tier.pinned[first.frame]);
    try t.expect(host.tier.dirty[id] and host.tier.valid(id, .{ .frame = first.frame, .gen = first.gen }));
    try t.expectEqual(first.frame, fids.items[0]);
    try t.expect(!selector.reset_pending);
    try host.tier.unpin(&other, id);
    try snap.restore(stream, &seq, null, &kept);
    try t.expectEqual(@as(usize, 150), seq.fed);
    try t.expectEqual(kept.history, seq.history);
    try t.expect(selector.reset_pending);
    try t.expectEqual(kvt_none, fids.items[0]);
    try t.expect(host.tier.resident(id) == null);
    const refilled = try host.tier.pin(&own, id);
    try t.expect(refilled.miss and refilled.gen != first.gen);
    try host.tier.unpin(&own, id);
}

const kvt_none = std.math.maxInt(u32);
