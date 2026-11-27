//! Actual shared-store lookup and GLM snapshot bytes across off-grid tails, two owners and deferred eviction.
const std = @import("std");
const glm = @import("glm");
const cuda = @import("cuda");
const core = @import("core");
const lanes = @import("lanes");
const adapter = @import("prompt_pages");
const common = @import("test_support/glm_snapshot_compare.zig");
const rec = @import("test_support/glm_snap_driver.zig").Driver;
const Fixture = @import("test_support/glm_snap_fixture.zig").Fixture;
const t = std.testing;
const a = t.allocator;

fn enabled() !?[:0]u8 {
    const old = if (core.profile.lookup("TF_GLM_REQUEST_STARTS")) |v| try a.dupeSentinel(u8, v, 0) else null;
    _ = @import("sys").env.setenv("TF_GLM_REQUEST_STARTS", "1", 1);
    return old;
}

fn reset(old: ?[:0]u8) void {
    if (old) |v| {
        _ = @import("sys").env.setenv("TF_GLM_REQUEST_STARTS", v, 1);
        a.free(v);
    } else _ = @import("sys").env.unsetenv("TF_GLM_REQUEST_STARTS");
}

test "GLM shared starts: actual committed bytes equal unshared state; two private tails never write shared pages" {
    const old = try enabled();
    defer reset(old);
    for ([_]usize{ 293, 511, 512, 513, 805, 1023, 1024, 1535, 2047, 2048, 2049 }) |at| {
        var d = rec.init();
        var f: Fixture = undefined;
        try f.init(&d, 0, 1, true);
        defer f.deinit();
        try f.e.prepareSnapshots(8 * f.e.stateBytes(4096));
        const page = 2 * (512 * f.e.c.kv_lora * 2 + 128 * f.e.c.i_dim * 2);
        var kv = try cuda.DeviceBuffer.alloc(&d, 24 * page);
        defer kv.free();
        var p: glm.pages.Pages = undefined;
        p.pool = try core.kv_pool.Pool.init(a, 24, page);
        defer p.pool.deinit();
        f.e.pages = &p;
        var ids: [24]u32 = undefined;
        try t.expect(p.pool.hold(&ids));
        defer for (ids) |id| p.pool.release(id);
        @memset(rec.bytes(kv.ptr, kv.len), 0xA5);
        var source = f.seq;
        source.st = try common.pagedSeq(&d, &f, kv.ptr, page, ids[6..12]);
        defer source.st.deinit();
        defer source.cuts.deinit(a);
        var seed = f.seq;
        seed.st = try common.pagedSeq(&d, &f, kv.ptr, page, ids[0..6]);
        defer seed.st.deinit();
        const store = (try adapter.Adapter.open(glm.native, a, &f.e, 512, 8)).?;
        defer store.close();
        var prompt: [4097]u32 = @splat(7);
        var q: core.kv_prompts.Query = .of(&prompt);
        var scratch: [8]u32 = undefined;
        var unused: glm.share.Owner = .{};
        if (at != 2048) {
            try f.mark(2048, 2055, 31);
            const cold = try f.e.keepState();
            f.e.cur = &seed;
            try f.e.restoreState(cold);
            f.e.dropState(cold);
            f.e.share_pages = true;
            _ = p.prompts.?.keep(q, 2048, &unused, ids[0..4], scratch[0..4]);
        }
        f.e.share_pages = false;
        f.e.cur = &f.seq;
        try f.mark(at, at + 7, 97);
        const cold = try f.e.keepState();
        f.e.cur = &source;
        try f.e.restoreState(cold);
        f.e.dropState(cold);
        try common.expectSamePrefix(f.e.c, &f.seq.st, &source.st, at, at + 7);
        try source.cuts.append(a, @intCast(at));
        const cuts = [_]u32{@intCast(at)};
        q.starts = &cuts;
        f.e.share_pages = true;
        const full = at / 512;
        if (at == 805) {
            const slot = glm.pages.shareSlot(&f.e);
            try t.expect(slot != std.math.maxInt(u64));
            for (f.e.capacity - 1024..f.e.capacity) |cut| try t.expect(f.e.stateBytes(cut) <= slot);
        }
        _ = p.prompts.?.keep(q, @intCast(at), &unused, ids[6..][0..full], scratch[0..full]);
        if (at != 2048 and full > 0) try t.expect(scratch[0] != ids[6]);
        var s = try lanes.Stream.init(a, .{ .id = "shared-starts", .prompt = &prompt, .max_new = 1, .history_len = @intCast(at), .chunks = &cuts });
        defer s.deinit(a);
        var owners = [_]glm.share.Owner{ .{}, .{} };
        defer for (&owners) |*o| o.close(&f.e);
        var adopted = [_]std.ArrayList(u32){ .empty, .empty };
        defer for (&adopted) |*list| {
            p.give(list.items);
            list.deinit(a);
        };
        for (&owners, &adopted) |*owner, *list| {
            const plan = try glm.share.begin(&f.e, &s, owner, list);
            defer a.free(plan.marks);
            try t.expectEqual(@as(u32, @intCast(at)), plan.from);
            try t.expectEqualSlices(u32, ids[6..][0..full], list.items);
        }
        const saved = owners[0].pending.?;
        try t.expectEqual(@as(u32, 2), f.e.snaps.get(saved.id).?.borrows);
        const held = try a.dupe(u8, rec.bytes(kv.ptr + ids[6] * page, full * page));
        defer a.free(held);
        store.store.limit(.{}); // the saved Kept is destroyed now; each owner holds its value and pins the bytes
        try t.expect(f.e.holds(saved.id) and f.e.snaps.get(saved.id).?.retired);
        for (&owners, &adopted, 0..) |*owner, *list, turn| {
            var mapping: [6]u32 = undefined;
            @memcpy(&mapping, ids[12 + turn * 6 ..][0..6]);
            @memcpy(mapping[0..full], list.items);
            var dest = f.seq;
            dest.st = try common.pagedSeq(&d, &f, kv.ptr, page, &mapping);
            defer dest.st.deinit();
            @memset(rec.bytes(dest.st.arena.buf.ptr, dest.st.arena.buf.len), 0xE9);
            f.e.cur = &dest;
            if (full > 0) {
                mapping[0] = ids[12 + turn * 6];
                const copies = rec.copies;
                try t.expectError(error.SnapshotPages, f.e.restoreState(saved.id));
                try t.expectEqual(copies, rec.copies);
                mapping[0] = list.items[0];
            }
            dest.resume_pin = owner.pin;
            owner.pin = null;
            try f.e.restoreState(saved.id);
            try t.expect(dest.resume_pin == null);
            try common.expectSamePrefix(f.e.c, &f.seq.st, &dest.st, at, at + 7);
            try t.expectEqualSlices(u8, held, rec.bytes(kv.ptr + ids[6] * page, full * page));
            for (rec.bytes(dest.st.table, mapping.len * 4)) |byte| try t.expectEqual(@as(u8, 0xE9), byte);
        }
        try t.expect(!f.e.holds(saved.id));
        f.e.cur = &f.seq;
    }
}

test "GLM shared starts: chunk-history mismatch refuses; cancellation pins, eviction and follower ids release once" {
    const old = try enabled();
    defer reset(old);
    var d = rec.init();
    var f: Fixture = undefined;
    try f.init(&d, 0, 2, false);
    defer f.deinit();
    try f.e.prepareSnapshots(2 * f.e.stateBytes(805));
    try f.mark(805, 805, 11);
    const id = try f.e.keepState();
    var k: glm.engine.Kept = .{ .id = id, .at = 805, .below = 0 };
    var p: glm.pages.Pages = undefined;
    p.prompts = .{ .ctx = &p, .vtable = undefined }; // only the presence is read by snap_restore
    f.e.pages = &p;
    f.e.share_pages = true;
    p.used = &.{};
    f.e.images = null;
    f.e.seqs = .empty;
    defer f.e.seqs.deinit(a);
    try f.e.seqs.append(a, &f.seq);
    var bad: glm.share.Owner = .{ .len = 2000, .cuts = &.{ 300, 805 } };
    try t.expectError(error.NonCanonicalResume, glm.native.snap_restore(&f.e, &bad, &k));
    try t.expectEqual(@as(u32, 0), f.e.snaps.get(id).?.borrows);
    var owner: glm.share.Owner = .{ .len = 2000, .cuts = &.{805} };
    try glm.native.snap_restore(&f.e, &owner, &k);
    f.e.announce = 0;
    f.e.dropState(id);
    try t.expect(f.e.holds(id) and f.e.dropped.items.len == 0);
    f.seq.resume_pin = owner.pin;
    owner.pin = null;
    f.e.release(0); // even the retained slot 0 releases a cancelled request's deferred restore
    try t.expect(!f.e.holds(id));
    try t.expectEqualSlices(u32, &.{@intCast(id)}, f.e.dropped.items);
}

test "GLM shared starts: a snapshot-only parent page is reclaimed while its child is leased" {
    const old = try enabled();
    defer reset(old);
    var d = rec.init();
    d.api.cuMemcpyHtoD_v2 = struct {
        fn put(dst: u64, src: ?*const anyopaque, n: usize) callconv(.c) cuda.abi.Result {
            const bytes: [*]const u8 = @ptrCast(src.?);
            @memcpy(rec.bytes(dst, n), bytes[0..n]);
            return 0;
        }
    }.put;
    var f: Fixture = undefined;
    try f.init(&d, 0, 1, false);
    defer f.deinit();
    try f.e.prepareSnapshots(4 * f.e.stateBytes(4096));
    const page = 2 * (512 * f.e.c.kv_lora * 2 + 128 * f.e.c.i_dim * 2);
    var memory = try cuda.DeviceBuffer.alloc(&d, 6 * page);
    defer memory.free();
    @memset(rec.bytes(memory.ptr, memory.len), 0xA5);
    var leases = [_]core.kv_pool.Lease{ .{}, .{}, .{} };
    var used = [_]bool{ false, false, false };
    var p: glm.pages.Pages = .{ .gpa = a, .e = &f.e, .layout = .{ .fixed_bytes = 0, .token_bytes = 1, .page_tokens = 512, .snap_bytes = 0 }, .page = page, .pool = try core.kv_pool.Pool.init(a, 6, page), .arena = memory, .leases = &leases, .used = &used, .leader = true, .poison = false };
    defer p.pool.deinit();
    defer for (&leases) |*l| p.pool.finish(l);
    var seqs = [_]@TypeOf(f.seq){ f.seq, f.seq, f.seq };
    for (&seqs, 0..) |*s, i| {
        s.st = try common.pagedSeq(&d, &f, memory.ptr, page, &.{});
        if (i == 0) f.e.seqs = .empty;
        try f.e.seqs.append(a, s);
    }
    defer f.e.seqs.deinit(a);
    defer for (&seqs) |*s| s.st.deinit();
    defer f.e.cur = &f.seq;
    f.e.pages = &p;
    const store = (try adapter.Adapter.open(glm.native, a, &f.e, 512, 4)).?;
    defer store.close();
    var prompt: [4097]u32 = @splat(7);
    var q: core.kv_prompts.Query = .of(&prompt);
    var saved: [2]u64 = undefined;
    var physical: [2]u32 = undefined;
    for ([_]usize{ 2048, 805 }, 0..) |at, slot| {
        q.starts = if (at == 805) &.{805} else &.{};
        const ids = try p.openWith(@intCast(slot), at, 0, &.{});
        physical[slot] = ids[0];
        f.e.share_pages = false;
        f.e.cur = &f.seq;
        try f.mark(at, at, @intCast(31 + slot));
        const cold = try f.e.keepState();
        f.e.cur = &seqs[slot];
        try f.e.restoreState(cold);
        f.e.dropState(cold);
        f.e.share_pages = true;
        var owner: glm.share.Owner = .{};
        var path: [4]u32 = undefined;
        _ = p.prompts.?.keep(q, @intCast(at), &owner, ids[0 .. at / 512], path[0 .. at / 512]);
        saved[slot] = f.e.next_snap - 1;
        try t.expect(f.e.holds(saved[slot]));
        p.close(@intCast(slot));
    }
    try t.expect(physical[0] != physical[1]);
    try t.expectEqual(@as(usize, 1), store.store.tree.root.kids.items.len);
    const parent = store.store.tree.root.kids.items[0];
    try t.expect(parent.tails.items.len == 1 and parent.kids.items.len == 1);
    var s = try lanes.Stream.init(a, .{ .id = "reclaim", .prompt = &prompt, .max_new = 1, .history_len = 2048 });
    defer s.deinit(a);
    var owner: glm.share.Owner = .{};
    defer owner.close(&f.e);
    var adopted: std.ArrayList(u32) = .empty;
    defer adopted.deinit(a);
    const plan = try glm.share.begin(&f.e, &s, &owner, &adopted);
    defer a.free(plan.marks);
    try t.expectEqual(@as(u32, 2048), plan.from);
    const live = try p.openWith(0, 2048, 0, adopted.items);
    adopted.clearRetainingCapacity();
    defer p.close(0);
    try t.expectEqual(@as(usize, 5), live.len);
    try t.expectEqual(@as(usize, 0), p.pool.available());
    try t.expectEqual(@as(usize, 1), p.pool.reclaimable());
    try t.expect(p.fits(1, 0));
    var miss_owner: glm.share.Owner = .{};
    var none: std.ArrayList(u32) = .empty;
    defer none.deinit(a);
    const miss = try p.prompts.?.begin(.of(&.{9}), 0, &.{}, &miss_owner, &none);
    defer a.free(miss.marks);
    try t.expect(miss.from == 0 and miss.marks.len == 0 and none.items.len == 0);
    try t.expect(glm.snapshots.refs.reclaimable(&f.e, saved[1]));
    try glm.snapshots.refs.pin(&f.e, saved[1]);
    try t.expect(!glm.snapshots.refs.reclaimable(&f.e, saved[1]));
    glm.snapshots.refs.unpin(&f.e, saved[1]);
    f.e.share_pages = false;
    try t.expect(!glm.snapshots.refs.reclaimable(&f.e, saved[1]));
    f.e.share_pages = true;
    const held = try a.dupe(u8, rec.bytes(memory.ptr + physical[0] * page, 4 * page));
    defer a.free(held);
    const admitted = try p.openWith(2, 1, 0, &.{});
    defer p.close(2);
    try t.expectEqualSlices(u32, &.{physical[1]}, admitted);
    try t.expect(!f.e.holds(saved[1]) and f.e.holds(saved[0]));
    try t.expectEqual(@as(u32, 1), f.e.snaps.get(saved[0]).?.borrows);
    try t.expectEqualSlices(u8, held, rec.bytes(memory.ptr + physical[0] * page, 4 * page));
    try t.expectEqual(@as(usize, 0), p.pool.reclaimable());
}

test "GLM shared starts: canonical snapshot and tree holds reclaim safely in a five-page pool" {
    const old = try enabled();
    defer reset(old);
    const Mode = enum { opening, admission, borrowed, leased, multiple, ledger };
    for (std.enums.values(Mode)) |mode| {
        var d = rec.init();
        d.api.cuMemcpyHtoD_v2 = struct {
            fn put(dst: u64, src: ?*const anyopaque, n: usize) callconv(.c) cuda.abi.Result {
                const bytes: [*]const u8 = @ptrCast(src.?);
                @memcpy(rec.bytes(dst, n), bytes[0..n]);
                return 0;
            }
        }.put;
        var f: Fixture = undefined;
        try f.init(&d, 0, 1, false);
        defer f.deinit();
        try f.e.prepareSnapshots(4 * f.e.stateBytes(4096));
        const page = 2 * (512 * f.e.c.kv_lora * 2 + 128 * f.e.c.i_dim * 2);
        var memory = try cuda.DeviceBuffer.alloc(&d, 5 * page);
        defer memory.free();
        @memset(rec.bytes(memory.ptr, memory.len), 0xA5);
        var leases = [_]core.kv_pool.Lease{ .{}, .{} };
        var used = [_]bool{ false, false };
        var p: glm.pages.Pages = .{ .gpa = a, .e = &f.e, .layout = .{ .fixed_bytes = 0, .token_bytes = 1, .page_tokens = 512, .snap_bytes = 0 }, .page = page, .pool = try core.kv_pool.Pool.init(a, 5, page), .arena = memory, .leases = &leases, .used = &used, .leader = true, .poison = false };
        defer p.pool.deinit();
        defer for (&leases) |*l| p.pool.finish(l);
        f.e.seqs = .empty;
        defer f.e.seqs.deinit(a);
        var seqs = [_]@TypeOf(f.seq){ f.seq, f.seq };
        for (&seqs) |*s| {
            s.st = try common.pagedSeq(&d, &f, memory.ptr, page, &.{});
            try f.e.seqs.append(a, s);
        }
        defer for (&seqs) |*s| s.st.deinit();
        defer f.e.cur = &f.seq;
        f.e.pages = &p;
        const store = (try adapter.Adapter.open(glm.native, a, &f.e, 512, 4)).?;
        defer store.close();
        const mine = try p.openWith(0, 2048, 0, &.{});
        var ids: [4]u32 = undefined;
        @memcpy(&ids, mine[0..4]);
        try f.mark(2048, 2048, 31);
        const cold = try f.e.keepState();
        f.e.cur = &seqs[0];
        try f.e.restoreState(cold);
        f.e.dropState(cold);
        f.e.share_pages = true;
        var prompt: [4097]u32 = @splat(7);
        var q: core.kv_prompts.Query = .of(&prompt);
        var owner: glm.share.Owner = .{};
        var path: [4]u32 = undefined;
        if (mode == .multiple) {
            try seqs[0].st.restoreHost(f.e.stream, 512, 512, &.{0}, .{ 0, 0 });
            q.starts = &.{512};
            _ = p.prompts.?.keep(q, 512, &owner, ids[0..1], path[0..1]);
            try t.expectEqual(@as(usize, 1), store.store.tree.snaps);
            try seqs[0].st.restoreHost(f.e.stream, 2048, 2048, &.{0}, .{ 0, 0 });
            q.starts = &.{};
        }
        _ = p.prompts.?.keep(q, 2048, &owner, &ids, &path);
        try t.expectEqualSlices(u32, &ids, &path);
        const saved = f.e.next_snap - 1;
        try t.expect(f.e.holds(saved));
        p.close(0);
        try t.expectEqual(@as(usize, 1), p.pool.available());
        try t.expectEqual(@as(usize, 0), p.pool.reclaimable());
        for (ids, 0..) |id, i| try t.expectEqual(@as(u32, if (mode == .multiple and i == 0) 3 else 2), p.pool.holders(id));
        const held = try a.dupe(u8, rec.bytes(memory.ptr, 4 * page));
        defer a.free(held);
        const copies = rec.copies;
        if (mode == .admission) {
            f.e.share_pages = false;
            try t.expect(!p.fits(513, 0));
            f.e.share_pages = true;
            _ = @import("sys").env.setenv("TF_GLM_REQUEST_STARTS", "0", 1);
            try t.expect(!p.fits(513, 0));
            _ = @import("sys").env.setenv("TF_GLM_REQUEST_STARTS", "1", 1);
            try t.expect(f.e.holds(saved) and p.pool.available() == 1);
        }
        var pinned = false;
        defer if (pinned) glm.snapshots.refs.unpin(&f.e, saved);
        if (mode == .borrowed) {
            try glm.snapshots.refs.pin(&f.e, saved);
            pinned = true;
            try t.expect(!p.fits(513, 0));
            try t.expectError(error.LaneMemory, p.openWith(1, 513, 0, &.{}));
            try t.expect(f.e.holds(saved) and p.pool.available() == 1);
            for (ids) |id| try t.expectEqual(@as(u32, 2), p.pool.holders(id));
            glm.snapshots.refs.unpin(&f.e, saved);
            pinned = false;
        }
        if (mode == .leased) {
            for (ids) |id| p.pool.retain(id);
            _ = try p.openWith(0, 2048, 0, &ids);
            try t.expect(!p.fits(513, 0));
            try t.expectError(error.LaneMemory, p.openWith(1, 513, 0, &.{}));
            for (ids) |id| {
                try t.expectEqual(@as(u32, 1), p.pool.holders(id));
                try t.expectEqual(@as(u32, 1), p.pool.leased[id]);
            }
            try t.expectEqualSlices(u8, held, rec.bytes(memory.ptr, 4 * page));
            p.close(0);
        }
        if (mode == .ledger) {
            p.pool.ledger = .{ .ctx = &owner, .want = struct {
                fn want(_: *const anyopaque, n: usize) usize {
                    return n + 5;
                }
            }.want };
            try t.expect(!p.fits(513, 0));
            try t.expectError(error.LaneMemory, p.openWith(1, 513, 0, &.{}));
            p.pool.ledger = null;
        }
        if (mode != .opening) try t.expect(p.fits(513, 0));
        const admitted = try p.openWith(1, 513, 0, &.{});
        defer p.close(1);
        try t.expectEqual(@as(usize, 2), admitted.len);
        try t.expect(!f.e.holds(saved) and store.store.tree.snaps == 0);
        try t.expectEqualSlices(u8, held, rec.bytes(memory.ptr, 4 * page));
        try t.expectEqual(copies, rec.copies);
        try t.expectEqual(@as(usize, 3), p.pool.available());
    }
}
