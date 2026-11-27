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
