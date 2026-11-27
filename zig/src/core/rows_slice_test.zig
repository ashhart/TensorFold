//! Sliced gather: per-file staging cap, slice bounds, verify-before-free and mid-job abandons (#3370 PR 1).
const std = @import("std");
const platform = @import("sys");
const pc = @import("prompt_cache.zig");
const rows = @import("rows_store.zig");
const Peak = @import("rows_peak_allocator.zig").Peak;
const fx = @import("rows_async_fixture.zig");
const waits = @import("rows_test_wait.zig");
const Fixture = fx.Fixture;
const Hold = fx.Gate;
const t = std.testing;

fn keepQueued(x: *Fixture, tokens: []const u32, at: u32) !void {
    x.f.prompt = tokens;
    x.f.at = at;
    try t.expect(x.s.keep(tokens, at, null, x.f.starts, x.f.spans));
}

fn waitUntil(check: anytype) !void {
    const end = waits.deadline();
    while (!check.done()) {
        if (rows.write.nowNs() >= end) return error.WaitTimedOut;
        rows.write.io.nap();
    }
}

test "rows slice: a cold save larger than the per-file permit hashes equal to whole staging and restores" {
    for ([_]bool{ false, true }) |planned| for ([_]bool{ false, true }) |verify| {
        const whole = try Fixture.open("slice-whole", .evicted, false, 1 << 20);
        defer whole.close();
        const sliced = try Fixture.open("slice-cap", .off, false, 6000);
        defer sliced.close();
        var peak: Peak = .{ .child = std.heap.page_allocator };
        sliced.r.sliced.payload_allocator = peak.allocator();
        sliced.r.verify = verify;
        const tokens = fx.prompt(201);
        for ([_]*Fixture{ whole, sliced }) |x| {
            x.f.spans = &.{.{ 400, 450 }};
            x.f.planned = planned;
            x.f.starts = if (planned) &.{ 0, 256, 512, 768, 1000 } else &.{};
            x.s.rules.planned = planned;
            x.r.keying.planned = planned;
            try x.keep(&tokens, 1000);
        }
        whole.s.remove(0);
        try whole.settle();
        try t.expectEqual(@as(u64, 0), sliced.r.counts.skipped);
        try sliced.idle(1);
        try t.expectEqual(@as(u64, 0), sliced.r.counts.skipped);
        try t.expectEqual(@as(usize, 1), sliced.r.states.items.len);
        try t.expectEqual(@as(usize, 0), peak.live);
        try t.expect(peak.peak <= sliced.r.staging_bytes);
        if (verify) {
            try t.expectEqual(@as(u64, 1), sliced.r.counts.verified);
            try t.expectEqual(@as(u64, 0), sliced.r.counts.differed);
        }
        const d = platform.c.opendir(whole.r.dir) orelse return error.OpenFailed;
        defer _ = platform.c.closedir(d);
        var checked: usize = 0;
        while (platform.c.readdir(d)) |ent| {
            const name = std.mem.sliceTo(&ent.name, 0);
            if (!std.mem.endsWith(u8, name, ".st") and !std.mem.endsWith(u8, name, ".blk")) continue;
            try t.expectEqual(try fx.digest(whole.r.dir, name), try fx.digest(sliced.r.dir, name));
            checked += 1;
        }
        try t.expect(checked > 3);
        try t.expectEqual(@as(usize, 0), fx.files(sliced.r.dir, ".part"));
        const saved = try rows.read.read(sliced.r, 0);
        defer sliced.f.snapshots().vtable.drop(&sliced.f, saved);
        try sliced.f.snapshots().vtable.restore(&sliced.f, null, saved);
        try t.expect(sliced.f.restored_ok);
    };
}

test "rows slice: one step gathers one slice and peak staged payload stays under the cap while the worker is held" {
    const x = try Fixture.open("slice-peak", .off, false, 6000);
    defer x.close();
    var peak: Peak = .{ .child = std.heap.page_allocator };
    x.r.sliced.payload_allocator = peak.allocator();
    var hold: Hold = .{};
    defer hold.release.store(true, .release);
    x.r.file_io = hold.io();
    const tokens = fx.prompt(202);
    try keepQueued(x, &tokens, 1000);
    try t.expect(rows.write.stepBudget(x.r, 0));
    try t.expectEqual(@as(usize, 1), x.f.gathered);
    try t.expect(x.r.sliced.active != null);
    try Hold.wait(&hold.entered);
    // Keep stepping while the worker is blocked on file 0: the cap must stop gathering at or under 6000 bytes.
    var steps: usize = 0;
    while (rows.write.stepBudget(x.r, 0)) : (steps += 1) try t.expect(steps < 16);
    try t.expect(x.r.sliced.active != null);
    try t.expect(peak.peak <= x.r.staging_bytes);
    try t.expect(x.r.worker.?.slicedUsed() <= x.r.staging_bytes);
    try t.expect(!rows.write.ready(x.r));
    hold.release.store(true, .release);
    try x.idle(1);
    try t.expect(x.r.sliced.active == null);
    try t.expectEqual(@as(usize, 0), peak.live);
    try t.expect(peak.peak <= x.r.staging_bytes);
    try t.expectEqual(@as(u64, 0), x.r.worker.?.bytes());
    try t.expectEqual(@as(usize, 1), x.r.states.items.len);
}

test "rows slice: a single file larger than the whole permit goes alone once earlier payloads free" {
    const x = try Fixture.open("slice-alone", .off, false, 1024);
    defer x.close();
    var peak: Peak = .{ .child = std.heap.page_allocator };
    x.r.sliced.payload_allocator = peak.allocator();
    const tokens = fx.prompt(203);
    try x.keep(&tokens, 1000);
    try x.idle(1);
    try t.expectEqual(@as(u64, 0), x.r.counts.skipped);
    try t.expectEqual(@as(usize, 1), x.r.states.items.len);
    try t.expectEqual(@as(usize, 0), peak.live);
    try t.expect(peak.peak > x.r.staging_bytes);
    try t.expect(peak.peak < x.s.entries.items[0].bytes);
    const saved = try rows.read.read(x.r, 0);
    defer x.f.snapshots().vtable.drop(&x.f, saved);
    try x.f.snapshots().vtable.restore(&x.f, null, saved);
    try t.expect(x.f.restored_ok);
}

test "rows slice: an entry leaving RAM mid-job abandons cleanly with no files, no .part and no dangling entry" {
    for ([_]rows.Spill{ .off, .finish }) |spill| {
        const x = try Fixture.open("slice-drop", spill, false, 1 << 20);
        defer x.close();
        var peak: Peak = .{ .child = std.heap.page_allocator };
        x.r.sliced.payload_allocator = peak.allocator();
        var hold: Hold = .{};
        defer hold.release.store(true, .release);
        x.r.file_io = hold.io();
        const tokens = fx.prompt(204);
        try keepQueued(x, &tokens, 1000);
        try t.expect(rows.write.stepBudget(x.r, 0));
        try t.expect(x.r.sliced.active != null);
        try Hold.wait(&hold.entered);
        x.s.remove(0);
        try t.expect(x.r.sliced.active == null);
        try t.expectEqual(@as(usize, 0), x.f.live);
        try t.expectEqual(@as(u64, 1), x.r.sliced.abandoned);
        hold.release.store(true, .release);
        try x.settle();
        try t.expectEqual(@as(usize, 0), x.r.states.items.len);
        try t.expectEqual(@as(usize, 0), fx.files(x.r.dir, ".st"));
        try t.expectEqual(@as(usize, 0), fx.files(x.r.dir, ".blk"));
        try t.expectEqual(@as(usize, 0), fx.files(x.r.dir, ".part"));
        try t.expectEqual(@as(u64, 0), x.r.admission.reserved);
        try t.expectEqual(@as(u64, 0), x.r.worker.?.bytes());
        try t.expectEqual(@as(usize, 0), peak.live);
        try t.expectEqual(@as(u64, 1), x.r.counts.failed);
        try t.expectEqual(error.EntryDropped, x.r.failures.records[0].?.err);
    }
}

test "rows slice: an image losing its pages mid-job abandons with no files, no .part and nothing to find" {
    const Whole = struct {
        var ok: bool = true;
        fn check(_: *anyopaque, _: pc.Saved) bool {
            return ok;
        }
    };
    Whole.ok = true;
    const x = try Fixture.open("slice-not-whole", .off, false, 1 << 20);
    defer x.close();
    var peak: Peak = .{ .child = std.heap.page_allocator };
    x.r.sliced.payload_allocator = peak.allocator();
    var img = x.r.img.*;
    img.whole = Whole.check;
    x.r.img = &img;
    const tokens = fx.prompt(205);
    try keepQueued(x, &tokens, 1000);
    try t.expect(rows.write.stepBudget(x.r, 0));
    try t.expect(x.r.sliced.active != null);
    // Let the worker seal file 0 to a .part file and free its payload before the image loses its pages.
    try waitUntil(struct {
        r: *rows.Rows,
        fn done(self: @This()) bool {
            return self.r.worker.?.slicedUsed() == 0;
        }
    }{ .r = x.r });
    try t.expectEqual(@as(usize, 1), fx.files(x.r.dir, ".part"));
    try t.expectEqual(@as(usize, 0), peak.live);
    Whole.ok = false;
    try t.expect(rows.write.stepBudget(x.r, 0));
    try t.expect(x.r.sliced.active == null);
    try t.expectEqual(@as(u64, 1), x.r.sliced.abandoned);
    try x.settle();
    try t.expectEqual(@as(usize, 0), x.r.states.items.len);
    try t.expectEqual(@as(usize, 0), fx.files(x.r.dir, ".st"));
    try t.expectEqual(@as(usize, 0), fx.files(x.r.dir, ".blk"));
    try t.expectEqual(@as(usize, 0), fx.files(x.r.dir, ".part"));
    try t.expectEqual(@as(u64, 0), x.r.admission.reserved);
    try t.expectEqual(@as(usize, 0), peak.live);
    try t.expectEqual(error.NotWhole, x.r.failures.records[0].?.err);
    const reopened = try rows.Rows.open(fx.a, x.f.snapshots(), .{ .root = x.root, .identity = 7, .cap = 1 << 30, .daily = 1 << 30, .floor = 0, .lookahead = 1 });
    defer reopened.deinit();
    try t.expectEqual(@as(usize, 0), reopened.states.items.len);
    try t.expectEqual(@as(usize, 0), reopened.blocks.count());
}

test "rows slice: clear, forgetAll and deinit under a running sliced job leave no file, no .part and nothing to find" {
    const Mode = enum { clear, forget, deinit };
    for ([_]Mode{ .clear, .forget, .deinit }) |mode| {
        const x = try Fixture.open("slice-close", .off, false, 1 << 20);
        var closed = false;
        defer if (!closed) x.close();
        const dir_copy = try fx.a.dupeSentinel(u8, x.r.dir, 0);
        defer fx.a.free(dir_copy);
        const root_copy = try fx.a.dupe(u8, x.root);
        defer fx.a.free(root_copy);
        var peak: Peak = .{ .child = std.heap.page_allocator };
        x.r.sliced.payload_allocator = peak.allocator();
        const tokens = fx.prompt(206);
        try keepQueued(x, &tokens, 1000);
        try t.expect(rows.write.stepBudget(x.r, 0));
        try t.expect(x.r.sliced.active != null);
        try waitUntil(struct {
            r: *rows.Rows,
            fn done(self: @This()) bool {
                return self.r.worker.?.slicedUsed() == 0;
            }
        }{ .r = x.r });
        try t.expectEqual(@as(usize, 1), fx.files(dir_copy, ".part"));
        try t.expectEqual(@as(usize, 0), peak.live);
        switch (mode) {
            .clear => {
                x.s.clear();
                try t.expect(x.r.sliced.active == null);
                try t.expectEqual(@as(usize, 0), x.r.states.items.len);
                try t.expectEqual(@as(usize, 0), fx.files(dir_copy, ".st"));
                try t.expectEqual(@as(usize, 0), fx.files(dir_copy, ".blk"));
                try t.expectEqual(@as(usize, 0), fx.files(dir_copy, ".part"));
                try t.expectEqual(@as(usize, 0), peak.live);
            },
            .forget => {
                _ = x.r.forgetAll();
                try t.expect(x.r.sliced.active == null);
                try x.settle();
                try t.expectEqual(@as(usize, 0), x.r.states.items.len);
                try t.expectEqual(@as(usize, 0), fx.files(dir_copy, ".st"));
                try t.expectEqual(@as(usize, 0), fx.files(dir_copy, ".blk"));
                try t.expectEqual(@as(usize, 0), fx.files(dir_copy, ".part"));
                try t.expectEqual(@as(usize, 0), peak.live);
            },
            .deinit => {
                x.s.deinit();
                x.r.deinit();
                closed = true;
                defer {
                    @import("prompt_imprint_test.zig").rmTree(x.root);
                    fx.a.free(x.root);
                    fx.a.destroy(x);
                }
                try t.expectEqual(@as(usize, 0), fx.files(dir_copy, ".st"));
                try t.expectEqual(@as(usize, 0), fx.files(dir_copy, ".blk"));
                try t.expectEqual(@as(usize, 0), fx.files(dir_copy, ".part"));
                try t.expectEqual(@as(usize, 0), peak.live);
                const reopened = try rows.Rows.open(fx.a, x.f.snapshots(), .{ .root = root_copy, .identity = 7, .cap = 1 << 30, .daily = 1 << 30, .floor = 0, .lookahead = 1 });
                defer reopened.deinit();
                try t.expectEqual(@as(usize, 0), reopened.states.items.len);
                try t.expectEqual(@as(usize, 0), reopened.blocks.count());
            },
        }
    }
}

test "rows slice: an eviction beside an active sliced job uses its separate allowance without deadlocking" {
    const x = try Fixture.sized("slice-evict-beside", .finish, false, 6000, 1100);
    defer x.close();
    const first = fx.prompt(207);
    const second = fx.prompt(208);
    try keepQueued(x, &first, 1000);
    try keepQueued(x, &second, 1000);
    // Start slicing `second` (newest) for one file, then evict `first` (still queued) into the eviction allowance.
    try t.expect(rows.write.stepBudget(x.r, 0));
    try t.expect(x.r.sliced.active != null);
    x.s.remove(0); // evicts `first`, which was kept first at index 0
    try t.expectEqual(@as(usize, 2), x.r.worker.?.pending());
    try x.idle(2);
    try t.expectEqual(@as(u64, 0), x.r.counts.skipped);
    try t.expectEqual(@as(usize, 2), x.r.states.items.len);
}

test "rows slice: hostBytes and startup plan report sliced cap alone when spill is off and plus eviction when on" {
    const off = try Fixture.sized("slice-plan-off", .off, false, 1 << 20, 1100);
    defer off.close();
    const on = try Fixture.sized("slice-plan-on", .finish, false, 1 << 20, 1100);
    defer on.close();
    try t.expectEqual(off.r.staging_bytes, rows.write.hostBytes(off.r));
    try t.expectEqual(on.r.staging_bytes + rows.write.slice.evictionAllowance(on.r), rows.write.hostBytes(on.r));
    var b1: [128]u8 = undefined;
    var b2: [128]u8 = undefined;
    const s1 = rows.write.formatHostPlan(off.r, &b1);
    const s2 = rows.write.formatHostPlan(on.r, &b2);
    try t.expect(std.mem.indexOf(u8, s1, "staging up to 1 MiB host RAM") != null);
    try t.expect(std.mem.indexOf(u8, s1, "eviction") == null);
    try t.expect(std.mem.indexOf(u8, s2, "staging up to 2 MiB host RAM (1 MiB sliced + 1 MiB eviction)") != null);
}
