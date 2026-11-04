//! Retained staging under restore bursts: bytes, read slots, failure notification, exact fallback and full release.
const std = @import("std");
const rows = @import("rows_store.zig");
const pf = @import("rows_prefetch.zig");
const fixture = @import("rows_scale_fixture.zig");
const t = std.testing;

const Signal = struct {
    calls: std.atomic.Value(usize) = .init(0),
    fn call(ctx: *anyopaque) void {
        const s: *Signal = @ptrCast(@alignCast(ctx));
        _ = s.calls.fetchAdd(1, .acq_rel);
    }
    fn fail(_: *pf.Prefetch, _: *pf.Read) !void {
        return error.InjectedRead;
    }
};

const Counts = struct { reads: usize, charged: u64, allocated: u64, settled: usize };

fn counts(f: *pf.Prefetch) Counts {
    f.mutex.lockUncancelable(f.io);
    defer f.mutex.unlock(f.io);
    var out: Counts = .{ .reads = f.held_reads, .charged = f.held_bytes, .allocated = 0, .settled = 0 };
    for (f.reads.items) |rd| if (rd.phase == .ready or rd.phase == .failed) {
        out.allocated += rd.buf.len;
        out.settled += 1;
    };
    return out;
}

fn wait(f: *pf.Prefetch, n: usize) !Counts {
    const deadline = rows.write.nowNs() + 5 * std.time.ns_per_s;
    while (true) {
        const c = counts(f);
        if (c.settled == n) return c;
        if (rows.write.nowNs() >= deadline) return error.RestoreWaiterStranded;
        rows.write.io.nap();
    }
}

test "restore scale: bursts bound allocated staging and failures wake every admitted read" {
    for ([_]bool{ false, true }) |fails| for ([_]usize{ 0, 2, 8 }) |slots| {
        var tag: [40]u8 = undefined;
        var x = try fixture.Fixture.init(try std.fmt.bufPrint(&tag, "burst-{d}-{}", .{ slots, fails }));
        defer x.close();
        try x.seed(32);
        const r = try rows.Rows.open(t.allocator, x.f.snapshots(), x.options());
        defer r.deinit();
        x.f.prompt = r.states.items[0].tokens;
        const saved = try rows.read.read(r, 0);
        x.f.snapshots().vtable.drop(&x.f, saved);
        const cap: u64 = if (slots == 8) 3 * rows.file.alignment else 16 << 30;
        const expected: usize = if (slots == 8) 3 else slots;
        const f = try pf.make(r, t.io, .{ .reads = slots, .bytes = cap });
        var signal: Signal = .{};
        f.notify = .{ .ctx = &signal, .call = Signal.call };
        if (fails) f.reader = Signal.fail;
        var peak: u64 = 0;
        for (0..8) |_| {
            var names: [32]rows.chain.Name = undefined;
            var made: usize = 0;
            for (0..32) |i| if (f.start(r, i)) |name| {
                names[made] = name;
                made += 1;
            };
            try t.expectEqual(expected, made);
            const c = try wait(f, made);
            try t.expectEqual(made, c.reads);
            try t.expectEqual(@as(u64, made) * rows.file.alignment, c.allocated);
            try t.expect(c.allocated <= c.charged and c.charged <= cap);
            peak = @max(peak, c.allocated + r.stage.len);
            for (names[0..made]) |name| try t.expect(!f.pending(name));
            if (made > 0) {
                const i = r.find(names[0]).?;
                x.f.prompt = r.states.items[i].tokens;
                const restored = try rows.read.read(r, i);
                defer x.f.snapshots().vtable.drop(&x.f, restored);
                try x.f.snapshots().vtable.restore(&x.f, null, restored);
                try t.expect(x.f.restored_ok);
            }
            for (names[0..made]) |name| f.forget(name);
            try t.expectEqual(@as(usize, 0), counts(f).reads);
            try t.expectEqual(@as(u64, 0), counts(f).charged);
        }
        f.stop();
        try t.expectEqual(expected * 8, signal.calls.load(.acquire));
        try t.expectEqual(f.counts.made, f.counts.freed);
        try t.expectEqual(@as(u64, if (fails) expected * 8 else 0), f.counts.failed);
        try t.expectEqual(@as(usize, 0), x.f.live);
        std.debug.print("RESTORE_BURST slots={d} byte_cap={d} fail={} attempted=256 admitted={d} peak_staging_bytes={d} retained_sync_bytes={d} notified={d} released={d}\n", .{ slots, cap, fails, f.counts.made, peak, r.stage.len, signal.calls.load(.acquire), f.counts.freed });
    };
}
