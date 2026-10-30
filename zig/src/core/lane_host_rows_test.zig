//! A lane host's kept states on disk: written while idle, finished on a stop, read back by the next host, asked about.
const std = @import("std");
const platform = @import("sys");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const rmTree = @import("prompt_imprint_test.zig").rmTree;
const LaneHost = api.LaneHost;
const Reason = api.Reason;
const Allocator = std.mem.Allocator;

/// The fake backend's kept states (their history's tokens) as one row a token, for the rows store.
const Snaps = struct {
    fn of(ptr: *anyopaque) *lanes.fake.Fake {
        return @ptrCast(@alignCast(ptr));
    }
    fn bytes(_: *anyopaque, at: u32) u64 {
        return 64 + 4 * @as(u64, at);
    }
    fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!pc.Saved {
        return of(ptr).save(@ptrCast(@alignCast(owner.?)), at);
    }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {
        return error.BackendRestores;
    }
    fn drop(ptr: *anyopaque, saved: pc.Saved) void {
        of(ptr).drop(saved);
    }
    fn kept(saved: pc.Saved) *std.ArrayList(u32) {
        return @ptrCast(@alignCast(saved));
    }
    fn parts(_: *anyopaque, at: u32, out: *std.ArrayList(pc.rows.image.Part), a: Allocator) anyerror!void {
        try out.append(a, .{ .bytes = @as(u64, at) * 4, .row = 4 });
    }
    fn gather(_: *anyopaque, saved: pc.Saved, ranges: []const pc.rows.image.Range, out: []u8) anyerror!void {
        const src = std.mem.sliceAsBytes(kept(saved).items);
        var o: usize = 0;
        for (ranges) |r| {
            @memcpy(out[o..][0..@intCast(r.len)], src[@intCast(r.off)..][0..@intCast(r.len)]);
            o += @intCast(r.len);
        }
    }
    fn blank(ptr: *anyopaque, at: u32) anyerror!pc.Saved {
        const k = try of(ptr).gpa.create(std.ArrayList(u32));
        k.* = .empty;
        try k.appendNTimes(of(ptr).gpa, 0, at);
        return k;
    }
    fn scatter(_: *anyopaque, saved: pc.Saved, ranges: []const pc.rows.image.Range, in: []const u8) anyerror!void {
        const dst = std.mem.sliceAsBytes(kept(saved).items);
        var o: usize = 0;
        for (ranges) |r| {
            @memcpy(dst[@intCast(r.off)..][0..@intCast(r.len)], in[o..][0..@intCast(r.len)]);
            o += @intCast(r.len);
        }
    }
    const vtable: pc.Snapshots.VTable = .{ .bytes = bytes, .save = save, .restore = restore, .drop = drop, .rows = &.{ .parts = parts, .gather = gather, .blank = blank, .scatter = scatter } };
};

const Box = struct {
    mutex: std.Io.Mutex = .init,
    tokens: std.ArrayList(u32) = .empty,
    cached: ?u32 = null,
    done: ?Reason = null,
    fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
        const b: *@This() = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(std.testing.io);
        defer b.mutex.unlock(std.testing.io);
        switch (e.*) {
            .prefilled => |c| b.cached = c,
            .logprobs, .prompt_logprobs, .audio => {},
            .tokens => |t| b.tokens.appendSlice(std.testing.allocator, t) catch {},
            .finished => |f| b.done = f.reason,
        }
    }
    fn wait(b: *@This()) Reason {
        while (true) {
            b.mutex.lockUncancelable(std.testing.io);
            const d = b.done;
            b.mutex.unlock(std.testing.io);
            if (d) |r| return r;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
    }
};

/// One host over the fake backend with a prompt cache and a rows store under `root`; `turns` run in order.
fn session(root: []const u8, spill: pc.rows.Spill, comptime check: fn (h: *LaneHost, r: *pc.rows.Rows, t: *lanes.fake.Fake) anyerror!void) !void {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    const family: pc.Snapshots = .{ .ptr = &target, .vtable = &Snaps.vtable };
    const rows = try pc.rows.Rows.open(gpa, family, .{ .root = root, .identity = 5, .cap = 1 << 30, .floor = 0, .daily = 1 << 30, .spill = spill });
    defer rows.deinit();
    var store = pc.Store.init(gpa, family, .{ .min_gap = 4, .min_prompt = 0 }, 1 << 20);
    defer store.deinit();
    store.disk = rows;
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    host.cache = &store;
    try host.start();
    defer host.stop();
    try check(&host, rows, &target);
}

fn prompt(n: usize) [700]u32 {
    var p: [700]u32 = undefined;
    for (&p, 0..) |*t, i| t.* = @intCast((i * 7919 + n) % 997);
    return p;
}

test "a host writes its kept states while idle and on its stop; the next host resumes from disk, its reply unchanged" {
    var buf: [128]u8 = undefined;
    const root = try std.fmt.bufPrint(&buf, "/tmp/tf-host-rows-{d}", .{platform.c.getpid()});
    defer rmTree(root);
    const S = struct {
        var first: [706]u32 = undefined; // turn one's prompt and its reply
        fn one(h: *LaneHost, r: *pc.rows.Rows, _: *lanes.fake.Fake) anyerror!void {
            const p = prompt(1);
            var b: Box = .{};
            defer b.tokens.deinit(std.testing.allocator);
            const req: api.Request = .{ .prompt = &p, .max_tokens = 6, .history_len = 600 };
            try h.engine().submit(1, &req, .{ .ctx = &b, .event = Box.event });
            try std.testing.expectEqual(Reason.length, b.wait());
            @memcpy(first[0..700], &p);
            @memcpy(first[700..], b.tokens.items);
            var waited: usize = 0;
            while (waited < 500) : (waited += 1) { // written while the host idles (asked on its thread)
                var ask: api.ImprintAsk = .{ .op = .where, .prompt = &first };
                try std.testing.expect(h.engine().imprintAsk(&ask));
                try std.testing.expect(ask.wait(std.testing.io, 5000));
                if (ask.out.disk == 600) break;
                std.Io.sleep(std.testing.io, .fromMilliseconds(10), .awake) catch {};
            }
            try std.testing.expect(waited < 500);
            _ = r;
            const p2 = prompt(2); // another conversation: kept, not yet written when the host stops
            var b2: Box = .{};
            defer b2.tokens.deinit(std.testing.allocator);
            const req2: api.Request = .{ .prompt = &p2, .max_tokens = 2, .history_len = 650 };
            try h.engine().submit(2, &req2, .{ .ctx = &b2, .event = Box.event });
            try std.testing.expectEqual(Reason.length, b2.wait());
        }
        fn two(h: *LaneHost, r: *pc.rows.Rows, _: *lanes.fake.Fake) anyerror!void {
            try std.testing.expectEqual(@as(usize, 2), r.states.items.len); // the stop wrote the second
            // this host keys no spans, so an ask that names one still finds the state
            var spanned: api.ImprintAsk = .{ .op = .where, .prompt = &first, .spans = &.{.{ 100, 110 }} };
            try std.testing.expect(h.engine().imprintAsk(&spanned));
            try std.testing.expect(spanned.wait(std.testing.io, 5000));
            try std.testing.expectEqual(@as(u32, 600), spanned.out.disk);
            var ask: api.ImprintAsk = .{ .op = .where, .prompt = &first };
            try std.testing.expect(h.engine().imprintAsk(&ask));
            try std.testing.expect(ask.wait(std.testing.io, 5000));
            try std.testing.expect(ask.disk and ask.out.disk == 600 and ask.out.ram == 0 and ask.out.from == .disk);
            var b: Box = .{};
            defer b.tokens.deinit(std.testing.allocator);
            const req: api.Request = .{ .prompt = &first, .max_tokens = 8, .history_len = 704 };
            try h.engine().submit(3, &req, .{ .ctx = &b, .event = Box.event });
            try std.testing.expectEqual(Reason.length, b.wait());
            try std.testing.expectEqual(@as(?u32, 600), b.cached);
            // the fake's tokens read the whole history: a wrong row changes them
            var history: std.ArrayList(u32) = .empty;
            defer history.deinit(std.testing.allocator);
            try history.appendSlice(std.testing.allocator, &first);
            for (b.tokens.items) |t| {
                try std.testing.expectEqual(lanes.fake.next(history.items, null, history.items.len), t);
                try history.append(std.testing.allocator, t);
            }
            var gone: api.ImprintAsk = .{ .op = .forget };
            try std.testing.expect(h.engine().imprintAsk(&gone));
            try std.testing.expect(gone.wait(std.testing.io, 5000));
            try std.testing.expect(gone.freed > 0 and r.states.items.len == 0 and r.blocks.count() == 0);
        }
    };
    try session(root, .off, S.one);
    try session(root, .off, S.two);
}

test "rows I/O: a host serves another turn while its disk writer is blocked" {
    const Fixture = @import("rows_async_fixture.zig");
    const Check = struct {
        fn run(h: *LaneHost, r: *pc.rows.Rows, _: *lanes.fake.Fake) !void {
            var gate: Fixture.Gate = .{};
            var first: Box = .{};
            var second: Box = .{};
            defer {
                gate.release.store(true, .release);
                h.engine().drain();
                first.tokens.deinit(std.testing.allocator);
                second.tokens.deinit(std.testing.allocator);
            }
            r.file_io = gate.io();
            const p = prompt(31);
            const req: api.Request = .{ .prompt = &p, .max_tokens = 6, .history_len = 600 };
            try h.engine().submit(31, &req, .{ .ctx = &first, .event = Box.event });
            try Fixture.Gate.wait(&gate.entered);
            const q = prompt(32);
            const next: api.Request = .{ .prompt = &q, .max_tokens = 6, .history_len = 600 };
            try h.engine().submit(32, &next, .{ .ctx = &second, .event = Box.event });
            const end = pc.rows.write.nowNs() + 3 * std.time.ns_per_s;
            var result: ?Reason = null;
            while (pc.rows.write.nowNs() < end) {
                second.mutex.lockUncancelable(std.testing.io);
                result = second.done;
                second.mutex.unlock(std.testing.io);
                if (result != null) break;
                pc.rows.write.io.nap();
            }
            gate.release.store(true, .release);
            _ = first.wait();
            _ = second.wait();
            try std.testing.expectEqual(@as(?Reason, .length), result);
            try std.testing.expectEqual(@as(usize, 6), second.tokens.items.len);
        }
    };
    const root = try std.fmt.allocPrint(std.testing.allocator, "{s}/tf-host-io-{d}", .{ if (platform.c.getenv("TMPDIR")) |p| std.mem.span(p) else ".", platform.c.getpid() });
    defer std.testing.allocator.free(root);
    defer rmTree(root);
    try session(root, .off, Check.run);
}

test "rows close: a spill-only host wakes with its RAM conversation on disk and replies exactly as a fresh host" {
    const S = struct {
        var history: [706]u32 = undefined;
        var resumed: [8]u32 = undefined;
        fn one(h: *LaneHost, _: *pc.rows.Rows, _: *lanes.fake.Fake) !void {
            const p = prompt(71);
            var b: Box = .{};
            defer b.tokens.deinit(std.testing.allocator);
            const req: api.Request = .{ .prompt = &p, .max_tokens = 6, .history_len = 600 };
            try h.engine().submit(71, &req, .{ .ctx = &b, .event = Box.event });
            try std.testing.expectEqual(Reason.length, b.wait());
            @memcpy(history[0..700], &p);
            @memcpy(history[700..], b.tokens.items);
            var ask: api.ImprintAsk = .{ .op = .where, .prompt = &history };
            try std.testing.expect(h.engine().imprintAsk(&ask));
            try std.testing.expect(ask.wait(std.testing.io, 5000));
            try std.testing.expectEqual(@as(u32, 600), ask.out.ram);
            try std.testing.expectEqual(@as(u32, 0), ask.out.disk);
        }
        fn two(h: *LaneHost, r: *pc.rows.Rows, _: *lanes.fake.Fake) !void {
            try std.testing.expectEqual(@as(usize, 1), r.states.items.len);
            try turn(h, r, true);
        }
        fn fresh(h: *LaneHost, r: *pc.rows.Rows, _: *lanes.fake.Fake) !void {
            try turn(h, r, false);
        }
        fn turn(h: *LaneHost, r: *pc.rows.Rows, wake: bool) !void {
            var b: Box = .{};
            defer b.tokens.deinit(std.testing.allocator);
            const req: api.Request = .{ .prompt = &history, .max_tokens = 8, .history_len = 704 };
            try h.engine().submit(72, &req, .{ .ctx = &b, .event = Box.event });
            try std.testing.expectEqual(Reason.length, b.wait());
            try std.testing.expectEqual(@as(?u32, if (wake) 600 else 0), b.cached);
            try std.testing.expectEqual(@as(u64, if (wake) 1 else 0), r.counts.reads);
            if (wake) @memcpy(&resumed, b.tokens.items) else try std.testing.expectEqualSlices(u32, &resumed, b.tokens.items);
        }
    };
    const a = std.testing.allocator;
    const tmp = if (platform.c.getenv("TMPDIR")) |p| std.mem.span(p) else ".";
    const root = try std.fmt.allocPrint(a, "{s}/tf-host-wake-{d}", .{ tmp, platform.c.getpid() });
    defer a.free(root);
    defer rmTree(root);
    const fresh = try std.fmt.allocPrint(a, "{s}-fresh", .{root});
    defer a.free(fresh);
    defer rmTree(fresh);
    try session(root, .evicted, S.one);
    try session(root, .evicted, S.two);
    try session(fresh, .evicted, S.fresh);
}
