//! Real Drafted wrapper and CUDA stage/selection around synthetic drafter rows and a history-dependent target.
const std = @import("std");
const q = @import("qwen35");
const lanes = @import("lanes");
const dr = lanes.drafter;
const be = lanes.backend;
const a = std.testing.allocator;
const F = q.draft.frame;
var own_trees = false;
var masked_mode = false;

const Odd = struct {
    fn rules() lanes.grammar.Rules {
        return .{ .ptr = @ptrFromInt(1), .vtable = &.{ .accept = accept, .rollback = rollback, .terminated = terminated, .fill = fill, .drafts = drafts, .free = free } };
    }
    fn accept(_: *anyopaque, token: u32) !bool {
        return token < lanes.fake.vocab and token % 2 == 1;
    }
    fn rollback(_: *anyopaque, _: usize) !void {}
    fn terminated(_: *anyopaque) bool {
        return false;
    }
    fn fill(_: *anyopaque, out: []u32) !void {
        @memset(out, 0);
        for (0..lanes.fake.vocab) |i| if (i % 2 == 1) {
            out[i / 32] |= @as(u32, 1) << @intCast(i % 32);
        };
    }
    fn drafts(_: *anyopaque, _: []u32) !usize {
        return 0;
    }
    fn free(_: *anyopaque) void {}
};

const Head = struct {
    d: q.draft.Draft = undefined,
    storage: [F.stagedLen(2)]u8 align(16) = undefined,
    book: [512]u16 = @splat(0),
    data: std.AutoHashMapUnmanaged(*lanes.Stream, Lane) = .empty,
    widest: usize = 0,
    calls: usize = 0,
    const Lane = struct { ctx: q.draft.Ctx, history: std.ArrayList(u32) = .empty };
    fn init(h: *Head) !void {
        h.d.frame_streams = 128;
        h.d.scratch_one = 0;
        h.d.target_taps = 0;
        h.d.c = @import("qwen35_draft_frame_host.zig").config();
        h.d.w.predecessor = &h.book;
        h.d.w.successor = &h.book;
        h.d.staged = .{ .d = undefined, .bytes = &h.storage };
        if (own_trees) h.d.c.block = 16;
        h.d.gpa = a;
        h.d.streams = .empty;
        h.d.landing = .{};
        try h.data.ensureTotalCapacity(a, 128);
        try h.d.streams.ensureTotalCapacity(a, 128);
        h.d.selection_base = 0;
        h.d.step_ms = 0.001;
        h.d.n_costs = 0;
    }
    fn deinit(h: *Head) void {
        var it = h.data.valueIterator();
        while (it.next()) |l| l.history.deinit(a);
        h.data.deinit(a);
        h.d.streams.deinit(a);
    }
    fn self(p: *anyopaque) *Head {
        return @ptrCast(@alignCast(p));
    }
    fn api(h: *Head) dr.Drafter {
        return .{ .ptr = h, .vtable = &.{ .taps = taps, .facts = facts, .open = open, .absorb = absorb, .hold = hold, .held = held, .release = release, .tree = tree, .chances = chances } };
    }
    fn tree(p: *anyopaque, s: *lanes.Stream, gpa: std.mem.Allocator) !?lanes.stream.Held {
        return self(p).d.drafter().tree(s, gpa);
    }
    fn chances(p: *anyopaque, s: *lanes.Stream, out: []f64) !bool {
        return self(p).d.drafter().chances(s, out);
    }
    fn taps(_: *anyopaque) []const u32 {
        return &.{0};
    }
    fn facts(p: *anyopaque) dr.Facts {
        var f = self(p).d.drafter().facts();
        f.plain_guard = false;
        return f;
    }
    fn open(p: *anyopaque, s: *lanes.Stream) !void {
        const h = self(p);
        try h.data.put(a, s, .{ .ctx = .{ .ring = .{ .d = undefined, .ptr = @intFromPtr(s), .len = 1 } } });
        try h.d.streams.put(a, s, &h.data.getPtr(s).?.ctx);
    }
    fn absorb(p: *anyopaque, items: []const dr.Absorb) !void {
        for (items) |it| {
            const l = self(p).data.getPtr(it.stream).?;
            while (l.history.items.len < it.start) try l.history.append(a, 0);
            l.history.shrinkRetainingCapacity(it.start);
            const rows: [*]const u32 = @ptrFromInt(it.features.buffer + it.features.offset);
            const n = if (it.rows) |rs| rs.len else it.features.rows;
            for (0..n) |i| try l.history.append(a, rows[if (it.rows) |rs| rs[i] else i]);
            l.ctx.end = l.history.items.len;
        }
    }
    fn hold(p: *anyopaque, items: []const dr.Hold) !void {
        const h = self(p);
        h.calls += 1;
        h.widest = @max(h.widest, items.len);
        var contexts: [128]*q.draft.Ctx = undefined;
        for (items, contexts[0..items.len]) |it, *ctx| ctx.* = &h.data.getPtr(it.stream).?.ctx;
        try F.stage(h.d.c, items, contexts[0..items.len], &h.storage);
        const words = std.mem.bytesAsSlice(u32, h.storage[F.words_at..F.hp_at]);
        @memset(words, 0);
        @memset(h.storage[F.hp_at..], 0);
        for (items, 0..) |it, index| {
            var guess: std.ArrayList(u32) = .empty;
            defer guess.deinit(a);
            try guess.appendSlice(a, h.data.getPtr(it.stream).?.history.items);
            try guess.append(a, it.pending);
            for (0..h.d.c.block - 1) |row| {
                const token = lanes.fake.next(guess.items, it.stream.sampling, it.position + row);
                const at = (index * (h.d.c.block - 1) + row) * 43;
                words[at + 2] = token;
                words[at + 3] = (token + 1) % 97;
                words[at + 22] = @bitCast(@as(f32, 1));
                words[at + 23] = @bitCast(@as(f32, 0));
                try guess.append(a, token);
            }
        }
        try F.publish(&h.d, items, contexts[0..items.len]);
        if (own_trees) for (contexts[0..items.len], items) |ctx, it| {
            try std.testing.expect(ctx.grown > 0);
            try std.testing.expect(ctx.grown <= it.depth);
        };
    }
    fn held(p: *anyopaque, streams: []const *lanes.Stream, outs: []const []u32) !void {
        for (streams, outs) |s, out| {
            const ctx = &self(p).data.getPtr(s).?.ctx;
            if (out.len > ctx.held_n) return error.TooFewDrafts;
            @memcpy(out, ctx.held[0..out.len]);
        }
    }
    fn release(p: *anyopaque, s: *lanes.Stream) void {
        _ = self(p).d.streams.remove(s);
        if (self(p).data.fetchRemove(s)) |kv| {
            var l = kv.value;
            l.history.deinit(a);
        }
    }
};
const Snapshot = struct {
    stream: ?*lanes.Stream = null,
    start: u64 = 0,
    n: usize = 0,
    ids: [64]u32 = undefined,
};
var snapshots: [8]Snapshot = @splat(.{});
fn verify(p: *anyopaque, windows: []const be.Window, out: []be.Verified) !void {
    const target: *lanes.fake.Fake = @ptrCast(@alignCast(p));
    for (windows) |w| {
        const slot = for (&snapshots) |*s| {
            if (s.stream == w.stream or s.stream == null) break s;
        } else return error.TooManyStreams;
        slot.stream = w.stream;
        slot.start = w.positions[0] - 1;
        slot.n = w.rows();
        slot.ids[0] = w.pending;
        @memcpy(slot.ids[1..][0..w.tokens.len], w.tokens);
    }
    return target.backend().vtable.verify(p, windows, out);
}
fn features(p: *anyopaque, s: *lanes.Stream, taps: []const u32, start: u64, count: u32) !be.Features {
    for (&snapshots) |*sn| if (sn.stream == s and start >= sn.start and start + count <= sn.start + sn.n) {
        return .{ .buffer = @intFromPtr(&sn.ids), .offset = (start - sn.start) * 4, .rows = count, .row_bytes = 4, .space = .host, .dtype = .u32 };
    };
    const target: *lanes.fake.Fake = @ptrCast(@alignCast(p));
    return target.backend().vtable.features.?(p, s, taps, start, count);
}
fn run(count: usize, depth: u32, restored: u32, seed: ?u64, salt: usize) ![8][32]u32 {
    snapshots = @splat(.{});
    var target: lanes.fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var head: Head = .{};
    try head.init();
    defer head.deinit();
    var b = target.backend();
    var vtable = b.vtable.*;
    vtable.verify = verify;
    vtable.features = features;
    b.vtable = &vtable;
    var wrapped = try lanes.drafted.Drafted.init(a, b, head.api());
    defer wrapped.deinit();
    var facts = wrapped.facts(q.native.laneFacts(64, 128));
    facts.drafts = depth;
    const costs: [64]lanes.config.Cost = blk: {
        var c: [64]lanes.config.Cost = undefined;
        for (&c, 1..) |*v, i| v.* = .{ .width = @intCast(i), .ms = 5 };
        break :blk c;
    };
    facts.window_costs = &costs;
    facts.lane_costs = &costs;
    var cfg = try lanes.Config.init(a, facts, 64, 63);
    defer cfg.deinit(a);
    cfg.timed = false;
    var clock: lanes.fake.FixedClock = .{};
    var engine = lanes.Engine.init(a, &cfg, wrapped.backend(), clock.clock());
    defer engine.deinit();
    var streams: [8]lanes.Stream = undefined;
    var constraints: [8]lanes.grammar.Constraint = @splat(.init(Odd.rules(), (lanes.fake.vocab + 31) / 32, null));
    var saved: [8]?*anyopaque = @splat(null);
    var prompts: [8][14]u32 = undefined;
    var made: usize = 0;
    defer for (streams[0..made], saved[0..made]) |*s, copy| {
        s.deinit(a);
        if (copy) |v| target.drop(v);
    };
    for (streams[0..count], prompts[0..count], saved[0..count], 0..) |*s, *prompt, *copy, i| {
        const id = salt + i;
        for (prompt, 0..) |*t, j| t.* = @intCast(1 + (j * 7 + id * 11) % 89);
        if (restored > 0) {
            var early = try lanes.Stream.init(a, .{ .id = "saved", .prompt = prompt[0..restored], .max_new = 1 });
            defer early.deinit(a);
            try b.prefill(&early);
            copy.* = try target.save(&early, restored);
            b.release(&early);
        }
        s.* = try lanes.Stream.init(a, .{ .id = "frame", .prompt = prompt, .max_new = 32, .drafts = depth > 0, .sampling = if (seed) |v| .{ .seed = v + id, .temperature = 0.7 } else null, .reuse = .{ .saved = copy.*, .at = restored }, .loop_guard = false, .grammar = if (masked_mode) &constraints[i] else null });
        made += 1;
        try engine.addStream(s);
    }
    while (engine.activeCount() > 0) try engine.step();
    var replies: [8][32]u32 = undefined;
    for (streams[0..count], replies[0..count]) |*s, *out| {
        @memcpy(out, s.emitted());
        try std.testing.expectEqual(restored, s.cached);
    }
    if (depth > 0) {
        try std.testing.expect(head.calls > 0);
        try std.testing.expectEqual(count, head.widest);
    }
    return replies;
}
test "Draft frames: each stream's real CUDA frame selection through Drafted equals plain fresh solo at every depth" {
    for ([_]?u64{ null, 17 }) |seed| {
        var expected: [8][32]u32 = undefined;
        for (&expected, 0..) |*out, salt| out.* = (try run(1, 0, 0, seed, salt))[0];
        for ([_]u32{ 0, 1, 3, 7 }) |depth| for ([_]u32{ 0, 5, 13 }) |restore| for ([_]usize{ 1, 2, 4, 8 }) |count| {
            const actual = try run(count, depth, restore, seed, 0);
            try std.testing.expectEqualSlices([32]u32, expected[0..count], actual[0..count]);
        };
    }
}

test "Qwen CUDA contracts: grown trees truncated to admitted nodes equal plain fresh resumed and concurrent" {
    own_trees = true;
    defer own_trees = false;
    for ([_]?u64{ null, 17 }) |seed| {
        const expected = try run(2, 0, 0, seed, 0);
        for ([_]u32{ 3, 15 }) |depth| for ([_]u32{ 0, 13 }) |restored| {
            const both = try run(2, depth, restored, seed, 0);
            for (0..2) |i| try std.testing.expectEqualSlices(u32, &expected[i], &both[i]);
            const solo = try run(1, depth, restored, seed, 1);
            try std.testing.expectEqualSlices(u32, &both[1], &solo[0]);
        };
    }
}

test "Qwen CUDA contracts: grammar-pruned own trees preserve greedy and sampled equality" {
    own_trees = true;
    masked_mode = true;
    defer {
        own_trees = false;
        masked_mode = false;
    }
    for ([_]?u64{ null, 17 }) |seed| {
        const expected = try run(2, 0, 0, seed, 0);
        for ([_]u32{ 3, 15 }) |depth| {
            const actual = try run(2, depth, 13, seed, 0);
            for (0..2) |i| {
                try std.testing.expectEqualSlices(u32, &expected[i], &actual[i]);
                for (actual[i]) |token| try std.testing.expect(token % 2 == 1);
            }
        }
    }
}
