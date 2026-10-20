//! An external drafter in front of the fake target: drafted rounds equal one-token rounds, the contract holds.
const std = @import("std");
const Config = @import("config.zig").Config;
const Model = @import("config.zig").Model;
const Cost = @import("config.zig").Cost;
const Engine = @import("engine.zig").Engine;
const sm = @import("stream.zig");
const fake = @import("fake.zig");
const be = @import("backend.zig");
const dr = @import("drafter.zig");
const Drafted = @import("drafted.zig").Drafted;
const SuffixLookup = @import("proposer.zig").SuffixLookup;
const Sampling = @import("sampling.zig").Sampling;
const grammar = @import("grammar.zig");

const gpa = std.testing.allocator;

/// Rebuilds a stream's history from the target's features and drafts its next tokens, now and then wrong.
pub const FakeDrafter = struct {
    lanes: std.AutoHashMapUnmanaged(*sm.Stream, Lane) = .empty,
    fail: enum { none, open, absorb } = .none,
    facts_: dr.Facts = .{ .depth = 6, .step_ms = 0.5 },
    opens: u32 = 0,
    absorbs: u32 = 0,
    holds: u32 = 0,
    helds: u32 = 0,
    releases: u32 = 0,
    widest_hold: usize = 0,
    rows: u64 = 0,
    prompt_rows: u64 = 0,
    live_rows: u64 = 0,
    observe_first: ?*bool = null,
    ordered: bool = true,
    first_start: ?u64 = null, // the last opened stream's first absorbed row

    const Lane = struct { history: std.ArrayList(u32) = .empty, held: std.ArrayList(u32) = .empty };
    const taps_ = [_]u32{ 44, 47, 49 };
    const unseen = fake.vocab + 1; // a row the drafter has no state for (restored by prompt reuse)

    pub fn deinit(x: *FakeDrafter) void {
        var it = x.lanes.valueIterator();
        while (it.next()) |l| free(l);
        x.lanes.deinit(gpa);
    }

    fn free(l: *Lane) void {
        l.history.deinit(gpa);
        l.held.deinit(gpa);
    }

    pub fn drafter(x: *FakeDrafter) dr.Drafter {
        return .{ .ptr = x, .vtable = &.{ .taps = taps, .facts = facts, .open = open, .absorb = absorb, .hold = hold, .held = held, .release = release } };
    }

    fn self(ptr: *anyopaque) *FakeDrafter {
        return @ptrCast(@alignCast(ptr));
    }

    fn taps(_: *anyopaque) []const u32 {
        return &taps_;
    }

    fn facts(ptr: *anyopaque) dr.Facts {
        return self(ptr).facts_;
    }

    fn open(ptr: *anyopaque, s: *sm.Stream) anyerror!void {
        const x = self(ptr);
        x.opens += 1;
        if (x.fail == .open) return error.Injected;
        const got = try x.lanes.getOrPut(gpa, s);
        if (got.found_existing) return error.OpenedTwice;
        got.value_ptr.* = .{};
        x.first_start = null;
    }

    fn absorb(ptr: *anyopaque, items: []const dr.Absorb) anyerror!void {
        const x = self(ptr);
        x.absorbs += 1;
        if (x.fail == .absorb) return error.Injected;
        for (items) |it| {
            const l = x.lanes.getPtr(it.stream) orelse return error.UnknownStream;
            const f = it.features;
            if (f.space != .host or f.dtype != .u32 or f.row_bytes != 4 or it.follow.len != f.rows or f.ready != .none) return error.Shape;
            const rows: [*]const u32 = @ptrFromInt(f.buffer + f.offset);
            if (x.first_start == null) x.first_start = it.start;
            while (l.history.items.len < it.start) try l.history.append(gpa, unseen);
            l.history.shrinkRetainingCapacity(it.start);
            try l.history.appendSlice(gpa, rows[0..f.rows]);
            for (it.follow[0 .. it.follow.len - 1], rows[1..f.rows]) |a, b| if (a != b) return error.FollowMismatch;
            x.rows += f.rows;
            if (it.prompt) x.prompt_rows += f.rows else x.live_rows += f.rows;
        }
    }

    fn hold(ptr: *anyopaque, items: []const dr.Hold) anyerror!void {
        if (self(ptr).observe_first) |seen| self(ptr).ordered = seen.* and items[0].stream.emitted().len == 1;
        const x = self(ptr);
        x.holds += 1;
        x.widest_hold = @max(x.widest_hold, items.len);
        for (items) |it| {
            const l = x.lanes.getPtr(it.stream) orelse return error.UnknownStream;
            // the row before `pending` must be one it absorbed (hold's contract): no drafting from nothing
            if (l.history.items.len + 1 != it.position) return error.PositionMismatch;
            if (l.history.items[l.history.items.len - 1] == unseen) return error.NoRowToDraftFrom;
            var guess: std.ArrayList(u32) = .empty;
            defer guess.deinit(gpa);
            try guess.appendSlice(gpa, l.history.items);
            try guess.append(gpa, it.pending);
            l.held.clearRetainingCapacity();
            for (0..it.depth) |j| {
                const at = it.position + j;
                var t = fake.next(guess.items, it.stream.sampling, at);
                if ((at *% 2654435761 + j) % 5 == 0) t = (t + 1) % fake.vocab;
                try l.held.append(gpa, t);
                try guess.append(gpa, t);
            }
        }
    }

    fn held(ptr: *anyopaque, streams: []const *sm.Stream, out: []const []u32) anyerror!void {
        const x = self(ptr);
        x.helds += 1;
        for (streams, out) |s, o| {
            const l = x.lanes.getPtr(s) orelse return error.UnknownStream;
            if (l.held.items.len < o.len) return error.TooFewDrafts;
            @memcpy(o, l.held.items[0..o.len]);
        }
    }

    fn release(ptr: *anyopaque, s: *sm.Stream) void {
        const x = self(ptr);
        x.releases += 1;
        if (x.lanes.fetchRemove(s)) |kv| {
            var l = kv.value;
            free(&l);
        }
    }
};

/// The fake target holding features as `Features` promises (a keep overwrites what it drops), refusing anything else.
pub const Strict = struct {
    inner: be.Backend,
    snaps: std.AutoHashMapUnmanaged(*sm.Stream, Snap) = .empty,
    refused: u32 = 0,
    taps: [8]u32 = undefined,
    tap_count: usize = 0, // the layers it keeps, told before any forward

    const Snap = struct { start: u64 = 0, rows: std.ArrayList(u32) = .empty, valid: usize = 0 };
    const poison: u32 = 0xdead;

    pub fn deinit(x: *Strict) void {
        var it = x.snaps.valueIterator();
        while (it.next()) |v| v.rows.deinit(gpa);
        x.snaps.deinit(gpa);
    }

    pub fn backend(x: *Strict) be.Backend {
        return .{ .ptr = x, .vtable = &.{ .prefill = prefill, .first = first, .first_masked = firstMasked, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draftNone, .prepare_features = prepare, .features = features, .release = release } };
    }

    fn self(ptr: *anyopaque) *Strict {
        return @ptrCast(@alignCast(ptr));
    }

    fn snap(x: *Strict, s: *sm.Stream) !*Snap {
        const got = try x.snaps.getOrPut(gpa, s);
        if (!got.found_existing) got.value_ptr.* = .{};
        return got.value_ptr;
    }

    fn prefill(ptr: *anyopaque, s: *sm.Stream) anyerror!void {
        const x = self(ptr);
        try x.inner.prefill(s);
        const sn = try x.snap(s);
        sn.start = s.cached;
        sn.rows.clearRetainingCapacity();
        try sn.rows.appendSlice(gpa, s.prompt()[s.cached..]);
        sn.valid = sn.rows.items.len;
    }

    fn first(ptr: *anyopaque, s: *sm.Stream, position: u64) anyerror!u64 {
        return self(ptr).inner.first(s, position);
    }

    fn firstMasked(ptr: *anyopaque, s: *sm.Stream, position: u64, mask: []const u32) anyerror!u64 {
        const in = self(ptr).inner;
        return in.vtable.first_masked.?(in.ptr, s, position, mask);
    }

    fn queue(ptr: *anyopaque, s: *sm.Stream, feed: be.Feed, position: u64) anyerror!u64 {
        const x = self(ptr);
        (try x.snap(s)).valid = 0;
        return x.inner.queue(s, feed, position);
    }

    fn read(ptr: *anyopaque, handle: u64) anyerror!u32 {
        return self(ptr).inner.read(handle);
    }

    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const x = self(ptr);
        try x.inner.verify(windows, out);
        for (windows) |w| {
            if (w.held != 0) return error.HeldReachedTarget; // Drafted hands host tokens
            const sn = try x.snap(w.stream);
            sn.start = w.positions[0] - 1;
            sn.rows.clearRetainingCapacity();
            try sn.rows.append(gpa, w.pending);
            try sn.rows.appendSlice(gpa, w.tokens);
            sn.valid = sn.rows.items.len;
        }
    }

    fn keep(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const x = self(ptr);
        try x.inner.keep(windows, paths);
        for (windows, paths) |w, path| {
            const sn = try x.snap(w.stream);
            sn.valid = @min(sn.valid, path.len);
            @memset(sn.rows.items[sn.valid..], poison);
        }
    }

    fn prepare(ptr: *anyopaque, taps: []const u32) anyerror!void {
        const x = self(ptr);
        if (x.snaps.count() > 0) return error.PreparedAfterForward;
        @memcpy(x.taps[0..taps.len], taps);
        x.tap_count = taps.len;
    }

    fn draftNone(_: *anyopaque, _: []const be.DraftRequest) anyerror!void {
        return error.TargetDraftCalled;
    }

    fn features(ptr: *anyopaque, s: *sm.Stream, taps: []const u32, start: u64, count: u32) anyerror!be.Features {
        const x = self(ptr);
        if (!std.mem.eql(u32, taps, x.taps[0..x.tap_count])) return error.TapsNotPrepared;
        const sn = x.snaps.getPtr(s) orelse return error.UnknownStream;
        if (start < sn.start or start + count > sn.start + sn.valid) {
            x.refused += 1;
            return error.RowsNotHeld;
        }
        return .{ .buffer = @intFromPtr(sn.rows.items.ptr), .offset = (start - sn.start) * 4, .rows = count, .row_bytes = 4, .space = .host, .dtype = .u32 };
    }

    fn release(ptr: *anyopaque, s: *sm.Stream) void {
        const x = self(ptr);
        x.inner.release(s);
        if (x.snaps.fetchRemove(s)) |kv| {
            var v = kv.value;
            v.rows.deinit(gpa);
        }
    }
};

pub const Mode = enum { plain, wrapped, wrapped_early };
const Case = struct { root_copies: bool = false, prompt: []const u32, max_new: u32 = 40, sampling: ?Sampling = null, drafts: bool = true, restored: u32 = 0, grammar: ?*grammar.Constraint = null };
pub const Out = struct {
    tokens: [][]u32,
    drafted: u64 = 0,
    accepted: u64 = 0,
    cached: u32 = 0,
    head: FakeDrafter,
    rounds: u64,
};

var costs: [16]Cost = undefined;

fn base() Model {
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 5.0 + 0.8 * @as(f64, @floatFromInt(w)) };
    return .{ .exact_width = 16, .gpu_tokens = true, .window_costs = &costs, .hidden_rows = true, .batch_rows = 32, .max_streams = 8 };
}

pub fn run(cases: []const Case, mode: Mode, batched: bool) !Out {
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var head: FakeDrafter = .{ .facts_ = .{ .depth = 6, .step_ms = 0.5, .batched = batched } };
    var strict: Strict = .{ .inner = target.backend() };
    defer strict.deinit();
    var target_backend = strict.backend();
    target_backend.first_before_draft = mode == .wrapped_early;
    var wrapped = try Drafted.init(gpa, target_backend, head.drafter());
    defer wrapped.deinit();
    try std.testing.expectEqual(mode == .wrapped_early, wrapped.backend().first_before_draft);
    var m = base();
    m.root_copies = cases[0].root_copies;
    var cfg = try Config.init(gpa, if (mode != .plain) wrapped.facts(m) else m, 16, 15);
    defer cfg.deinit(gpa);
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, if (mode != .plain) wrapped.backend() else target.backend(), clock.clock());
    defer engine.deinit();
    const streams = try gpa.alloc(sm.Stream, cases.len);
    defer gpa.free(streams);
    const proposers = try gpa.alloc(SuffixLookup, cases.len);
    defer gpa.free(proposers);
    const kept = try gpa.alloc(?*anyopaque, cases.len);
    defer gpa.free(kept);
    defer for (kept) |k| if (k) |saved| target.drop(saved);
    for (cases, streams, proposers, kept) |c, *s, *p, *k| {
        k.* = null;
        if (c.restored > 0) { // the prompt's first `restored` tokens as a kept state, saved by an earlier pass
            var early = try sm.Stream.init(gpa, .{ .id = "early", .prompt = c.prompt[0..c.restored], .max_new = 1, .eos = &.{96} });
            defer early.deinit(gpa);
            const b = target.backend();
            try b.prefill(&early);
            k.* = try target.save(&early, c.restored);
            b.release(&early);
        }
        p.* = try SuffixLookup.init(gpa, .{ .min_match = 4 });
        s.* = try sm.Stream.init(gpa, .{ .id = "s", .prompt = c.prompt, .max_new = c.max_new, .eos = &.{96}, .sampling = c.sampling, .drafts = c.drafts, .proposer = p.proposer(), .reuse = .{ .saved = k.*, .at = c.restored }, .grammar = c.grammar });
    }
    defer for (streams, proposers) |*s, *p| {
        s.deinit(gpa);
        p.deinit();
    };
    for (streams) |*s| try engine.addStream(s);
    while (engine.activeCount() > 0) try engine.step();
    var out: Out = .{ .tokens = try gpa.alloc([]u32, cases.len), .cached = streams[0].cached, .head = head, .rounds = target.rounds };
    out.head.lanes = .empty; // the counters only; the lanes are freed below
    for (out.tokens, streams) |*o, *s| {
        o.* = try gpa.dupe(u32, s.emitted());
        out.drafted += s.drafted;
        out.accepted += s.accepted;
    }
    head.deinit();
    return out;
}

pub fn freeOut(o: Out) void {
    for (o.tokens) |t| gpa.free(t);
    gpa.free(o.tokens);
}

pub const p1 = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5 };
const p2 = [_]u32{ 2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5, 9 };
pub const long = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4 };
pub const sampled: Sampling = .{ .seed = 5, .temperature = 0.7, .top_k = 0, .top_p = 0.95 };

test "drafts off on the wrapped backend never reach the drafter" {
    for ([_]?Sampling{ null, sampled }) |s| {
        const off = try run(&.{.{ .prompt = &p1, .sampling = s, .drafts = false }}, .wrapped, false);
        defer freeOut(off);
        const plain = try run(&.{.{ .prompt = &p1, .sampling = s, .drafts = false }}, .plain, false);
        defer freeOut(plain);
        try std.testing.expectEqualSlices(u32, plain.tokens[0], off.tokens[0]);
        const h = off.head;
        try std.testing.expectEqual(@as(u32, 0), h.opens + h.absorbs + h.holds + h.helds + h.releases);
    }
}

test "shared rounds commit what each stream commits alone, and reach the drafter as one batch" {
    const together = try run(&.{ .{ .prompt = &p1 }, .{ .prompt = &p2, .sampling = sampled, .max_new = 30 } }, .wrapped, true);
    defer freeOut(together);
    const one = try run(&.{.{ .prompt = &p1, .drafts = false }}, .plain, false);
    defer freeOut(one);
    const two = try run(&.{.{ .prompt = &p2, .sampling = sampled, .max_new = 30, .drafts = false }}, .plain, false);
    defer freeOut(two);
    try std.testing.expectEqualSlices(u32, one.tokens[0], together.tokens[0]);
    try std.testing.expectEqualSlices(u32, two.tokens[0], together.tokens[1]);
    try std.testing.expect(together.head.widest_hold >= 2); // both streams' drafts in one hold
    try std.testing.expect(together.head.helds <= together.rounds); // at most one readback a verify
}

test "a restored prompt, part or whole: the drafter starts past it and the output is the one-token decode" {
    for ([_]u32{ 12, long.len }) |at| {
        for ([_]?Sampling{ null, sampled }) |s| {
            const drafted = try run(&.{.{ .prompt = &long, .sampling = s, .restored = at }}, .wrapped, false);
            defer freeOut(drafted);
            const plain = try run(&.{.{ .prompt = &long, .sampling = s, .drafts = false }}, .plain, false);
            defer freeOut(plain);
            try std.testing.expectEqual(at, drafted.cached);
            try std.testing.expectEqualSlices(u32, plain.tokens[0], drafted.tokens[0]);
            try std.testing.expect(drafted.head.first_start.? >= at);
        }
    }
}

test "the depth rule reads the drafter's facts, not the target's" {
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    const prior = [_]f64{ 0.7, 0.5, 0.3 };
    var head: FakeDrafter = .{ .facts_ = .{ .depth = 3, .step_ms = 1.25, .prior = &prior, .plain_guard = true, .batched = true } };
    defer head.deinit();
    var wrapped: Drafted = .{ .gpa = gpa, .target = target.backend(), .drafter = head.drafter() };
    defer wrapped.deinit();
    var t = base();
    t.mtp_step_ms = 0.1;
    t.plain_guard = false;
    t.draft_streams = false;
    const m = wrapped.facts(t);
    try std.testing.expectEqual(@as(f64, 1.25), m.mtp_step_ms);
    try std.testing.expect(m.plain_guard and m.draft_streams and m.mtp and !m.speculate_early);
    try std.testing.expectEqualSlices(f64, &prior, m.draft_prior);
    try std.testing.expectEqual(@as(u32, 3), m.drafts);
    head.facts_.stream_bytes = 4096;
    try std.testing.expectEqual(@as(usize, 1000 + 4096), wrapped.streamBytes(1000)); // admission counts both
}

fn always(_: *anyopaque) bool {
    return true;
}

test "every failure around the prompt pass cleans up once" {
    const Fail = enum { open, absorb, features, cancel };
    for ([_]Fail{ .open, .absorb, .features, .cancel }) |which| {
        var target: fake.Fake = .{ .gpa = gpa };
        defer target.deinit();
        var head: FakeDrafter = .{};
        defer head.deinit();
        if (which == .open) head.fail = .open;
        if (which == .absorb) head.fail = .absorb;
        var tb = target.backend();
        var vt = tb.vtable.*;
        if (which == .features) vt.features = null;
        tb.vtable = &vt;
        var wrapped: Drafted = .{ .gpa = gpa, .target = tb, .drafter = head.drafter() };
        defer wrapped.deinit();
        var cfg = try Config.init(gpa, wrapped.facts(base()), 16, 15);
        defer cfg.deinit(gpa);
        var clock: fake.FixedClock = .{};
        var engine = Engine.init(gpa, &cfg, wrapped.backend(), clock.clock());
        defer engine.deinit();
        var dummy: u8 = 0;
        var s = try sm.Stream.init(gpa, .{ .id = "s", .prompt = &p1, .max_new = 8, .eos = &.{96}, .cancel_check = if (which == .cancel) .{ .ptr = &dummy, .check = always } else null });
        defer s.deinit(gpa);
        const got = engine.addStream(&s);
        try std.testing.expect(std.meta.isError(got));
        try std.testing.expectEqual(@as(u32, 0), target.lanes.count()); // the target released exactly where it should
        try std.testing.expectEqual(@as(u32, 0), head.lanes.count());
        try std.testing.expectEqual(@as(usize, 0), wrapped.lanes.count());
        // the drafter is released only if it opened the stream, and then once
        const want: u32 = if (which == .absorb or which == .features) 1 else 0;
        try std.testing.expectEqual(want, head.releases);
    }
}

test "the strict target holds only what Features promises: a keep drops the rejected rows" {
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var strict: Strict = .{ .inner = target.backend() };
    defer strict.deinit();
    const b = strict.backend();
    try b.vtable.prepare_features.?(b.ptr, &.{49});
    var s = try sm.Stream.init(gpa, .{ .id = "s", .prompt = &p1, .max_new = 8, .eos = &.{96} });
    defer s.deinit(gpa);
    try b.prefill(&s);
    defer b.release(&s);
    _ = try b.features(&s, &.{49}, 0, p1.len);
    try std.testing.expectError(error.TapsNotPrepared, b.features(&s, &.{ 44, 49 }, 0, 1));
    const t = try b.read(try b.first(&s, p1.len));
    const drafts = [_]u32{ 1, 2, 3 };
    const positions = [_]u64{ p1.len + 1, p1.len + 2, p1.len + 3, p1.len + 4 };
    const w = [_]be.Window{.{ .stream = &s, .pending = t, .held = 0, .tokens = &drafts, .parents = null, .positions = &positions }};
    var drawn: [4]u32 = undefined;
    var echo: [3]u32 = undefined;
    var out = [_]be.Verified{.{ .sampled = &drawn, .drafts = &echo }};
    try b.verify(&w, &out);
    try std.testing.expectError(error.RowsNotHeld, b.features(&s, &.{49}, 0, 1)); // the prompt rows are gone
    _ = try b.features(&s, &.{49}, p1.len, 4); // the verify's rows, all held until a keep
    try b.keep(&w, &.{&.{ 0, 1 }});
    _ = try b.features(&s, &.{49}, p1.len, 2); // the kept rows stay after the keep
    try std.testing.expectError(error.RowsNotHeld, b.features(&s, &.{49}, p1.len, 3));
}

/// The wrapper over the strict target, streaming `long` restored whole from a kept state: the head has no row yet.
pub const Restored = struct {
    target: fake.Fake = .{ .gpa = gpa },
    strict: Strict = undefined,
    head: FakeDrafter = .{},
    wrapped: Drafted = undefined,
    saved: *anyopaque = undefined,
    s: sm.Stream = undefined,

    pub fn init(r: *Restored) !be.Backend {
        r.* = .{};
        r.strict = .{ .inner = r.target.backend() };
        r.wrapped = try Drafted.init(gpa, r.strict.backend(), r.head.drafter());
        var early = try sm.Stream.init(gpa, .{ .id = "early", .prompt = &long, .max_new = 1, .eos = &.{96} });
        defer early.deinit(gpa);
        try r.target.backend().prefill(&early);
        r.saved = try r.target.save(&early, long.len);
        r.target.backend().release(&early);
        r.s = try sm.Stream.init(gpa, .{ .id = "s", .prompt = &long, .max_new = 32, .eos = &.{96}, .reuse = .{ .saved = r.saved, .at = long.len } });
        const b = r.wrapped.backend();
        try b.prefill(&r.s);
        return b;
    }

    pub fn deinit(r: *Restored) void {
        r.wrapped.backend().release(&r.s);
        r.s.deinit(gpa);
        r.target.drop(r.saved);
        r.wrapped.deinit();
        r.head.deinit();
        r.strict.deinit();
        r.target.deinit();
    }
};

test "a copy round after a whole restore does not leave fillers in place of the drafter's first real drafts" {
    var r: Restored = undefined;
    const b = try r.init();
    defer r.deinit();
    const s = &r.s;
    const head = &r.head;
    const n: u64 = long.len;
    const t = try b.read(try b.first(s, n));
    // the head is asked for 3 drafts but has no row: fillers
    try b.draft(&.{.{ .stream = s, .follow = &.{}, .first = .{ .value = t }, .rows = null, .start = n, .position = n + 1, .depth = 3 }});
    // the round loop takes a copy proposal instead: the fillers are never asked for
    const copy = [_]u32{ 7, 8 };
    var drawn: [3]u32 = undefined;
    var echo: [2]u32 = undefined;
    const w1 = [_]be.Window{.{ .stream = s, .pending = t, .held = 0, .tokens = &copy, .parents = null, .positions = &.{ n + 1, n + 2, n + 3 } }};
    var o1 = [_]be.Verified{.{ .sampled = &drawn, .drafts = &echo }};
    try b.verify(&w1, &o1);
    try b.keep(&w1, &.{&.{ 0, 1 }});
    // the next request has real rows: the drafter holds real drafts
    try b.draft(&.{.{ .stream = s, .follow = &.{ copy[0], drawn[1] }, .rows = &.{ 0, 1 }, .start = n, .position = n + 3, .depth = 3 }});
    try std.testing.expectEqual(@as(u32, 1), head.holds);
    var drawn2: [4]u32 = undefined;
    var echo2: [3]u32 = undefined;
    const w2 = [_]be.Window{.{ .stream = s, .pending = drawn[1], .held = 3, .tokens = &.{}, .parents = null, .positions = &.{ n + 3, n + 4, n + 5, n + 6 } }};
    var o2 = [_]be.Verified{.{ .sampled = &drawn2, .drafts = &echo2 }};
    try b.verify(&w2, &o2);
    // the verify read the drafter (not stale fillers) and the target saw its drafts
    try std.testing.expectEqual(@as(u32, 1), head.helds);
    try std.testing.expectEqualSlices(u32, head.lanes.getPtr(s).?.held.items[0..3], &echo2);
}

test "committed first crosses the external drafter wrapper without changing sampled or shared decode" {
    for ([_]?Sampling{ null, sampled }) |sampling| {
        const cases = [_]Case{
            .{ .prompt = &p1, .sampling = sampling },
            .{ .prompt = &p2, .sampling = sampling },
        };
        const old = try run(&cases, .wrapped, false);
        defer freeOut(old);
        const early = try run(&cases, .wrapped_early, false);
        defer freeOut(early);
        for (old.tokens, early.tokens) |want, got| try std.testing.expectEqualSlices(u32, want, got);
        try std.testing.expectEqual(old.rounds, early.rounds);
        try std.testing.expectEqual(old.drafted, early.drafted);
        try std.testing.expectEqual(old.accepted, early.accepted);
        try std.testing.expectEqual(old.head.rows, early.head.rows);
        try std.testing.expectEqual(old.head.holds, early.head.holds);
        try std.testing.expectEqual(@as(u32, cases.len), early.head.releases);
    }
}

test "the early first token equals the late one after a complete prompt restore" {
    for ([_]?Sampling{ null, sampled }) |sampling| {
        const cases = [_]Case{.{ .prompt = &p1, .sampling = sampling, .restored = p1.len }};
        const old = try run(&cases, .wrapped, false);
        defer freeOut(old);
        const early = try run(&cases, .wrapped_early, false);
        defer freeOut(early);
        try std.testing.expectEqualSlices(u32, old.tokens[0], early.tokens[0]);
        try std.testing.expectEqual(old.cached, early.cached);
        try std.testing.expectEqual(@as(u32, 1), early.head.releases);
    }
}

test "the drafter wrapper sees the committed callback before its first hold" {
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var seen = false;
    var head: FakeDrafter = .{ .observe_first = &seen };
    defer head.deinit();
    var backend = target.backend();
    backend.first_before_draft = true;
    var wrapped = try Drafted.init(gpa, backend, head.drafter());
    defer wrapped.deinit();
    try std.testing.expect(wrapped.backend().first_before_draft);
    var cfg = try Config.init(gpa, wrapped.facts(base()), 16, 15);
    defer cfg.deinit(gpa);
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, wrapped.backend(), clock.clock());
    defer engine.deinit();
    var stream = try sm.Stream.init(gpa, .{ .id = "order", .prompt = &p1, .max_new = 8 });
    defer stream.deinit(gpa);
    const Hook = struct {
        fn call(ptr: *anyopaque) void {
            const flag: *bool = @ptrCast(@alignCast(ptr));
            flag.* = true;
        }
    };
    try engine.addStreamCommitted(&stream, .{ .ptr = &seen, .call = Hook.call });
    try std.testing.expect(seen and head.ordered and head.holds > 0);
    engine.discard(&stream);
}

test {
    _ = @import("drafted_role_test.zig");
}
