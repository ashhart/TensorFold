//! A --drafter server's memory plan: one charge a stream from admission to release, its drafter ring made after it.
const std = @import("std");
const lanes = @import("lanes");
const cfg = @import("config.zig");
const st = @import("state.zig");
const budget = @import("budget.zig");
const dc = @import("dflash2_config.zig");
const fixture = lanes.drafted_fixture;
const be = lanes.backend;
const dr = lanes.drafter;
const Stream = lanes.Stream;

const gpa = std.testing.allocator;

/// The lanes fake as Ripple's target: each prompt pass admits the stream's charge as backend.prefillFn does.
const Target = struct {
    fake: lanes.fake.Fake,
    c: *const cfg.Config,
    ring: u64,
    ledger: budget.Ledger,
    charges: std.AutoHashMapUnmanaged(*Stream, u64) = .empty,
    live: u64 = 0, // bytes made: caches at admission, rings when the drafter makes them
    peak: u64 = 0,
    vt: be.Backend.VTable = undefined,

    fn deinit(t: *Target) void {
        t.charges.deinit(gpa);
        t.fake.deinit();
    }

    fn backend(t: *Target) be.Backend {
        var b = t.fake.backend();
        t.vt = b.vtable.*;
        t.vt.prefill = prefill;
        t.vt.prefill_features = prefillFeatures;
        t.vt.release = release;
        b.vtable = &t.vt;
        return b;
    }

    fn of(ptr: *anyopaque) *Target {
        const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        return @fieldParentPtr("fake", f);
    }

    fn made(t: *Target, bytes: u64) !void {
        t.live += bytes;
        t.peak = @max(t.peak, t.live);
        try std.testing.expect(t.live <= t.ledger.held and t.ledger.held <= t.ledger.limit);
    }

    /// The whole prompt pass (a stream drafting off): its charge admitted, then the fake's pass.
    fn prefill(ptr: *anyopaque, s: *Stream) anyerror!void {
        try of(ptr).admitted(s);
        try of(ptr).fake.backend().vtable.prefill(ptr, s);
    }

    /// The streamed pass (a stream drafting on): its charge admitted, the fake's pass, then its rows to the drafter.
    fn prefillFeatures(ptr: *anyopaque, s: *Stream, taps: []const u32, sink: be.FeatureSink) anyerror!void {
        const t = of(ptr);
        try t.admitted(s);
        try t.fake.backend().vtable.prefill(ptr, s);
        const n: u32 = @intCast(s.prompt().len);
        try sink.take(sink.ptr, s, try t.fake.backend().vtable.features.?(ptr, s, taps, 0, n), 0);
    }

    /// The stream's charge taken as backend.prefillFn takes it, and its caches made.
    fn admitted(t: *Target, s: *Stream) !void {
        const capacity = s.prompt().len + s.max_new + 1;
        const need = budget.charge(t.c, capacity, s.drafts, t.ring);
        const Admit = struct {
            pub fn growth(_: *const @This()) u64 {
                return 0;
            }
            pub fn shed(_: *const @This(), _: u64) void {}
            pub fn restoreBytes(_: *const @This()) u64 {
                return 0;
            }
            pub fn dropRestore(_: *const @This()) void {}
            pub fn trim(_: *const @This(), _: u64) void {}
            pub fn prepare(_: *const @This()) !void {}
        };
        try budget.admitStream(&t.ledger, need, &Admit{});
        try t.charges.put(gpa, s, need);
        try t.made(st.cacheBytes(t.c, capacity));
    }

    fn release(ptr: *anyopaque, s: *Stream) void {
        const t = of(ptr);
        if (t.charges.fetchRemove(s)) |kv| {
            t.ledger.give(kv.value); // the same bytes admission took
            t.live -= kv.value - (if (s.drafts) t.ring else 0);
        }
        t.fake.backend().vtable.release(ptr, s);
    }
};

/// The fake drafter with Ripple's ring rule: none at open, made at the stream's first absorb or hold.
const Rings = struct {
    inner: fixture.FakeDrafter = .{},
    t: *Target,
    rings: std.AutoHashMapUnmanaged(*Stream, void) = .empty,
    vt: dr.Drafter.VTable = undefined,

    fn deinit(r: *Rings) void {
        r.rings.deinit(gpa);
        r.inner.deinit();
    }

    fn drafter(r: *Rings) dr.Drafter {
        var d = r.inner.drafter();
        r.vt = d.vtable.*;
        r.vt.absorb = absorb;
        r.vt.hold = hold;
        r.vt.release = release;
        d.vtable = &r.vt;
        return d;
    }

    fn of(ptr: *anyopaque) *Rings {
        const f: *fixture.FakeDrafter = @ptrCast(@alignCast(ptr));
        return @fieldParentPtr("inner", f);
    }

    fn make(r: *Rings, s: *Stream) !void {
        if (r.rings.contains(s)) return;
        try std.testing.expect(r.t.charges.contains(s)); // the target admitted the stream's charge first
        try r.rings.put(gpa, s, {});
        try r.t.made(r.t.ring);
    }

    fn absorb(ptr: *anyopaque, items: []const dr.Absorb) anyerror!void {
        const r = of(ptr);
        for (items) |it| try r.make(it.stream);
        return r.inner.drafter().vtable.absorb(ptr, items);
    }

    fn hold(ptr: *anyopaque, items: []const dr.Hold) anyerror!void {
        const r = of(ptr);
        for (items) |it| try r.make(it.stream);
        return r.inner.drafter().vtable.hold(ptr, items);
    }

    fn release(ptr: *anyopaque, s: *Stream) void {
        const r = of(ptr);
        if (r.rings.remove(s)) r.t.live -= r.t.ring;
        r.inner.drafter().vtable.release(ptr, s);
    }
};

/// A Ripple config and its drafter's ring, and a stream's caches at the fixture prompt's capacity.
const Plan = struct { c: cfg.Config, ring: u64, caches: u64 };

const max_new = 40;

fn plan() !Plan {
    const c = try cfg.parse(gpa, @embedFile("fixtures/config.json"));
    const d = try dc.parse(gpa, @embedFile("fixtures/dflash2_config.json"), &c);
    const ring: u64 = @import("dflash2.zig").Drafter.laneBytes(&d);
    return .{ .c = c, .ring = ring, .caches = st.cacheBytes(&c, fixture.p1.len + max_new + 1) };
}

/// The wrapper's prompt pass for `streams` under `limit`: each one's outcome, in order, none released between them.
fn admit(p: *const Plan, limit: u64, drafts: []const bool, out: []anyerror!void) !struct { held: u64, peak: u64, rings: usize } {
    var t: Target = .{ .fake = .{ .gpa = gpa }, .c = &p.c, .ring = p.ring, .ledger = .{ .limit = limit } };
    defer t.deinit();
    var r: Rings = .{ .t = &t };
    defer r.deinit();
    var w = try dc.wrap(gpa, t.backend(), r.drafter());
    defer w.deinit();
    const b = w.backend();
    var ss: [2]Stream = undefined;
    for (drafts, 0..) |on, i| ss[i] = try Stream.init(gpa, .{ .id = "s", .prompt = &fixture.p1, .max_new = max_new, .eos = &.{96}, .drafts = on });
    defer for (ss[0..drafts.len]) |*s| s.deinit(gpa);
    for (ss[0..drafts.len], out) |*s, *o| o.* = b.vtable.prefill(b.ptr, s);
    const held = t.ledger.held;
    const peak = t.peak;
    const rings = r.rings.count();
    for (ss[0..drafts.len]) |*s| b.vtable.release(b.ptr, s);
    try std.testing.expectEqual(@as(u64, 0), t.ledger.held); // every charge given back as taken
    try std.testing.expectEqual(@as(u64, 0), t.live);
    return .{ .held = held, .peak = peak, .rings = rings };
}

test "two drafting streams are charged their rings too: the second waits when both do not fit" {
    const p = try plan();
    const one = p.caches + p.ring;
    var out: [2]anyerror!void = undefined;
    // room for one drafting stream and a second's caches, not its ring: the second is refused, not admitted short
    const got = try admit(&p, 2 * one - 1, &.{ true, true }, &out);
    try out[0];
    try std.testing.expectError(error.LaneMemory, out[1]);
    try std.testing.expectEqual(one, got.held); // the first stream's ring stays in the ledger after its admission
    try std.testing.expectEqual(@as(usize, 1), got.rings);
    try std.testing.expect(got.peak <= 2 * one - 1);
    // with room for both, both hold their caches and rings
    const both = try admit(&p, 2 * one, &.{ true, true }, &out);
    try out[0];
    try out[1];
    try std.testing.expectEqual(2 * one, both.held);
    try std.testing.expectEqual(@as(usize, 2), both.rings);
}

test "a stream with drafting off is charged its caches only: it fits exactly where a drafting one does not" {
    const p = try plan();
    var out: [1]anyerror!void = undefined;
    const off = try admit(&p, p.caches, &.{false}, &out);
    try out[0];
    try std.testing.expectEqual(p.caches, off.held);
    try std.testing.expectEqual(@as(usize, 0), off.rings);
    _ = try admit(&p, p.caches + p.ring - 1, &.{true}, &out);
    try std.testing.expectError(error.LaneMemory, out[0]);
    const on = try admit(&p, p.caches + p.ring, &.{true}, &out);
    try out[0];
    try std.testing.expectEqual(p.caches + p.ring, on.held);
}

test "the streamed prompt pass makes a drafter ring only after the target admitted the stream's charge" {
    const p = try plan();
    var out: [2]anyerror!void = undefined;
    // Rings.make checks the charge is held before a ring is made, and Target.made that nothing made passes the ledger
    const got = try admit(&p, 2 * (p.caches + p.ring), &.{ true, false }, &out);
    try out[0];
    try out[1];
    try std.testing.expectEqual(@as(usize, 1), got.rings);
    try std.testing.expectEqual(2 * p.caches + p.ring, got.peak);
}

test "Ripple's drafter opens a stream with no ring: its buffers wait for the first absorb or hold" {
    var d: @import("dflash2.zig").Drafter = undefined;
    d.gpa = gpa;
    d.lanes = .empty;
    d.follow = null;
    defer d.lanes.deinit(gpa);
    var s = try Stream.init(gpa, .{ .id = "s", .prompt = &fixture.p1, .max_new = max_new, .eos = &.{96} });
    defer s.deinit(gpa);
    const x = d.drafter();
    try x.vtable.open(x.ptr, &s);
    try std.testing.expect(!d.lanes.get(&s).?.made);
    x.vtable.release(x.ptr, &s);
    try std.testing.expectEqual(@as(usize, 0), d.lanes.count());
}
