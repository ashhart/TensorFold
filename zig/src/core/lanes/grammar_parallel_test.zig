//! Serial windows are the byte oracle; different matchers deliberately allow different tokens.
const std = @import("std");
const t = std.testing;
const grammar = @import("grammar.zig");
const parallel = @import("grammar_parallel.zig");
const win = @import("windows.zig");
const sm = @import("stream.zig");
const fake = @import("fake.zig");
const et = @import("engine_test.zig");
const Engine = @import("engine.zig").Engine;
const Mode = enum { plain, forced, copied, tree, growing, inactive, opening, finished, budget };

const Matcher = struct {
    id: u32 = 0,
    at: u32 = 0,
    fail: bool = false,
    fills: usize = 0,
    budgets: usize = 0,
    fault_owner: ?std.Thread.Id = null,
    worker_faults: usize = 0,
    busy: std.atomic.Value(bool) = .init(false),
    fn cast(p: *anyopaque) *Matcher {
        return @ptrCast(@alignCast(p));
    }
    fn token(m: *const Matcher, at: u32) u32 {
        return 1000 + 8 * m.id + at;
    }
    fn rules(m: *Matcher) grammar.Rules {
        return .{ .ptr = m, .parallel_walk = true, .vtable = &.{ .accept = accept, .rollback = rollback, .terminated = terminated, .fill = fill, .drafts = drafts, .free = free, .budget = budget } };
    }
    fn accept(p: *anyopaque, token_id: u32) anyerror!bool {
        const m = cast(p);
        if (token_id != m.token(m.at)) return false;
        m.at += 1;
        return true;
    }
    fn rollback(p: *anyopaque, n: usize) anyerror!void {
        cast(p).at -= @intCast(n);
    }
    fn terminated(p: *anyopaque) bool {
        return cast(p).at > 3;
    }
    fn fill(p: *anyopaque, words: []u32) anyerror!void {
        const m = cast(p);
        if (m.busy.swap(true, .acq_rel)) return error.ConcurrentMatcher;
        defer m.busy.store(false, .release);
        m.fills += 1;
        if (m.fail) return if (m.id == 1) error.FirstStreamFailure else error.LaterStreamFailure;
        @memset(words, 0);
        const next = m.token(m.at);
        words[next / 32] |= @as(u32, 1) << @intCast(next % 32);
        if (m.fault_owner) |owner| if (std.Thread.getCurrentId() != owner) {
            words[0] ^= 1;
            m.worker_faults += 1;
        };
    }
    fn drafts(p: *anyopaque, out: []u32) anyerror!usize {
        const m = cast(p);
        const n = @min(out.len, 3 -| m.at);
        for (out[0..n], 0..) |*v, i| v.* = m.token(m.at + @as(u32, @intCast(i)));
        return n;
    }
    fn budget(ptr: *anyopaque, words: []u32, left: u32) anyerror!bool {
        cast(ptr).budgets += 1;
        if (words.len > 0 and left == 0) @memset(words, 0);
        return true;
    }
    fn free(_: *anyopaque) void {}
};

const Fixture = struct {
    cfg: @import("config.zig").Config = undefined,
    target: fake.Fake = .{ .gpa = t.allocator },
    clock: @import("backend.zig").WallClock = .{ .io = t.io },
    engine: Engine = undefined,
    streams: [32]sm.Stream = undefined,
    matchers: [32]Matcher = undefined,
    constraints: [32]grammar.Constraint = undefined,
    plans: [32]win.Plan = undefined,
    tokens: [32][4]u32 = undefined,
    n: usize = 0,

    fn init(f: *Fixture, n: usize, mode: Mode) !void {
        f.cfg = try et.model();
        f.engine = Engine.init(t.allocator, &f.cfg, f.target.backend(), f.clock.clock());
        f.engine.steps = 1;
        f.engine.alone = mode == .growing;
        for (0..n) |i| {
            f.streams[i] = try sm.Stream.init(t.allocator, .{ .id = "masks", .prompt = &.{1}, .max_new = 16, .drafts = mode == .growing });
            f.n += 1;
            f.streams[i].pending = 9;
            f.matchers[i] = .{ .id = @intCast(i), .at = if (mode == .finished) 4 else 0 };
            f.constraints[i] = .init(f.matchers[i].rules(), 7760, if (mode == .inactive or mode == .opening) 99 else null);
            f.streams[i].grammar = &f.constraints[i];
            if (mode == .budget) {
                try f.constraints[i].arm();
                f.constraints[i].budget.?.max_new = 16;
            }
            const m = &f.matchers[i];
            f.tokens[i] = if (mode == .opening) .{ 99, m.token(0), 777, m.token(1) } else .{ m.token(0), m.token(1), 777, m.token(0) };
            f.plans[i] = .{ .stream = &f.streams[i], .position = 1, .kind = if (mode == .forced) .forced else if (mode == .plain) .none else .copy, .tokens = if (mode == .plain or mode == .growing) &.{} else &f.tokens[i], .parents = if (mode == .tree) &.{ -1, 0, -1, -1 } else null, .branch_from = if (mode == .tree) 3 else 0, .graft_recalled = if (mode == .tree) &.{ true, false } else &.{} };
        }
    }
    fn deinit(f: *Fixture) void {
        f.engine.deinit();
        for (f.streams[0..f.n]) |*s| s.deinit(t.allocator);
        f.target.deinit();
        f.cfg.deinit(t.allocator);
    }
    fn run(f: *Fixture, threaded: bool) !void {
        if (threaded) try parallel.constrain(&f.engine, f.plans[0..f.n]) else for (f.plans[0..f.n]) |*p| try win.constrain(&f.engine, p);
    }
};

fn masksEqual(a: *const Fixture, b: *const Fixture) bool {
    for (a.plans[0..a.n], b.plans[0..b.n]) |x, y| if (!std.mem.eql(u32, x.masks, y.masks)) return false;
    return true;
}
fn compare(a: *const Fixture, b: *const Fixture) !void {
    try t.expect(masksEqual(a, b));
    for (a.plans[0..a.n], b.plans[0..b.n], 0..) |x, y, i| {
        try t.expectEqualSlices(u32, x.tokens, y.tokens);
        try t.expectEqualSlices(i32, x.parents orelse &.{}, y.parents orelse &.{});
        try t.expectEqual(x.kind, y.kind);
        try t.expectEqual(x.grammar_from, y.grammar_from);
        try t.expectEqual(x.branch_from, y.branch_from);
        try t.expectEqualSlices(bool, x.graft_recalled, y.graft_recalled);
        try t.expectEqual(a.matchers[i].at, b.matchers[i].at);
        try t.expectEqual(a.matchers[i].budgets, b.matchers[i].budgets);
        try t.expect(a.constraints[i].profile == null and b.constraints[i].profile == null);
    }
}

test "grammar parallel: 1/4/8/16/32 streams keep every serial mask byte and pruned plan" {
    for ([_]usize{ 1, 4, 8, 16, 32 }) |n| for (std.meta.tags(Mode)) |mode| {
        var serial: Fixture = .{};
        try serial.init(n, mode);
        defer serial.deinit();
        var threaded: Fixture = .{};
        try threaded.init(n, mode);
        defer threaded.deinit();
        try serial.run(false);
        try threaded.run(true);
        try compare(&serial, &threaded);
    };
}

test "grammar parallel: real mask results catch wrong-stream and changed-bit planted controls" {
    var serial: Fixture = .{};
    try serial.init(8, .plain);
    defer serial.deinit();
    var threaded: Fixture = .{};
    try threaded.init(8, .plain);
    defer threaded.deinit();
    try serial.run(false);
    try threaded.run(true);
    try compare(&serial, &threaded);
    const a = threaded.plans[0].masks;
    threaded.plans[0].masks = threaded.plans[1].masks;
    try t.expect(!masksEqual(&serial, &threaded));
    threaded.plans[0].masks = a;
    const bits: []u32 = @constCast(threaded.plans[0].masks);
    bits[0] ^= 1;
    try t.expect(!masksEqual(&serial, &threaded));
}

test "grammar parallel: worker errors join, release profiles and return the first stream failure" {
    var f: Fixture = .{};
    try f.init(8, .plain);
    defer f.deinit();
    f.matchers[1].fail = true;
    f.matchers[3].fail = true;
    try t.expectError(error.FirstStreamFailure, f.run(true));
    for (f.matchers[0..f.n], f.constraints[0..f.n]) |*m, c| {
        try t.expectEqual(@as(usize, 1), m.fills);
        try t.expect(!m.busy.load(.acquire));
        try t.expect(c.profile == null);
    }
    try t.expect(f.plans[0].masks.len > 0);
    for (f.plans[1..f.n]) |p| try t.expectEqual(@as(usize, 0), p.masks.len);
}

test "grammar parallel: aliased constraints and matchers retain serial ownership" {
    for ([_]bool{ false, true }) |same_constraint| {
        var f: Fixture = .{};
        try f.init(4, .plain);
        defer f.deinit();
        if (same_constraint) f.streams[1].grammar = f.streams[0].grammar else f.constraints[1].rules = f.constraints[0].rules;
        try t.expect(!parallel.independent(f.plans[0..f.n]));
        try f.run(true);
        try t.expect(f.engine.grammar_workers.pool == null);
        try t.expectEqualSlices(u32, f.plans[0].masks, f.plans[1].masks);
        try t.expectEqual(@as(usize, 2), f.matchers[0].fills);
    }
}

test "grammar parallel: unavailable async workers run every walk inline with the same bytes" {
    var serial: Fixture = .{};
    try serial.init(8, .plain);
    defer serial.deinit();
    var threaded: Fixture = .{};
    try threaded.init(8, .plain);
    defer threaded.deinit();
    try threaded.run(true);
    threaded.engine.grammar_workers.pool.?.threaded.setAsyncLimit(.nothing);
    threaded.engine.steps += 1;
    try serial.run(false);
    try threaded.run(true);
    try compare(&serial, &threaded);
    for (threaded.matchers[0..threaded.n]) |m| try t.expectEqual(@as(usize, 2), m.fills);
}

test "grammar parallel: dormant thinking and finished matchers keep the serial fast path" {
    for ([_]Mode{ .inactive, .finished }) |mode| {
        var f: Fixture = .{};
        try f.init(16, mode);
        defer f.deinit();
        try f.run(true);
        try t.expect(f.engine.grammar_workers.pool == null);
        for (f.matchers[0..f.n]) |m| try t.expectEqual(@as(usize, 0), m.fills);
    }
}

test "grammar parallel: callbacks without a concurrency contract keep serial ownership" {
    var f: Fixture = .{};
    try f.init(8, .plain);
    defer f.deinit();
    f.constraints[2].rules.parallel_walk = false;
    try f.run(true);
    try t.expect(f.engine.grammar_workers.pool == null);
    for (f.matchers[0..f.n]) |m| try t.expectEqual(@as(usize, 1), m.fills);
}

test "grammar parallel: a custom clock without a thread-safety contract stays on the owner" {
    var f: Fixture = .{};
    try f.init(8, .plain);
    defer f.deinit();
    var vtable = f.engine.clock.vtable.*;
    vtable.parallel_host_ns = false;
    f.engine.clock.vtable = &vtable;
    try f.run(true);
    try t.expect(f.engine.grammar_workers.pool == null);
    for (f.matchers[0..f.n]) |m| try t.expectEqual(@as(usize, 1), m.fills);
}

const HeldProbe = struct {
    owner: std.Thread.Id,
    calls: usize = 0,
    fn read(ptr: *anyopaque, _: *sm.Stream, out: []u32) anyerror!void {
        const p: *HeldProbe = @ptrCast(@alignCast(ptr));
        if (std.Thread.getCurrentId() != p.owner) return error.GpuReadOffOwner;
        p.calls += 1;
        for (out, 0..) |*token_id, i| token_id.* = 1000 + @as(u32, @intCast(i));
    }
};

test "grammar parallel: held device drafts are read on the owner before any matcher work" {
    var serial: Fixture = .{};
    try serial.init(8, .plain);
    defer serial.deinit();
    var threaded: Fixture = .{};
    try threaded.init(8, .plain);
    defer threaded.deinit();
    var serial_probe: HeldProbe = .{ .owner = std.Thread.getCurrentId() };
    var threaded_probe: HeldProbe = .{ .owner = std.Thread.getCurrentId() };
    var serial_vtable = serial.engine.backend.vtable.*;
    var threaded_vtable = threaded.engine.backend.vtable.*;
    serial_vtable.held = HeldProbe.read;
    threaded_vtable.held = HeldProbe.read;
    serial.engine.backend.ptr = &serial_probe;
    serial.engine.backend.vtable = &serial_vtable;
    threaded.engine.backend.ptr = &threaded_probe;
    threaded.engine.backend.vtable = &threaded_vtable;
    for ([_]*Fixture{ &serial, &threaded }) |f| for (f.plans[0..f.n]) |*p| {
        p.held = 2;
        p.tokens = &.{};
        p.kind = .head;
    };
    try serial.run(false);
    try threaded.run(true);
    try compare(&serial, &threaded);
    try t.expectEqual(@as(usize, 8), serial_probe.calls);
    try t.expectEqual(@as(usize, 8), threaded_probe.calls);
}

test "grammar parallel: byte oracle catches a fault produced inside worker fill callbacks" {
    if ((std.Thread.getCpuCount() catch 1) < 2) return error.SkipZigTest;
    var serial: Fixture = .{};
    try serial.init(32, .plain);
    defer serial.deinit();
    var faulty: Fixture = .{};
    try faulty.init(32, .plain);
    defer faulty.deinit();
    for ([_]*Fixture{ &serial, &faulty }) |f| for (f.matchers[0..f.n]) |*m| {
        m.fault_owner = std.Thread.getCurrentId();
    };
    try serial.run(false);
    try faulty.run(true);
    var planted: usize = 0;
    for (faulty.matchers[0..faulty.n]) |m| planted += m.worker_faults;
    try t.expect(planted > 0);
    try t.expect(!masksEqual(&serial, &faulty));
    for (serial.matchers[0..serial.n]) |m| try t.expectEqual(@as(usize, 0), m.worker_faults);
    var clean: Fixture = .{};
    try clean.init(32, .plain);
    defer clean.deinit();
    try clean.run(true);
    try compare(&serial, &clean);
}

const Mixed = struct {
    cfg: @import("config.zig").Config = undefined,
    target: fake.Fake = .{ .gpa = t.allocator },
    clock: @import("backend.zig").WallClock = .{ .io = t.io },
    vtable: @import("backend.zig").Clock.VTable = undefined,
    backend_vtable: @import("backend.zig").Backend.VTable = undefined,
    engine: Engine = undefined,
    streams: [8]sm.Stream = undefined,
    classes: [8]et.Classes = undefined,
    constraints: [8]grammar.Constraint = undefined,
    masks: [8][4]u32 = undefined,
    mask_lengths: [8]usize = @splat(0),
    n: usize = 0,

    fn init(f: *Mixed, threaded: bool) !void {
        f.cfg = try et.model();
        f.vtable = f.clock.clock().vtable.*;
        f.vtable.parallel_host_ns = threaded;
        var clock = f.clock.clock();
        clock.vtable = &f.vtable;
        var backend = f.target.backend();
        f.backend_vtable = backend.vtable.*;
        f.backend_vtable.verify = verify;
        backend.vtable = &f.backend_vtable;
        f.engine = Engine.init(t.allocator, &f.cfg, backend, clock);
        for (0..8) |i| {
            f.classes[i] = .{ .length = if (i == 0) 2 else 7, .spell = true };
            var rules = f.classes[i].rules();
            rules.parallel_walk = true;
            f.constraints[i] = .init(rules, (fake.vocab + 31) / 32, null);
            f.streams[i] = try sm.Stream.init(t.allocator, .{ .id = "mixed", .prompt = &et.p1, .max_new = 12, .eos = &.{96}, .drafts = false, .grammar = if (i < 6) &f.constraints[i] else null });
            f.n += 1;
            try f.engine.addStream(&f.streams[i]);
        }
    }
    fn deinit(f: *Mixed) void {
        f.engine.deinit();
        for (f.streams[0..f.n]) |*s| s.deinit(t.allocator);
        f.target.deinit();
        f.cfg.deinit(t.allocator);
    }
    fn verify(ptr: *anyopaque, windows: []const @import("backend.zig").Window, out: []@import("backend.zig").Verified) anyerror!void {
        const target: *fake.Fake = @ptrCast(@alignCast(ptr));
        const f: *Mixed = @alignCast(@fieldParentPtr("target", target));
        for (windows) |w| for (&f.streams, 0..) |*s, i| {
            if (w.stream != s) continue;
            if (w.masks.len > f.masks[i].len) return error.UnexpectedMixedMaskRows;
            @memcpy(f.masks[i][0..w.masks.len], w.masks);
            f.mask_lengths[i] = w.masks.len;
        };
        try target.backend().verify(windows, out);
    }
};

test "grammar parallel: mixed constrained and free streams keep serial bytes when one finishes mid-batch" {
    var serial: Mixed = .{};
    try serial.init(false);
    defer serial.deinit();
    var threaded: Mixed = .{};
    try threaded.init(true);
    defer threaded.deinit();
    var continued = false;
    var steps: usize = 0;
    while (serial.engine.activeCount() > 0) : (steps += 1) {
        try t.expect(steps < 20);
        serial.mask_lengths = @splat(0);
        threaded.mask_lengths = @splat(0);
        try serial.engine.step();
        try threaded.engine.step();
        for (&serial.streams, &threaded.streams, 0..) |*a, *b, i| {
            try t.expectEqualSlices(u32, a.emitted(), b.emitted());
            try t.expectEqual(a.finished, b.finished);
            try t.expectEqual(a.reason, b.reason);
            try t.expectEqual(serial.mask_lengths[i], threaded.mask_lengths[i]);
            try t.expectEqualSlices(u32, serial.masks[i][0..serial.mask_lengths[i]], threaded.masks[i][0..threaded.mask_lengths[i]]);
            if (i >= 6) try t.expectEqual(@as(usize, 0), threaded.mask_lengths[i]);
        }
        if (serial.streams[0].finished and !serial.streams[1].finished) {
            continued = true;
            try t.expect(threaded.mask_lengths[1] > 0);
        }
    }
    try t.expect(continued);
    try t.expect(threaded.engine.grammar_workers.pool != null);
    try t.expect(serial.engine.grammar_workers.pool == null);
    try t.expectEqual(@as(usize, 0), threaded.engine.activeCount());
    for (0..6) |i| try t.expectEqual(serial.classes[i].at, threaded.classes[i].at);
}
