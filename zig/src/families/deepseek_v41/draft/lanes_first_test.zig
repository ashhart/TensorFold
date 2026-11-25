//! Backend.first_masked on DeepSeek's lanes: the prompt pass draws the first token under the bits it is later given.
const std = @import("std");
const lanes = @import("lanes");
const twin = @import("twin.zig");
const iface = @import("iface.zig");
const ln = @import("lanes.zig");
const servedCosts = @import("lanes_test.zig").servedCosts;
const gt = @import("lanes_grammar_test.zig");

const gpa = std.testing.allocator;
const grammar = lanes.grammar;
const Steps = gt.Steps;
const Case = gt.Case;

/// The twin with prompts as the GPU target runs them: a prefix in place, rows in pieces, then the last row alone.
const Pieces = struct {
    t: *twin.Twin,
    pass: iface.Pass,
    cached: u64 = 0, // rows `begin` puts in place before the pieces, as a prompt cache hit does (run again here)
    ran_tail: bool = false, // `tail` runs the last row anyway and returns its pick (Lanes.join then returns false)
    drop_bits: bool = false, // a broken target: it draws the prompt's pick without the slot's first bits
    bits: [8][]const u32 = @splat(&.{}), // each slot's first-token bits (Target.first_mask)

    fn target(p: *Pieces) iface.Target {
        return .{ .ptr = p, .vtable = &.{ .prefill = prefill, .window = window, .keep = keep, .taps = taps, .release = release, .first_mask = firstMask, .begin = begin, .piece = piece, .finish = finish, .tail = tail } };
    }

    fn cast(ptr: *anyopaque) *Pieces {
        return @ptrCast(@alignCast(ptr));
    }

    fn inner(ptr: *anyopaque) iface.Target {
        return cast(ptr).t.target();
    }

    fn firstMask(ptr: *anyopaque, slot: u32, bits: []const u32) void {
        cast(ptr).bits[slot] = bits;
    }

    /// The slot's first bits, once (GpuTarget.takeFirst).
    fn take(p: *Pieces, slot: u32) []const u32 {
        defer p.bits[slot] = &.{};
        return if (p.drop_bits) &.{} else p.bits[slot];
    }

    fn prefill(ptr: *anyopaque, slot: u32, ids: []const u32, smp: ?lanes.Sampling, draw: u64) anyerror!u32 {
        try inner(ptr).firstMask(slot, cast(ptr).take(slot));
        return inner(ptr).prefill(slot, ids, smp, draw);
    }

    fn window(ptr: *anyopaque, segments: []const iface.Segment, choices: [][]u32) anyerror!void {
        return inner(ptr).window(segments, choices);
    }

    fn keep(ptr: *anyopaque, slot: u32, path: []const u32) anyerror!void {
        return inner(ptr).keep(slot, path);
    }

    fn taps(ptr: *anyopaque, slot: u32) iface.Taps {
        return inner(ptr).taps(slot);
    }

    fn release(ptr: *anyopaque, slot: u32) void {
        inner(ptr).release(slot);
    }

    /// Prompt rows [start, end): the slot's first rows prefill, later ones run as a window; their taps go to the pass.
    fn run(p: *Pieces, slot: u32, ids: []const u32, start: u64, end: u64) !void {
        const t = p.t.target();
        const n: usize = @intCast(end - start);
        if (start == 0) {
            _ = try t.prefill(slot, ids[0..n], null, end);
        } else {
            var draws: [64]u64 = undefined;
            for (draws[0..n], 0..) |*w, i| w.* = start + i + 1;
            var out: [64]u32 = undefined;
            var rows = [_][]u32{out[0..n]};
            try t.window(&.{.{ .slot = slot, .start = start, .tokens = ids[@intCast(start)..@intCast(end)], .parents = null, .draws = draws[0..n], .sampling = null }}, &rows);
        }
        const tp = t.taps(slot);
        var at: [64]u32 = undefined;
        for (at[0..tp.rows], 0..) |*r, i| r.* = @intCast(i);
        try p.pass.ingest(slot, start, tp, at[0..tp.rows]);
    }

    fn begin(ptr: *anyopaque, slot: u32, ids: []const u32) anyerror!iface.Target.Begun {
        const p = cast(ptr);
        if (p.cached > 0) try p.run(slot, ids, 0, p.cached);
        return .{ .at = p.cached };
    }

    fn piece(ptr: *anyopaque, slot: u32, ids: []const u32, start: u64, end: u64, _: bool) anyerror!void {
        return cast(ptr).run(slot, ids, start, end);
    }

    /// The prompt's last row as its own window, drawn under the slot's first bits (GpuTarget.finish).
    fn last(p: *Pieces, slot: u32, ids: []const u32, smp: ?lanes.Sampling, draw: u64) !u32 {
        const bits = p.take(slot);
        var one: [1]u32 = undefined;
        var out = [_][]u32{&one};
        try p.t.target().window(&.{.{ .slot = slot, .start = ids.len - 1, .tokens = ids[ids.len - 1 ..], .parents = null, .draws = &.{draw}, .sampling = smp, .masks = bits }}, &out);
        return one[0];
    }

    fn finish(ptr: *anyopaque, slot: u32, ids: []const u32, smp: ?lanes.Sampling, draw: u64) anyerror!u32 {
        return cast(ptr).last(slot, ids, smp, draw);
    }

    fn tail(ptr: *anyopaque, slot: u32, ids: []const u32, smp: ?lanes.Sampling, draw: u64) anyerror!?u32 {
        const p = cast(ptr);
        if (p.ran_tail) return try p.last(slot, ids, smp, draw);
        _ = p.take(slot); // the first token comes from the round's window, its bits the lane core's
        return null;
    }
};

/// A prompt's way to its first token; joined: its first round draws it, tail: the target runs the tail row anyway.
const Path = enum { whole, pieces, cached, joined, tail };

/// What the prompt pass left: the slot's KV rows and taps, the drafter's rings, and the lanes' state for the stream.
const Seen = struct {
    first: u32 = 0,
    kv: []f32,
    taps: []f32,
    ring: []f32,
    tags: []i64,
    st: [4]u64,

    fn take(t: *twin.Twin, d: *twin.Drafter, x: *ln.Lanes, s: *lanes.Stream) !Seen {
        const sl = &t.slots[0];
        const st = x.state(s).?;
        return .{ .kv = try std.mem.concat(gpa, f32, &.{ sl.k.items, sl.v.items }), .taps = try gpa.dupe(f32, sl.taps.items), .ring = try gpa.dupe(f32, d.rings[0]), .tags = try gpa.dupe(i64, d.tags[0]), .st = .{ st.len, st.base, st.n, x.drawn.items.len } };
    }

    fn deinit(v: Seen) void {
        gpa.free(v.kv);
        gpa.free(v.taps);
        gpa.free(v.ring);
        gpa.free(v.tags);
    }

    /// The same KV rows, taps and drafter rings; `lanes`: the same lanes state too (one handle more after a draw).
    fn expectSame(a: Seen, b: Seen, lanes_: bool) !void {
        try std.testing.expectEqualSlices(f32, a.kv, b.kv);
        try std.testing.expectEqualSlices(f32, a.taps, b.taps);
        try std.testing.expectEqualSlices(f32, a.ring, b.ring);
        try std.testing.expectEqualSlices(i64, a.tags, b.tags);
        if (lanes_) try std.testing.expectEqualSlices(u64, a.st[0..3], b.st[0..3]);
    }
};

/// The case's prompt pass on `path`, then the first draw as Engine.drawFirst makes it; `masked`: under the grammar.
fn probe(cs: Case, path: Path, masked: bool) !Seen {
    const t = try twin.Twin.init(gpa, .{}, 2);
    defer t.deinit();
    const d = try twin.Drafter.init(gpa, t, 2);
    defer d.deinit();
    var p: Pieces = .{ .t = t, .pass = d.pass(), .cached = if (path == .cached) 4 else 0, .ran_tail = path == .tail };
    var c = try servedCosts(gpa);
    defer c.deinit(gpa);
    const x = try ln.Lanes.init(gpa, if (path == .whole) t.target() else p.target(), d.pass(), c, .{ .shape = d.shape, .slots = 1 });
    defer x.deinit();
    var g: Steps = .{ .length = cs.length, .r = cs.r };
    var k: grammar.Constraint = .init(g.rules(), Steps.words, null);
    var s = try lanes.Stream.init(gpa, .{ .id = "s", .prompt = cs.prompt, .max_new = cs.length + 8, .sampling = cs.sampling, .grammar = if (masked) &k else null });
    defer s.deinit(gpa);
    const b = x.backend();
    if (path == .whole) try b.prefill(&s) else {
        const begun = try b.vtable.begin.?(b.ptr, &s);
        try std.testing.expectEqual(p.cached, begun.cached);
        const n: u64 = s.prompt_len - 1;
        const mid = (begun.cached + n) / 2;
        try b.vtable.piece.?(b.ptr, &s, begun.cached, mid, false);
        try b.vtable.piece.?(b.ptr, &s, mid, n, false);
        if (path == .tail) try std.testing.expect(!try b.vtable.join.?(b.ptr, &s)) else try b.vtable.finish.?(b.ptr, &s);
    }
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var seen = try Seen.take(t, d, x, &s);
    errdefer seen.deinit();
    const mask = if (s.grammar) |gr| try gr.next(arena.allocator()) else null;
    const h = if (mask) |m| try b.vtable.first_masked.?(b.ptr, &s, s.prompt_len, m) else try b.first(&s, s.prompt_len);
    seen.first = try b.read(h);
    const after = try Seen.take(t, d, x, &s);
    defer after.deinit();
    try seen.expectSame(after, true); // first_masked ran no forward, wrote no KV row and drafted nothing
    try std.testing.expectEqual(seen.st[3] + 1, after.st[3]);
    // only the pass's own bits are its contract: other bits, or any bits for an unmasked pass, are refused
    var other: [Steps.words]u32 = @splat(std.math.maxInt(u32));
    if (mask) |m| other[0] = ~m[0];
    // bits that still allow the pick but differ by one other token: refused too (a pick-only check takes them)
    if (mask) |m| {
        var near: [Steps.words]u32 = undefined;
        @memcpy(&near, m);
        const flip: u32 = if (seen.first == Steps.stop) 0 else Steps.stop;
        near[flip / 32] ^= @as(u32, 1) << @intCast(flip % 32);
        try std.testing.expect(grammar.allows(&near, seen.first));
        try std.testing.expectError(error.FirstMaskChanged, b.vtable.first_masked.?(b.ptr, &s, s.prompt_len, &near));
    }
    try std.testing.expectError(error.FirstMaskChanged, b.vtable.first_masked.?(b.ptr, &s, s.prompt_len, &other));
    const refused = try Seen.take(t, d, x, &s);
    defer refused.deinit();
    try after.expectSame(refused, true);
    try std.testing.expectEqual(after.st[3], refused.st[3]);
    b.release(&s);
    return seen;
}

const p1 = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4 };
const p2 = [_]u32{ 2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5, 9, 0, 4, 5 };
const p3 = [_]u32{ 10, 20, 30, 40, 10, 20, 30, 40, 10, 20, 30 };
const warm: lanes.Sampling = .{ .seed = 5, .temperature = 1.0, .top_k = 20, .top_p = 0.95, .min_p = 0.02 };

/// A case on `prompt` whose grammar refuses the twin's unmasked first token, so the mask decides the pick.
fn refusing(prompt: []const u32, smp: ?lanes.Sampling) !Case {
    const t = try twin.Twin.init(gpa, .{}, 1);
    defer t.deinit();
    const free = try t.serial(0, prompt, 1, smp, null);
    defer gpa.free(free);
    var r: u32 = 0;
    while ((free[0] + r) % 3 == 0) r += 1;
    return .{ .prompt = prompt, .length = 12, .r = r, .sampling = smp };
}

test "first_masked: a pick the pass drew without its bits is refused (error.FirstNotMasked)" {
    const cs = try refusing(&p1, null);
    const t = try twin.Twin.init(gpa, .{}, 2);
    defer t.deinit();
    const d = try twin.Drafter.init(gpa, t, 2);
    defer d.deinit();
    var p: Pieces = .{ .t = t, .pass = d.pass(), .drop_bits = true };
    var c = try servedCosts(gpa);
    defer c.deinit(gpa);
    const x = try ln.Lanes.init(gpa, p.target(), d.pass(), c, .{ .shape = d.shape, .slots = 1 });
    defer x.deinit();
    var g: Steps = .{ .length = cs.length, .r = cs.r };
    var k: grammar.Constraint = .init(g.rules(), Steps.words, null);
    var s = try lanes.Stream.init(gpa, .{ .id = "s", .prompt = cs.prompt, .max_new = cs.length + 8, .sampling = cs.sampling, .grammar = &k });
    defer s.deinit(gpa);
    const b = x.backend();
    const begun = try b.vtable.begin.?(b.ptr, &s);
    try b.vtable.piece.?(b.ptr, &s, begun.cached, s.prompt_len - 1, false);
    try b.vtable.finish.?(b.ptr, &s);
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    // the bits are the pass's own (the lanes kept them), but the target drew the unmasked pick, which they refuse
    const mask = (try k.next(arena.allocator())).?;
    try std.testing.expectError(error.FirstNotMasked, b.vtable.first_masked.?(b.ptr, &s, s.prompt_len, mask));
    b.release(&s);
}

test "first_masked: the pass's pick under the first bits, whole, in pieces, cached or tail-run; KV and drafter unchanged" {
    for ([_][]const u32{ &p1, &p2, &p3 }) |prompt| for ([_]?lanes.Sampling{ null, warm }) |smp| {
        const cs = try refusing(prompt, smp);
        const want = try gt.serial(cs);
        defer gpa.free(want);
        for ([_]Path{ .whole, .pieces, .cached, .tail }) |path| {
            const masked = try probe(cs, path, true);
            defer masked.deinit();
            const plain = try probe(cs, path, false);
            defer plain.deinit();
            try std.testing.expectEqual(want[0], masked.first);
            try std.testing.expect((masked.first + cs.r) % 3 == 0 and (plain.first + cs.r) % 3 != 0);
            // the grammar changed only the pick: the pass left the same KV rows, taps and drafter rings
            try masked.expectSame(plain, false);
        }
    };
}

/// The cases through the lane engine with their prompts in pieces on `path` (`.whole`: added whole), four at once.
fn runPath(cases: []const Case, path: Path, slots: u32) ![][]u32 {
    const t = try twin.Twin.init(gpa, .{}, slots + 1);
    defer t.deinit();
    const d = try twin.Drafter.init(gpa, t, slots + 1);
    defer d.deinit();
    var p: Pieces = .{ .t = t, .pass = d.pass(), .cached = if (path == .cached) 4 else 0, .ran_tail = path == .tail };
    var c = try servedCosts(gpa);
    defer c.deinit(gpa);
    const x = try ln.Lanes.init(gpa, if (path == .whole) t.target() else p.target(), d.pass(), c, .{ .shape = d.shape, .slots = slots });
    defer x.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var cfg = try lanes.Config.init(gpa, try x.model(arena.allocator(), 64), 16, 15);
    defer cfg.deinit(gpa);
    cfg.join_tail = path == .joined or path == .tail;
    var clock: lanes.fake.FixedClock = .{};
    var e = lanes.Engine.init(gpa, &cfg, x.backend(), clock.clock());
    defer e.deinit();
    x.attach(&e);
    const gs = try gpa.alloc(Steps, cases.len);
    defer gpa.free(gs);
    const cons = try gpa.alloc(grammar.Constraint, cases.len);
    defer gpa.free(cons);
    const streams = try gpa.alloc(lanes.Stream, cases.len);
    defer gpa.free(streams);
    for (cases, gs, cons, streams) |cs, *g, *k, *s| {
        g.* = .{ .length = cs.length, .r = cs.r };
        k.* = .init(g.rules(), Steps.words, null);
        s.* = try lanes.Stream.init(gpa, .{ .id = "s", .prompt = cs.prompt, .max_new = cs.length + 8, .sampling = cs.sampling, .grammar = k });
    }
    defer for (streams) |*s| s.deinit(gpa);
    for (streams) |*s| {
        if (path == .whole) {
            try e.addStream(s);
            continue;
        }
        const begun = try e.beginStream(s);
        try e.piece(s, begun.cached, s.prompt_len - 1, false);
        try e.finishStream(s);
    }
    while (e.activeCount() > 0) try e.step();
    const out = try gpa.alloc([]u32, cases.len);
    for (out, streams) |*o, *s| {
        try std.testing.expectEqualStrings("", s.problem);
        try std.testing.expect(s.reason == .stop);
        o.* = try gpa.dupe(u32, s.emitted());
    }
    return out;
}

test "first_masked: replies whose prompts came whole, in pieces, cached, joined or tail-run equal their masked serial decode" {
    for ([_]?lanes.Sampling{ null, warm }) |smp| {
        var cases: [3]Case = undefined;
        for (&cases, [_][]const u32{ &p1, &p2, &p3 }) |*cs, prompt| cs.* = try refusing(prompt, smp);
        for ([_]Path{ .whole, .pieces, .cached, .joined, .tail }) |path| {
            const got = try runPath(&cases, path, 4);
            defer {
                for (got) |o| gpa.free(o);
                gpa.free(got);
            }
            for (cases, got) |cs, reply| {
                const want = try gt.serial(cs);
                defer gpa.free(want);
                try std.testing.expectEqualSlices(u32, want, reply);
            }
        }
    }
}
