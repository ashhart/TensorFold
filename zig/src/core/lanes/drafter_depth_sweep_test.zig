//! The depth sweep on the lane core's FakeDrafter over the fake target, and three faulty drafters it must refuse.
const std = @import("std");
const sweep = @import("drafter_depth_sweep.zig");
const dr = @import("drafter.zig");
const fake = @import("fake.zig");
const Stream = @import("stream.zig").Stream;
const Spec = @import("stream.zig").Spec;
const Sampling = @import("sampling.zig").Sampling;
const Features = @import("backend.zig").Features;
const FakeDrafter = @import("drafted_test.zig").FakeDrafter;

const gpa = std.testing.allocator;
const sampled: Sampling = .{ .seed = 7 };

/// The fake target's rule (fake.next on the whole context) behind a FakeDrafter, a deeper twin, and a fault.
const Harness = struct {
    sampling: ?Sampling,
    use_deeper: bool = true,
    use_want: bool = true,
    fault: Fault = .none,
    under: FakeDrafter = .{},
    twin: FakeDrafter = .{},
    faulty: Faulty = undefined,
    stream: Stream = undefined,
    context: std.ArrayList(u32) = .empty, // prompt and kept rows
    rows: std.ArrayList(u32) = .empty, // the last absorb's rows and their follow
    follow: std.ArrayList(u32) = .empty,

    fn harness(h: *Harness) sweep.Harness {
        return .{ .ctx = h, .begin = begin, .verify = verify, .want = if (h.use_want) want else null, .end = end };
    }

    fn self(ctx: *anyopaque) *Harness {
        return @ptrCast(@alignCast(ctx));
    }

    fn begin(ctx: *anyopaque, n: usize) anyerror!sweep.Case {
        const h = self(ctx);
        h.under = .{};
        h.twin = .{};
        h.context = .empty;
        for (0..n) |i| try h.context.append(gpa, @intCast((i * 7 + 3) % (fake.vocab - 1)));
        const p = h.context.items;
        const spec: Spec = .{ .id = "s", .prompt = p, .max_new = 64, .eos = &.{}, .sampling = h.sampling };
        h.stream = try Stream.init(gpa, spec);
        h.faulty = .{ .inner = h.under.drafter(), .fault = h.fault };
        const prompt = try h.kept(0, p, fake.next(p, h.sampling, n));
        return .{ .drafter = h.faulty.drafter(), .stream = &h.stream, .prompt = prompt, .deeper = h.twinIf() };
    }

    fn twinIf(h: *Harness) ?dr.Drafter {
        return if (h.use_deeper) h.twin.drafter() else null;
    }

    /// Rows [start, start + rows.len) and the token after each (the last: `next`), appended to the context.
    fn kept(h: *Harness, start: usize, rows: []const u32, next: u32) !sweep.Step {
        h.rows.clearRetainingCapacity();
        try h.rows.appendSlice(gpa, rows);
        h.follow.clearRetainingCapacity();
        try h.follow.appendSlice(gpa, rows[1..]);
        try h.follow.append(gpa, next);
        if (start != 0) try h.context.appendSlice(gpa, rows);
        const n: u32 = @intCast(rows.len);
        const buf = @intFromPtr(h.rows.items.ptr);
        const f: Features = .{ .buffer = buf, .rows = n, .row_bytes = 4, .space = .host, .dtype = .u32 };
        var a: dr.Absorb = .{ .stream = &h.stream, .features = f, .start = start, .follow = h.follow.items };
        a.prompt = start == 0;
        return .{ .absorb = a, .next = .{ .pending = next, .position = h.context.items.len + 1 } };
    }

    /// The target's window [pending, drafts..]: drafts kept while they match its draws.
    fn verify(ctx: *anyopaque, r: sweep.Round, drafts: []const u32) anyerror!sweep.Step {
        const h = self(ctx);
        const start = h.context.items.len;
        var row: std.ArrayList(u32) = .empty;
        defer row.deinit(gpa);
        try row.appendSlice(gpa, h.context.items);
        try row.append(gpa, r.pending);
        var draw = fake.next(row.items, h.sampling, row.items.len);
        for (drafts) |t| {
            if (t != draw) break;
            try row.append(gpa, t);
            draw = fake.next(row.items, h.sampling, row.items.len);
        }
        const rows = try gpa.dupe(u32, row.items[start..]);
        defer gpa.free(rows);
        return h.kept(start, rows, draw);
    }

    /// FakeDrafter's rule written out once more, as a family's CPU reference would be: its own chain, every fifth off.
    fn want(ctx: *anyopaque, r: sweep.Round, out: []u32) anyerror!void {
        const h = self(ctx);
        var guess: std.ArrayList(u32) = .empty;
        defer guess.deinit(gpa);
        try guess.appendSlice(gpa, h.context.items);
        try guess.append(gpa, r.pending);
        for (out, 0..) |*o, j| {
            const at = r.position + j;
            var t = fake.next(guess.items, h.sampling, at);
            if ((at *% 2654435761 + j) % 5 == 0) t = (t + 1) % fake.vocab;
            o.* = t;
            try guess.append(gpa, t);
        }
    }

    fn end(ctx: *anyopaque) void {
        const h = self(ctx);
        h.under.deinit();
        h.twin.deinit();
        h.stream.deinit(gpa);
        h.context.deinit(gpa);
    }

    fn deinit(h: *Harness) void {
        h.rows.deinit(gpa);
        h.follow.deinit(gpa);
    }
};

const Fault = enum { none, ignore_depth, over_by_one, never_shrinks, shifted };

/// A drafter's hold with a planted fault: the depth it asks its inner drafter for, or a wrong pending token.
const Faulty = struct {
    inner: dr.Drafter,
    fault: Fault,
    widest: u32 = 0,

    const vtable: dr.Drafter.VTable = .{
        .taps = taps,
        .facts = facts,
        .open = open,
        .absorb = absorb,
        .hold = hold,
        .held = held,
        .release = release,
    };

    fn drafter(x: *Faulty) dr.Drafter {
        return .{ .ptr = x, .vtable = &vtable };
    }

    fn self(ptr: *anyopaque) *Faulty {
        return @ptrCast(@alignCast(ptr));
    }

    fn taps(ptr: *anyopaque) []const u32 {
        return self(ptr).inner.taps();
    }

    fn facts(ptr: *anyopaque) dr.Facts {
        return self(ptr).inner.facts();
    }

    fn open(ptr: *anyopaque, s: *Stream) anyerror!void {
        return self(ptr).inner.open(s);
    }

    fn absorb(ptr: *anyopaque, items: []const dr.Absorb) anyerror!void {
        return self(ptr).inner.absorb(items);
    }

    fn hold(ptr: *anyopaque, items: []const dr.Hold) anyerror!void {
        const x = self(ptr);
        var it = items[0];
        x.widest = @max(x.widest, it.depth);
        it.depth = switch (x.fault) {
            .none => it.depth,
            .ignore_depth => x.inner.facts().depth,
            .over_by_one => it.depth + 1,
            .never_shrinks => x.widest,
            .shifted => it.depth,
        };
        if (x.fault == .shifted) it.pending = (it.pending + 1) % fake.vocab;
        return x.inner.hold(&.{it});
    }

    fn held(ptr: *anyopaque, streams: []const *Stream, out: []const []u32) anyerror!void {
        return self(ptr).inner.held(streams, out);
    }

    fn release(ptr: *anyopaque, s: *Stream) void {
        self(ptr).inner.release(s);
    }
};

fn run(h: *Harness) !sweep.Report {
    defer h.deinit();
    return sweep.sweep(gpa, h.harness(), .{});
}

test "depth sweep: FakeDrafter holds exactly each depth's drafts, greedy and sampled, against both references" {
    try std.testing.expectEqual(@as(u32, 6), (FakeDrafter{}).facts_.depth);
    for ([_]?Sampling{ null, sampled }) |s| {
        for ([_][2]bool{ .{ true, true }, .{ true, false }, .{ false, true } }) |refs| {
            var h: Harness = .{ .sampling = s, .use_deeper = refs[0], .use_want = refs[1] };
            const r = try run(&h);
            // two prompts x (depths 1..6 + one zig-zag run) x 12 rounds; drafts 12 x 21 fixed + 42 zig-zag a prompt
            try std.testing.expectEqual(sweep.Report{ .runs = 14, .rounds = 168, .drafts = 2 * (12 * 21 + 42) }, r);
        }
    }
}

test "depth sweep: a drafter ignoring Hold.depth, over-drafting, never shrinking its count or off its pending fails" {
    for ([_]Fault{ .ignore_depth, .over_by_one, .never_shrinks }) |f| {
        var h: Harness = .{ .sampling = null, .fault = f };
        try std.testing.expectError(error.HeldMore, run(&h));
    }
    var h: Harness = .{ .sampling = null, .fault = .shifted };
    try std.testing.expectError(error.DraftsDiffer, run(&h));
}

test "depth sweep: no reference and a facts().depth past the block are refused" {
    var none: Harness = .{ .sampling = null, .use_deeper = false, .use_want = false };
    try std.testing.expectError(error.NoReference, run(&none));
    var h: Harness = .{ .sampling = null };
    defer h.deinit();
    try std.testing.expectError(error.DepthOverBlock, sweep.sweep(gpa, h.harness(), .{ .block = 3 }));
}
