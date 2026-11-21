//! Any core Drafter at every depth 1..block holds exactly the asked drafts, the reference's (EugeneClaw, #2460).
const std = @import("std");
const dr = @import("drafter.zig");
const Stream = @import("stream.zig").Stream;

/// A round: the pending token and Hold.position (one past the last absorbed row).
pub const Round = struct { pending: u32, position: u64 };

/// The rows the target kept (absorbed by both drafters as given) and the round after them.
pub const Step = struct { absorb: dr.Absorb, next: Round };

/// One run's drafters over a fresh target; nothing opened yet, the sweep opens and releases `stream` itself.
pub const Case = struct {
    drafter: dr.Drafter,
    stream: *Stream,
    prompt: Step,
    deeper: ?dr.Drafter = null, // a second instance fed the same rows, held at `block`: depth d holds its first d
};

/// The family's side: its target, and its reference when a shorter round is not a longer one's prefix.
pub const Harness = struct {
    ctx: *anyopaque,
    begin: *const fn (ctx: *anyopaque, prompt_len: usize) anyerror!Case,
    verify: *const fn (ctx: *anyopaque, round: Round, drafts: []const u32) anyerror!Step,
    want: ?*const fn (ctx: *anyopaque, round: Round, out: []u32) anyerror!void = null, // the out.len drafts expected
    end: *const fn (ctx: *anyopaque) void,
};

pub const Options = struct {
    block: ?u32 = null, // the most a hold may ask (the family's block); null: the drafter's facts().depth
    prompts: []const usize = &.{ 6, 40 },
    rounds: usize = 12,
};

pub const Report = struct { runs: usize = 0, rounds: usize = 0, drafts: usize = 0 };

pub const Error = error{ NoReference, DepthOverBlock, HeldFewer, HeldMore, DraftsDiffer, ReferencesDisagree };

/// Fixed-depth runs for every depth 1..block and prompt, then one zig-zag run a prompt (block, 1, block - 1, 2, ..).
pub fn sweep(gpa: std.mem.Allocator, h: Harness, o: Options) !Report {
    var r: Report = .{};
    for (o.prompts) |n| {
        const block = try blockOf(h, n, o.block);
        var d: u32 = 1;
        while (d <= block) : (d += 1) try run(gpa, h, n, block, .{ .fixed = d }, o.rounds, &r);
        try run(gpa, h, n, block, .zigzag, o.rounds, &r);
    }
    return r;
}

const Schedule = union(enum) {
    fixed: u32,
    zigzag,

    fn at(s: Schedule, block: u32, i: usize) u32 {
        const k: u32 = @intCast((i / 2) % block);
        return switch (s) {
            .fixed => |d| d,
            .zigzag => if (i % 2 == 0) block - k else 1 + k,
        };
    }
};

fn blockOf(h: Harness, n: usize, asked: ?u32) !u32 {
    const c = try h.begin(h.ctx, n);
    defer h.end(h.ctx);
    const most = c.drafter.facts().depth;
    const block = asked orelse most;
    const at: Where = .{ .depth = most, .round = 0, .prompt = n };
    if (block == 0 or most > block) return fail(error.DepthOverBlock, at, "facts().depth over block {d}", .{block});
    if (h.want == null and c.deeper == null) return error.NoReference;
    return block;
}

fn run(gpa: std.mem.Allocator, h: Harness, n: usize, block: u32, s: Schedule, rounds: usize, rep: *Report) !void {
    const c = try h.begin(h.ctx, n);
    defer h.end(h.ctx);
    const bufs = try gpa.alloc(u32, 3 * (block + 1));
    defer gpa.free(bufs);
    const got = bufs[0 .. block + 1];
    const deep = bufs[block + 1 ..][0 .. block + 1];
    const want = bufs[2 * (block + 1) ..][0 .. block + 1];
    try c.drafter.open(c.stream);
    defer c.drafter.release(c.stream);
    if (c.deeper) |x| try x.open(c.stream);
    defer if (c.deeper) |x| x.release(c.stream);
    try absorb(c, c.prompt.absorb);
    var r = c.prompt.next;
    for (0..rounds) |i| {
        const at: Where = .{ .depth = s.at(block, i), .round = i, .prompt = n };
        const d = at.depth;
        // facts().depth again every round, not only for the block: a drafter may grow it after a hold
        const f = c.drafter.facts();
        if (f.depth > block) return fail(error.DepthOverBlock, at, "facts().depth {d} over block", .{f.depth});
        const ref = try reference(h, c, r, block, at, deep, want);
        try c.drafter.hold(&.{holdAt(c, r, d)});
        c.drafter.held(&.{c.stream}, &.{got[0..d]}) catch |e| return fail(error.HeldFewer, at, "{s}", .{@errorName(e)});
        if (!std.mem.eql(u32, ref, got[0..d])) return differ(error.DraftsDiffer, at, ref, got[0..d]);
        // the count contract (Drafter.held): exactly d held; Facts.slack deepens the asked depth, not the held count
        if (c.drafter.held(&.{c.stream}, &.{got[0 .. d + 1]})) |_| {
            return fail(error.HeldMore, at, "{d} drafts held", .{d + 1});
        } else |_| {}
        const step = try h.verify(h.ctx, r, got[0..d]);
        try absorb(c, step.absorb);
        r = step.next;
        rep.rounds += 1;
        rep.drafts += d;
    }
    rep.runs += 1;
}

/// The drafts expected at depth d: the family's `want`, else the deeper instance's first d (both must agree).
fn reference(h: Harness, c: Case, r: Round, block: u32, at: Where, deep: []u32, want: []u32) ![]const u32 {
    const d = at.depth;
    var out: ?[]const u32 = null;
    if (c.deeper) |x| {
        try x.hold(&.{holdAt(c, r, block)});
        try x.held(&.{c.stream}, &.{deep[0..block]});
        out = deep[0..d];
    }
    if (h.want) |f| {
        try f(h.ctx, r, want[0..d]);
        if (out) |o| if (!std.mem.eql(u32, o, want[0..d])) return differ(error.ReferencesDisagree, at, want[0..d], o);
        out = want[0..d];
    }
    return out orelse error.NoReference;
}

/// A round's hold as Drafted asks it: the stream's sampling with it, so a sampled family draws as it serves.
fn holdAt(c: Case, r: Round, depth: u32) dr.Hold {
    const s = c.stream;
    return .{ .stream = s, .pending = r.pending, .position = r.position, .depth = depth, .sampling = s.sampling };
}

fn absorb(c: Case, a: dr.Absorb) !void {
    try c.drafter.absorb(&.{a});
    if (c.deeper) |x| try x.absorb(&.{a});
}

const Where = struct { depth: u32, round: usize, prompt: usize };

fn differ(e: Error, at: Where, want: []const u32, got: []const u32) Error {
    return fail(e, at, "want {any} got {any}", .{ want, got });
}

fn fail(e: Error, at: Where, comptime fmt: []const u8, args: anytype) Error {
    const head = "drafter depth sweep: {s} at depth {d} round {d} prompt {d}: ";
    std.debug.print(head ++ fmt ++ "\n", .{ @errorName(e), at.depth, at.round, at.prompt } ++ args);
    return e;
}
