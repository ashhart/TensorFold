//! Qwen3.8-27B on Metal as one stage of the ring (RFC #2005 v1): its binding of fabric/stage's driver.
const std = @import("std");
const mtl = @import("metal");
const stage = @import("stage");
const q = @import("tensorfold").qwen27;

const boot = stage.boot;
const plan = stage.plan;
const driver = stage.driver;
const fwd = q.forward;
const Round = q.round_plan.Round;

/// Every stage of one placement reports this; rank 0 refuses another. This file is the driver's binding `B`.
pub const version = "tensorfold qwen27-metal stage-v1";
pub const backend: stage.payload.Backend = .metal;
pub const most_streams = 8;

/// One process's 27B: the model, a runner of `parallel` slots, and the stage's layers (null: the whole model).
pub const Engine = struct {
    gpa: std.mem.Allocator,
    model: *q.model.Model,
    runner: q.decode_round.Runner,
    span: ?fwd.Span,
    caches: u32 = 0, // slots handed out by `open`, slot i the i-th
    /// The last window's round, kept until every stream in it says how many rows it keeps (or opens anew).
    last: ?Round = null,
    last_caches: [most_streams]?*Cache = @splat(null),
    hop: ?driver.Hop = null,
    /// A test's hop for a whole-model span without a ring (the rows go out and come back through the host).
    loopback: ?driver.Hop = null,
    bytes: []u8, // the boundary's host rows: h then r, as the ring sends them

    /// `span` null runs the whole model; else the stage's layers, and rank 0 (`heads`) also its embedding and head.
    pub fn init(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, parallel: u32, context: u32, span: ?fwd.Span) !*Engine {
        if (parallel == 0 or parallel > most_streams) return error.TooManyStreams;
        const model = try if (span) |sp| q.model.Model.loadStage(gpa, io, dir, q.attention_plan.max_rows, sp.first, sp.end) else q.model.Model.loadDeferred(gpa, io, dir, q.attention_plan.max_rows);
        errdefer model.deinit();
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        e.* = .{ .gpa = gpa, .model = model, .runner = undefined, .span = span, .bytes = &.{} };
        e.runner = try q.decode_round.Runner.init(gpa, model, parallel, context);
        errdefer e.runner.deinit();
        e.runner.span = span; // for the runner's whole life: its passes and keeps touch only these layers
        e.bytes = try gpa.alloc(u8, 2 * @as(usize, q.attention_plan.max_rows) * model.config.hidden * 2);
        return e;
    }

    pub fn deinit(e: *Engine) void {
        if (e.last) |*r| r.deinit();
        e.gpa.free(e.bytes);
        e.runner.deinit();
        e.model.deinit();
        e.gpa.destroy(e);
    }

    /// Logits row `row` of the last round (rank 0 or the whole model): a prompt chunk's in row 0.
    pub fn rowLogits(e: *const Engine, row: usize) []const u16 {
        const vocab = e.model.config.vocab;
        return e.model.frame.get(.logits).buffer.slice(u16, (row + 1) * vocab)[row * vocab ..];
    }

    /// The last window kept before the runner's next pass: each stream its reported rows, one yet to say all of them.
    fn settle(e: *Engine) !void {
        var r = e.last orelse return;
        var store: [most_streams][q.attention_plan.max_rows]u32 = undefined;
        var paths: [most_streams][]const u32 = undefined;
        for (r.slots, 0..) |_, i| {
            const c = e.last_caches[i].?;
            const rows = c.accepted orelse c.window.?.rows;
            for (store[i][0..rows], 0..) |*v, k| v.* = @intCast(k);
            paths[i] = store[i][0..rows];
            if (c.accepted != null) {
                c.window = null;
                c.accepted = null;
            } else c.kept = true; // its report, in a later round, must say all of them
        }
        defer {
            r.deinit();
            e.last = null;
            e.last_caches = @splat(null);
        }
        try e.runner.keep(&r, paths[0..r.slots.len]);
    }
};

/// A slot of the runner and its unsettled window.
pub const Cache = struct {
    e: *Engine,
    slot: u32,
    window: ?driver.Window = null,
    accepted: ?u32 = null,
    /// Kept whole before the stream reported (another round ran first): only all its rows may be reported now.
    kept: bool = false,
};

/// One stream's rows in a round: the driver fills these in stream order.
pub const Part = struct { cache: *Cache, rows: u32 };

/// The ring's boundary for this engine: h and r as bf16 planes of the hidden width, no drafter taps (v1).
pub fn boundary(e: *const Engine) boot.Boundary {
    return .{ .hidden = @intCast(e.model.config.hidden), .row_dtype = .bf16, .planes = 2, .draft_width = 0 };
}

/// A prompt chunk and a round at most 128 rows: the pre-M5 prompt pass and the logits frame.
pub fn shapes(parallel: u32, context: u32) boot.Shapes {
    return .{ .prompt_chunk = q.attention_plan.max_rows, .max_rows = q.attention_plan.max_rows, .parallel = parallel, .context = context };
}

pub fn kvDtype(_: *const Engine) stage.payload.Dtype {
    return .bf16;
}

pub fn open(_: std.mem.Allocator, e: *Engine, context: u32) !Cache {
    if (context > e.runner.capacity or e.caches >= e.runner.slots) return error.NoFreeSlot;
    defer e.caches += 1;
    return .{ .e = e, .slot = e.caches };
}

pub fn close(_: std.mem.Allocator, _: *Cache) void {}

pub fn reset(e: *Engine, c: *Cache) !void {
    try e.settle();
    c.window = null;
    c.accepted = null;
    c.kept = false;
    try e.runner.reset(c.slot);
    e.runner.offsets[c.slot] = 0;
}

/// The stream's rows so far: an unsettled window counts its reported rows.
pub fn position(c: *const Cache) u32 {
    if (c.window) |w| if (c.accepted) |k| if (!c.kept) return w.start + k;
    return c.e.runner.offsets[c.slot];
}

pub fn window(c: *const Cache) ?driver.Window {
    if (c.accepted != null) return null;
    return c.window;
}

/// The last window kept to its first `accepted` rows, a prefix path, once every stream in it has said.
pub fn rollback(_: *Engine, c: *Cache, accepted: u32) !void {
    const w = c.window orelse return error.BadAccepted;
    if (accepted > w.rows) return error.BadAccepted;
    if (c.kept) { // a round without this stream ran since: Metal's runner kept the whole window then (v1)
        if (accepted != w.rows) return error.WindowKeptWhole;
        c.window = null;
        c.kept = false;
        return;
    }
    c.accepted = accepted;
}

fn give(ptr: *anyopaque, h: []const u8, r: []const u8, rows: u32) anyerror!void {
    const e: *Engine = @ptrCast(@alignCast(ptr));
    const hop = e.hop orelse return error.StageHopMissing;
    const buf = e.bytes[0 .. h.len + r.len];
    @memcpy(buf[0..h.len], h);
    @memcpy(buf[h.len..], r);
    try hop.give(hop.ctx, rows, buf);
}

fn take(ptr: *anyopaque, h: []u8, r: []u8, rows: u32) anyerror!void {
    const e: *Engine = @ptrCast(@alignCast(ptr));
    const hop = e.hop orelse return error.StageHopMissing;
    const buf = e.bytes[0 .. h.len + r.len];
    try hop.take(hop.ctx, rows, buf);
    @memcpy(h, buf[0..h.len]);
    @memcpy(r, buf[h.len..]);
}

/// This stage's layers over the round's rows: a prompt chunk keeps its last row's logits, a window every row's.
pub fn forward(e: *Engine, parts: []const Part, ids: []const u32, mode: plan.Mode, hop: ?driver.Hop) !void {
    try e.settle();
    if (parts.len > most_streams) return error.TooManyStreams;
    if (mode == .prompt and parts.len != 1) return error.PromptParts;
    var inputs: [most_streams]q.round_plan.Input = undefined;
    var at: usize = 0;
    for (parts, inputs[0..parts.len]) |p, *in| {
        in.* = .{ .slot = p.cache.slot, .start = e.runner.offsets[p.cache.slot], .capacity = e.runner.capacity, .ids = ids[at..][0..p.rows] };
        at += p.rows;
    }
    const m = e.model;
    var round = try Round.init(e.gpa, inputs[0..parts.len], @intCast(m.config.conv_kernel), e.runner.slots, @intCast(m.shard.global.vocab));
    var kept = false;
    defer if (!kept) round.deinit();
    e.hop = hop orelse e.loopback;
    defer e.hop = null;
    if (e.runner.span) |*sp| sp.hop = if (e.hop != null) .{ .ptr = e, .give = give, .take = take } else null;
    defer if (e.runner.span) |*sp| {
        sp.hop = null;
    };
    const heads = if (e.span) |sp| sp.heads else true;
    if (mode == .prompt) {
        m.kernels.prompt = true;
        defer m.kernels.prompt = false;
        return e.runner.advance(&round, if (heads) .last else .none);
    }
    try e.runner.verifyHead(&round, if (heads) .all else .none);
    for (parts, 0..) |p, i| {
        p.cache.window = .{ .start = inputs[i].start, .rows = p.rows };
        p.cache.accepted = null;
        p.cache.kept = false;
        e.last_caches[i] = p.cache;
    }
    e.last = round;
    kept = true;
}

pub const Driver = driver.Driver(@This());
pub const Entry = driver.Entry;
pub const Slots = Driver.Slots;
pub const Lead = Driver.Lead;
pub const serve = Driver.serve;
pub const describe = Driver.describe;

/// The checkpoint's and layout's fingerprint: SHA-256 over config.json, the version and every rank's range.
pub fn fingerprint(config: []const u8, ranges: []const boot.Range) [32]u8 {
    return driver.fingerprint(version, config, ranges);
}

/// Boot on the ring, every rank a Metal stage; the next round returned.
pub fn bootRing(gpa: std.mem.Allocator, ring: *stage.Ring, own: *const boot.Description, model: *const boot.Model, ranges: []const boot.Range) !u64 {
    return driver.bootRing(gpa, ring, own, model, ranges, .metal);
}
