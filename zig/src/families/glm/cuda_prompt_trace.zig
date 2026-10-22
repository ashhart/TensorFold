//! Paired same-stream events describe one prompt chunk without serializing its compute and transport streams.
const std = @import("std");
const cuda = @import("cuda");
pub const Stage = enum {
    control_reduce,
    token_gather,
    cache_write,
    prompt_score,
    drafter,
    replicated_qakva,
    projections,
    index_update,
    index_score,
    index_select,
    mla,
    kda,
    ffn,
    router,
    experts,
    shared_expert,
    output_projection,
    glue,
    partials,
    glued,
    all_gather,
    wait_filled,
    wait_gathered,
    wait_glues,
    wait_shared,
    embed,
    head,
    gate,
    draw,
    mtp,
    commit,
};
pub const Site = enum { chunk, mixer, ffn };
pub const Meta = struct { rank: usize, slot: u32, start: usize, rows: usize };
const Pair = struct {
    first: cuda.Event,
    last: cuda.Event,
    stream: cuda.Stream = undefined,
    stage: Stage = undefined,
    layer: ?usize = null,
    piece: ?usize = null,
    site: Site = .chunk,
    complete: bool = false,
    ms: f32 = 0,
};
pub const Scope = struct {
    trace: ?*Trace = null,
    index: usize = 0,
    pub inline fn end(s: Scope) void {
        const t = s.trace orelse return;
        const p = &t.events[s.index];
        if (p.complete) {
            t.invalid = true;
            return;
        }
        p.last.record(p.stream) catch {
            t.invalid = true;
            return;
        };
        p.complete = true;
    }
};
pub const Trace = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    events: []Pair,
    n: usize = 0,
    meta: Meta = undefined,
    layer: ?usize = null,
    piece: ?usize = null,
    site: Site = .chunk,
    usable: bool = true,
    active: bool = false,
    invalid: bool = false,
    t0: i96 = 0,
    pub fn init(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, capacity: usize) !*Trace {
        return initWith(gpa, io, d, capacity, cuda.Event.init);
    }
    pub fn initWith(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, capacity: usize, comptime create: anytype) !*Trace {
        const t = try gpa.create(Trace);
        errdefer gpa.destroy(t);
        const events = try gpa.alloc(Pair, capacity);
        errdefer gpa.free(events);
        var made: usize = 0;
        errdefer for (events[0..made]) |*p| {
            p.first.deinit();
            p.last.deinit();
        };
        for (events) |*p| {
            p.* = .{ .first = try create(d, true), .last = undefined };
            p.last = create(d, true) catch |err| {
                p.first.deinit();
                return err;
            };
            made += 1;
        }
        t.* = .{ .gpa = gpa, .io = io, .events = events };
        return t;
    }
    pub fn deinit(t: *Trace) void {
        for (t.events) |*p| {
            p.first.deinit();
            p.last.deinit();
        }
        t.gpa.free(t.events);
        t.gpa.destroy(t);
    }
    pub fn begin(t: *Trace, meta: Meta) void {
        t.n = 0;
        t.invalid = false;
        t.active = true;
        t.meta = meta;
        t.layer = null;
        t.piece = null;
        t.site = .chunk;
        t.t0 = std.Io.Clock.awake.now(t.io).toNanoseconds();
    }
    pub fn span(t: *Trace, stream: cuda.Stream, stage: Stage) Scope {
        if (!t.active) return .{};
        if (t.n == t.events.len) {
            t.invalid = true;
            return .{};
        }
        const index = t.n;
        t.n += 1;
        const p = &t.events[index];
        p.stream = stream;
        p.stage = stage;
        p.layer = t.layer;
        p.piece = t.piece;
        p.site = t.site;
        p.complete = false;
        p.first.record(stream) catch {
            t.invalid = true;
            return .{};
        };
        return .{ .trace = t, .index = index };
    }
    pub fn collect(t: *Trace) !void {
        if (!t.active or t.invalid or t.n == 0) return error.InvalidPromptProfile;
        for (t.events[0..t.n]) |p| {
            if (!p.complete) return error.IncompletePromptProfile;
            try p.last.synchronize();
        }
        for (t.events[0..t.n]) |*p| {
            p.ms = try cuda.Event.elapsedMs(p.first, p.last);
            if (!std.math.isFinite(p.ms) or p.ms < 0) return error.InvalidPromptProfile;
        }
    }
    pub fn drainIncomplete(t: *Trace) !void {
        for (t.events[0..t.n]) |*p| if (!p.complete) {
            t.invalid = true;
            try p.last.record(p.stream);
            p.complete = true;
        };
        for (t.events[0..t.n]) |p| try p.last.synchronize();
    }
    pub fn finish(t: *Trace) void {
        if (!t.active) return;
        defer t.active = false;
        const wall: f64 = @as(f64, @floatFromInt(std.Io.Clock.awake.now(t.io).toNanoseconds() - t.t0)) / 1e6;
        if (t.invalid or blk: {
            for (t.events[0..t.n]) |p| if (!p.complete) break :blk true;
            break :blk false;
        }) t.drainIncomplete() catch {
            t.usable = false;
        };
        t.collect() catch |err| {
            std.log.warn("glm prompt invalid rank={d} slot={d} start={d} rows={d} reason={s}", .{ t.meta.rank, t.meta.slot, t.meta.start, t.meta.rows, @errorName(err) });
            return;
        };
        std.log.info("glm prompt chunk rank={d} slot={d} start={d} rows={d} wall_ms={d:.4} spans={d}", .{ t.meta.rank, t.meta.slot, t.meta.start, t.meta.rows, wall, t.n });
        for (t.events[0..t.n]) |p| std.log.info("glm prompt stage rank={d} slot={d} start={d} rows={d} layer={?d} piece={?d} site={s} stage={s} stream={s} ms={d:.4}", .{ t.meta.rank, t.meta.slot, t.meta.start, t.meta.rows, p.layer, p.piece, @tagName(p.site), @tagName(p.stage), if (p.stream.handle == t.events[0].stream.handle) "main" else "side", p.ms });
    }
};
