//! The first cold prompt's rank-local ledger, independent of TF_GLM_PROF and its prompt-shape policy.
const std = @import("std");
const cuda = @import("cuda");
const plan = @import("cuda_prefill_plan.zig");
const Pair = @import("cuda_prefill_events.zig").Pair;
const a = std.heap.page_allocator;
var owner: ?*Trace = null;
const limit = 262144;
pub const Row = struct {
    sequence: usize,
    parent: ?usize,
    stage: plan.Stage,
    class: plan.Class,
    layer: ?usize,
    layer_kind: plan.Kind,
    ffn_kind: plan.Ffn,
    site: plan.Site,
    piece: ?usize,
    start: usize,
    rows: usize,
    stream: usize,
    wall_begin_ns: i96,
    wall_end_ns: ?i96 = null,
    host_gap_before_ns: i96,
    cuda_event_ns: ?u64 = null,
    event_status: enum { pending, complete, unavailable } = .pending,
};
const Entry = struct { row: Row, pair: ?Pair, stream: cuda.Stream, stopped: bool = false };
pub const Scope = struct {
    trace: ?*Trace = null,
    index: usize = 0,
    pub inline fn end(s: Scope) void {
        if (s.trace) |t| t.stop(s.index);
    }
};
pub const Chunk = struct {
    trace: ?*Trace = null,
    scope: Scope = .{},
    last: bool = false,
    pub fn fail(c: Chunk) void {
        if (c.trace) |t| t.invalid = true;
    }
    pub fn end(c: Chunk) void {
        const t = c.trace orelse return;
        c.scope.end();
        if (c.last) {
            t.request.end();
            t.finished = true;
        }
        t.active = false;
        poll(t);
    }
};
pub const Trace = struct {
    d: *const cuda.Driver,
    io: std.Io,
    path: []const u8,
    rank: usize,
    world: usize,
    layers: []plan.Layer,
    entries: std.ArrayList(Entry) = .empty,
    streams: std.ArrayList(cuda.abi.Stream) = .empty,
    stack: [32]usize = undefined,
    depth: usize = 0,
    thread: std.Thread.Id = undefined,
    active: bool = false,
    begun: bool = false,
    finished: bool = false,
    invalid: bool = false,
    written: bool = false,
    dropped: usize = 0,
    slot: u32 = 0,
    tokens: usize = 0,
    step_start: usize = 0,
    start: usize = 0,
    rows: usize = 0,
    layer: ?usize = null,
    piece: ?usize = null,
    site: plan.Site = .chunk,
    t0: i96 = 0,
    last_hook: i96 = 0,
    request: Scope = .{},

    pub fn init(d: *const cuda.Driver, io: std.Io, prefix: []const u8, rank: usize, world: usize, layers: []const plan.Layer) !*Trace {
        if (owner != null) return error.PrefillTraceAlreadyOpen;
        const t = try a.create(Trace);
        errdefer a.destroy(t);
        const path = try std.fmt.allocPrint(a, "{s}.rank{d}.json", .{ prefix, rank });
        errdefer a.free(path);
        const kinds = try a.dupe(plan.Layer, layers);
        errdefer a.free(kinds);
        const file = try std.Io.Dir.cwd().createFile(io, path, .{});
        file.close(io);
        t.* = .{ .d = d, .io = io, .path = path, .rank = rank, .world = world, .layers = kinds };
        owner = t;
        return t;
    }
    fn now(t: *const Trace) i96 {
        return std.Io.Clock.awake.now(t.io).toNanoseconds() - t.t0;
    }
    fn recording(t: *const Trace) bool {
        return t.active and t.thread == std.Thread.getCurrentId();
    }
    fn begin(t: *Trace, stream: cuda.Stream, stage: plan.Stage) Scope {
        if (!t.recording()) return .{};
        if (t.entries.items.len == limit or t.depth == t.stack.len) {
            t.dropped += 1;
            return .{};
        }
        const begin_ns = t.now();
        const sid = for (t.streams.items, 0..) |s, i| {
            if (s == stream.handle) break i;
        } else blk: {
            t.streams.append(a, stream.handle) catch {
                t.dropped += 1;
                return .{};
            };
            break :blk t.streams.items.len - 1;
        };
        const kind: plan.Layer = if (t.layer) |l| if (l < t.layers.len) t.layers[l] else .{ .kind = .outside_layer, .ffn = .outside_layer } else .{ .kind = .outside_layer, .ffn = .outside_layer };
        const index = t.entries.items.len;
        const pair = Pair.init(&t.d.api, stream.handle);
        t.entries.append(a, .{ .pair = pair, .stream = stream, .row = .{
            .sequence = index,
            .parent = if (t.depth > 0) t.stack[t.depth - 1] else null,
            .stage = stage,
            .class = plan.class(stage),
            .layer = t.layer,
            .layer_kind = kind.kind,
            .ffn_kind = kind.ffn,
            .site = t.site,
            .piece = t.piece,
            .start = t.start,
            .rows = t.rows,
            .stream = sid,
            .wall_begin_ns = begin_ns,
            .host_gap_before_ns = @max(0, begin_ns - t.last_hook),
            .event_status = if (pair == null) .unavailable else .pending,
        } }) catch {
            if (pair) |p| p.deinit(&t.d.api);
            t.dropped += 1;
            return .{};
        };
        if (pair == null) t.invalid = true;
        t.stack[t.depth] = index;
        t.depth += 1;
        t.last_hook = t.now();
        return .{ .trace = t, .index = index };
    }
    fn stop(t: *Trace, index: usize) void {
        const e = &t.entries.items[index];
        if (e.row.wall_end_ns != null) {
            t.invalid = true;
            return;
        }
        if (e.pair) |p| {
            e.stopped = p.stop(&t.d.api, e.stream.handle);
            if (!e.stopped) {
                e.row.event_status = .unavailable;
                t.invalid = true;
            }
        }
        e.row.wall_end_ns = t.now();
        if (t.depth > 0 and t.stack[t.depth - 1] == index) t.depth -= 1 else t.invalid = true;
        t.last_hook = t.now();
    }
    pub fn collect(t: *Trace, final: bool) void {
        for (t.entries.items) |*e| {
            if (e.row.event_status != .pending) continue;
            if (e.stopped) if (e.pair) |p| if (p.elapsed(&t.d.api)) |ns| {
                e.row.cuda_event_ns = ns;
                e.row.event_status = .complete;
            };
            if (final and e.row.event_status == .pending) {
                e.row.event_status = .unavailable;
                t.invalid = true;
            }
        }
    }
    fn write(t: *Trace) !void {
        const rows = try a.alloc(Row, t.entries.items.len);
        defer a.free(rows);
        var complete = t.begun and t.finished and !t.invalid and t.dropped == 0;
        for (t.entries.items, rows) |e, *row| {
            row.* = e.row;
            complete = complete and e.row.event_status == .complete and e.row.wall_end_ns != null;
        }
        const bytes = try std.json.Stringify.valueAlloc(a, .{
            .schema = "tf-glm-prefill-v1",
            .rank = t.rank,
            .world = t.world,
            .slot = t.slot,
            .prompt_tokens = t.tokens,
            .complete = complete,
            .dropped_stages = t.dropped,
            .timings = "inclusive same-stream spans; parent spans and concurrent streams must not be summed",
            .host_gaps = "rank-local wall intervals between hooks; includes CPU work, scheduling and existing waits",
            .rows = rows,
        }, .{});
        defer a.free(bytes);
        try std.Io.Dir.cwd().writeFile(t.io, .{ .sub_path = t.path, .data = bytes });
        t.written = true;
    }
    pub fn deinit(t: *Trace) void {
        t.active = false;
        t.collect(true);
        t.write() catch |err| std.log.err("GLM prefill ledger write failed: {t}", .{err});
        for (t.entries.items) |e| if (e.pair) |p| p.deinit(&t.d.api);
        t.entries.deinit(a);
        t.streams.deinit(a);
        a.free(t.layers);
        a.free(t.path);
        if (owner == t) owner = null;
        a.destroy(t);
    }
};
pub fn enter(t0: ?*Trace, stream: cuda.Stream, slot: u32, start: usize, rows: usize, tokens: usize, resumed: bool) Chunk {
    const t = t0 orelse return .{};
    if (t.finished) return .{};
    if (!t.begun) {
        if (resumed or start != 0) return .{};
        t.begun = true;
        t.t0 = std.Io.Clock.awake.now(t.io).toNanoseconds();
        t.slot = slot;
        t.tokens = tokens;
    } else if (resumed or slot != t.slot or tokens != t.tokens) return .{};
    t.thread = std.Thread.getCurrentId();
    t.active = true;
    t.step_start = start;
    t.start = start;
    t.rows = rows;
    context(null, null, .chunk);
    if (t.entries.items.len == 0) t.request = t.begin(stream, .request);
    return .{ .trace = t, .scope = t.begin(stream, .chunk), .last = start + rows == tokens };
}
pub inline fn span(stream: cuda.Stream, stage: plan.Stage) Scope {
    const t = owner orelse return .{};
    return t.begin(stream, stage);
}
pub inline fn context(layer: ?usize, piece: ?usize, site: plan.Site) void {
    const t = owner orelse return;
    if (!t.recording()) return;
    t.layer = layer;
    t.piece = piece;
    t.site = site;
}
pub fn part(offset: usize, rows: usize) void {
    const t = owner orelse return;
    if (!t.recording()) return;
    t.start = t.step_start + offset;
    t.rows = rows;
}
pub fn poll(trace: ?*Trace) void {
    const t = trace orelse return;
    if (!t.finished or t.written) return;
    t.collect(false);
    for (t.entries.items) |e| if (e.row.event_status == .pending) return;
    t.write() catch |err| std.log.err("GLM prefill ledger write failed: {t}", .{err});
}
pub fn close(t: ?*Trace) void {
    if (t) |v| v.deinit();
}
test {
    _ = @import("cuda_prefill_test.zig");
}
