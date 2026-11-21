//! The shared serving vtable through Nemotron's actual prompt adapters, chunk and layer-cancel callback.
const std = @import("std");
const hip = @import("hip");
const lanes = @import("lanes");
const serve = @import("hip_serve");
const source = @import("lanes.zig");
const Engine = @import("engine.zig").Engine;
const forward = @import("forward.zig");
const model = @import("model.zig");
const pages = @import("pages.zig");
const T = std.testing;

const Flag = struct {
    on: bool = false,
    fn check(ptr: *anyopaque) bool {
        const f: *Flag = @ptrCast(@alignCast(ptr));
        return f.on;
    }
};

const Record = struct {
    flag: *Flag,
    cancel_inside: bool,
    syncs: usize = 0,
    layers: usize = 0,
    passes: usize = 0,
    draws: usize = 0,
    logits: usize = 0,
    traced: bool = false,
    rows: usize = 0,
    sync_error: bool = false,
};

fn synchronize(handle: hip.abi.Stream) callconv(.c) hip.abi.Result {
    const r: *Record = @ptrCast(@alignCast(handle.?));
    r.syncs += 1;
    return if (r.sync_error) 1 else 0;
}

fn current(_: hip.abi.Context) callconv(.c) hip.abi.Result {
    return 0;
}

const Inert = struct {
    var runtime: hip.Runtime = .{ .lib = undefined, .api = undefined };
};

const RecordedEngine = struct {
    rec: *Record,
    ctx: hip.Context,
    stream: hip.Stream,
    o: struct { world: usize },
    act: model.Kind = .bf16,
    pool: pages.Pool = undefined,
    m: model.Model = undefined,
    arena: hip.Arena = undefined,

    pub const poll = Engine.poll;

    pub fn fresh(_: *RecordedEngine) !void {}

    pub fn ops(_: *RecordedEngine, _: anytype) hip.ops.Ops {
        return .{ .l = undefined, .bf16 = true, .stream = hip.counting, .arena = undefined };
    }

    pub fn ids(_: *RecordedEngine, _: hip.ops.Ops, _: []const u32) !u64 {
        return 0;
    }

    pub fn model_(e: *RecordedEngine) *const model.Model {
        return &e.m;
    }

    pub fn logits(e: *RecordedEngine, _: hip.ops.Ops, x: hip.ops.Tensor, _: usize) !hip.ops.Tensor {
        e.rec.logits += 1;
        return x;
    }
};

const NoHead = struct {
    pub fn absorb(_: *NoHead, _: hip.ops.Ops, _: *pages.Stream, _: hip.ops.Tensor, _: []const u32) !void {}
};

const Sampler = struct {
    rec: *Record,
    pub fn draw(d: *Sampler, _: hip.ops.Ops, _: hip.Stream, _: hip.ops.Tensor, _: anytype, out: []u32, _: bool) !void {
        d.rec.draws += 1;
        out[0] = 7;
    }
};

const Drawer = struct {
    vocab: usize = 8,
    pub fn drawHost(_: *Drawer, _: []const u16, _: anytype) !u32 {
        return 7;
    }
};

const Fake = struct {
    gpa: std.mem.Allocator,
    e: *RecordedEngine,
    streams: serve.backend.Streams(source.Lane),
    tp: serve.protocol.Sender,
    prefix: void = {},
    lane: *source.Lane,
    head: ?*NoHead = null,
    sampler: Sampler,
    drawer_state: Drawer = .{},
    chunk_rows: usize,
    start: usize,
    cut: usize = 0,
    state: u64 = 0,
    saved_at: usize = 0,
    saved_state: u64 = 0,
    saved_valid: bool = false,
    kept: usize = 0,

    pub const cancelOf = source.Hip.cancelOf;
    pub const advance = source.Hip.advance;
    pub const finish = source.Hip.finish;
    pub const chunk = source.Hip.chunk;

    pub fn runPrompt(h: *Fake, _: hip.ops.Ops, ctx: forward.Ctx, _: u64, rows: usize, pos: usize) !hip.ops.Tensor {
        const r = h.e.rec;
        r.passes += 1;
        r.rows = rows;
        r.traced = ctx.trace != null;
        try T.expectEqual(pos, ctx.stream.len);
        if (r.cancel_inside) r.flag.on = true;
        for (0..2) |layer| {
            r.layers += 1;
            try forward.traceLayer(ctx.trace, layer, .{ .ptr = 1, .kind = .bf16 }, rows);
        }
        for (pos..pos + rows) |i| h.state = (h.state *% 16777619) ^ @as(u64, @intCast(i + 1));
        return .{ .ptr = 1, .kind = .bf16 };
    }

    pub fn drawer(h: *Fake) *Drawer {
        return &h.drawer_state;
    }

    pub fn begin(h: *Fake, s: *lanes.Stream) !*source.Lane {
        const from: usize = if (s.reuse.saved != null) s.reuse.at else h.start;
        const cut = if (h.cut != 0) h.cut else from + h.chunk_rows;
        h.lane.fill = serve.backend.Fill.init(from, &.{@intCast(cut)});
        h.lane.s.len = from;
        if (s.reuse.saved != null) {
            try T.expect(h.saved_valid);
            try T.expectEqual(h.saved_at, from);
            h.state = h.saved_state;
        } else h.state = 0;
        s.cached = @intCast(from);
        h.lane.id = try h.streams.idOf(s);
        try h.streams.map.put(h.gpa, s, h.lane);
        return h.lane;
    }

    pub fn reach(h: *Fake, _: *source.Lane, at: usize, limit: usize) !usize {
        return @min(at + h.chunk_rows, limit);
    }

    pub fn keepCut(h: *Fake, _: *source.Lane, _: []const u32, at: usize) !void {
        h.kept += 1;
        h.saved_at = at;
        h.saved_state = h.state;
        h.saved_valid = true;
    }

    pub fn keepRows(_: *Fake, _: *source.Lane, _: usize) !void {}
    pub fn flush(_: *Fake) !void {}
    pub fn verify(_: *Fake, _: []const lanes.backend.Window, _: []lanes.backend.Verified) !void {}
    pub fn draft(_: *Fake, _: []const lanes.backend.DraftRequest) !void {}
    pub fn release(_: *Fake, _: *source.Lane) void {}
};

fn run(final: bool, rows: usize, world: usize, hook: bool, cancel_inside: bool, start: usize) !void {
    var tokens: [4096]u32 = @splat(1);
    var flag: Flag = .{};
    var s = try lanes.Stream.init(T.allocator, .{
        .id = "prompt",
        .prompt = tokens[0 .. start + rows + @intFromBool(!final)],
        .max_new = 8,
        .cancel_check = if (hook) .{ .ptr = &flag, .check = Flag.check } else null,
    });
    defer s.deinit(T.allocator);
    var r: Record = .{ .flag = &flag, .cancel_inside = cancel_inside };
    Inert.runtime.api.hipCtxSetCurrent = current;
    Inert.runtime.api.hipStreamSynchronize = synchronize;
    var e: RecordedEngine = .{
        .rec = &r,
        .ctx = .{ .r = &Inert.runtime, .device = 0, .handle = null },
        .stream = .{ .r = &Inert.runtime, .handle = @ptrCast(&r) },
        .o = .{ .world = world },
    };
    e.m.dims.hidden = 1;
    var lane: source.Lane = undefined;
    lane.fill = null;
    lane.first_row = null;
    lane.pending = null;
    lane.held_n = 0;
    lane.resumed = false;
    lane.hidden = .{ .r = &Inert.runtime, .ptr = @ptrFromInt(1), .len = 2 };
    var h: Fake = .{
        .gpa = T.allocator,
        .e = &e,
        .streams = serve.backend.Streams(source.Lane).init(T.allocator),
        .tp = .{ .gpa = T.allocator },
        .lane = &lane,
        .sampler = .{ .rec = &r },
        .chunk_rows = rows,
        .start = start,
    };
    defer h.streams.deinit();
    const b = serve.backend.Backend(Fake).backend(&h);
    const step = b.vtable.prefill_step.?;
    const polls = hook and world == 1 and rows >= 512;
    if (polls and cancel_inside) {
        try T.expectError(error.Cancelled, step(b.ptr, &s, 0));
        try T.expectEqual(start, lane.s.len);
        try T.expectEqual(start, lane.fill.?.at);
        try T.expectEqual(@as(usize, 0), lane.fill.?.next);
        try T.expectEqual(@as(usize, 0), h.kept);
        try T.expectEqual(@as(usize, 0), r.draws);
        try T.expectEqual(@as(usize, 0), r.logits);
        try T.expectEqual(@as(usize, 1), r.layers);
    } else {
        try T.expectEqual(final, try step(b.ptr, &s, 0));
        try T.expectEqual(start + rows, lane.s.len);
        try T.expectEqual(@as(usize, @intFromBool(final)), r.draws);
        try T.expectEqual(@as(usize, @intFromBool(final)), r.logits);
        if (final) try T.expect(lane.fill == null) else {
            try T.expectEqual(start + rows, lane.fill.?.at);
            try T.expectEqual(@as(usize, 1), h.kept);
        }
        try T.expectEqual(@as(usize, 2), r.layers);
    }
    try T.expectEqual(@as(usize, 1), r.passes);
    try T.expectEqual(rows, r.rows);
    try T.expectEqual(polls, r.traced);
    try T.expectEqual(if (polls) r.layers else @as(usize, 0), r.syncs);
}

test "Nemotron intermediate prompt chunks retain the stream cancel through the production adapter and trace" {
    for ([_]usize{ 0, 64 }) |start| for ([_]usize{ 511, 512, 1024 }) |rows| for ([_]usize{ 1, 2 }) |world| {
        try run(false, rows, world, true, true, start);
        try run(false, rows, world, true, false, start);
        try run(false, rows, world, false, true, start);
    };
}

test "Nemotron final prompt chunks cancel before progress logits or a first draw" {
    for ([_]usize{ 0, 64 }) |start| for ([_]usize{ 511, 512, 1024 }) |rows| for ([_]usize{ 1, 2 }) |world| {
        try run(true, rows, world, true, true, start);
        try run(true, rows, world, true, false, start);
        try run(true, rows, world, false, true, start);
    };
}

test "Nemotron cancellation between prompt cuts resumes with the exact fresh recurrent state" {
    const rows = 64;
    var tokens: [3 * rows]u32 = @splat(1);
    var flag: Flag = .{};
    var r: Record = .{ .flag = &flag, .cancel_inside = false };
    Inert.runtime.api.hipCtxSetCurrent = current;
    Inert.runtime.api.hipStreamSynchronize = synchronize;
    var e: RecordedEngine = .{
        .rec = &r,
        .ctx = .{ .r = &Inert.runtime, .device = 0, .handle = null },
        .stream = .{ .r = &Inert.runtime, .handle = @ptrCast(&r) },
        .o = .{ .world = 1 },
    };
    e.m.dims.hidden = 1;
    var lane: source.Lane = undefined;
    lane.fill = null;
    lane.first_row = null;
    lane.pending = null;
    lane.held_n = 0;
    lane.resumed = false;
    lane.hidden = .{ .r = &Inert.runtime, .ptr = @ptrFromInt(1), .len = 2 };
    var h: Fake = .{
        .gpa = T.allocator,
        .e = &e,
        .streams = serve.backend.Streams(source.Lane).init(T.allocator),
        .tp = .{ .gpa = T.allocator },
        .lane = &lane,
        .sampler = .{ .rec = &r },
        .chunk_rows = rows,
        .start = 0,
        .cut = 2 * rows,
    };
    defer h.streams.deinit();
    const b = serve.backend.Backend(Fake).backend(&h);
    const step = b.vtable.prefill_step.?;

    var cancelled = try lanes.Stream.init(T.allocator, .{ .id = "cancelled", .prompt = &tokens, .max_new = 8 });
    defer cancelled.deinit(T.allocator);
    try T.expect(!try step(b.ptr, &cancelled, 0));
    try T.expectEqual(rows, lane.fill.?.at);
    try T.expect(!h.saved_valid);
    if (b.vtable.cancel_prefill) |keep| try keep(b.ptr, &cancelled);
    try T.expect(h.saved_valid);
    try T.expectEqual(rows, h.saved_at);
    try T.expectEqual(h.state, h.saved_state);
    b.vtable.release(b.ptr, &cancelled);
    h.state = 0;

    var resumed = try lanes.Stream.init(T.allocator, .{
        .id = "resumed",
        .prompt = &tokens,
        .max_new = 8,
        .reuse = .{ .saved = @ptrCast(&h.saved_state), .at = rows },
    });
    defer resumed.deinit(T.allocator);
    try T.expect(!try step(b.ptr, &resumed, 0));
    try T.expectEqual(rows, resumed.cached);
    try T.expect(try step(b.ptr, &resumed, 0));
    const resumed_state = h.state;
    b.vtable.release(b.ptr, &resumed);

    h.saved_valid = false;
    var fresh = try lanes.Stream.init(T.allocator, .{ .id = "fresh", .prompt = &tokens, .max_new = 8 });
    defer fresh.deinit(T.allocator);
    try T.expect(!try step(b.ptr, &fresh, 0));
    try T.expect(!try step(b.ptr, &fresh, 0));
    try T.expect(try step(b.ptr, &fresh, 0));
    try T.expectEqual(tokens.len, lane.s.len);
    try T.expectEqual(resumed_state, h.state);
    b.vtable.release(b.ptr, &fresh);
}

test "Nemotron layer traces keep non-cancel failures as BadShape" {
    const Fail = struct {
        fn layer(_: *anyopaque, _: usize, _: hip.ops.Tensor, _: usize) anyerror!void {
            return error.TestFailure;
        }
    };
    var ctx: u8 = 0;
    try T.expectError(error.BadShape, forward.traceLayer(.{ .ctx = &ctx, .layer = Fail.layer }, 0, undefined, 512));
}

test "chunk cuts leaving 1, 8, 15, 16, 17 or 65 trailing rows partition every token in order" {
    const prompt_len: usize = 193;
    for ([_]usize{ 1, 8, 15, 16, 17, 65 }) |tail| {
        const cut = prompt_len - tail;
        var at: usize = 0;
        var last_rows: usize = 0;
        while (at < prompt_len) {
            const n = @min(cut, prompt_len - at);
            try T.expect(n > 0);
            last_rows = n;
            at += n;
        }
        try T.expectEqual(prompt_len, at);
        try T.expectEqual(tail, last_rows);
    }
}
