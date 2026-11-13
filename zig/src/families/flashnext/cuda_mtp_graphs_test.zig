//! The head's pieces and groups and the rollback replay as graphs with a rank's collective attached, and their keys.
const std = @import("std");
const t = std.testing;
const cuda = @import("cuda");
const abi = cuda.abi;
const wg = @import("cuda_window_graph.zig");
const parts = @import("cuda_batch_state.zig");
const pieces = @import("cuda_mtp_pieces.zig");
const roll = @import("cuda_mtp_roll.zig");
const linear_n = @import("cuda_prompt_wt.zig").linear_n;

/// A driver that records graphs and counts their launches; no kernel runs.
const Fake = struct {
    var capturing = false;
    var launches: usize = 0;
    fn begin(_: abi.Stream, _: abi.CaptureMode) callconv(.c) abi.Result {
        capturing = true;
        return 0;
    }
    fn end(_: abi.Stream, out: *abi.Graph) callconv(.c) abi.Result {
        capturing = false;
        out.* = @ptrFromInt(1);
        return 0;
    }
    fn status(_: abi.Stream, s: *abi.CaptureStatus) callconv(.c) abi.Result {
        s.* = if (capturing) .active else .none;
        return 0;
    }
    fn instantiate(e: *abi.GraphExec, _: abi.Graph, _: u64) callconv(.c) abi.Result {
        e.* = @ptrFromInt(2);
        return 0;
    }
    fn update(_: abi.GraphExec, _: abi.Graph, info: *abi.ExecUpdateResultInfo) callconv(.c) abi.Result {
        info.result = .success;
        return 0;
    }
    fn graphDestroy(_: abi.Graph) callconv(.c) abi.Result {
        return 0;
    }
    fn execDestroy(_: abi.GraphExec) callconv(.c) abi.Result {
        return 0;
    }
    fn upload(_: abi.GraphExec, _: abi.Stream) callconv(.c) abi.Result {
        return 0;
    }
    fn launch(_: abi.GraphExec, _: abi.Stream) callconv(.c) abi.Result {
        launches += 1;
        return 0;
    }

    fn driver(d: *cuda.Driver) void {
        d.api.cuStreamBeginCapture_v2 = begin;
        d.api.cuStreamEndCapture = end;
        d.api.cuStreamIsCapturing = status;
        d.api.cuGraphInstantiateWithFlags = instantiate;
        d.api.cuGraphExecUpdate_v2 = update;
        d.api.cuGraphDestroy = graphDestroy;
        d.api.cuGraphUpload = upload;
        d.api.cuGraphLaunch = launch;
        d.api.cuGraphExecDestroy = execDestroy;
        launches = 0;
    }
};

/// Passes run outside a capture: they ran eagerly.
var eager: usize = 0;

fn ran() void {
    if (!Fake.capturing) eager += 1;
}

/// A rank's engine as the passes see it: its stream, its graph caches and, as at --tp 2, a collective.
const Engine = struct {
    gpa: std.mem.Allocator,
    stream: *const cuda.Stream,
    graphs: wg.Graphs = .{},
    collective: ?*const anyopaque = @ptrFromInt(0x10),

    fn deinit(e: *Engine) void {
        e.graphs.clear(e.gpa);
    }
};

const Buf = struct { ptr: u64 };

/// A draft's buffers and length, as the head's keys read them.
const Draft = struct {
    gen: u64 = 1,
    len: usize = 9000,
    cap: usize = 262144,
    media: ?*const anyopaque = null,
    kc: Buf = .{ .ptr = 0x1000_0000 },
    out: Buf = .{ .ptr = 0x2000_0000 },
    dlogits: Buf = .{ .ptr = 0x2100_0000 },
    saved: Buf = .{ .ptr = 0x2200_0000 },
    keep: parts.Keep = .{},
};

const Scratch = struct { s: *const parts.Scratch };

const Piece = struct {
    o: struct { e: *Engine },
    p: *const Scratch,
    d: *const Draft,
    m: usize = 3,
    last: bool = true,
    score: bool = true,
    input: u64 = 0x3000_0000,

    pub fn run(_: Piece) !void {
        ran();
    }
};

const Lanes = struct { ds: []const *const Draft, nexts: []const []const u32, rows: []const u64, scores: bool = true };

const Group = struct {
    o: struct { e: *Engine },
    p: *const Scratch,
    group: Lanes,
    total: usize,

    pub fn run(_: Group) !void {
        ran();
    }
};

/// A sequence's DeltaNet buffers, as the rollback's key reads them.
const Seq = struct {
    gen: u64 = 7,
    tail: Buf = .{ .ptr = 0x4000_0000 },
    conv: [linear_n]Buf = undefined,
    state: [linear_n]Buf = undefined,

    fn at(base: u64) Seq {
        var q: Seq = .{};
        for (&q.conv, &q.state, 0..) |*c, *s, i| {
            c.ptr = base + i * 0x10000;
            s.ptr = base + 0x100_0000 + i * 0x10000;
        }
        return q;
    }
};

const Replay = struct {
    pub fn run(_: Replay) !void {
        ran();
    }
};

fn scratch() parts.Scratch {
    var s: parts.Scratch = undefined;
    s.h = 0x5000_0000;
    s.cap = 64;
    return s;
}

/// The cache's captures, updates and replays, the passes run eagerly and the graphs launched, in that order.
fn counts(c: anytype) [5]usize {
    return .{ c.captures, c.updates, c.replays, eager, Fake.launches };
}

/// A fake driver's stream with a current context, as a capture needs; `restore` puts the old context back.
const Rig = struct {
    d: cuda.Driver,
    stream: cuda.Stream,
    previous: abi.Context,

    fn init(r: *Rig) void {
        r.d = undefined;
        Fake.driver(&r.d);
        r.stream = .{ .d = &r.d, .handle = @ptrFromInt(1) };
        r.previous = cuda.copy_source.current();
        cuda.copy_source.setCurrent(@ptrFromInt(0x3254));
        eager = 0;
    }

    fn restore(r: *Rig) void {
        cuda.copy_source.setCurrent(r.previous);
    }
};

test "rank graphs: a head piece with a collective attached runs as its graph; with media, eagerly" {
    var r: Rig = undefined;
    r.init();
    defer r.restore();
    var e: Engine = .{ .gpa = t.allocator, .stream = &r.stream };
    defer e.deinit();
    const s = scratch();
    const p: Scratch = .{ .s = &s };
    var d: Draft = .{};
    const one: Piece = .{ .o = .{ .e = &e }, .p = &p, .d = &d };
    try pieces.runPiece(one, 0x6000_0000);
    try t.expectEqual([5]usize{ 0, 0, 0, 1, 0 }, counts(e.graphs.head)); // a new slot runs eagerly once
    try pieces.runPiece(one, 0x6000_0000);
    try t.expectEqual([5]usize{ 1, 0, 0, 1, 1 }, counts(e.graphs.head)); // then is captured and launched
    d.len += 1; // the next round at the same geometry: positions are device values, so the key stays
    try pieces.runPiece(one, 0x6000_0000);
    try t.expectEqual([5]usize{ 1, 0, 1, 1, 2 }, counts(e.graphs.head));
    var media = d;
    media.media = @ptrFromInt(0x20);
    try pieces.runPiece(Piece{ .o = .{ .e = &e }, .p = &p, .d = &media }, 0x6000_0000);
    try pieces.runPiece(Piece{ .o = .{ .e = &e }, .p = &p, .d = &media }, 0x6000_0000);
    try t.expectEqual([5]usize{ 1, 0, 1, 3, 2 }, counts(e.graphs.head));
}

test "rank graphs: a group of heads with a collective attached runs as its graph" {
    var r: Rig = undefined;
    r.init();
    defer r.restore();
    var e: Engine = .{ .gpa = t.allocator, .stream = &r.stream };
    defer e.deinit();
    const s = scratch();
    const p: Scratch = .{ .s = &s };
    const a: Draft = .{ .gen = 1 };
    const b: Draft = .{ .gen = 2, .kc = .{ .ptr = 0x1100_0000 }, .out = .{ .ptr = 0x2300_0000 }, .dlogits = .{ .ptr = 0x2400_0000 } };
    const ds = [_]*const Draft{ &a, &b };
    const nexts = [_][]const u32{ &.{ 5, 6 }, &.{7} };
    const run: Group = .{ .o = .{ .e = &e }, .p = &p, .group = .{ .ds = &ds, .nexts = &nexts, .rows = &.{ 0x3000_0000, 0x3100_0000 } }, .total = 3 };
    for (0..3) |_| try pieces.runGroup(run, 0x6000_0000);
    try t.expectEqual([5]usize{ 1, 0, 1, 1, 2 }, counts(e.graphs.head_many));
}

test "rank graphs: a rollback with a collective attached runs as its graph; swapped states update it in place" {
    var r: Rig = undefined;
    r.init();
    defer r.restore();
    var e: Engine = .{ .gpa = t.allocator, .stream = &r.stream };
    defer e.deinit();
    const s = scratch();
    var q: Seq = .at(0x7000_0000);
    const d: Draft = .{};
    for (0..3) |_| try roll.graphed(&e, &s, &q, &d, 2, Replay{});
    try t.expectEqual([5]usize{ 1, 0, 1, 1, 2 }, counts(e.graphs.rollback));
    // the round's verify swapped every state with the draft's spare (Draft.settle): the held graph takes them
    for (&q.state, 0..) |*b, i| b.ptr = 0x9000_0000 + i * 0x10000;
    for (0..2) |_| try roll.graphed(&e, &s, &q, &d, 2, Replay{});
    try t.expectEqual([5]usize{ 1, 1, 2, 1, 4 }, counts(e.graphs.rollback));
}

fn headKey(d: Draft, rows: usize, last: bool, scores: bool, input: u64, s: parts.Scratch, score_rows: u64) wg.HeadKey {
    return .init(&d, rows, last, scores, input, &s, score_rows);
}

/// `k` holds another value than `base` on the same slot: the held graph is updated, never replayed stale.
fn expectMoved(base: anytype, k: @TypeOf(base)) !void {
    try t.expect(!std.meta.eql(base, k));
    try t.expectEqual(wg.slotOf(base), wg.slotOf(k));
}

test "rank graphs: a head piece's key holds every host value its kernels take" {
    const d: Draft = .{};
    const s = scratch();
    const base = headKey(d, 3, true, true, 0x3000_0000, s, 0x6000_0000);
    var other_s = s;
    other_s.h += 0x100;
    var cap_s = s;
    cap_s.cap += 1;
    const moved = [_]wg.HeadKey{
        headKey(.{ .gen = 2 }, 3, true, true, 0x3000_0000, s, 0x6000_0000), // another draft at the same addresses
        headKey(.{ .cap = 131072 }, 3, true, true, 0x3000_0000, s, 0x6000_0000),
        headKey(.{ .kc = .{ .ptr = 0x1000_0100 } }, 3, true, true, 0x3000_0000, s, 0x6000_0000),
        headKey(.{ .out = .{ .ptr = 0x2000_0100 } }, 3, true, true, 0x3000_0000, s, 0x6000_0000),
        headKey(.{ .dlogits = .{ .ptr = 0x2100_0100 } }, 3, true, true, 0x3000_0000, s, 0x6000_0000),
        headKey(d, 3, true, true, 0x3000_0100, s, 0x6000_0000), // the piece's main-model rows
        headKey(d, 3, true, true, 0x3000_0000, other_s, 0x6000_0000),
        headKey(d, 3, true, true, 0x3000_0000, cap_s, 0x6000_0000),
        headKey(d, 3, true, true, 0x3000_0000, s, 0x6000_0100), // the block scores
        headKey(.{ .len = 9000 + 256 }, 3, true, true, 0x3000_0000, s, 0x6000_0000), // a wider score grid
    };
    for (moved) |k| try expectMoved(base, k);
    // below the scored length the dense chunks move with it
    const short = headKey(.{ .len = 1000 }, 3, true, true, 0x3000_0000, s, 0x6000_0000);
    try expectMoved(short, headKey(.{ .len = 1600 }, 3, true, true, 0x3000_0000, s, 0x6000_0000));
    const slots = [_]wg.HeadKey{
        short, // no score grid: other kernels
        headKey(d, 4, true, true, 0x3000_0000, s, 0x6000_0000),
        headKey(d, 3, false, true, 0x3000_0000, s, 0x6000_0000),
        headKey(d, 3, true, false, 0x3000_0000, s, 0x6000_0000),
    };
    for (slots) |k| try t.expect(!std.meta.eql(wg.slotOf(base), wg.slotOf(k)));
}

fn manyKey(ds: []const *const Draft, nexts: []const []const u32, rows: []const u64, scores: bool, s: parts.Scratch, score_rows: u64) wg.HeadManyKey {
    var total: usize = 0;
    for (nexts) |nx| total += nx.len;
    return .init(ds, nexts, rows, total, scores, &s, score_rows);
}

test "rank graphs: a head group's key holds every lane's host values" {
    const s = scratch();
    const a: Draft = .{ .gen = 1 };
    const b: Draft = .{ .gen = 2, .kc = .{ .ptr = 0x1100_0000 }, .out = .{ .ptr = 0x2300_0000 }, .dlogits = .{ .ptr = 0x2400_0000 } };
    const nexts = [_][]const u32{ &.{ 5, 6 }, &.{7} };
    const rows = [_]u64{ 0x3000_0000, 0x3100_0000 };
    const base = manyKey(&.{ &a, &b }, &nexts, &rows, true, s, 0x6000_0000);
    try t.expectEqual(base, manyKey(&.{ &a, &b }, &nexts, &rows, true, s, 0x6000_0000));
    const lane: [6]Draft = .{ .{ .gen = 3 }, .{ .cap = 131072 }, .{ .kc = .{ .ptr = 0x1000_0100 } }, .{ .out = .{ .ptr = 0x2000_0100 } }, .{ .dlogits = .{ .ptr = 0x2100_0100 } }, .{ .len = 9000 + 256 } };
    for (&lane) |*x| try expectMoved(base, manyKey(&.{ x, &b }, &nexts, &rows, true, s, 0x6000_0000));
    try expectMoved(base, manyKey(&.{ &a, &b }, &nexts, &.{ 0x3000_0000, 0x3100_0100 }, true, s, 0x6000_0000));
    try expectMoved(base, manyKey(&.{ &a, &b }, &nexts, &rows, true, s, 0x6000_0100));
    var other_s = s;
    other_s.h += 0x100;
    try expectMoved(base, manyKey(&.{ &a, &b }, &nexts, &rows, true, other_s, 0x6000_0000));
    const slots = [_]wg.HeadManyKey{
        manyKey(&.{ &a, &b }, &.{ &.{5}, &.{ 6, 7 } }, &rows, true, s, 0x6000_0000), // the rows split another way
        manyKey(&.{ &a, &b }, &nexts, &rows, false, s, 0x6000_0000),
        manyKey(&.{&a}, nexts[0..1], rows[0..1], true, s, 0x6000_0000),
    };
    for (slots) |k| try t.expect(!std.meta.eql(wg.slotOf(base), wg.slotOf(k)));
}

test "rank graphs: a rollback's key holds every buffer the replay touches and its kept rows" {
    const s = scratch();
    const q: Seq = .at(0x7000_0000);
    var d: Draft = .{};
    d.keep.layers[0] = .{ .proj = 0x8000_0000, .hist = 0x8100_0000 };
    const base: wg.RollbackKey = .init(&q, &d, 2, &s);
    for (0..linear_n) |i| {
        var conv = q;
        conv.conv[i].ptr += 0x100;
        try expectMoved(base, wg.RollbackKey.init(&conv, &d, 2, &s));
        var state = q;
        state.state[i].ptr += 0x100;
        try expectMoved(base, wg.RollbackKey.init(&state, &d, 2, &s));
    }
    var tail = q;
    tail.tail.ptr += 0x100;
    try expectMoved(base, wg.RollbackKey.init(&tail, &d, 2, &s));
    var gen = q;
    gen.gen += 1;
    try expectMoved(base, wg.RollbackKey.init(&gen, &d, 2, &s));
    const drafts: [4]Draft = .{ .{ .gen = 2, .keep = d.keep }, .{ .saved = .{ .ptr = 0x2200_0100 }, .keep = d.keep }, kept(d, 0x8000_0100, 0x8100_0000), kept(d, 0x8000_0000, 0x8100_0100) };
    for (&drafts) |*x| try expectMoved(base, wg.RollbackKey.init(&q, x, 2, &s));
    var other_s = s;
    other_s.h += 0x100;
    try expectMoved(base, wg.RollbackKey.init(&q, &d, 2, &other_s));
    try t.expect(!std.meta.eql(wg.slotOf(base), wg.slotOf(wg.RollbackKey.init(&q, &d, 3, &s)))); // kept rows: the grids
}

fn kept(d: Draft, proj: u64, hist: u64) Draft {
    var x = d;
    x.keep.layers[0].proj = proj;
    x.keep.layers[0].hist = hist;
    return x;
}
