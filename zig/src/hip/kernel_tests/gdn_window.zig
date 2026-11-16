//! A Flash Next window through gdn_replay: fill, send, the plan's conv split and delta rule, then keep(n).
const std = @import("std");
const rig = @import("rig.zig");
const hip = @import("../root.zig");
const flashnext = @import("../kernel_tests.zig").flashnext();
const cache = flashnext.cache;
const gdn = flashnext.gdn_replay;
const model = flashnext.model;
const mixers = flashnext.mixers;
const forward = flashnext.forward;

const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;

const key_heads = 1;
const value_heads = 8;
const key_dim = 128;
const value_dim = 128;
const conv = 4;
const taps = conv - 1;
const kw = key_heads * key_dim;
const vw = value_heads * value_dim;
const channels = 2 * kw + vw;
const hidden = 32;
const vocab = 8;
const cap = 8;
/// Rows the linear attention's products take a window of: the pending token and three drafts.
const window_rows = 4;
const eps: f32 = 1e-6;
const state_words = value_heads * key_dim * value_dim;

/// Flash Next's dimensions at this shape, one linear layer then a sparse one so both descriptor forms run.
fn dimsOf() model.Dims {
    return .{
        .hidden = hidden,
        .streams = 1,
        .low = 1,
        .vocab = vocab,
        .layers = 2,
        .eps = eps,
        .heads = 1,
        .kv_heads = 1,
        .head_dim = 2,
        .rotary = 1,
        .theta = 10_000,
        .key_heads = key_heads,
        .value_heads = value_heads,
        .key_dim = key_dim,
        .value_dim = value_dim,
        .conv = conv,
        .index_heads = 1,
        .index_dim = 2,
        .index_ratio = 4,
        .index_top = 1,
        .ple_dim = 1,
        .ple_taps = 2,
        .ngram = 1,
        .vocab_rows = vocab,
        .gate = .sigmoid,
    };
}

fn dense(weight: u64, n: usize, k: usize) model.Projection {
    return .{ .n = @intCast(n), .k = @intCast(k), .handle = .{ .dense = .{ .weight = weight, .dtype = .f32 } } };
}

/// `n` rows of `k` weights, scaled so a row's product stays order one.
fn matrix(t: *Rig, rng: *rig.Rng, n: usize, k: usize) !rig.Buffer {
    const host = try gpa.alloc(f32, n * k);
    defer gpa.free(host);
    const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(k))));
    for (host) |*e| e.* = rng.unit() * scale;
    return rig.Buffer.fromHost(t, host);
}

fn vector(t: *Rig, rng: *rig.Rng, n: usize, scale: f32) !rig.Buffer {
    const host = try gpa.alloc(f32, n);
    defer gpa.free(host);
    for (host) |*e| e.* = rng.unit() * scale;
    return rig.Buffer.fromHost(t, host);
}

/// The linear layer's weight fields, in the order `free` takes them.
const names = struct {
    const all = [_][]const u8{ "qkv", "z", "a", "b", "out", "conv", "a_log", "dt_bias", "norm", "qw", "kw_w" };
};

const Weights = struct {
    qkv: rig.Buffer,
    z: rig.Buffer,
    a: rig.Buffer,
    b: rig.Buffer,
    out: rig.Buffer,
    conv: rig.Buffer,
    a_log: rig.Buffer,
    dt_bias: rig.Buffer,
    norm: rig.Buffer,
    qw: rig.Buffer,
    kw_w: rig.Buffer,

    fn make(t: *Rig, rng: *rig.Rng) !Weights {
        const shapes = [_][2]usize{
            .{ channels, hidden },    .{ vw, hidden },
            .{ value_heads, hidden }, .{ value_heads, hidden },
            .{ hidden, vw },
        };
        var w: Weights = undefined;
        var made: usize = 0;
        // every buffer made so far, in the order `free` takes them
        errdefer inline for (names.all, 0..) |name, i| {
            if (i < made) @field(w, name).free();
        };
        inline for (names.all[0..5], 0..) |name, i| {
            @field(w, name) = try matrix(t, rng, shapes[i][0], shapes[i][1]);
            made += 1;
        }
        w.conv = try vector(t, rng, channels * conv, 0.25);
        w.a_log = try vector(t, rng, value_heads, 0.5);
        w.dt_bias = try vector(t, rng, value_heads, 1.0);
        w.norm = try vector(t, rng, value_dim, 0.5);
        w.qw = try vector(t, rng, key_dim, 0.5);
        w.kw_w = try vector(t, rng, key_dim, 0.5);
        return w;
    }

    fn free(w: *Weights) void {
        inline for (names.all) |name| @field(w, name).free();
    }

    fn linear(w: *const Weights) model.Linear {
        return .{
            .qkv = dense(w.qkv.ptr, channels, hidden),
            .z = dense(w.z.ptr, vw, hidden),
            .a = dense(w.a.ptr, value_heads, hidden),
            .b = dense(w.b.ptr, value_heads, hidden),
            .out = dense(w.out.ptr, hidden, vw),
            .conv = w.conv.ptr,
            .a_log = w.a_log.ptr,
            .dt_bias = w.dt_bias.ptr,
            .norm = w.norm.ptr,
        };
    }

    /// The model this layer runs under: the linear mixer, the plan's slots (null: the per-row window), no sparse layer.
    fn modelOf(w: *const Weights, act: model.Kind, slots: ?*gdn.Slots) model.Model {
        return .{
            .dims = dimsOf(),
            .qk = .{ .q = w.qw.ptr, .k = w.kw_w.ptr, .eps = eps },
            .act = act,
            .embed = dense(w.qkv.ptr, hidden, hidden),
            .layers = &.{},
            .mix = .{ .scale = 0, .down = dense(w.z.ptr, vw, hidden), .up = dense(w.out.ptr, hidden, vw) },
            .head = dense(w.out.ptr, vocab, hidden),
            .ple = .{
                .layer = 0,
                .kv = dense(w.z.ptr, vw, hidden),
                .key_scale = 0,
                .query_scale = 0,
                .conv_scale = 0,
                .conv = 0,
            },
            .gdn = slots,
        };
    }
};

/// A stream's caches over the two layers, its plan slots, and the window's start state and conv tail.
const Window = struct {
    c: cache.Caches,
    slots: gdn.Slots,
    hs: []f32,
    hc: []f32,

    fn open(t: *Rig, rng: *rig.Rng, act: model.Kind) !Window {
        const kinds = [_]bool{ true, false };
        var win: Window = undefined;
        win.c = try cache.Caches.init(gpa, &t.r, dimsOf(), &kinds, cap, act, 0);
        errdefer win.c.deinit(gpa);
        win.slots = try gdn.Slots.init(gpa, &t.r, dimsOf(), 1);
        errdefer win.slots.deinit();
        win.hs = try gpa.alloc(f32, state_words);
        errdefer gpa.free(win.hs);
        win.hc = try gpa.alloc(f32, taps * channels);
        for (win.hs) |*e| e.* = 0.1 * rng.unit();
        for (win.hc) |*e| e.* = rng.unit();
        return win;
    }

    fn deinit(w: *Window) void {
        gpa.free(w.hc);
        gpa.free(w.hs);
        w.slots.deinit();
        w.c.deinit(gpa);
    }

    /// The window's start state and conv tail under the linear layer, the state's snapshots a nonzero byte.
    fn reset(w: *Window, mark: u8) !void {
        const l0 = w.c.linearAt(0);
        try w.c.mem.upload(l0.state - w.c.mem.base(), std.mem.sliceAsBytes(w.hs));
        try w.c.mem.upload(l0.conv - w.c.mem.base(), std.mem.sliceAsBytes(w.hc));
        try w.c.window.fill8(mark);
    }
};

/// A view of `b`'s bytes at `addr`, for reading what a launch wrote there.
fn bytesAt(b: hip.DeviceBuffer, addr: u64, n: usize) hip.DeviceBuffer {
    return .{ .r = b.r, .ptr = @ptrFromInt(addr), .len = n };
}

/// `words` 32-bit words of `b` at `addr`, for reading what a launch wrote there.
fn readWords(b: hip.DeviceBuffer, addr: u64, words: usize) ![]u32 {
    const out = try gpa.alloc(u32, words);
    errdefer gpa.free(out);
    try bytesAt(b, addr, words * 4).download(0, std.mem.sliceAsBytes(out));
    return out;
}

/// The per-row window's snapshots of the rows it ran: each row's state and conv tail, and the window's start.
const Snaps = struct {
    states: [][]u32,
    convs: [][]u32,
    start: []u32,
    start_conv: []u32,

    fn read(w: *const Window, rows: usize) !Snaps {
        var s: Snaps = undefined;
        s.states = try gpa.alloc([]u32, rows);
        errdefer gpa.free(s.states);
        s.convs = try gpa.alloc([]u32, rows);
        var made: usize = 0;
        errdefer for (s.convs[0..made]) |b| gpa.free(b);
        errdefer for (s.states[0..made]) |a| gpa.free(a);
        for (0..rows) |r| {
            s.states[r] = try readWords(w.c.window, w.c.snapshot(0, r).state, state_words);
            made += 1;
            s.convs[r] = try readWords(w.c.window, w.c.snapshot(0, r).conv, taps * channels);
        }
        s.start = try gpa.alloc(u32, state_words);
        errdefer gpa.free(s.start);
        s.start_conv = try gpa.alloc(u32, taps * channels);
        for (s.start, w.hs) |*e, v| e.* = @bitCast(v);
        for (s.start_conv, w.hc) |*e, v| e.* = @bitCast(v);
        return s;
    }

    fn deinit(s: *Snaps) void {
        gpa.free(s.start_conv);
        gpa.free(s.start);
        for (s.states, s.convs) |a, b| {
            gpa.free(a);
            gpa.free(b);
        }
        gpa.free(s.convs);
        gpa.free(s.states);
    }
};

test "a window's plan: fill writes every slot's words and the linear layer's descriptor where the kernels read them" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x2728 };
    var win = try Window.open(t, &rng, if (t.bf16) .bf16 else .f16);
    defer win.deinit();
    for ([_]usize{ 1, 2, 4 }) |rows| try fillCase(t, &win, rows);
}

/// The plan's words after a fill for `rows`: where every kernel reads them.
fn fillCase(t: *Rig, win: *Window, rows: usize) !void {
    const p = win.slots.planFor(0).?;
    const l = p.layout();
    const d = dimsOf();
    try p.fill(&win.c, rows);
    try p.send(t.stream.handle);
    try t.stream.synchronize();
    const bytes = try gpa.alloc(u8, p.dev.len);
    defer gpa.free(bytes);
    try p.dev.download(0, bytes);
    const held = std.mem.bytesAsSlice(u64, @as([]align(8) u8, @alignCast(bytes)));
    const words = std.mem.bytesAsSlice(u32, @as([]align(4) u8, @alignCast(bytes)));
    // the window slot's rows from its first, kept as asked; the empty slot runs and keeps nothing
    try std.testing.expectEqual(@as(u32, 0), words[l.first]);
    try std.testing.expectEqual(@as(u32, @intCast(rows)), words[l.count]);
    try std.testing.expectEqual(@as(u32, @bitCast(@as(i32, -1))), words[l.keep]);
    try std.testing.expectEqual(@as(u32, 0), words[l.first + 1]);
    try std.testing.expectEqual(@as(u32, 0), words[l.count + 1]);
    try std.testing.expectEqual(@as(u32, @bitCast(@as(i32, -1))), words[l.keep + 1]);
    // the slot pointers name the plan's own descriptors, one after another
    const d0 = p.dev.base() + 4 * std.mem.alignForward(usize, l.words, 2);
    try std.testing.expectEqual(d0, held[l.ptr / 2]);
    try std.testing.expectEqual(d0 + 4 * gdn.Layout.desc(d.layers), held[l.ptr / 2 + 1]);
    // the linear layer's descriptor: no kept final row, then its conv and state; the sparse layer's zeros beside
    const desc = std.mem.bytesAsSlice(u64, @as([]align(8) u8, @alignCast(bytes)));
    const first = std.mem.alignForward(usize, l.words, 2) / 2;
    const l0 = win.c.linearAt(0);
    try std.testing.expectEqual(@as(u64, 0), desc[first]);
    try std.testing.expectEqual(@as(u64, 0), desc[first + 1]);
    try std.testing.expectEqual(l0.conv, desc[first + 2]);
    try std.testing.expectEqual(l0.state, desc[first + 3]);
    try std.testing.expectEqual(@as(u64, 0), desc[first + 4]);
    try std.testing.expectEqual(@as(u64, 0), desc[first + 5]);
    // the samples and kept rows the linear layer keeps, and the sparse layer's zeros
    try std.testing.expectEqual(p.samplesOf(0), desc[l.snaps / 2]);
    try std.testing.expectEqual(p.kept.base(), desc[l.snaps / 2 + 1]);
    try std.testing.expectEqual(@as(u64, 0), desc[l.snaps / 2 + 2]);
    try std.testing.expectEqual(@as(u64, 0), desc[l.snaps / 2 + 3]);
}

test "a window's conv split on its plan, norms and gates on, writes the per-row launch's q, k, v, gate and beta" {
    const t = try Rig.open(1 << 22);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x2729 };
    var w = try Weights.make(t, &rng);
    defer w.free();
    var win = try Window.open(t, &rng, if (t.bf16) .bf16 else .f16);
    defer win.deinit();
    for ([_]usize{ 1, 2, 3, 4 }) |rows| try convSplitCase(t, &w, &win, &rng, rows);
}

/// Both paths' conv split over `rows`: today's launch keeps the state a row, the plan's the input a row.
fn convSplitCase(t: *Rig, w: *const Weights, win: *Window, rng: *rig.Rng, rows: usize) !void {
    const o = t.ops(&t.on);
    const act = w.modelOf(if (t.bf16) .bf16 else .f16, null).act;
    const p = win.slots.planFor(0).?;
    const l0 = win.c.linearAt(0);
    const hx = try gpa.alloc(u16, rows * channels);
    defer gpa.free(hx);
    for (hx) |*e| e.* = t.bits(rng.unit());
    const ha = try gpa.alloc(u16, rows * value_heads);
    defer gpa.free(ha);
    for (ha) |*e| e.* = t.bits(rng.unit());
    const hb = try gpa.alloc(u16, rows * value_heads);
    defer gpa.free(hb);
    for (hb) |*e| e.* = t.bits(rng.unit());
    const ng = rows * value_heads;
    var x = try rig.Buffer.fromHost(t, hx);
    defer x.free();
    var ga = try rig.Buffer.fromHost(t, ha);
    defer ga.free();
    var gb = try rig.Buffer.fromHost(t, hb);
    defer gb.free();
    // the gate's values both launches write over the same a and b
    var gate = try rig.Buffer.alloc(t, ng * 4);
    defer gate.free();
    var beta = try rig.Buffer.alloc(t, ng * 4);
    defer beta.free();
    try o.gdnGatePrefill(
        .{ .ptr = ga.ptr, .kind = act },
        .{ .ptr = gb.ptr, .kind = act },
        w.a_log.ptr,
        w.dt_bias.ptr,
        gate.ptr,
        beta.ptr,
        ng,
        value_heads,
    );
    // the plan writes q beside its rows and k, v by plan_ops.convSplit's signature: only q has a buffer of its own
    var planned: rig.Buffer = try rig.Buffer.alloc(t, rows * kw * 4);
    defer planned.free();
    var today: [3]rig.Buffer = .{ undefined, undefined, undefined };
    var made: usize = 0;
    defer for (today[0..made]) |*a| a.free();
    for ([_]usize{ rows * kw, rows * kw, rows * vw }) |n| {
        today[made] = try rig.Buffer.alloc(t, n * 4);
        made += 1;
    }
    try win.reset(0x5A);
    try o.l.tf_conv_split(.{
        .x = x.ptr,
        .kind = @backingInt(act),
        .weight = w.conv.ptr,
        .state = l0.conv,
        .states = win.c.snapshot(win.c.slot[0], 0).conv,
        .qn = today[0].ptr,
        .kn = today[1].ptr,
        .v = today[2].ptr,
        .channels = channels,
        .kernel = conv,
        .rows = @intCast(rows),
        .kw = kw,
        .vw = vw,
        .qw = w.qw.ptr,
        .kw_w = w.kw_w.ptr,
        .eps = eps,
        .norm = 128,
        .ga = ga.ptr,
        .gb = gb.ptr,
        .a_log = w.a_log.ptr,
        .dt_bias = w.dt_bias.ptr,
        .gate = gate.ptr,
        .beta = beta.ptr,
        .gcount = @intCast(ng),
        .heads = value_heads,
    }, o.stream);
    try t.stream.synchronize();
    // the plan's: the same launch from the window's words, which keeps the rows' conv input in place
    try win.reset(0x5A);
    try p.fill(&win.c, rows);
    try p.send(o.stream);
    const kept = p.keptOf(0, 0);
    try hip.plan_ops.convSplit(
        o,
        .{ .ptr = x.ptr, .kind = act },
        w.conv.ptr,
        planned.ptr,
        kept.k,
        kept.v,
        p.ref(),
        0,
        channels,
        conv,
        kw,
        vw,
        .{ .q = w.qw.ptr, .k = w.kw_w.ptr, .eps = eps },
        .{
            .a = .{ .ptr = ga.ptr, .kind = act },
            .b = .{ .ptr = gb.ptr, .kind = act },
            .a_log = w.a_log.ptr,
            .dt_bias = w.dt_bias.ptr,
            .gate = kept.gate,
            .beta = kept.beta,
            .count = ng,
            .heads = value_heads,
        },
    );
    try t.stream.synchronize();
    for ([_]struct { what: []const u8, n: usize, a: hip.DeviceBuffer, b: hip.DeviceBuffer }{
        .{ .what = "q", .n = rows * kw, .a = today[0].buf, .b = planned.buf },
        .{ .what = "k", .n = rows * kw, .a = today[1].buf, .b = bytesAt(p.kept, kept.k, rows * kw * 4) },
        .{ .what = "v", .n = rows * vw, .a = today[2].buf, .b = bytesAt(p.kept, kept.v, rows * vw * 4) },
        .{ .what = "gate", .n = ng, .a = gate.buf, .b = bytesAt(p.kept, kept.gate, ng * 4) },
        .{ .what = "beta", .n = ng, .a = beta.buf, .b = bytesAt(p.kept, kept.beta, ng * 4) },
    }) |c2| {
        const n = c2.n;
        const want = try readWords(c2.a, at(c2.a), n);
        defer gpa.free(want);
        const got = try readWords(c2.b, at(c2.b), n);
        defer gpa.free(got);
        try std.testing.expectEqualSlices(u32, want, got);
    }
}

test "a window through its plan equals the per-row window at every keep n: y, the state and the conv tail" {
    const t = try Rig.open(1 << 24);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x2730 };
    var w = try Weights.make(t, &rng);
    defer w.free();
    var win = try Window.open(t, &rng, if (t.bf16) .bf16 else .f16);
    defer win.deinit();
    var tokens: [window_rows]u32 = undefined;
    for (&tokens, 0..) |*tok, i| tok.* = @intCast(i);
    for ([_]usize{ 1, 2, 3, 4 }) |rows| try planCase(t, &w, &win, &rng, tokens[0..rows], rows);
}

/// A window of `rows` both ways: today's forward, then the plan's, then every keep n against today's snapshots.
fn planCase(t: *Rig, w: *const Weights, win: *Window, rng: *rig.Rng, tokens: []const u32, rows: usize) !void {
    const o = t.ops(&t.on);
    const act: model.Kind = if (t.bf16) .bf16 else .f16;
    const p = win.slots.planFor(0).?;
    const l0 = win.c.linearAt(0);
    const m = w.modelOf(act, null);
    const planned = w.modelOf(act, &win.slots);
    const hx = try gpa.alloc(u16, rows * channels);
    defer gpa.free(hx);
    for (hx) |*e| e.* = t.bits(rng.unit());
    var x = try rig.Buffer.fromHost(t, hx);
    defer x.free();
    const xt: hip.ops.Tensor = .{ .ptr = x.ptr, .kind = act };
    const seg = [_]forward.Seg{.{ .c = &win.c, .tokens = tokens, .pos = 0 }};
    // today's window: one conv split and delta rule a stream, a snapshot a row, the state written back
    try win.reset(0x5A);
    o.arena.reset();
    const y0 = try mixers.linear(o, &m, w.linear(), &seg, 0, xt, rows, true);
    try t.stream.synchronize();
    const y_a = try readWords(win.c.mem, y0.ptr, rows * hidden * act.size() / 4);
    defer gpa.free(y_a);
    var snaps = try Snaps.read(win, rows);
    defer snaps.deinit();
    // the plan's: the same forward from the window's words, every row's state left to the keep
    try win.reset(0x12);
    try p.fill(&win.c, rows);
    try p.send(o.stream);
    o.arena.reset();
    const y1 = try mixers.linear(o, &planned, w.linear(), &seg, 0, xt, rows, true);
    try t.stream.synchronize();
    const y_b = try readWords(win.c.mem, y1.ptr, rows * hidden * act.size() / 4);
    defer gpa.free(y_b);
    try std.testing.expectEqualSlices(u32, y_a, y_b);
    // what the plan's round left where it found it: a one-row window writes it back, a longer one leaves the start
    const plan_left = try readWords(win.c.mem, l0.state, state_words);
    defer gpa.free(plan_left);
    const plan_tail = try readWords(win.c.mem, l0.conv, taps * channels);
    defer gpa.free(plan_tail);
    // the round writes its state and tail back only for a window of one row; a longer one leaves the start for its keep
    if (rows == 1) {
        try std.testing.expectEqualSlices(u32, snaps.states[0], plan_left);
        try std.testing.expectEqualSlices(u32, snaps.convs[0], plan_tail);
    } else {
        try std.testing.expectEqualSlices(u32, snaps.start, plan_left);
        try std.testing.expectEqualSlices(u32, snaps.start_conv, plan_tail);
    }
    // a one-row window's round left its state in place; a longer one keeps from the start its round left
    if (rows > 1) for (1..rows + 1) |n| try keepCase(t, p, win, o, n, snaps.states[n - 1], snaps.convs[n - 1]);
}

/// A keep of the window's first `n` rows: the state and conv tail it leaves are today's snapshot of row n.
fn keepCase(
    t: *Rig,
    p: *gdn.Plan,
    win: *Window,
    o: hip.ops.Ops,
    n: usize,
    want_state: []const u32,
    want_conv: []const u32,
) !void {
    // the keep reads the round's samples and kept rows, which the plan's forward above wrote
    try win.reset(0x12);
    try p.fill(&win.c, p.rows);
    try p.send(o.stream);
    try p.keep(o, n);
    try t.stream.synchronize();
    const l0 = win.c.linearAt(0);
    const got_state = try readWords(win.c.mem, l0.state, state_words);
    defer gpa.free(got_state);
    try std.testing.expectEqualSlices(u32, want_state, got_state);
    const got_conv = try readWords(win.c.mem, l0.conv, taps * channels);
    defer gpa.free(got_conv);
    try std.testing.expectEqualSlices(u32, want_conv, got_conv);
}
