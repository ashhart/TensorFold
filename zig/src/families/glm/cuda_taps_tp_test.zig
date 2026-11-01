//! DFlash2 prompt taps at TP>1: a recording CUDA driver and emulated NCCL check every rank holds every row.
const std = @import("std");
const cuda = @import("cuda");
const fw = @import("cuda_forward.zig");
const pieces_mod = @import("cuda_forward_pieces.zig");
const taps = @import("cuda_taps.zig");
const dflash = @import("dflash.zig");
const cfg = @import("config.zig");
const shard = @import("cuda_shard.zig");
const weights = @import("cuda_weights.zig");

const Forward = fw.Forward;
const Rank = fw.Rank;
const Buffers = fw.Buffers;
const Weights = weights.Weights;

const MAXR: usize = 32; // rows a scenario uses at most (the dispatch arithmetic is row-count independent)
const step_base: u64 = 0x1_0000_0000; // one buffer's address space, wider than any buffer below
const tap_fn: usize = 0xDF7A9; // the recording driver accepts only df_tap launches at this handle
const POISON: u64 = 0xBADD_0000_0BAD_0000; // a non-owner row never gathered keeps this and fails the row check
fn trueX(row: usize) u64 {
    return 0x7A05_0000_0000_0000 | @as(u64, @intCast(row)); // a row's residual tag, distinct from POISON
}
fn base(k: usize) u64 {
    return @as(u64, @intCast(k)) * step_base;
}

/// One device buffer by base address: residual rows (one tag a row) or taps (layers.len tags a row).
const Buf = struct { base: u64, stride: usize, slots: usize, tags: []u64 };

/// Host memory standing in for device memory at row granularity, indexed back through the base and stride.
const Device = struct {
    bufs: [16]Buf = undefined,
    n: usize = 0,

    fn add(d: *Device, b: u64, stride: usize, slots: usize, tags: []u64) void {
        d.bufs[d.n] = .{ .base = b, .stride = stride, .slots = slots, .tags = tags };
        d.n += 1;
    }
    fn find(d: *Device, ptr: u64) struct { buf: *Buf, row: usize } {
        for (d.bufs[0..d.n]) |*b| {
            const rows = b.tags.len / b.slots;
            if (ptr >= b.base and ptr < b.base + b.stride * rows) {
                std.debug.assert((ptr - b.base) % b.stride == 0);
                return .{ .buf = b, .row = (ptr - b.base) / b.stride };
            }
        }
        unreachable; // every pointer the tap path builds is a registered buffer at an aligned row
    }
};
var dev: Device = .{};

fn memcpy(dst: cuda.abi.DevicePtr, src: cuda.abi.DevicePtr, bytes: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
    const s = dev.find(src);
    const d = dev.find(dst);
    std.debug.assert(s.buf.slots == d.buf.slots and bytes % s.buf.stride == 0);
    const n = (bytes / s.buf.stride) * s.buf.slots;
    @memcpy(d.buf.tags[d.row * d.buf.slots ..][0..n], s.buf.tags[s.row * s.buf.slots ..][0..n]);
    return cuda.abi.success;
}

/// df_tap's mode 0: tap slot k of n rows takes the residual row's tag (identity keeps the row check readable).
fn launchTap(f: cuda.abi.Function, gx: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, params: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
    std.debug.assert(f == @as(cuda.abi.Function, @ptrFromInt(tap_fn)));
    const p = params.?;
    const xptr = @as(*align(1) const u64, @ptrCast(p[0].?)).*;
    const outptr = @as(*align(1) const u64, @ptrCast(p[1].?)).*;
    const k: usize = @intCast(@as(*align(1) const i32, @ptrCast(p[2].?)).*);
    const x = dev.find(xptr);
    const o = dev.find(outptr);
    for (0..gx) |i| o.buf.tags[(o.row + i) * o.buf.slots + k] = x.buf.tags[x.row + i];
    return cuda.abi.success;
}

fn errName(_: cuda.abi.Result, out: *?[*:0]const u8) callconv(.c) cuda.abi.Result {
    out.* = "fixture";
    return cuda.abi.success;
}
fn driver() cuda.Driver {
    var api: cuda.abi.Api = undefined;
    api.cuMemcpyDtoDAsync_v2 = memcpy;
    api.cuLaunchKernel = launchTap;
    api.cuGetErrorName = errName;
    api.cuGetErrorString = errName;
    return .{ .lib = undefined, .api = api, .func_name = null };
}

const Kind = enum { send, recv };
const Post = struct { kind: Kind, rank: usize, group: usize, peer: usize, row: usize, rows: usize };
const Msg = struct { vals: [MAXR * taps.layers.len]u64 = undefined, len: usize = 0, used: bool = false };

/// NCCL for the split taps exchange, recorded then replayed a group at a time so sends land before receives.
const Wire = struct {
    posts: [512]Post = undefined,
    n: usize = 0,
    cur: usize = 0, // the rank whose shareTaps is posting now
    groups: [3]usize = .{ 0, 0, 0 },
    gidx: usize = 0,

    fn record(w: *Wire, kind: Kind, peer: usize, ptr: u64, count: usize) void {
        const loc = dev.find(ptr);
        w.posts[w.n] = .{ .kind = kind, .rank = w.cur, .group = w.gidx, .peer = peer, .row = loc.row, .rows = count / (taps.layers.len * taps.hidden) };
        w.n += 1;
    }

    fn replay(w: *Wire, outs: [][]u64) void {
        const S = taps.layers.len;
        var maxg: usize = 0;
        for (w.posts[0..w.n]) |p| maxg = @max(maxg, p.group + 1);
        var g: usize = 0;
        while (g < maxg) : (g += 1) {
            var mail: [3][3]Msg = undefined;
            for (0..3) |a| for (0..3) |b| {
                mail[a][b] = .{};
            };
            for (w.posts[0..w.n]) |p| if (p.group == g and p.kind == .send) {
                const m = &mail[p.rank][p.peer];
                m.len = p.rows * S;
                @memcpy(m.vals[0..m.len], outs[p.rank][p.row * S ..][0..m.len]);
                m.used = true;
            };
            for (w.posts[0..w.n]) |p| if (p.group == g and p.kind == .recv) {
                const m = &mail[p.peer][p.rank]; // the sender is this receive's peer
                std.debug.assert(m.used and m.len == p.rows * S);
                @memcpy(outs[p.rank][p.row * S ..][0..m.len], m.vals[0..m.len]);
            };
        }
    }
};
var wire: Wire = .{};

fn grpStart() callconv(.c) cuda.nccl.Result {
    wire.gidx = wire.groups[wire.cur];
    wire.groups[wire.cur] += 1;
    return 0;
}
fn grpEnd() callconv(.c) cuda.nccl.Result {
    return 0;
}
fn nSend(ptr: cuda.abi.DevicePtr, count: usize, _: cuda.nccl.DataType, peer: c_int, _: cuda.nccl.Comm, _: cuda.abi.Stream) callconv(.c) cuda.nccl.Result {
    wire.record(.send, @intCast(peer), ptr, count);
    return 0;
}
fn nRecv(ptr: cuda.abi.DevicePtr, count: usize, _: cuda.nccl.DataType, peer: c_int, _: cuda.nccl.Comm, _: cuda.abi.Stream) callconv(.c) cuda.nccl.Result {
    wire.record(.recv, @intCast(peer), ptr, count);
    return 0;
}
fn fakeLib() cuda.nccl.Library {
    var api: cuda.nccl.Api = undefined;
    api.ncclGroupStart = grpStart;
    api.ncclGroupEnd = grpEnd;
    api.ncclSend = nSend;
    api.ncclRecv = nRecv;
    return .{ .lib = undefined, .api = api };
}

/// Rank that owns row i: the piece holding it, then the share within that piece, through the engine's own split.
fn ownerOf(i: usize, cut: []const [2]usize, world: usize) usize {
    for (cut) |p| if (i >= p[0] and i < p[1]) {
        const n = p[1] - p[0];
        for (0..world) |r| {
            const sh = Forward.share(n, world, r);
            if (i - p[0] >= sh[0] and i - p[0] < sh[1]) return r;
        }
    };
    unreachable;
}

fn makeWeights(world: usize, rank: usize) Weights {
    var w: Weights = undefined;
    w.config = cfg.Config{};
    w.share = shard.Share.of(&w.config, rank, world) catch unreachable;
    return w;
}
fn makeBuffers(xb: u64) Buffers {
    var b: Buffers = undefined;
    b.x = xb;
    return b;
}
fn expectComplete(ot: []const u64, R: usize) !void {
    for (0..R) |i| for (0..taps.layers.len) |k|
        try std.testing.expectEqual(trueX(i), ot[i * taps.layers.len + k]);
}

/// In-process ranks sharing one tap buffer (the local or gather exchange): split gathers residual rows by copy.
fn runShared(world: usize, R: usize, cut: []const [2]usize, split: bool) !void {
    dev = .{};
    const drv = driver();
    var ws: [3]Weights = undefined;
    var bs: [3]Buffers = undefined;
    var rk: [3]Rank = undefined;
    var xstore: [3][MAXR]u64 = undefined;
    var ostore: [MAXR * taps.layers.len]u64 = undefined;
    for (0..world) |r| {
        ws[r] = makeWeights(world, r);
        const xb = base(1 + r);
        bs[r] = makeBuffers(xb);
        rk[r] = .{ .w = &ws[r], .st = undefined, .b = &bs[r] };
        for (0..R) |i| xstore[r][i] = if (!split or ownerOf(i, cut, world) == r) trueX(i) else POISON;
        dev.add(xb, 4 * taps.hidden * 2, 1, xstore[r][0..R]);
    }
    const ob = base(99);
    @memset(ostore[0 .. R * taps.layers.len], POISON);
    dev.add(ob, taps.layers.len * taps.hidden * 2, taps.layers.len, ostore[0 .. R * taps.layers.len]);
    const stream: cuda.Stream = .{ .d = &drv, .handle = @ptrFromInt(1) };
    const f: Forward = .{ .d = &drv, .m = undefined, .stream = stream, .ranks = rk[0..world], .exchange = .local, .split = split, .ring = false };
    const t: taps.Taps = .{ .f = .{ .d = &drv, .handle = @ptrFromInt(tap_fn) }, .out = ob };
    for (taps.layers) |j| try pieces_mod.tapPieces(&f, t, j, cut);
    try pieces_mod.shareTaps(&f, t, cut);
    try expectComplete(ostore[0 .. R * taps.layers.len], R);
}

/// One Forward a rank (the NCCL exchange): split taps its own rows, then shareTaps all-gathers through the fake wire.
fn runNccl(world: usize, R: usize, cut: []const [2]usize, split: bool, ring: bool) !void {
    dev = .{};
    wire = .{};
    const drv = driver();
    const lib = fakeLib();
    var ws: [3]Weights = undefined;
    var bs: [3]Buffers = undefined;
    var rk: [3]Rank = undefined;
    var fwd: [3]Forward = undefined;
    var tp: [3]taps.Taps = undefined;
    var xstore: [3][MAXR]u64 = undefined;
    var ostore: [3][MAXR * taps.layers.len]u64 = undefined;
    for (0..world) |r| {
        ws[r] = makeWeights(world, r);
        const xb = base(1 + r);
        bs[r] = makeBuffers(xb);
        rk[r] = .{ .w = &ws[r], .st = undefined, .b = &bs[r] };
        for (0..R) |i| xstore[r][i] = if (!split or ownerOf(i, cut, world) == r) trueX(i) else POISON;
        dev.add(xb, 4 * taps.hidden * 2, 1, xstore[r][0..R]);
        const ob = base(100 + r);
        @memset(ostore[r][0 .. R * taps.layers.len], POISON);
        dev.add(ob, taps.layers.len * taps.hidden * 2, taps.layers.len, ostore[r][0 .. R * taps.layers.len]);
        const comm: cuda.nccl.Comm = @ptrFromInt(r + 1);
        fwd[r] = .{ .d = &drv, .m = undefined, .stream = .{ .d = &drv, .handle = @ptrFromInt(1) }, .ranks = rk[r .. r + 1], .exchange = .{ .nccl = .{ .lib = &lib, .comm = comm } }, .split = split, .ring = ring };
        tp[r] = .{ .f = .{ .d = &drv, .handle = @ptrFromInt(tap_fn) }, .out = ob };
    }
    for (0..world) |r| for (taps.layers) |j| try pieces_mod.tapPieces(&fwd[r], tp[r], j, cut);
    for (0..world) |r| {
        wire.cur = r;
        try pieces_mod.shareTaps(&fwd[r], tp[r], cut);
    }
    var outs: [3][]u64 = undefined;
    for (0..world) |r| outs[r] = ostore[r][0 .. R * taps.layers.len];
    wire.replay(outs[0..world]);
    for (0..world) |r| try expectComplete(outs[r], R);
}

const cut_a = [_][2]usize{ .{ 0, 5 }, .{ 5, 11 }, .{ 11, 18 } }; // three uneven pieces, nonzero offsets
const cut_b = [_][2]usize{.{ 0, 9 }}; // one piece
const cut_c = [_][2]usize{ .{ 0, 4 }, .{ 4, 13 }, .{ 13, 20 } }; // pieces off the rank split boundaries

test "GLM prompt taps: local split copies gather every row into one tap buffer at TP2 and TP3" {
    for ([_]usize{ 2, 3 }) |world| {
        try runShared(world, 18, &cut_a, true);
        try runShared(world, 9, &cut_b, true);
        try runShared(world, 20, &cut_c, true);
    }
}

test "GLM prompt taps: the gather control taps the whole complete chunk at TP2 and TP3" {
    for ([_]usize{ 2, 3 }) |world| {
        try runShared(world, 18, &cut_a, false);
        try runNccl(world, 18, &cut_a, false, false);
    }
}

test "GLM prompt taps: NCCL split direct sends gather every rank's taps at TP2 and TP3" {
    for ([_]usize{ 2, 3 }) |world| {
        try runNccl(world, 18, &cut_a, true, false);
        try runNccl(world, 9, &cut_b, true, false);
        try runNccl(world, 20, &cut_c, true, false);
    }
}

test "GLM prompt taps: NCCL split ring relay gathers every rank's taps at TP2 and TP3" {
    for ([_]usize{ 2, 3 }) |world| {
        try runNccl(world, 18, &cut_a, true, true);
        try runNccl(world, 9, &cut_b, true, true);
        try runNccl(world, 20, &cut_c, true, true);
    }
}

test "GLM prompt taps: the pre-patch whole-chunk tap leaves non-owner rows poisoned" {
    const world = 3;
    const R = 18;
    const cut: []const [2]usize = &cut_a;
    dev = .{};
    const drv = driver();
    var ws: [3]Weights = undefined;
    var bs: [3]Buffers = undefined;
    var rk: [3]Rank = undefined;
    var xstore: [3][MAXR]u64 = undefined;
    var ostore: [MAXR * taps.layers.len]u64 = undefined;
    for (0..world) |r| {
        ws[r] = makeWeights(world, r);
        const xb = base(1 + r);
        bs[r] = makeBuffers(xb);
        rk[r] = .{ .w = &ws[r], .st = undefined, .b = &bs[r] };
        for (0..R) |i| xstore[r][i] = if (ownerOf(i, cut, world) == r) trueX(i) else POISON;
        dev.add(xb, 4 * taps.hidden * 2, 1, xstore[r][0..R]);
    }
    const ob = base(99);
    @memset(ostore[0 .. R * taps.layers.len], POISON);
    dev.add(ob, taps.layers.len * taps.hidden * 2, taps.layers.len, ostore[0 .. R * taps.layers.len]);
    const stream: cuda.Stream = .{ .d = &drv, .handle = @ptrFromInt(1) };
    const t: taps.Taps = .{ .f = .{ .d = &drv, .handle = @ptrFromInt(tap_fn) }, .out = ob };
    // the old path tapped the whole chunk from rank 0's residual, which at TP>1 holds only rank 0's rows
    for (taps.layers) |j| try t.at(stream, j, rk[0].b.x, 0, R);
    var poisoned = false;
    for (0..R) |i| if (ownerOf(i, cut, world) != 0 and ostore[i * taps.layers.len] != trueX(i)) {
        poisoned = true;
    };
    try std.testing.expect(poisoned);
}

test "GLM prompt taps: keep_from boundary follows dflash.promptRun, one row each side" {
    const WIN: usize = @intCast(dflash.WIN);
    const len = WIN + 4;
    const keep = len - WIN; // the first window row DFlash2 reads
    try std.testing.expect((try dflash.promptRun(0, 0, 0, keep, len, 0)) == null);
    const run = (try dflash.promptRun(0, 0, 0, keep + 1, len, 3)).?;
    try std.testing.expectEqual(@as(usize, 1), run.n);
    try std.testing.expectEqual(@as(i64, @intCast(keep)), run.pos0);
    try std.testing.expectEqual(@as(u64, (3 + keep) * dflash.TAP_ROW), run.taps);
    // the engine's f.taps = null predicate agrees with promptRun returning null at every row across the boundary
    for ([_]usize{ 0, 1, keep - 1, keep, keep + 1, WIN }) |rows| {
        if (rows > len) continue;
        const skip = rows <= len -| WIN;
        const none = (try dflash.promptRun(0, 0, 0, rows, len, 0)) == null;
        try std.testing.expectEqual(skip, none);
    }
}
