//! Ling's kept states on disk (--learn-conversations, --spill-gib): every rank's share in rank 0's one image.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const api = @import("engine_api");
const engine = @import("engine.zig");
const kept = @import("kept.zig");
const tp = @import("tp.zig");
const state = @import("state.zig");
const rows = api.prompt_cache.rows;
const Engine = engine.Engine;
const Snap = kept.Snap;
const Allocator = std.mem.Allocator;

/// The store's view of a kept state: words, each rank's KDA parts, the MTP layer's rows, then each MLA layer's rows.
pub const image: rows.Image = .{ .parts = parts, .gather = gather, .blank = blank, .scatter = scatter, .whole = whole };

const row = state.cache_width * 2; // a latent cache row's bytes
const chunk = @import("graph.zig").latent.chunk; // the split's unit: positions [c * chunk, (c + 1) * chunk) on rank c % world
const scratch_bytes = 64 << 20; // the ranks' transfer buffer, each
const piece_max = 16 << 20; // the most bytes one copied piece holds
const batch_max = 128; // pieces one step names

const Words = extern struct { at: u32, hd_pos: u32, world: u32, hd: u32, head: u32, hd_head: u32, mla: u32, hd_mla: u32 };
const words = @sizeOf(Words);

/// The image's shape: rank count, each rank's KDA bytes (backbone, MTP layer), MLA layers, and the cache split.
const Shape = struct { world: usize, head: usize, hd_head: usize, mla: usize, hd_mla: usize, hd: bool, split: ?state.Split };

fn shape(e: *const Engine) !Shape {
    const q = e.seqs.items[0] orelse return error.NoSequence;
    const hd = q.hd;
    return .{ .world = e.w.world, .head = q.st.keptHead(), .hd_head = if (hd) |h| h.keptHead() else 0, .mla = q.st.mla_layers, .hd_mla = if (hd) |h| h.mla_layers else 0, .hd = hd != null, .split = q.st.split };
}

/// Offsets of the image's parts at `at`: the KDA parts, the MTP rows (positions below at + 1), the MLA rows.
const Layout = struct {
    heads: u64,
    hd_rows: u64,
    rows: u64,
    end: u64,
    fn of(sh: Shape, at: u64) Layout {
        const heads: u64 = words;
        const hd_rows = heads + sh.world * (sh.head + sh.hd_head);
        const mla = hd_rows + sh.hd_mla * (at + 1) * row;
        return .{ .heads = heads, .hd_rows = hd_rows, .rows = mla, .end = mla + sh.mla * at * row };
    }
};

fn parts(ptr: *anyopaque, at: u32, out: *std.ArrayList(rows.image.Part), a: Allocator) anyerror!void {
    const sh = try shape(eng(ptr));
    const l = Layout.of(sh, at);
    try out.append(a, .{ .bytes = words });
    if (l.hd_rows > l.heads) try out.append(a, .{ .bytes = l.hd_rows - l.heads });
    if (l.rows > l.hd_rows) try out.append(a, .{ .bytes = l.rows - l.hd_rows }); // MTP rows read token p + 1: own
    for (0..sh.mla) |_| try out.append(a, .{ .bytes = @as(u64, at) * row, .row = row });
}

fn eng(ptr: *anyopaque) *Engine {
    return @ptrCast(@alignCast(ptr));
}

/// One piece of a range: a rank's region (0: KDA, 1: MTP KDA, 2 + mi: MLA layer mi, 2 + mla + mi: MTP layer mi).
const Seg = struct { len: usize, rank: usize = 0, region: u32 = 0, local: u64 = 0, kind: enum { words, data, zero } = .data, all: bool = false };

/// The pieces of image bytes [off, off + len) for a state at `at` whose MTP rows stand at `hd_pos`.
fn segments(sh: Shape, at: u64, hd_pos: u64, off: u64, len: u64, out: *std.ArrayList(Seg), a: Allocator) !void {
    const l = Layout.of(sh, at);
    var o = off;
    const end = off + len;
    if (end > l.end) return error.BadKeptRange;
    while (o < end) {
        var s: Seg = .{ .len = 0 };
        if (o < words) {
            s = .{ .len = @intCast(@min(end, words) - o), .kind = .words, .local = o };
        } else if (o < l.hd_rows) {
            const x = o - l.heads;
            const kda = sh.world * sh.head;
            const one: u64 = if (x < kda) sh.head else sh.hd_head;
            const y = if (x < kda) x else x - kda;
            s = .{ .len = 0, .rank = @intCast(y / one), .region = if (x < kda) 0 else 1, .local = y % one };
            s.len = @intCast(@min(end - o, one - s.local, piece_max));
        } else if (o < l.rows) {
            const x = o - l.hd_rows;
            const n = (at + 1) * row;
            s = positional(sh, x % n, @min(end - o, n - x % n), hd_pos, @intCast(2 + sh.mla + x / n));
        } else {
            const x = o - l.rows;
            const n = at * row;
            s = positional(sh, x % n, @min(end - o, n - x % n), at, @intCast(2 + x / n));
        }
        try out.append(a, s);
        o += s.len;
    }
}

/// A layer's bytes from `b` (of `n`) by position: the rank holding its chunk, rows past `limit` zero.
fn positional(sh: Shape, b: u64, most: u64, limit: u64, region: u32) Seg {
    const p = b / row;
    const c = p / chunk;
    var len = @min(most, (c + 1) * chunk * row - b, piece_max);
    if (b >= limit * row) return .{ .len = @intCast(len), .kind = .zero };
    len = @min(len, limit * row - b);
    const s = sh.split orelse return .{ .len = @intCast(len), .region = region, .local = b, .all = true };
    return .{ .len = @intCast(len), .rank = c % s.world, .region = region, .local = ((c / s.world) * chunk + p % chunk) * row + b % row };
}

/// Where region `region` of this rank's state `s` starts: its KDA buffers, or its rows (in place, or in the copy).
fn regionAt(e: *const Engine, s: *const Snap, region: u32) !u64 {
    const q = e.seqs.items[0] orelse return error.NoSequence;
    if (region == 0) return s.buf.ptr;
    const hb = if (s.hd) |*x| x.ptr else 0;
    if (region == 1) return if (s.hd != null) hb else error.BadKeptRange;
    const mi: usize = region - 2;
    if (mi < q.st.mla_layers) {
        if (s.src) |src| return src.st.cacheAt(mi);
        return s.buf.ptr + q.st.keptHead() + mi * q.st.below(s.pos) * row;
    }
    const h = q.hd orelse return error.BadKeptRange;
    const hi = mi - q.st.mla_layers;
    if (s.src) |src| return src.hd.?.cacheAt(hi);
    if (s.hd == null) return error.BadKeptRange;
    return hb + h.keptHead() + hi * h.below(if (s.hd_room > 0) s.hd_room else s.hd_pos) * row;
}

fn snapOf(e: *Engine, saved: api.prompt_cache.Saved) !struct { id: u64, at: u32, s: *Snap } {
    const k: *const engine.Kept = @ptrCast(@alignCast(saved));
    return .{ .id = k.id, .at = k.at, .s = e.snaps.getPtr(k.id) orelse return error.NoKeptState };
}

fn scratch(e: *Engine) !u64 {
    if (e.disk_buf == null) e.disk_buf = try cuda.DeviceBuffer.alloc(e.d, scratch_bytes);
    return e.disk_buf.?.ptr;
}

fn toHost(e: *const Engine, out: []u8, src: u64) !void {
    try e.d.check(e.d.api.cuMemcpyDtoH_v2(@ptrCast(out.ptr), src, out.len), "cuMemcpyDtoH");
}

fn toDevice(e: *const Engine, dst: u64, in: []const u8) !void {
    try e.d.check(e.d.api.cuMemcpyHtoD_v2(dst, @ptrCast(in.ptr), in.len), "cuMemcpyHtoD");
}

/// Pieces of rank 1's share one step moves: (region, local offset low, high, bytes) each, and where they sit here.
const Batch = struct {
    ids: [batch_max * 4]u32 = undefined,
    at: [batch_max]usize = undefined,
    n: usize = 0,
    bytes: usize = 0,

    fn fits(b: *const Batch, len: usize) bool {
        return b.n < batch_max and b.bytes + len <= scratch_bytes;
    }

    fn add(b: *Batch, s: Seg, at: usize) void {
        b.ids[4 * b.n ..][0..4].* = .{ s.region, @truncate(s.local), @intCast(s.local >> 32), @intCast(s.len) };
        b.at[b.n] = at;
        b.n += 1;
        b.bytes += s.len;
    }
};

/// Rank 1's pieces of state `id` into rank 0's `out`: one step, one NCCL transfer into the scratch, then to the host.
fn flushOut(e: *Engine, b: *Batch, id: u64, out: []u8) !void {
    if (b.n == 0) return;
    const g = e.g.tp orelse return error.NoGroup;
    const buf = try scratch(e);
    try e.sendOn(e.slot, .{ .kind = @backingInt(tp.Kind.disk_out), .a = id }, b.ids[0 .. 4 * b.n]);
    try g.recv(e.stream, buf, b.bytes, 1);
    try e.stream.synchronize();
    var o: usize = 0;
    for (0..b.n) |i| {
        const len: usize = b.ids[4 * i + 3];
        try toHost(e, out[b.at[i]..][0..len], buf + o);
        o += len;
    }
    b.* = .{};
}

/// Rank 0's gather: its own pieces straight to the host, rank 1's through the scratch, rows past a position zero.
fn gather(ptr: *anyopaque, saved: api.prompt_cache.Saved, ranges: []const rows.image.Range, out: []u8) anyerror!void {
    const e = eng(ptr);
    const k = try snapOf(e, saved);
    const sh = try shape(e);
    const w: Words = .{ .at = k.at, .hd_pos = @intCast(k.s.hd_pos), .world = @intCast(sh.world), .hd = @intFromBool(sh.hd), .head = @intCast(sh.head), .hd_head = @intCast(sh.hd_head), .mla = @intCast(sh.mla), .hd_mla = @intCast(sh.hd_mla) };
    try e.stream.synchronize();
    var segs: std.ArrayList(Seg) = .empty;
    defer segs.deinit(e.gpa);
    var b: Batch = .{};
    var o: usize = 0;
    for (ranges) |g| {
        segs.clearRetainingCapacity();
        try segments(sh, k.at, k.s.hd_pos, g.off, g.len, &segs, e.gpa);
        for (segs.items) |s| {
            const dst = out[o..][0..s.len];
            switch (s.kind) {
                .words => @memcpy(dst, std.mem.asBytes(&w)[s.local..][0..s.len]),
                .zero => @memset(dst, 0),
                .data => if (s.rank == 0) try toHost(e, dst, try regionAt(e, k.s, s.region) + s.local) else {
                    if (!b.fits(s.len)) try flushOut(e, &b, k.id, out);
                    b.add(s, o);
                },
            }
            o += s.len;
        }
    }
    try flushOut(e, &b, k.id, out);
}

/// Rank 1: the pieces rank 0 names of kept state `id`, back to back into the scratch, sent to rank 0.
pub fn serveOut(e: *Engine, id: u64, ids: []const u32) !void {
    const s = e.snaps.getPtr(id) orelse return error.NoKeptState;
    const g = e.g.tp orelse return error.NoGroup;
    const buf = try scratch(e);
    var o: usize = 0;
    var i: usize = 0;
    while (i + 4 <= ids.len) : (i += 4) {
        const len: usize = ids[i + 3];
        try e.copy(buf + o, try regionAt(e, s, ids[i]) + (@as(u64, ids[i + 2]) << 32 | ids[i + 1]), len);
        o += len;
    }
    try g.send(e.stream, buf, o, 0);
    try e.stream.synchronize();
}

/// Rank 1's pieces from the scratch's `b.bytes` (filled here) to rank 1; `hd_pos` + 1 names its MTP position (0: same).
fn flushIn(e: *Engine, b: *Batch, id: u64, hd_pos: u64) !void {
    if (b.n == 0 and hd_pos == 0) return;
    if (e.g.tp == null) {
        b.* = .{};
        return;
    }
    try e.sendOn(e.slot, .{ .kind = @backingInt(tp.Kind.disk_in), .a = id, .seed = hd_pos }, b.ids[0 .. 4 * b.n]);
    if (b.bytes > 0) try e.g.tp.?.send(e.stream, try scratch(e), b.bytes, 1);
    try e.stream.synchronize();
    b.* = .{};
}

/// Rank 0's scatter into a blank: its own pieces from the host, rank 1's (and every rank's rows without a split) sent.
fn scatter(ptr: *anyopaque, saved: api.prompt_cache.Saved, ranges: []const rows.image.Range, in: []const u8) anyerror!void {
    const e = eng(ptr);
    const k = try snapOf(e, saved);
    const sh = try shape(e);
    try e.stream.synchronize();
    var segs: std.ArrayList(Seg) = .empty;
    defer segs.deinit(e.gpa);
    var b: Batch = .{};
    var hd_pos: u64 = 0;
    var o: usize = 0;
    for (ranges) |g| {
        segs.clearRetainingCapacity();
        try segments(sh, k.at, k.s.hd_pos, g.off, g.len, &segs, e.gpa);
        for (segs.items) |s| {
            const src = in[o..][0..s.len];
            o += s.len;
            switch (s.kind) {
                .zero => {},
                .words => {
                    var w: Words = undefined;
                    @memcpy(std.mem.asBytes(&w)[s.local..][0..s.len], src);
                    if (s.local + s.len < words) return error.KeptWordsDiffer; // the words arrive whole
                    if (w.at != k.at or w.world != sh.world or w.head != sh.head or w.hd_head != sh.hd_head or w.mla != sh.mla or w.hd_mla != sh.hd_mla or w.hd != @intFromBool(sh.hd) or w.hd_pos > k.at + 1) return error.KeptWordsDiffer;
                    k.s.hd_pos = w.hd_pos;
                    hd_pos = @as(u64, w.hd_pos) + 1;
                },
                .data => {
                    if (s.rank == 0) try toDevice(e, try regionAt(e, k.s, s.region) + s.local, src);
                    if (sh.world > 1 and (s.rank != 0 or s.all)) {
                        if (!b.fits(s.len)) try flushIn(e, &b, k.id, 0);
                        try toDevice(e, try scratch(e) + b.bytes, src);
                        b.add(s, 0);
                    }
                },
            }
        }
    }
    try flushIn(e, &b, k.id, hd_pos);
}

/// Rank 1: rank 0's pieces of kept state `id` from the scratch into place, and its MTP position when named.
pub fn serveIn(e: *Engine, id: u64, hd_pos: u64, ids: []const u32) !void {
    const s = e.snaps.getPtr(id) orelse return error.NoKeptState;
    if (hd_pos > 0) s.hd_pos = @intCast(hd_pos - 1);
    var n: usize = 0;
    var i: usize = 0;
    while (i + 4 <= ids.len) : (i += 4) n += ids[i + 3];
    if (n == 0) return;
    const buf = try scratch(e);
    try e.g.tp.?.recv(e.stream, buf, n, 0);
    var o: usize = 0;
    i = 0;
    while (i + 4 <= ids.len) : (i += 4) {
        const len: usize = ids[i + 3];
        try e.copy(try regionAt(e, s, ids[i]) + (@as(u64, ids[i + 2]) << 32 | ids[i + 1]), buf + o, len);
        o += len;
    }
    try e.stream.synchronize();
}

fn blank(ptr: *anyopaque, at: u32) anyerror!api.prompt_cache.Saved {
    const e = eng(ptr);
    const k = try e.gpa.create(engine.Kept);
    errdefer e.gpa.destroy(k);
    k.* = .{ .id = try kept.blank(e, at), .at = at };
    return k;
}

/// A state whose slot rewrote its rows (an in-place one dropped) is gone: it stays off disk.
fn whole(ptr: *anyopaque, saved: api.prompt_cache.Saved) bool {
    const k: *const engine.Kept = @ptrCast(@alignCast(saved));
    return eng(ptr).holds(k.id);
}

/// Host bytes the store may stage at once: a ~250K state's image at the native window, half that past it (1M).
pub fn staging(e: *const Engine) u64 {
    return if (e.inplace) 2 << 30 else 4 << 30;
}

/// What the states also depend on: the rank count, the cache split, the MTP layer.
pub fn facts(e: *const Engine) [3]u64 {
    return .{ e.w.world, @intFromBool(e.w.kv_split), @intFromBool(e.mtp != null) };
}

/// A fixed prompt's state at a mark, its whole image hashed (a kernel or arithmetic change moves it), and its rate.
pub fn probe(a: Allocator, io: std.Io, backend: lanes.backend.Backend, e: *Engine) !struct { hash: u64, rate: f64 } {
    var toks: [4 * engine.prompt_rows + 47]u32 = undefined;
    for (&toks, 0..) |*x, i| x.* = @intCast(1000 + i * 7919 % 50000);
    const Hook = struct {
        e: *Engine,
        id: ?u64 = null,
        failed: ?anyerror = null,
        fn at(ptr: *anyopaque, _: *lanes.Stream, _: u32) void {
            const p: *@This() = @ptrCast(@alignCast(ptr));
            p.id = p.e.keepState() catch |err| {
                p.failed = err;
                return;
            };
        }
    };
    var hook: Hook = .{ .e = e };
    defer if (hook.id) |id| e.dropState(id);
    const mark: u32 = 4 * engine.prompt_rows;
    const marks = [_]u32{mark};
    var s = try lanes.Stream.init(a, .{ .id = "probe", .prompt = &toks, .max_new = 1, .reuse = .{ .marks = &marks, .hook = .{ .ptr = &hook, .at = Hook.at } } });
    defer s.deinit(a);
    const ns = pass: {
        defer backend.vtable.release(backend.ptr, &s);
        const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
        try backend.vtable.prefill(backend.ptr, &s);
        try e.stream.synchronize();
        break :pass std.Io.Clock.awake.now(io).toNanoseconds() - t0;
    };
    var k: engine.Kept = .{ .id = hook.id orelse return hook.failed orelse error.ProbeUnmarked, .at = mark };
    const sh = try shape(e);
    const bytes = try a.alloc(u8, @intCast(Layout.of(sh, mark).end));
    defer a.free(bytes);
    try gather(e, &k, &.{.{ .off = 0, .len = bytes.len }}, bytes);
    return .{ .hash = std.hash.Wyhash.hash(0x6c69, bytes), .rate = @as(f64, @floatFromInt(toks.len)) / (@as(f64, @floatFromInt(@max(ns, 1))) / 1e9) };
}

test "the image: rows by position from the rank holding their chunk, rows past the MTP position zero" {
    const sh: Shape = .{ .world = 2, .head = 100, .hd_head = 0, .mla = 2, .hd_mla = 1, .hd = true, .split = .{ .world = 2, .rank = 0 } };
    const at = 3 * chunk;
    const l = Layout.of(sh, at);
    try std.testing.expectEqual(@as(u64, words + 200), l.hd_rows);
    try std.testing.expectEqual(l.hd_rows + (at + 1) * row, l.rows);
    var segs: std.ArrayList(Seg) = .empty;
    defer segs.deinit(std.testing.allocator);
    try segments(sh, at, at - 1, l.rows + at * row, at * row, &segs, std.testing.allocator); // MLA layer 1, whole
    try std.testing.expectEqual(@as(usize, 3), segs.items.len);
    for (segs.items, 0..) |s, c| {
        try std.testing.expectEqual(@as(usize, chunk * row), s.len);
        try std.testing.expectEqual(c % 2, s.rank);
        try std.testing.expectEqual(@as(u32, 3), s.region);
        try std.testing.expectEqual(@as(u64, (c / 2) * chunk * row), s.local);
    }
    segs.clearRetainingCapacity();
    try segments(sh, at, at - 1, l.hd_rows + (at - 2) * row, 3 * row, &segs, std.testing.allocator); // MTP rows at the end
    try std.testing.expectEqual(@as(usize, 3), segs.items.len); // at - 2 held; at - 1 and at (the next chunk) zero
    try std.testing.expect(segs.items[0].kind == .data and segs.items[0].len == row and segs.items[1].kind == .zero and segs.items[2].kind == .zero);
    segs.clearRetainingCapacity();
    try segments(sh, at, at, words + 50, 100, &segs, std.testing.allocator); // across the two ranks' KDA parts
    try std.testing.expectEqual(@as(usize, 2), segs.items.len);
    try std.testing.expect(segs.items[0].rank == 0 and segs.items[0].local == 50 and segs.items[1].rank == 1 and segs.items[1].local == 0);
    const one: Shape = .{ .world = 1, .head = 8, .hd_head = 0, .mla = 1, .hd_mla = 0, .hd = false, .split = null };
    segs.clearRetainingCapacity();
    try segments(one, 10, 0, words + 8, 10 * row, &segs, std.testing.allocator);
    try std.testing.expect(segs.items.len == 1 and segs.items[0].all and segs.items[0].local == 0);
    try std.testing.expectError(error.BadKeptRange, segments(one, 10, 0, 0, Layout.of(one, 10).end + 1, &segs, std.testing.allocator));
}
