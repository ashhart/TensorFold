//! Ripple's forward on GB10s: rows of one or more streams' cache slots, one rank or two (TP=2, rank 0 drives).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const config = @import("config.zig");
const weights = @import("tp2_weights.zig");
const Kernels = @import("tp2_kernels.zig").Kernels;
const kept = @import("tp2_kept.zig");

const hd = config.head_dim;
const Tree = cuda.tree_attention.TreeAttention;
const max_depth = cuda.tree_attention.max_depth;
/// Rows a forward call runs at once: the EXL3 linears' widest call, and tree attention's longest path.
pub const max_rows = 128;
/// Rows a tree attention call takes when prompt chunks go to the prompt rows (decode rows only; sizes its scratch).
pub const tree_rows = 16;
/// Streams one decode call holds (each its own slot's rows).
pub const max_streams = 16;
/// Tap rows a slot's ring holds for a drafter (a prompt chunk and a window, read right after the call that wrote them).
pub const tap_rows = 256;
/// Query heads a key head in the padded prompt layout (head dim 128 prompt rows take 4 or 8 a block).
const pad_group = 12;

/// A layer's K then V caches, bf16 [slots, ring, kv heads, 128]: a ring is the window (or context) plus a chunk.
pub const Cache = struct { kv: cuda.DeviceBuffer, ring: usize };

/// Where a rank stands: rank of world (the server's --rank and --tp).
pub const Place = struct { rank: usize = 0, world: usize = 2 };

/// What has been validated: sm_121 (GB10) at exactly two ranks; every other SM or rank count is refused by name.
pub fn admit(sm: u32, world: usize, rank: usize) !void {
    if (sm != 121) return error.RicursiveRequiresSm121;
    if (world != 2) return error.RicursiveNeedsTwoRanks;
    if (rank >= world) return error.RicursiveRankPastWorld;
}

test "ricursive two-rank admission: sm_121 at exactly two ranks" {
    try admit(121, 2, 0);
    try admit(121, 2, 1);
    for ([_]u32{ 80, 86, 89, 90, 100, 120 }) |sm| try std.testing.expectError(error.RicursiveRequiresSm121, admit(sm, 2, 0));
    for ([_]usize{ 0, 1, 3, 4, 8 }) |w| try std.testing.expectError(error.RicursiveNeedsTwoRanks, admit(121, w, 0));
    try std.testing.expectError(error.RicursiveRankPastWorld, admit(121, 2, 2));
}

pub const Options = struct {
    context: usize = 4096,
    /// Cache slots (streams); rank 0 tells the followers its slots and context, which must equal theirs.
    slots: usize = 1,
    truncated: bool = false,
    place: Place = .{},
    /// Full layers' prompt chunks on the window prompt rows (q padded); false: tree rows (prompt == decode bits).
    full_prompt: bool = true,
    /// Two ranks' small gathers by RoCE writes (else NCCL carries every gather; the same bits).
    roce: bool = false,
    /// Decode rows (one row, its logits) replayed as CUDA graphs keyed by 64-group key buckets.
    graphs: bool = false,
    /// --drafter on a follower: the drafter it opens its share of when rank 0 opens one.
    drafter: ?[]const u8 = null,
};

/// Plan ops: stop, decode rows, a prompt chunk, kept states (save, restore, drop, row copy), hello, a drafter's calls.
pub const Op = enum(i32) { stop = 0, rows = 1, prompt = 2, save = 3, restore = 4, drop = 5, copy = 6, hello = 7, taps = 8, draft_open = 9, draft_absorb = 10, draft_hold = 11 };

/// Rank 0's plan word before each call: op, rows, p0, head rows, slot, a, b; rows' ids/positions/slots; streams.
pub const Header = extern struct {
    op: i32 = 0,
    n: i32 = 0,
    p0: i32 = 0,
    head: i32 = 0,
    slot: i32 = 0,
    ns: i32 = 0,
    a: i32 = 0,
    b: i32 = 0,
    ids: [max_rows]i32 = @splat(0),
    pos: [max_rows]i32 = @splat(0),
    slots: [max_rows]i32 = @splat(0),
    streams: [max_streams][4]i32 = @splat(@splat(0)),
};

/// One stream's rows in a decode call: its slot, its cache length before them, its tokens.
pub const Entry = struct { slot: usize, p0: usize, tokens: []const u32 };

/// Staged inputs: ids, positions, slots, row streams (int32 [R] each), streams [S, 4], then offs (int64 [S, 2]).
const stage_i32 = 4 * max_rows + 4 * max_streams;
const stage_bytes = stage_i32 * 4 + max_streams * 16;

/// Device scratch: staged inputs, activations, gathered partials, tree and prompt-path buffers.
const Bufs = struct { st: u64, x: u64, seed: u64, y: u64, vin: u64, q: u64, k: u64, v: u64, qo: u64, ko: u64, vo: u64, g: u64, gated: u64, a: u64, parts: u64, gu: u64, act: u64, logits: u64, hdr: u64, paths: u64, depths: u64, inv: u64, tree: u64, qpad: u64 = 0, opad: u64 = 0 };

pub const Model = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    d: *const cuda.Driver,
    stream: cuda.Stream,
    c: config.Config,
    o: Options,
    w: weights.Weights,
    dense: cuda.exl3.Dense,
    norms: cuda.norms.Norms,
    qk: cuda.qk_rows.QkRows,
    attn: cuda.window_attention.WindowAttention,
    tree: Tree,
    k: Kernels,
    group: cuda.tp_group.Group = undefined,
    caches: []Cache = &.{},
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,
    // a rank's heads, key heads, FFN width and vocab share
    hq: usize,
    hk: usize,
    ff: usize,
    vr: usize,
    context: usize,
    slots: usize,
    b: Bufs = undefined,
    graphs: std.AutoHashMapUnmanaged(usize, cuda.graph.Exec) = .empty,
    warm: std.AutoHashMapUnmanaged(usize, void) = .empty,
    states: kept.States = .{},
    // a drafter's taps (rank 0): the layers' residuals after them, bf16 [slots, tap_rows, taps, hidden]
    taps: [8]u32 = @splat(0),
    n_taps: usize = 0,
    tap_ring: u64 = 0,
    // rank 1: a drafter's share, run on rank 0's plan words (its follow)
    draft_hook: ?*const fn (*Model, *const Header) anyerror!void = null,
    draft_ptr: ?*anyopaque = null,

    /// Loads `dir` (a rank's share at two ranks) on the context's device; sm_121 only.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, d: *const cuda.Driver, dir: []const u8, o: Options) !*Model {
        try admit(try ctx.capability(), o.place.world, o.place.rank);
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
        defer gpa.free(bytes);
        const c = try config.parse(gpa, bytes);
        if (!o.truncated) _ = try c.qualification(); // a truncation only through the parity tool's --truncated
        if (o.context == 0 or o.context > std.math.maxInt(i32) - max_rows) return error.RicursiveContext;
        if (o.slots == 0 or o.slots > max_streams) return error.RicursiveSlots;
        const world = o.place.world;
        if (c.heads % world != 0 or c.kv_heads % world != 0) return error.RicursiveCut;
        var ck = try core.Checkpoint.openModel(gpa, io, dir);
        defer ck.close();
        const m = try gpa.create(Model);
        errdefer gpa.destroy(m);
        m.* = .{ .gpa = gpa, .io = io, .d = d, .stream = try cuda.Stream.init(d, false), .c = c, .o = o, .w = undefined, .dense = undefined, .norms = undefined, .qk = undefined, .attn = undefined, .tree = undefined, .k = undefined, .hq = c.heads / world, .hk = c.kv_heads / world, .ff = c.intermediate / world, .vr = c.vocab / world, .context = o.context, .slots = o.slots };
        m.group = try cuda.tp_group.Group.open(d, o.place.rank, world);
        if (o.roce) try m.group.openRoce(gpa, io, d, m.stream, max_rows * c.hidden * 4);
        m.w = try weights.load(gpa, d, &ck, c, .{ .rank = o.place.rank, .world = world });
        m.dense = try cuda.exl3.Dense.load(d);
        m.norms = try cuda.norms.Norms.load(d);
        m.qk = try cuda.qk_rows.QkRows.load(d);
        m.attn = try cuda.window_attention.WindowAttention.load(d);
        m.tree = try Tree.load(d);
        m.tree.sms = @intCast(try ctx.attribute(.multiprocessor_count));
        m.k = try Kernels.load(d);
        try m.alloc(c);
        try m.hello();
        return m;
    }

    /// The device bytes `slots` slots' caches take at `context` (a rank's key heads `hk`).
    pub fn cacheBytes(c: config.Config, hk: usize, context: usize, slots: usize) u64 {
        var total: u64 = 0;
        for (0..c.layers) |i| total += 2 * slots * (if (c.global[i]) context + max_rows else c.window + max_rows) * hk * hd * 2;
        return total;
    }

    fn buf(m: *Model, n: usize) !u64 {
        try m.bufs.ensureUnusedCapacity(m.gpa, 1);
        const b = try cuda.DeviceBuffer.alloc(m.d, @max(n, 16));
        m.bufs.appendAssumeCapacity(b);
        return b.ptr;
    }

    pub fn up(m: *const Model, p: u64, bytes: []const u8) !void {
        return (cuda.DeviceBuffer{ .d = m.d, .ptr = p, .len = bytes.len }).upload(0, bytes);
    }

    fn down(m: *const Model, p: u64, out: []u8) !void {
        return (cuda.DeviceBuffer{ .d = m.d, .ptr = p, .len = out.len }).download(0, out);
    }

    fn alloc(m: *Model, c: config.Config) !void {
        const R = max_rows;
        const h = c.hidden;
        const qw = m.hq * hd;
        const kw = m.hk * hd;
        const world = m.o.place.world;
        m.caches = try m.gpa.alloc(Cache, c.layers);
        for (m.caches) |*ca| ca.* = .{ .kv = .{ .d = m.d, .ptr = 0, .len = 0 }, .ring = 0 };
        for (m.caches, 0..) |*ca, i| {
            const ring = if (c.global[i]) m.context + R else c.window + R;
            ca.* = .{ .kv = try cuda.DeviceBuffer.alloc(m.d, 2 * m.slots * ring * kw * 2), .ring = ring };
        }
        var inv: [hd / 2]f32 = undefined;
        c.inverseFrequencies(&inv);
        m.b = .{
            .st = try m.buf(stage_bytes),
            .x = try m.buf(R * h * 4),
            .seed = try m.buf(R * h * 2),
            .y = try m.buf(R * h * 2),
            .vin = try m.buf(R * h * 2),
            .q = try m.buf(R * qw * 2),
            .k = try m.buf(R * kw * 2),
            .v = try m.buf(R * kw * 2),
            .qo = try m.buf(R * qw * 2),
            .ko = try m.buf(R * kw * 2),
            .vo = try m.buf(R * kw * 2),
            .g = try m.buf(R * qw * 2),
            .gated = try m.buf(R * qw * 2),
            .a = try m.buf(R * h * 4),
            .parts = try m.buf(world * R * h * 4),
            .gu = try m.buf(2 * R * m.ff * 2),
            .act = try m.buf(R * m.ff * 2),
            .logits = try m.buf(world * R * m.vr * 4),
            .hdr = try m.buf(world * @sizeOf(Header)),
            .paths = try m.buf(R * max_depth * 4),
            .depths = try m.buf(R * 4),
            .inv = try m.buf(inv.len * 4),
            .tree = try m.buf(Tree.scratchBytes(if (m.o.full_prompt) tree_rows else R, m.hq, m.hk, std.mem.alignForward(usize, Tree.groups(m.context + R), 64), hd)),
        };
        try m.up(m.b.inv, std.mem.sliceAsBytes(&inv));
        if (m.o.full_prompt) { // the pad heads stay zero: never written
            const pb = R * m.hk * pad_group * hd * 2;
            const z = try m.gpa.alloc(u8, pb);
            defer m.gpa.free(z);
            @memset(z, 0);
            m.b.qpad = try m.buf(pb);
            try m.up(m.b.qpad, z);
            m.b.opad = try m.buf(pb);
        }
    }

    /// A drafter's taps: from now on each call keeps the residual after these layers, per slot and position.
    pub fn setTaps(m: *Model, taps: []const u32) !void {
        if (taps.len == 0 or taps.len > m.taps.len or m.tap_ring != 0) return error.RicursiveTaps;
        for (taps) |t| if (t >= m.c.layers) return error.RicursiveTaps;
        if (m.rank() == 0) {
            var h: Header = .{ .op = @backingInt(Op.taps), .n = @intCast(taps.len) };
            for (taps, 0..) |t, i| h.ids[i] = @intCast(t);
            try m.plan(&h);
        }
        @memcpy(m.taps[0..taps.len], taps);
        m.n_taps = taps.len;
        m.tap_ring = try m.buf(m.slots * tap_rows * taps.len * m.c.hidden * 2);
        var it = m.graphs.valueIterator(); // graphs captured before hold no tap writes: capture them again
        while (it.next()) |e| e.deinit();
        m.graphs.clearRetainingCapacity();
        m.warm.clearRetainingCapacity();
    }

    pub fn deinit(m: *Model) void {
        m.stream.synchronize() catch {};
        m.states.deinit(m.gpa);
        var it = m.graphs.valueIterator();
        while (it.next()) |e| e.deinit();
        m.graphs.deinit(m.gpa);
        m.warm.deinit(m.gpa);
        for (m.bufs.items) |*b| b.free();
        m.bufs.deinit(m.gpa);
        for (m.caches) |*ca| if (ca.kv.ptr != 0) ca.kv.free();
        m.gpa.free(m.caches);
        m.k.unload();
        m.tree.unload();
        m.attn.unload();
        m.qk.unload();
        m.norms.unload();
        m.dense.unload();
        m.w.deinit();
        m.group.close();
        m.stream.deinit();
        m.gpa.destroy(m);
    }

    pub fn rank(m: *const Model) usize {
        return m.o.place.rank;
    }

    /// Slot 0's rows `ids` at p0.. in calls of <= `chunk` rows (> 1: prompt chunks) into host fp32 logits `out`.
    pub fn forward(m: *Model, ids: []const u32, p0: usize, chunk: usize, out: []f32, capture: ?[]f32) !void {
        if (chunk == 0 or chunk > max_rows or p0 + ids.len > m.context) return error.RicursiveRows;
        const last = out.len == m.c.vocab and ids.len > 1;
        if (!last and out.len != ids.len * m.c.vocab) return error.RicursiveRows;
        var at: usize = 0;
        while (at < ids.len) {
            const n = @min(chunk, ids.len - at);
            var h: Header = .{ .op = @backingInt(if (chunk > 1) Op.prompt else Op.rows), .n = @intCast(n), .p0 = @intCast(p0 + at) };
            h.head = if (!last) h.n else if (at + n == ids.len) 1 else 0;
            fill(&h, &.{.{ .slot = 0, .p0 = p0 + at, .tokens = ids[at..][0..n] }});
            try m.plan(&h);
            const o = if (last) out else out[at * m.c.vocab ..][0 .. n * m.c.vocab];
            try m.rows(&h, if (h.head > 0) o else null, capture, at, ids.len);
            at += n;
        }
    }

    /// A header's rows and streams from `entries` (each stream's rows in order).
    fn fill(h: *Header, entries: []const Entry) void {
        var r: usize = 0;
        for (entries, 0..) |e, si| {
            h.streams[si] = .{ @intCast(r), @intCast(e.tokens.len), @intCast(e.p0), @intCast(e.slot) };
            for (e.tokens, 0..) |t, i| {
                h.ids[r] = @intCast(t);
                h.pos[r] = @intCast(e.p0 + i);
                h.slots[r] = @intCast(e.slot);
                r += 1;
            }
        }
        h.ns = @intCast(entries.len);
        h.slot = if (entries.len > 0) @intCast(entries[0].slot) else 0;
    }

    /// One prompt chunk of `slot` at p0 (device logits of its last row kept when `last`; logitsTo reads them).
    pub fn prompt(m: *Model, slot: usize, tokens: []const u32, p0: usize, last: bool) !void {
        if (tokens.len == 0 or tokens.len > max_rows or slot >= m.slots or p0 + tokens.len > m.context) return error.RicursiveRows;
        var h: Header = .{ .op = @backingInt(Op.prompt), .n = @intCast(tokens.len), .p0 = @intCast(p0), .head = @intFromBool(last) };
        fill(&h, &.{.{ .slot = slot, .p0 = p0, .tokens = tokens }});
        try m.plan(&h);
        try m.rows(&h, null, null, 0, 0);
    }

    /// Decode rows of several streams in one call (every row's logits; logitsTo reads them).
    pub fn decode(m: *Model, entries: []const Entry) !void {
        if (entries.len == 0 or entries.len > max_streams) return error.RicursiveRows;
        var n: usize = 0;
        for (entries) |e| {
            if (e.slot >= m.slots or e.tokens.len == 0 or e.p0 + e.tokens.len > m.context + tree_rows) return error.RicursiveRows;
            n += e.tokens.len;
        }
        if (n > tree_rows) return error.RicursiveRows;
        var h: Header = .{ .op = @backingInt(Op.rows), .n = @intCast(n), .p0 = @intCast(entries[0].p0), .head = @intCast(n) };
        fill(&h, entries);
        try m.plan(&h);
        try m.rows(&h, null, null, 0, 0);
    }

    /// The last call's `rows` head rows as fp32 [rows, vocab] at `dst` (device), the ranks' vocab halves interleaved.
    pub fn logitsTo(m: *Model, dst: u64, rows_: usize) !void {
        const world = m.o.place.world;
        for (0..world) |r| {
            const cp: cuda.abi.Memcpy2D = .{ .src = m.b.logits + r * rows_ * m.vr * 4, .src_pitch = m.vr * 4, .dst = dst + r * m.vr * 4, .dst_pitch = m.c.vocab * 4, .width = m.vr * 4, .height = rows_ };
            try m.d.check(m.d.api.cuMemcpy2DAsync_v2(&cp, m.stream.handle), "cuMemcpy2DAsync");
        }
    }

    /// A kept-state op (save, restore, drop, copy) on every rank.
    pub fn keptOp(m: *Model, op: Op, slot: usize, a: i32, n: usize) !void {
        var h: Header = .{ .op = @backingInt(op), .slot = @intCast(slot), .a = a, .p0 = @intCast(n) };
        try m.plan(&h);
        try kept.run(m, &h);
    }

    /// Rank 0's header to every rank (rank 0 uploads it; the others read rank 0's).
    fn exchange(m: *Model, h: *Header) !void {
        const g = &m.group;
        try m.stream.synchronize();
        if (m.rank() == 0) try m.up(m.b.hdr, std.mem.asBytes(h));
        // by NCCL, which waits as long as rank 0 idles (RoCE writes give up after their spin limit)
        try g.lib.check(g.lib.api.ncclAllGather(m.b.hdr + m.rank() * @sizeOf(Header), m.b.hdr, @sizeOf(Header), .u8, g.comm, m.stream.handle), "ncclAllGather");
        try m.stream.synchronize();
        try g.check();
        if (m.rank() != 0) try m.down(m.b.hdr, std.mem.asBytes(h));
    }

    /// Rank 0: the next call's plan word to every rank (rank 1 runs the same call), before it runs.
    pub fn plan(m: *Model, h: *Header) !void {
        try m.exchange(h);
    }

    /// The ranks agree on context and slots before any call.
    fn hello(m: *Model) !void {
        var h: Header = .{ .op = @backingInt(Op.hello), .n = @intCast(m.slots), .p0 = @intCast(m.context) };
        try m.exchange(&h);
        if (h.op != @backingInt(Op.hello) or h.n != m.slots or h.p0 != m.context) {
            std.log.err("ricursive rank {d}: rank 0 serves {d} slots at {d}, this rank {d} at {d} (--parallel, --context)", .{ m.rank(), h.n, h.p0, m.slots, m.context });
            return error.RicursiveRanksDisagree;
        }
    }

    /// Rank 0: rank 1's follow loop returns.
    pub fn stop(m: *Model) !void {
        var h: Header = .{};
        try m.plan(&h);
        try m.stream.synchronize();
    }

    /// Rank 1: rank 0's calls, until its stop.
    pub fn follow(m: *Model) !void {
        while (true) {
            var h: Header = .{};
            try m.exchange(&h);
            const op = std.enums.fromInt(Op, h.op) orelse return error.RicursiveRows;
            switch (op) {
                .stop => return,
                .rows, .prompt => {
                    if (h.n <= 0 or h.n > max_rows or h.p0 < 0 or h.head < 0 or h.head > h.n or h.ns <= 0 or h.ns > max_streams) return error.RicursiveRows;
                    try m.rows(&h, null, null, 0, 0);
                },
                .save, .restore, .drop, .copy => try kept.run(m, &h),
                .hello => return error.RicursiveRows,
                .taps => {
                    var t: [8]u32 = undefined;
                    for (t[0..@intCast(@min(@max(h.n, 0), 8))], 0..) |*x, i| x.* = @intCast(h.ids[i]);
                    try m.setTaps(t[0..@intCast(@min(@max(h.n, 0), 8))]);
                },
                .draft_open, .draft_absorb, .draft_hold => try (m.draft_hook orelse return error.RicursiveNoDrafter)(m, &h),
            }
        }
    }

    /// `bytes` a rank at parts + rank * bytes gathered from every rank, then summed in rank order into `into`.
    pub fn sumRanks(m: *Model, into: u64, bytes: usize) !void {
        const g = &m.group;
        try g.allGather(m.stream, m.b.parts + m.rank() * bytes, m.b.parts, bytes);
        try m.k.rankSum(m.stream, m.b.parts, into, bytes / 4, g.world);
    }

    /// Stages a call's inputs: ids, positions, slots, the streams' tree entries and key offsets, their chain paths.
    fn stage(m: *Model, hw: *const Header) !usize {
        const n: usize = @intCast(hw.n);
        const ns: usize = @intCast(hw.ns);
        var host: [stage_i32]i32 = @splat(0);
        @memcpy(host[0..max_rows], &hw.ids);
        @memcpy(host[max_rows..][0..max_rows], &hw.pos);
        @memcpy(host[2 * max_rows ..][0..max_rows], &hw.slots);
        var top: usize = 0; // the furthest key a row reads: the tree groups to walk
        var offs: [max_streams][2]i64 = @splat(.{ 0, 0 });
        var paths: [tree_rows * max_depth]i32 = @splat(0);
        var depths: [max_rows]i32 = @splat(0);
        const kw = m.hk * hd;
        for (0..ns) |si| {
            const e = hw.streams[si];
            host[4 * max_rows + 4 * si ..][0..4].* = .{ e[0], e[1], e[2], 0 };
            top = @max(top, @as(usize, @intCast(e[2] + e[1])));
            const slot: usize = @intCast(e[3]);
            const ring = m.context + max_rows; // a full layer's ring: its slot's K, then V after every slot's K
            offs[si] = .{ @intCast(slot * ring * kw), @intCast((m.slots + slot) * ring * kw) };
            const r0: usize = @intCast(e[0]);
            for (0..@intCast(e[1])) |i| {
                host[3 * max_rows + r0 + i] = @intCast(si);
                depths[r0 + i] = @intCast(i + 1);
                if (r0 + i < tree_rows) for (0..max_depth) |j| { // a chain: row i's path is its stream's rows 0..i
                    paths[(r0 + i) * max_depth + j] = @intCast(r0 + @min(i, j));
                };
            }
        }
        try m.stream.synchronize();
        try m.up(m.b.st, std.mem.sliceAsBytes(&host));
        try m.up(m.b.st + stage_i32 * 4, std.mem.sliceAsBytes(offs[0..ns]));
        const pr: usize = @min(n, tree_rows);
        try m.up(m.b.paths, std.mem.sliceAsBytes(paths[0 .. pr * max_depth]));
        if (n > tree_rows) { // a prompt chunk on tree rows (one stream from row 0): its longer chain
            const big = try m.gpa.alloc(i32, n * max_depth);
            defer m.gpa.free(big);
            for (0..n) |i| for (0..max_depth) |j| {
                big[i * max_depth + j] = @intCast(@min(i, j));
            };
            try m.up(m.b.paths, std.mem.sliceAsBytes(big));
        }
        try m.up(m.b.depths, std.mem.sliceAsBytes(depths[0..n]));
        return top;
    }

    fn rows(m: *Model, hw: *const Header, out: ?[]f32, capture: ?[]f32, at: usize, total: usize) !void {
        const c = m.c;
        const s = m.stream;
        const b = &m.b;
        const n: usize = @intCast(hw.n);
        const world = m.o.place.world;
        const top = try m.stage(hw);
        const hr: usize = @intCast(hw.head);
        const one = hw.op == @backingInt(Op.rows) and n == 1 and hr == 1 and capture == null and m.o.graphs;
        // a graphed decode row keys by its tree groups rounded up to 64 (32K keys): the kernels skip the groups past it
        const ng = if (one) std.mem.alignForward(usize, Tree.groups(top), 64) else Tree.groups(top);
        if (!one) try m.body(hw, ng, capture, at, total) else if (m.graphs.get(ng)) |e| try e.launchOn(s) else if (m.warm.contains(ng)) {
            try cuda.graph.beginCapture(s, .thread_local);
            m.body(hw, ng, null, 0, 0) catch |err| {
                var g = cuda.graph.endCapture(s) catch return err;
                g.deinit();
                return err;
            };
            var g = try cuda.graph.endCapture(s);
            defer g.deinit();
            const e = try g.instantiate();
            try m.graphs.put(m.gpa, ng, e);
            try e.launchOn(s);
        } else {
            try m.warm.put(m.gpa, ng, {});
            try m.body(hw, ng, null, 0, 0); // a bucket's first row eagerly: scratch grows here, never in a capture
        }
        if (hr == 0) return;
        const o = out orelse return;
        try s.synchronize();
        try m.group.check();
        const all = try m.gpa.alloc(f32, world * hr * m.vr);
        defer m.gpa.free(all);
        try m.down(b.logits, std.mem.sliceAsBytes(all));
        for (0..world) |r| for (0..hr) |i| @memcpy(o[i * c.vocab + r * m.vr ..][0..m.vr], all[(r * hr + i) * m.vr ..][0..m.vr]);
    }

    /// One call's kernels and gathers on the stream (no host work: a decode row captures it as a graph).
    fn body(m: *Model, hw: *const Header, ng: usize, capture: ?[]f32, at: usize, total: usize) !void {
        const c = m.c;
        const s = m.stream;
        const b = &m.b;
        const n: usize = @intCast(hw.n);
        const p0: usize = @intCast(hw.p0);
        const h = c.hidden;
        const qw = m.hq * hd;
        const kw = m.hk * hd;
        const pos = b.st + max_rows * 4;
        const slots = b.st + 2 * max_rows * 4;
        const row_stream = b.st + 3 * max_rows * 4;
        const streams = b.st + 4 * max_rows * 4;
        const offs = b.st + stage_i32 * 4;
        const slot: usize = @intCast(hw.slot);
        try m.k.embed(s, b.st, m.w.embed, b.x, b.seed, n, h, c.embed_scale);
        for (m.w.layers, m.caches, 0..) |*l, *ca, li| {
            const kc = ca.kv.ptr;
            const vc = ca.kv.ptr + m.slots * ca.ring * kw * 2;
            try m.norms.rms(s, b.x, .f32, l.input_norm, b.y, .f16, n, h, c.eps);
            try m.dense.prompt(s, l.q, b.y, .f16, n, b.q, .bf16);
            try m.dense.prompt(s, l.k, b.y, .f16, n, b.k, .bf16);
            try m.dense.prompt(s, l.g, b.y, .f16, n, b.g, .f16);
            try m.k.lerp(s, b.seed, b.y, b.vin, n * h, l.blend); // only V sees the value residual
            try m.dense.prompt(s, l.v, b.vin, .f16, n, b.v, .bf16);
            const src: cuda.qk_rows.Src = .{ .q = b.q, .q_ld = @intCast(qw), .q_hs = hd, .k = b.k, .k_ld = @intCast(kw), .k_hs = hd, .v = b.v, .v_ld = @intCast(kw), .v_hs = hd };
            const shape: cuda.qk_rows.Shape = if (l.global) .{ .norms = l.qk_norm, .eps = c.eps } else .{ .norms = l.qk_norm, .inv = b.inv, .rot = hd, .eps = c.eps };
            try m.qk.rowsFrom(s, src, .{ .h = m.hq, .hk = m.hk, .dim = hd }, shape, pos, slots, .{ .q = b.qo, .k = b.ko, .v = b.vo, .kc = kc, .vc = vc, .len = ca.ring }, n);
            if (!l.global) {
                const win: cuda.window_attention.Window = .{ .h = m.hq, .hk = m.hk, .d = hd, .win = c.window, .ring = ca.ring, .scale = c.scale() };
                try m.attn.decode(s, b.qo, kc, vc, b.q, pos, slots, win, b.tree, n);
            } else if (hw.op == @backingInt(Op.prompt) and m.o.full_prompt) { // prompt rows: q padded to 12 heads a key head
                const g = m.hq / m.hk;
                const so = slot * ca.ring * kw * 2;
                try m.k.heads(s, b.qo, b.qpad, n * m.hk, g, g, pad_group);
                const win: cuda.window_attention.Window = .{ .h = m.hk * pad_group, .hk = m.hk, .d = hd, .win = 0, .ring = ca.ring, .scale = c.scale() };
                try m.attn.prompt(s, b.qpad, kc + so, vc + so, b.opad, p0, n, win);
                try m.k.heads(s, b.opad, b.q, n * m.hk, g, pad_group, g);
            } else { // every key before the row: the cache's committed keys in 512-key groups, then the call's own rows
                if (m.o.full_prompt and n > tree_rows) return error.RicursiveRows; // decode rows only, scratch for 16
                const rws: cuda.tree_attention.Rows = .{ .row_stream = row_stream, .streams = streams, .offs = offs, .origin = kc, .paths = b.paths, .depths = b.depths };
                try m.tree.run(s, b.qo, b.ko, b.vo, rws, Tree.scratch(b.tree, n, m.hq, m.hk, ng, hd), b.q, n, m.hq, m.hk, hd, ng, c.scale());
            }
            try m.k.gate(s, b.q, b.g, b.gated, n * qw);
            try m.dense.prompt(s, l.o, b.gated, .f16, n, b.parts + m.rank() * n * h * 4, .f32);
            try m.sumRanks(b.a, n * h * 4);
            try m.k.normAdd(s, b.a, l.post_attn, b.x, n, h, c.eps);
            try m.norms.rms(s, b.x, .f32, l.pre_ffn, b.y, .f16, n, h, c.eps);
            const gu_up = b.gu + n * m.ff * 2;
            try m.dense.prompt(s, l.gate, b.y, .f16, n, b.gu, .f16);
            try m.dense.prompt(s, l.up, b.y, .f16, n, gu_up, .f16);
            try m.k.swiglu(s, b.gu, gu_up, b.act, n * m.ff, c.clamp);
            try m.dense.prompt(s, l.down, b.act, .f16, n, b.parts + m.rank() * n * h * 4, .f32);
            try m.sumRanks(b.a, n * h * 4);
            try m.k.normAdd(s, b.a, l.post_ffn, b.x, n, h, c.eps);
            for (m.taps[0..m.n_taps], 0..) |t, ti| if (t == li) try m.k.taps(s, b.x, pos, slots, m.tap_ring, tap_rows, m.n_taps, ti, h, n);
            if (capture) |cap| {
                try s.synchronize();
                try m.down(b.x, std.mem.sliceAsBytes(cap[(li * total + at) * h ..][0 .. n * h]));
            }
        }
        const hr: usize = @intCast(hw.head);
        if (hr == 0) return;
        try m.norms.rms(s, b.x + (n - hr) * h * 4, .f32, m.w.final_norm, b.y, .f16, hr, h, c.eps);
        const lg = b.logits + m.rank() * hr * m.vr * 4;
        try m.dense.prompt(s, m.w.head, b.y, .f16, hr, lg, .f32);
        try m.group.allGather(s, lg, b.logits, hr * m.vr * 4);
    }
};
