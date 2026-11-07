//! Ripple's DFlash2 drafter on CUDA: dflash2.zig's op order over the target's taps, streams batched in 16-row blocks.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const spec = @import("dflash2_spec.zig");
const dw = @import("cuda_dflash2_weights.zig");
const dk = @import("cuda_dflash2_kernels.zig");
const da = @import("cuda_dflash2_attn.zig");
const sel = @import("cuda_dflash2_select.zig");
const fwd = @import("cuda_model.zig");
const be = @import("cuda_backend.zig");
const cfg = @import("config.zig");

const dr = lanes.drafter;
const Stream = lanes.Stream;
const Cost = lanes.config.Cost;
const DeviceBuffer = cuda.DeviceBuffer;
const Weight = cuda.qmmf.Weight;

pub const Config = spec.Config;
const dcfg = @import("cuda_dflash2_config.zig"); // the qualified config and the kernels' gate
pub const gate = dcfg.gate;
pub const readConfig = dcfg.readConfig;

/// Context rows one absorb pass projects (dflash2.zig's absorb_rows: a prompt chunk's).
pub const absorb_rows = fwd.prompt_rows;
/// Rows a stream's absorb may read from the tap ring: a chunk's (the ring's other 16 hold a verify window).
const held_rows = fwd.prompt_rows;
/// Row runs an absorb pass takes (a stream's rows split at the tap ring's wrap).
const max_pieces = 64;
/// The target's verify windows the sweep times: 1 to 16 rows.
const windows = be.max_window;
const head_dim = dk.head_dim;

/// Device bytes one stream's rings take: every layer's K and V, ring(l) rows of kv_heads x 128 bf16 each.
pub fn laneBytes(c: *const Config) usize {
    var n: usize = 0;
    for (0..c.layers) |l| n += @as(usize, c.ring(l)) * c.kv_heads * head_dim * 2 * 2;
    return n;
}

/// Device bytes a stream adds to the target's: its drafter rings and its slot's tap ring (A's, 5 taps).
pub fn streamBytes(c: *const Config, target: cfg.Config) usize {
    return laneBytes(c) + fwd.tapBytes(target, spec.max_taps, 1);
}

/// The scratch's regions: a batch of `streams` blocks, an absorb pass, the segment tables.
const Bufs = struct {
    ids: u64 = 0, // u32 [R]: the blocks' tokens, then the sweep's draws
    resid: u64 = 0, // bf16 [R, H]
    x: u64 = 0,
    xc: u64 = 0,
    y: u64 = 0,
    yc: u64 = 0,
    dyn: u64 = 0, // bf16 [R, dyn]: a conv's coefficients
    qkv: u64 = 0, // bf16 [R, q + 2 kv]
    q: u64 = 0, // bf16 [R, heads, 128]: prep's queries
    att: u64 = 0, // bf16 [R, heads, 128]: the attention
    gu: u64 = 0, // bf16 [R, 2 ffn]
    act: u64 = 0, // bf16 [R, ffn]
    logits: u64 = 0, // bf16 [R, vocab]
    pm: u64 = 0, // the attention's partials (D's Scratch)
    pl: u64 = 0,
    po: u64 = 0,
    tv: u64 = 0, // the top-32's chunk bests
    ti: u64 = 0,
    out: u64 = 0, // f32 [R, 32] values, u32 [R, 32] ids, bf16 [R, rank] hidden: one download
    part: u64 = 0, // the lane matmul's slice partials
    inv: u64 = 0, // f32 [64]: RoPE's inverse frequencies
    psegs: u64 = 0, // C's Segment [layers][streams]: a block's prep
    asegs: u64 = 0, // D's Segment [layers][streams]: a block's attention
    fcx: u64 = 0, // bf16 [A, H]: fc of the tap rows
    hctx: u64 = 0, // bf16 [A, H]: normed
    kv: u64 = 0, // bf16 [A, 2 kv]
    csegs: u64 = 0, // C's Segment [layers][max_pieces]: an absorb's prep
};

/// The widest span a layer attends over (the global layer's when there is one).
fn widest(c: *const Config) usize {
    var s: usize = 0;
    for (0..c.layers) |l| s = @max(s, c.span(l));
    return s;
}

/// Slice partials the lane matmul needs at a block batch's rows for every projection it runs (0 when none reduces).
fn partBytes(c: *const Config, vocab: usize, clusters: bool, streams: usize) usize {
    const h: usize = c.hidden;
    const qw = @as(usize, c.heads) * head_dim;
    const kvw = @as(usize, c.kv_heads) * head_dim;
    const shapes = [_][2]usize{ .{ qw + 2 * kvw, h }, .{ h, qw }, .{ 2 * @as(usize, c.ffn), h }, .{ h, c.ffn }, .{ c.dyn(), h }, .{ vocab, h }, .{ c.rank, h } };
    var most: usize = 0;
    for (1..streams + 1) |s| for (shapes) |nk| {
        const rows = s * c.block;
        const p = cuda.qmmf.planFor(rows, cuda.modelopt.bf16Weight(0, nk[0], nk[1]), clusters);
        if (p.reduce) most = @max(most, @as(usize, p.sk) * rows * nk[0] * 4);
    };
    return most;
}

/// The scratch for `streams` blocks at `base` (0: only its bytes), every region on a 256-byte boundary.
fn place(c: *const Config, vocab: usize, clusters: bool, streams: usize, base: u64, b: *Bufs) usize {
    const r = streams * c.block;
    const a: usize = absorb_rows;
    const h: usize = c.hidden;
    const qw = @as(usize, c.heads) * head_dim;
    const kvw = @as(usize, c.kv_heads) * head_dim;
    const attn = da.scratch(.{ .q = 0, .segs = 0, .out = 0, .pm = 0, .pl = 0, .po = 0, .streams = streams, .span = widest(c), .heads = c.heads, .kv_heads = c.kv_heads });
    const tops = da.top32Scratch(.{ .logits = 0, .stride = vocab, .n = vocab, .rows = r, .vals = 0, .ids = 0, .part_vals = 0, .part_ids = 0 });
    const sizes = [_]struct { *u64, usize }{
        .{ &b.ids, r * 4 },                                       .{ &b.resid, r * h * 2 },                                .{ &b.x, r * h * 2 },
        .{ &b.xc, r * h * 2 },                                    .{ &b.y, r * h * 2 },                                    .{ &b.yc, r * h * 2 },
        .{ &b.dyn, r * c.dyn() * 2 },                             .{ &b.qkv, r * (qw + 2 * kvw) * 2 },                     .{ &b.q, r * qw * 2 },
        .{ &b.att, r * qw * 2 },                                  .{ &b.gu, r * 2 * @as(usize, c.ffn) * 2 },               .{ &b.act, r * @as(usize, c.ffn) * 2 },
        .{ &b.logits, r * vocab * 2 },                            .{ &b.pm, attn.max },                                    .{ &b.pl, attn.max },
        .{ &b.po, attn.values },                                  .{ &b.tv, tops },                                        .{ &b.ti, tops },
        .{ &b.out, r * da.top * 8 + r * @as(usize, c.rank) * 2 }, .{ &b.part, partBytes(c, vocab, clusters, streams) },    .{ &b.inv, head_dim / 2 * 4 },
        .{ &b.psegs, c.layers * streams * @sizeOf(dk.Segment) },  .{ &b.asegs, c.layers * streams * @sizeOf(da.Segment) }, .{ &b.fcx, a * h * 2 },
        .{ &b.hctx, a * h * 2 },                                  .{ &b.kv, a * 2 * kvw * 2 },                             .{ &b.csegs, c.layers * max_pieces * @sizeOf(dk.Segment) },
    };
    var at: usize = 0;
    for (sizes) |s| {
        at = std.mem.alignForward(usize, at, 256);
        s[0].* = base + at;
        at += s[1];
    }
    return at;
}

/// Device bytes the runtime takes for streams beside its weights: the scratch, each stream's rings and its tap ring.
pub fn runBytes(c: *const Config, target: cfg.Config, clusters: bool, streams: usize) usize {
    var b: Bufs = .{};
    return place(c, target.vocab, clusters, streams, 0, &b) + streams * streamBytes(c, target);
}

/// Device bytes a drafter over `streams` streams takes: B's weights and runBytes (admission's, decision 8).
pub fn openBytes(c: *const Config, target: cfg.Config, clusters: bool, streams: usize) usize {
    return dw.deviceBytes(c.*) + runBytes(c, target, clusters, streams);
}

/// A stream's drafter context: its rings, the positions [from, end) absorbed, its held drafts.
const Lane = struct {
    ring: DeviceBuffer,
    from: u64 = 0,
    end: u64 = 0,
    held: [sel.max_slots]u32 = undefined,
    held_len: usize = 0,
};

/// One absorb run: a stream's rows [pos, pos + rows) at batch row `begin`.
const Piece = struct { lane: *const Lane, begin: usize, pos: u64 };

pub const Drafter = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    m: *const fwd.Model,
    w: dw.Weights,
    c: Config,
    gk: dk.Kernels,
    ak: da.Kernels,
    clusters: bool,
    /// Bytes from a ring's base to each layer's K ring (V follows K a ring further, as D reads it).
    layer_at: [spec.max_layers]usize = @splat(0),
    lanes: std.AutoHashMapUnmanaged(*Stream, *Lane) = .empty,
    pool: std.ArrayList(*Lane) = .empty,
    made: usize = 0,
    streams: usize = 0, // the blocks the scratch holds
    scratch: DeviceBuffer,
    b: Bufs = .{},
    host: Host = .{},
    pieces: std.ArrayList(Piece) = .empty,
    rows: usize = 0, // the absorb pass's rows so far
    step_ms: f64 = 0,
    costs: [windows]Cost = undefined,
    n_costs: usize = 0,

    /// Host staging for `streams` blocks.
    const Host = struct {
        ids: []u32 = &.{},
        psegs: []dk.Segment = &.{},
        asegs: []da.Segment = &.{},
        csegs: []dk.Segment = &.{},
        out: []align(16) u8 = &.{},
        lanes: []*Lane = &.{},
        pending: []u32 = &.{},
        anchors: []u64 = &.{},

        fn free(h: *Host, gpa: std.mem.Allocator) void {
            gpa.free(h.ids);
            gpa.free(h.psegs);
            gpa.free(h.asegs);
            gpa.free(h.csegs);
            gpa.free(h.out);
            gpa.free(h.lanes);
            gpa.free(h.pending);
            gpa.free(h.anchors);
            h.* = .{};
        }
    };

    /// The drafter in `dir` (B's load: the qualified config, every tensor checked) over target `m`, for `streams`.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, m: *const fwd.Model, dir: []const u8, streams: usize) !*Drafter {
        const w = try dw.load(gpa, io, m.d, dir, &m.c);
        return init(gpa, io, m, w, streams);
    }

    /// The runtime over `w` (taken; freed on failure): kernels, scratch and rings for `streams` streams, a block timed.
    pub fn init(gpa: std.mem.Allocator, io: std.Io, m: *const fwd.Model, weights: dw.Weights, streams: usize) !*Drafter {
        var w = weights;
        errdefer w.deinit();
        const c = w.config;
        try gate(&c);
        if (c.vocab != m.c.vocab or c.hidden != m.c.hidden or c.mask >= c.vocab) return error.UnqualifiedRippleDFlash2;
        if (streams == 0 or streams * c.block > 65535) return error.UnsupportedRicursiveDFlash2Shape;
        const d = try gpa.create(Drafter);
        errdefer gpa.destroy(d);
        d.* = .{ .gpa = gpa, .io = io, .m = m, .w = w, .c = c, .gk = undefined, .ak = undefined, .clusters = m.k.lane.clusters, .scratch = undefined };
        var at: usize = 0;
        for (0..c.layers) |l| {
            d.layer_at[l] = at;
            at += @as(usize, c.ring(l)) * c.kv_heads * head_dim * 2 * 2;
        }
        d.gk = try dk.Kernels.load(m.d);
        errdefer d.gk.unload();
        d.ak = try da.Kernels.load(m.d);
        errdefer d.ak.unload();
        d.scratch = try DeviceBuffer.alloc(m.d, 0);
        errdefer d.freeRun();
        try d.reserve(@intCast(streams));
        try d.warm();
        return d;
    }

    /// Lanes, scratch and staging.
    fn freeRun(d: *Drafter) void {
        d.m.s.synchronize() catch {};
        var it = d.lanes.valueIterator();
        while (it.next()) |l| d.dropLane(l.*);
        d.lanes.deinit(d.gpa);
        for (d.pool.items) |l| d.dropLane(l);
        d.pool.deinit(d.gpa);
        d.pieces.deinit(d.gpa);
        d.scratch.free();
        d.host.free(d.gpa);
    }

    pub fn deinit(d: *Drafter) void {
        d.freeRun();
        d.ak.unload();
        d.gk.unload();
        d.w.deinit();
        d.gpa.destroy(d);
    }

    /// Scratch for `streams` blocks and a lane for each (the target's slots at most): made once, at open.
    pub fn reserve(d: *Drafter, asked: u32) !void {
        const streams = @min(@as(usize, asked), d.m.arena.slots);
        if (streams > d.streams) {
            d.m.s.synchronize() catch {};
            var b: Bufs = .{};
            var next = try DeviceBuffer.alloc(d.m.d, place(&d.c, d.c.vocab, d.clusters, streams, 0, &b));
            errdefer next.free();
            _ = place(&d.c, d.c.vocab, d.clusters, streams, next.ptr, &b);
            const inv = dk.inverseFrequencies(d.c.theta);
            try cuda.copy_source.enqueue(d.m.d, b.inv, std.mem.sliceAsBytes(&inv), d.m.s);
            var h: Host = .{};
            errdefer h.free(d.gpa);
            const r = streams * d.c.block;
            h.ids = try d.gpa.alloc(u32, r);
            h.psegs = try d.gpa.alloc(dk.Segment, d.c.layers * streams);
            h.asegs = try d.gpa.alloc(da.Segment, d.c.layers * streams);
            h.csegs = try d.gpa.alloc(dk.Segment, d.c.layers * max_pieces);
            h.out = try d.gpa.alignedAlloc(u8, .@"16", r * da.top * 8 + r * @as(usize, d.c.rank) * 2);
            h.lanes = try d.gpa.alloc(*Lane, streams);
            h.pending = try d.gpa.alloc(u32, streams);
            h.anchors = try d.gpa.alloc(u64, streams);
            try d.pieces.ensureTotalCapacity(d.gpa, max_pieces);
            d.scratch.free();
            d.scratch = next;
            d.b = b;
            d.host.free(d.gpa);
            d.host = h;
            d.streams = streams;
        }
        try d.pool.ensureTotalCapacity(d.gpa, streams);
        while (d.made < streams) {
            var ring = try DeviceBuffer.alloc(d.m.d, laneBytes(&d.c));
            errdefer ring.free();
            try ring.fill8(0, d.m.s.handle); // a key read past a window's end reads a written one, never stale NaN
            const l = try d.gpa.create(Lane);
            l.* = .{ .ring = ring };
            d.pool.appendAssumeCapacity(l);
            d.made += 1;
        }
    }

    fn dropLane(d: *Drafter, l: *Lane) void {
        l.ring.free();
        d.gpa.destroy(l);
    }

    fn ringK(d: *const Drafter, l: *const Lane, layer: usize) u64 {
        return l.ring.ptr + d.layer_at[layer];
    }

    fn ringV(d: *const Drafter, l: *const Lane, layer: usize) u64 {
        return d.ringK(l, layer) + @as(usize, d.c.ring(layer)) * d.c.kv_heads * head_dim * 2;
    }

    pub fn drafter(d: *Drafter) dr.Drafter {
        return .{ .kind = "DFlash2", .ptr = d, .vtable = &.{ .taps = tapsFn, .facts = factsFn, .open = openFn, .absorb = absorbFn, .hold = holdFn, .held = heldFn, .release = releaseFn } };
    }

    fn cast(ptr: *anyopaque) *Drafter {
        return @ptrCast(@alignCast(ptr));
    }

    fn tapsFn(ptr: *anyopaque) []const u32 {
        return &cast(ptr).c.taps;
    }

    /// Chains of up to block - 1 drafts, a round's streams one batch; the windows' costs once the sweep has run.
    pub fn facts(d: *const Drafter) dr.Facts {
        const costs = d.costs[0..d.n_costs];
        return .{ .depth = d.c.block - 1, .step_ms = d.step_ms, .plain_guard = true, .batched = true, .stream_bytes = streamBytes(&d.c, d.m.c), .window_costs = costs, .lane_costs = costs };
    }

    fn factsFn(ptr: *anyopaque) dr.Facts {
        return cast(ptr).facts();
    }

    fn openFn(ptr: *anyopaque, s: *Stream) anyerror!void {
        const d = cast(ptr);
        if (d.lanes.contains(s)) return error.DraftStreamOpen;
        if (d.pool.items.len == 0) try d.reserve(@intCast(d.made + 1));
        const l = d.pool.pop() orelse return error.DraftStreamsPastReserve;
        errdefer d.pool.appendAssumeCapacity(l);
        l.* = .{ .ring = l.ring };
        try d.lanes.put(d.gpa, s, l);
    }

    fn releaseFn(ptr: *anyopaque, s: *Stream) void {
        const d = cast(ptr);
        if (d.lanes.fetchRemove(s)) |e| d.pool.appendAssumeCapacity(e.value);
    }

    fn heldFn(ptr: *anyopaque, streams: []const *Stream, out: []const []u32) anyerror!void {
        const d = cast(ptr);
        for (streams, out) |s, o| {
            const l = d.lanes.get(s) orelse return error.DraftStreamUnknown;
            if (o.len > l.held_len) return error.DraftHeldFewer;
            @memcpy(o, l.held[0..o.len]);
        }
    }

    /// Rows [start, start + rows) replace the context from `start`; prompt rows past the longest span are skipped.
    fn absorbFn(ptr: *anyopaque, items: []const dr.Absorb) anyerror!void {
        const d = cast(ptr);
        const c = &d.c;
        const width: usize = spec.max_taps * @as(usize, c.hidden) * 2;
        for (items) |it| {
            const l = d.lanes.get(it.stream) orelse return error.DraftStreamUnknown;
            if (it.rows != null) return error.DraftTreesUnsupported; // chain verify only (decision 2)
            const f = it.features;
            if (f.space != .device or f.dtype != .bf16 or f.row_bytes != width) return error.DraftFeaturesUnsupported;
            if (it.start > l.end or l.end == 0) l.from = it.start; // the rows before it were restored, not absorbed
            const end = it.start + f.rows;
            var keep_from = if (it.prompt) @max(it.start, it.stream.prompt_len -| (spec.global_window + c.block)) else it.start;
            // a whole prompt pass leaves only its newest rows in the tap ring: older ones are not absorbed
            if (f.rows > held_rows) keep_from = @max(keep_from, end - held_rows);
            if (keep_from > it.start) l.from = @max(l.from, keep_from); // a long prompt's oldest rows: never absorbed
            var at: u64 = keep_from;
            while (at < end) {
                const row: usize = @intCast((f.offset + (at - it.start)) % fwd.tap_rows);
                const n: usize = @intCast(@min(end - at, fwd.tap_rows - row, absorb_rows));
                if (d.rows + n > absorb_rows or d.pieces.items.len == max_pieces) try d.flush();
                // fc of the piece's tap rows straight from the ring, into the pass's rows
                try d.m.k.prompt.matmulOut(d.m.s, f.buffer + row * width, width / 2, n, d.w.fc, d.b.fcx + d.rows * c.hidden * 2, .bf16);
                d.pieces.appendAssumeCapacity(.{ .lane = l, .begin = d.rows, .pos = at });
                d.rows += n;
                at += n;
            }
            l.end = end;
            l.held_len = 0;
        }
        try d.flush();
    }

    /// The pass's rows: fc's rows normed, then each layer's k and v, k normed and rotated, into every piece's ring.
    fn flush(d: *Drafter) !void {
        const rows = d.rows;
        if (rows == 0) return;
        const c = &d.c;
        const m = d.m;
        const ps = d.pieces.items;
        try dk.norm(&d.gk, m.s, .{ .resid = d.b.fcx, .w = d.w.hidden_norm, .x = d.b.hctx, .rows = rows, .width = c.hidden, .eps = c.eps });
        for (0..c.layers) |i| {
            const segs = d.host.csegs[i * ps.len ..][0..ps.len];
            for (segs, ps) |*sg, p| sg.* = .{ .k = d.ringK(p.lane, i), .v = d.ringV(p.lane, i), .begin = @intCast(p.begin), .pos = @intCast(p.pos) };
            try dk.checkSegments(segs, rows, c.ring(i));
        }
        try cuda.copy_source.enqueue(m.d, d.b.csegs, std.mem.sliceAsBytes(d.host.csegs[0 .. c.layers * ps.len]), m.s);
        for (0..c.layers) |i| {
            const L = &d.w.layers[i];
            try m.k.prompt.matmulOut(m.s, d.b.hctx, c.hidden, rows, L.kv, d.b.kv, .bf16);
            try dk.prep(&d.gk, m.s, .{ .x = d.b.kv, .qw = L.q_norm, .kw = L.k_norm, .inv = d.b.inv, .segs = d.b.csegs + i * ps.len * @sizeOf(dk.Segment), .streams = ps.len, .q = 0, .rows = rows, .kv_heads = c.kv_heads, .at = dk.Layout.context(c.kv_heads), .ring = c.ring(i), .rope = c.rope[i], .eps = c.eps });
        }
        d.pieces.clearRetainingCapacity();
        d.rows = 0;
    }

    /// y = x W^T over a batch's rows: the lane matmul, as the target's verify rows run it.
    fn lane(d: *const Drafter, x: u64, rows: usize, w: Weight, y: u64) !void {
        try d.m.k.lane.matmulOut(d.m.s, x, w.k, rows, w, y, .bf16, d.b.part);
    }

    /// x through `cv`'s prepare: its coefficients into dyn, side 0 into xc.
    fn convPrepare(d: *const Drafter, cv: dw.Conv, rows: usize) !void {
        try d.lane(d.b.x, rows, cv.proj, d.b.dyn);
        try d.conv(cv, d.b.x, d.b.xc, rows, 0);
    }

    fn conv(d: *const Drafter, cv: dw.Conv, in: u64, out: u64, rows: usize, side: usize) !void {
        const c = &d.c;
        try dk.conv(&d.gk, d.m.s, .{ .x = in, .dyn = d.b.dyn, .base = cv.base, .y = out, .rows = rows, .width = c.hidden, .block = c.block, .groups = c.hidden / c.group, .side = side, .taps = c.conv });
    }

    fn norm(d: *const Drafter, branch: u64, w: u64, rows: usize) !void {
        try dk.norm(&d.gk, d.m.s, .{ .resid = d.b.resid, .branch = branch, .w = w, .x = d.b.x, .rows = rows, .width = d.c.hidden, .eps = d.c.eps });
    }

    /// Each stream's block (pending, 15 masks) as encodeBlock, stream j at rows 16 j: each row's top 32 and selector.
    fn block(d: *Drafter, ls: []const *Lane, pending: []const u32, anchors: []const u64) !void {
        const c = &d.c;
        const m = d.m;
        const s = m.s;
        const nb: usize = c.block;
        const rows = ls.len * nb;
        const h: usize = c.hidden;
        for (ls, pending, 0..) |_, p, j| {
            d.host.ids[j * nb] = p;
            @memset(d.host.ids[j * nb + 1 ..][0 .. nb - 1], c.mask);
        }
        for (0..c.layers) |i| for (ls, anchors, 0..) |l, a, j| {
            d.host.psegs[i * ls.len + j] = .{ .k = d.ringK(l, i), .v = d.ringV(l, i), .begin = @intCast(j * nb), .pos = @intCast(a) };
            d.host.asegs[i * ls.len + j] = .{ .ring = d.ringK(l, i), .anchor = @intCast(a), .from = @intCast(l.from), .span = @intCast(c.span(i)), .rows = @intCast(c.ring(i)) };
        };
        for (0..c.layers) |i| try dk.checkSegments(d.host.psegs[i * ls.len ..][0..ls.len], rows, c.ring(i));
        try cuda.copy_source.enqueue(m.d, d.b.ids, std.mem.sliceAsBytes(d.host.ids[0..rows]), s);
        try cuda.copy_source.enqueue(m.d, d.b.psegs, std.mem.sliceAsBytes(d.host.psegs[0 .. c.layers * ls.len]), s);
        try cuda.copy_source.enqueue(m.d, d.b.asegs, std.mem.sliceAsBytes(d.host.asegs[0 .. c.layers * ls.len]), s);
        try dk.embed(&m.k.glue, s, d.b.ids, m.w.embed, d.b.resid, d.b.x, rows, h);
        for (0..c.layers) |i| {
            const L = &d.w.layers[i];
            // input norm (the residual takes the last sublayer's output first, from the second layer on)
            try d.norm(if (i > 0) d.b.yc else 0, L.in_norm, rows);
            try d.convPrepare(L.attn_conv, rows);
            try d.lane(d.b.xc, rows, L.qkv, d.b.qkv);
            try dk.prep(&d.gk, s, .{ .x = d.b.qkv, .qw = L.q_norm, .kw = L.k_norm, .inv = d.b.inv, .segs = d.b.psegs + i * ls.len * @sizeOf(dk.Segment), .streams = ls.len, .q = d.b.q, .rows = rows, .kv_heads = c.kv_heads, .at = dk.Layout.block(c.heads, c.kv_heads), .ring = c.ring(i), .rope = c.rope[i], .eps = c.eps });
            try da.attention(&d.ak, s, .{ .q = d.b.q, .segs = d.b.asegs + i * ls.len * @sizeOf(da.Segment), .out = d.b.att, .pm = d.b.pm, .pl = d.b.pl, .po = d.b.po, .streams = ls.len, .span = c.span(i), .heads = c.heads, .kv_heads = c.kv_heads });
            try d.lane(d.b.att, rows, L.o, d.b.y);
            // the attention conv's finish (side 1, its prepare's coefficients), then the post-attention norm
            try d.conv(L.attn_conv, d.b.y, d.b.yc, rows, 1);
            try d.norm(d.b.yc, L.post_norm, rows);
            try d.convPrepare(L.mlp_conv, rows);
            try d.lane(d.b.xc, rows, L.gate_up, d.b.gu);
            try dk.silu(&d.gk, s, d.b.gu, d.b.act, rows, c.ffn);
            try d.lane(d.b.act, rows, L.down, d.b.y);
            try d.conv(L.mlp_conv, d.b.y, d.b.yc, rows, 1);
        }
        try d.norm(d.b.yc, d.w.norm, rows);
        // every row: the target's head, its top 32, the selector's projection (rows 1-15 of a block are walked)
        const vals = d.b.out;
        const ids = vals + rows * da.top * 4;
        const hid = ids + rows * da.top * 4;
        try d.lane(d.b.x, rows, m.w.head, d.b.logits);
        try d.lane(d.b.x, rows, d.w.hproj, hid);
        try da.top32(&d.ak, s, .{ .logits = d.b.logits, .stride = c.vocab, .n = c.vocab, .rows = rows, .vals = vals, .ids = ids, .part_vals = d.b.tv, .part_ids = d.b.ti });
    }

    /// Each stream's `depth` drafts after `pending` at `position`: one batch of blocks, then B's walk on the host.
    fn holdFn(ptr: *anyopaque, items: []const dr.Hold) anyerror!void {
        const d = cast(ptr);
        const c = &d.c;
        if (items.len == 0) return;
        if (items.len > d.streams) return error.DraftStreamsPastReserve;
        const ls = d.host.lanes[0..items.len];
        for (items, ls, d.host.pending[0..items.len], d.host.anchors[0..items.len]) |it, *l, *p, *a| {
            l.* = d.lanes.get(it.stream) orelse return error.DraftStreamUnknown;
            if (it.depth == 0 or it.depth > c.block - 1 or it.position == 0) return error.DraftDepth;
            const anchor = it.position - 1;
            if (l.*.end != anchor) return error.DraftContextGap;
            p.* = it.pending;
            a.* = anchor;
        }
        try d.block(ls, d.host.pending[0..items.len], d.host.anchors[0..items.len]);
        const rows = items.len * c.block;
        const rank: usize = c.rank;
        const out = d.host.out[0 .. rows * da.top * 8 + rows * rank * 2];
        try d.m.download(d.b.out, out);
        const vals = std.mem.bytesAsSlice(f32, out[0 .. rows * da.top * 4]);
        const ids = std.mem.bytesAsSlice(u32, out[rows * da.top * 4 ..][0 .. rows * da.top * 4]);
        const hid = std.mem.bytesAsSlice(u16, out[rows * da.top * 8 ..]);
        for (items, ls, 0..) |it, l, j| {
            const r0 = j * c.block + 1;
            const in: sel.Inputs = .{ .top = c.top, .rank = c.rank, .vocab = c.vocab, .values = @alignCast(vals[r0 * da.top ..]), .ids = @alignCast(ids[r0 * da.top ..]), .hidden = @alignCast(hid[r0 * rank ..]), .pred = d.w.pred, .succ = d.w.succ };
            // a row with no candidate (the Metal walk's id past the vocabulary) refuses, as holdFn does there
            if (sel.walk(in, it.pending, l.held[0..it.depth]) < it.depth) return error.DraftNonfinite;
            l.held_len = it.depth;
        }
    }

    /// The kernels' first launch, then a block timed on a free lane for the depth rule's price, as dflash2.zig's warm.
    fn warm(d: *Drafter) !void {
        const l = d.pool.items[0];
        const a = [_]u64{0};
        const p = [_]u32{0};
        const one = [_]*Lane{l};
        var ms: f64 = 0;
        for (0..3) |run| {
            const t0 = now(d.io);
            try d.block(&one, &p, &a);
            try d.m.s.synchronize();
            if (run == 2) ms = msSince(d.io, t0);
        }
        try l.ring.fill8(0, d.m.s.handle);
        try d.m.s.synchronize();
        d.step_ms = ms / @as(f64, @floatFromInt(d.c.block - 1));
    }

    /// Verify windows of 1 to 16 rows timed on slot 0 before any stream (Qwen's sweep): median of 3, slot 0 reset.
    pub fn timeWindows(d: *Drafter) !void {
        if (d.lanes.count() != 0) return error.WindowsAfterStreams;
        const m = d.m;
        const ids: [windows]u32 = @splat(0);
        for (1..windows + 1) |w| {
            var ms: [3]f64 = undefined;
            for (0..4) |run| {
                const t0 = now(d.io);
                try m.round(&.{.{ .slot = 0, .p0 = 0, .tokens = w }}, ids[0..w]);
                try m.rowLogits(w);
                try m.drawRows(w, d.b.ids);
                try m.s.synchronize();
                if (run > 0) ms[run - 1] = msSince(d.io, t0);
            }
            std.mem.sort(f64, &ms, {}, std.sort.asc(f64));
            d.costs[w - 1] = .{ .width = @intCast(w), .ms = ms[1] };
        }
        try m.reset(0, windows);
        try m.s.synchronize();
        d.n_costs = windows;
        std.log.info("Ripple DFlash2 on CUDA: windows of 1 and 16 rows in {d:.2} and {d:.2} ms, a draft {d:.3} ms ({d} streams a batch)", .{ d.costs[0].ms, d.costs[windows - 1].ms, d.step_ms, d.streams });
    }
};

fn now(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).toNanoseconds();
}

fn msSince(io: std.Io, t0: i96) f64 {
    return @as(f64, @floatFromInt(now(io) - t0)) / 1e6;
}

const testing = std.testing;

fn release() !Config {
    const target = try cfg.parse(testing.allocator, @embedFile("fixtures/config.json"));
    return spec.parse(testing.allocator, @embedFile("fixtures/dflash2_config.json"), &target);
}

test "ricursive dflash2 cuda: the release's rings, a stream's bytes and the scratch at 1 and 8 streams" {
    const c = try release();
    try gate(&c);
    try testing.expectEqual(@as(usize, 136_839_168), laneBytes(&c)); // 4 x 4,224 + 16,512 rows of 8 x 128 K and V
    const target = try cfg.parse(testing.allocator, @embedFile("fixtures/config.json"));
    try testing.expectEqual(laneBytes(&c) + 105_676_800, streamBytes(&c, target));
    var b: Bufs = .{};
    const one = place(&c, c.vocab, true, 1, 0, &b);
    const eight = place(&c, c.vocab, true, 8, 0, &b);
    // the attention's partials (86.5 MB a stream at the global span) are most of it; the absorb pass is fixed
    try testing.expect(one > 86_000_000 and one < 200_000_000);
    try testing.expect(eight - one > 7 * 86_000_000 and eight - one < 7 * 100_000_000);
    try testing.expectEqual(@as(usize, 0), partBytes(&c, c.vocab, true, 8)); // GB10's clusters: no lane reduce
    try testing.expectEqual(dw.deviceBytes(c) + runBytes(&c, target, true, 8), openBytes(&c, target, true, 8));
    try testing.expectEqual(eight + 8 * streamBytes(&c, target), runBytes(&c, target, true, 8));
}

test "ricursive dflash2 cuda: the scratch's regions do not overlap and sit on 256-byte boundaries" {
    const c = try release();
    var b: Bufs = .{};
    const total = place(&c, c.vocab, false, 3, 4096, &b);
    var prev: u64 = 0;
    inline for (@typeInfo(Bufs).@"struct".field_names) |name| {
        const at = @field(b, name);
        try testing.expect(at % 256 == 0 and at >= prev and at <= 4096 + total);
        prev = at;
    }
}
