//! One TP rank's MiniMax-M3.1-Flash CUDA engine: sequences by slot, prompt chunks, verify windows and their draws.

const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("config.zig");
const shard = @import("cuda_shard.zig");
const weights = @import("cuda_weights.zig");
const fw = @import("cuda_forward.zig");
const state = @import("cuda_state.zig");
const kern = @import("cuda_kernels.zig");
const sampler = @import("cuda_sampler.zig");
const dspark = @import("cuda_dspark.zig");
const vision = @import("cuda_vision.zig");

pub const prompt_rows = state.prompt_rows;
pub const window_rows = state.window_rows;
pub const shared_rows = state.batch_rows;
pub const max_shared = 16;

/// Rank 0's own word on its request (cancelled or not); the other ranks have none and hear rank 0's.
pub const Cancel = struct { ptr: ?*anyopaque, check: *const fn (?*anyopaque) bool };

/// A stream's sequence: caches, sampling (null: greedy), draft caches, its last verify's taps, its prompt's logits.
pub const Seq = struct {
    st: state.State,
    sampling: ?sampler.Sampling = null,
    draft: ?dspark.Caches = null,
    stash: ?cuda.DeviceBuffer = null,
    first: ?cuda.DeviceBuffer = null,
    images: ?cuda.DeviceBuffer = null, // this pass's image rows, bf16 [tokens][hidden]
    spans: std.ArrayList(vision.Span) = .empty,
    drafted: [2]usize = .{ 0, 0 }, // the positions [from, to) the draft caches hold context for
};

/// A kept prompt state: its id (the same on every rank) and the position it stands at.
pub const Kept = struct { id: u64, at: u32 };
/// Where a prompt pass resumes: a kept state's id and position.
pub const Resume = struct { id: u64, at: usize };
const Snap = struct { buf: cuda.DeviceBuffer, at: usize, drafted: [2]usize };

/// A stream's draft request: its slot, the rows of its last verify it kept (0: the prompt's), the token after them.
pub const Req = struct { slot: u32, n: u32, anchor: u32, out: []u32 };

/// One stream's window in a shared forward: its slot, its ids (pending token, then drafts), its draws.
pub const Part = struct { slot: u32, ids: []const u32, out: []u32, bits: u64 = 0 };

pub const Options = struct {
    capacity: usize, // positions a sequence holds
    layers: u32 = 0, // 0: every layer; fewer for a thin check
    drafts: bool = true,
    drafter: bool = false, // the release's DSpark head (dspark/) drafts each stream's next block
    vision: ?vision.Limits = null, // --vision: the last rank's tower, every rank's image rows
};

pub const Engine = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    d: *const cuda.Driver,
    c: cfg.Config,
    share: shard.Share,
    m: fw.Modules,
    stream: cuda.Stream,
    w: weights.Weights,
    bp: state.Buffers, // prompt chunks
    bd: state.Buffers, // one stream's verify windows
    bs: state.Buffers, // several streams' windows together
    seqs: std.ArrayList(?*Seq) = .empty,
    cur: *Seq = undefined,
    slot: u32 = 0,
    capacity: usize,
    exchange: fw.Exchange = .local,
    picks: cuda.DeviceBuffer, // every rank's (value, id) of each row's greedy pick: fp32 then i32, [world][rows]
    picks_host: []u8,
    smp: sampler.Sampler,
    drafter: ?*dspark.Drafter = null,
    images: ?*vision.Images = null,
    tap_at: []i32 = &.{},
    snaps: std.AutoHashMapUnmanaged(u64, Snap) = .empty, // kept prompt states by id (the same ids on every rank)
    next_snap: u64 = 1, // rank 0: the next id
    dropped: std.ArrayList(u64) = .empty, // rank 0: ids dropped since the other ranks last heard
    spans: []shard.Span, // every rank's vocabulary span
    checkpoint_digest: [32]u8 = @splat(0),
    structured: bool = false, // a step under a grammar: every rank gathers whole rows, rank 0 masks and draws

    pub fn init(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, ctx: *const cuda.Context, dir: []const u8, rank: usize, world: usize, o: Options) !*Engine {
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        var c = try cfg.Config.read(gpa, io, dir);
        if (o.layers > 0) {
            if (o.layers > c.layers) return error.BadLayers;
            c.layers = o.layers;
        }
        const share = try shard.Share.of(&c, rank, world);
        var stream = try cuda.Stream.init(d, false);
        errdefer stream.deinit();
        var m = try fw.Modules.load(gpa, d, ctx);
        errdefer m.deinit();
        const layers = try gpa.alloc(u32, c.layers);
        defer gpa.free(layers);
        for (layers, 0..) |*l, i| l.* = @intCast(i);
        e.* = .{ .gpa = gpa, .io = io, .d = d, .c = c, .share = share, .m = m, .stream = stream, .w = undefined, .bp = undefined, .bd = undefined, .bs = undefined, .capacity = o.capacity, .picks = undefined, .picks_host = &.{}, .smp = undefined, .spans = &.{} };
        e.w = try weights.load(gpa, io, m.tools(d, stream), dir, c, share, layers);
        errdefer e.w.deinit();
        var extra: std.ArrayList(weights.Mx) = .empty;
        defer extra.deinit(gpa);
        if (o.drafter and o.drafts) {
            const dw = try dspark.load(gpa, io, e.m.tools(d, stream), dir, &e.w);
            const dr = try gpa.create(dspark.Drafter);
            errdefer gpa.destroy(dr);
            dr.* = try dspark.Drafter.init(d, dw, world, share.vocab.len(), c.hidden);
            e.drafter = dr;
            e.tap_at = try gpa.alloc(i32, c.layers);
            dspark.tapAt(&dw.c, c.layers, e.tap_at);
            try extra.append(gpa, dw.fc);
            for (dw.layers[0..dw.c.layers]) |l| try extra.appendSlice(gpa, &.{ l.qkv, l.kv, l.o, l.mlp.gu, l.mlp.down });
        }
        errdefer if (e.drafter) |dr| {
            dr.deinit();
            gpa.destroy(dr);
            gpa.free(e.tap_at);
        };
        const taps = e.drafter != null;
        e.bp = try state.Buffers.init(d, &e.w, prompt_rows, true, o.capacity, world, &e.m.lane, extra.items, taps);
        errdefer e.bp.deinit();
        e.bd = try state.Buffers.init(d, &e.w, window_rows, false, o.capacity, world, &e.m.lane, extra.items, taps);
        errdefer e.bd.deinit();
        e.bs = try state.Buffers.init(d, &e.w, shared_rows, false, o.capacity, world, &e.m.lane, extra.items, taps);
        errdefer e.bs.deinit();
        try e.m.reserve(&c, share, prompt_rows, true);
        try e.m.reserve(&c, share, @max(window_rows, shared_rows), false);
        e.picks = try cuda.DeviceBuffer.alloc(d, world * shared_rows * 8);
        errdefer e.picks.free();
        e.picks_host = try gpa.alloc(u8, world * shared_rows * 8);
        errdefer gpa.free(e.picks_host);
        e.smp = try sampler.Sampler.init(d, c.vocab, shared_rows);
        errdefer e.smp.deinit();
        e.spans = try gpa.alloc(shard.Span, world);
        errdefer gpa.free(e.spans);
        for (e.spans, 0..) |*sp, q| sp.* = (try shard.Share.of(&c, q, world)).vocab;
        if (o.vision) |lim| e.images = try vision.Images.init(gpa, io, d, dir, lim, rank + 1 == world);
        errdefer if (e.images) |v| v.deinit();
        try e.bind(0);
        try e.stream.synchronize();
        return e;
    }

    pub fn deinit(e: *Engine) void {
        e.stream.synchronize() catch {};
        var it = e.snaps.valueIterator();
        while (it.next()) |x| x.buf.free();
        e.snaps.deinit(e.gpa);
        e.dropped.deinit(e.gpa);
        for (0..e.seqs.items.len) |i| e.free(i);
        e.seqs.deinit(e.gpa);
        if (e.images) |v| v.deinit();
        e.gpa.free(e.spans);
        e.smp.deinit();
        e.gpa.free(e.picks_host);
        e.picks.free();
        e.bs.deinit();
        e.bd.deinit();
        e.bp.deinit();
        if (e.drafter) |dr| {
            dr.deinit();
            e.gpa.destroy(dr);
            e.gpa.free(e.tap_at);
        }
        e.w.deinit();
        e.m.deinit();
        e.stream.deinit();
        const gpa = e.gpa;
        gpa.destroy(e);
    }

    /// Bytes one more stream's sequence takes on this rank (its draft caches and stash too, with a drafter).
    pub fn seqBytes(e: *const Engine) u64 {
        const draft: u64 = if (e.drafter) |dr| dspark.Caches.bytes(dr.w.c.layers, e.share.kv.len(), e.capacity) + window_rows * 5 * e.c.hidden * 2 else 0;
        const images: u64 = if (e.images) |v| v.streamBytes() else 0;
        return state.cacheBytes(&e.c, e.share.kv.len(), e.capacity) + draft + e.share.vocab.len() * 2 + images;
    }

    /// Binds slot `slot`'s sequence for the steps that follow, making it on first use.
    pub fn bind(e: *Engine, slot: u32) !void {
        if (slot >= e.seqs.items.len) try e.seqs.appendNTimes(e.gpa, null, slot + 1 - e.seqs.items.len);
        if (e.seqs.items[slot] == null) {
            const q = try e.gpa.create(Seq);
            errdefer e.gpa.destroy(q);
            q.* = .{ .st = try state.State.init(e.gpa, e.d, &e.c, e.share.kv.len(), e.capacity) };
            errdefer {
                q.st.deinit();
                if (q.first) |*x| x.free();
                if (q.draft) |*x| x.deinit();
                if (q.stash) |*x| x.free();
            }
            q.first = try cuda.DeviceBuffer.alloc(e.d, e.share.vocab.len() * 2);
            if (e.drafter) |dr| {
                q.draft = try dspark.Caches.init(e.d, dr.w.c.layers, e.share.kv.len(), e.capacity);
                q.stash = try cuda.DeviceBuffer.alloc(e.d, window_rows * 5 * e.c.hidden * 2);
            }
            // a pass's image rows at the request limit, made here so nothing allocates after a step goes out
            if (e.images) |v| q.images = try cuda.DeviceBuffer.alloc(e.d, v.streamBytes());
            e.seqs.items[slot] = q;
        }
        e.cur = e.seqs.items[slot].?;
        e.slot = slot;
    }

    /// Frees slot `slot`'s sequence (slot 0 stays) and binds slot 0.
    pub fn release(e: *Engine, slot: u32) void {
        if (slot != 0) e.free(slot);
        e.cur = e.seqs.items[0].?;
        e.slot = 0;
    }

    fn free(e: *Engine, slot: usize) void {
        const q = e.seqs.items[slot] orelse return;
        e.stream.synchronize() catch {};
        q.st.deinit();
        if (q.draft) |*x| x.deinit();
        if (q.stash) |*x| x.free();
        if (q.first) |*x| x.free();
        if (q.images) |*x| x.free();
        q.spans.deinit(e.gpa);
        e.gpa.destroy(q);
        e.seqs.items[slot] = null;
    }

    /// The committed length: the pending token sits at this position, a window's row r draws position pos + 1 + r.
    pub fn pos(e: *const Engine) usize {
        return e.cur.st.pos;
    }

    /// A fresh sequence on the bound slot with `sampling` (null or temperature 0: greedy).
    pub fn begin(e: *Engine, sampling: ?sampler.Sampling) void {
        e.cur.spans.clearRetainingCapacity(); // no earlier request's rows splice into this one
        e.cur.drafted = .{ 0, 0 };
        e.cur.st.reset();
        e.cur.sampling = sampler.check(sampling);
    }

    fn forward(e: *Engine, b: *state.Buffers) fw.Forward {
        const im: ?fw.ImageRows = if (b == &e.bp and e.cur.spans.items.len > 0) .{ .rows = e.cur.images.?.ptr, .spans = e.cur.spans.items } else null;
        return .{ .d = e.d, .m = &e.m, .w = &e.w, .stream = e.stream, .b = b, .exchange = e.exchange, .taps = if (e.drafter != null) b.taps else null, .tap_at = e.tap_at, .images = im };
    }

    /// A pass's images (gh, gw, at, tokens each): rank 0's patches to the tower's rank, its rows to every rank.
    pub fn encodeImages(e: *Engine, spans: []const [4]u32, pixels: ?[]const []const f32) !void {
        const v = e.images orelse return error.NoVision;
        try imageLimits(v, spans);
        const q = e.cur;
        q.spans.clearRetainingCapacity();
        const world = e.share.world;
        var row: u32 = 0;
        for (spans, 0..) |sp, i| {
            const P: usize = @as(usize, sp[0]) * sp[1];
            if (pixels) |px| try v.pixels.upload(0, std.mem.sliceAsBytes(px[i]));
            if (world > 1) {
                const x = e.exchange.nccl;
                try x.lib.check(x.lib.api.ncclBroadcast(v.pixels.ptr, v.pixels.ptr, P * vision.patch_width * 4, .u8, 0, x.comm, e.stream.handle), "ncclBroadcast");
            }
            if (v.tower) |t| {
                try e.d.check(e.d.api.cuMemcpyDtoDAsync_v2(t.px32, v.pixels.ptr, P * vision.patch_width * 4, e.stream.handle), "cuMemcpyDtoDAsync");
                try t.encode(&e.m.k, e.stream, sp[0], sp[1], q.images.?.ptr + @as(usize, row) * vision.hidden * 2);
            }
            try q.spans.append(e.gpa, .{ .at = sp[2], .n = sp[3], .row = row });
            row += sp[3];
        }
        if (world > 1 and row > 0) {
            const x = e.exchange.nccl;
            try x.lib.check(x.lib.api.ncclBroadcast(q.images.?.ptr, q.images.?.ptr, @as(usize, row) * vision.hidden * 2, .u8, @intCast(world - 1), x.comm, e.stream.handle), "ncclBroadcast");
        }
        try e.stream.synchronize();
    }

    /// A pass's images within the tower's and the request's limits (checked on rank 0 before the step goes out).
    pub fn imageLimits(v: *const vision.Images, spans: []const [4]u32) !void {
        var total: usize = 0;
        for (spans) |sp| {
            const P: usize = @as(usize, sp[0]) * sp[1];
            if (P == 0 or P > vision.max_patches or sp[0] % 2 != 0 or sp[1] % 2 != 0 or sp[3] != P / 4) return error.ImageTooLarge;
            total += sp[3];
        }
        if (total > v.limits.image_tokens or spans.len > v.limits.max_images) return error.ImageTooLarge;
    }

    /// A forward's tapped rows [row0, row0 + rows) into the bound sequence's stash (its next draft absorbs the kept).
    fn stashTaps(e: *Engine, b: *const state.Buffers, row0: usize, rows: usize) !void {
        const stash = e.cur.stash orelse return;
        const row = 5 * @as(usize, e.c.hidden) * 2;
        try e.d.check(e.d.api.cuMemcpyDtoDAsync_v2(stash.ptr, b.taps + row0 * row, rows * row, e.stream.handle), "cuMemcpyDtoDAsync");
    }

    /// The whole prompt in chunks, then the first token from the last row.
    pub fn prefill(e: *Engine, ids: []const u32, cancel: ?Cancel) !u32 {
        if (ids.len == 0 or ids.len + window_rows >= e.capacity) return error.PromptTooLong;
        var start: usize = 0;
        while (true) : (start += prompt_rows) {
            if (try e.prefillChunk(ids, start, cancel)) |first| return first;
        }
    }

    /// One chunk of a prompt pass from `start`: the first token at the end, else null.
    pub fn prefillChunk(e: *Engine, ids: []const u32, start: usize, cancel: ?Cancel) !?u32 {
        if (start >= ids.len or start % prompt_rows != 0 or start != e.cur.st.pos) return error.BadChunk;
        if (cancel) |x| if (x.check(x.ptr)) return error.Cancelled;
        const rows = @min(ids.len - start, prompt_rows);
        const f = e.forward(&e.bp);
        const seg = [_]fw.Seg{.{ .st = &e.cur.st, .row0 = 0, .rows = rows }};
        try f.run(@ptrCast(ids[start..][0..rows]), &seg);
        try fw.Forward.commit(&seg, &.{rows});
        var first: ?u32 = null;
        if (start + rows == ids.len) {
            var t: [1]u32 = undefined;
            try e.copy(e.cur.first.?.ptr, e.bp.logits, e.share.vocab.len() * 2); // a grammar's redraw reads it
            try e.draw(&e.bp, 0, 1, ids.len, 0, &t);
            first = t[0];
        }
        // the drafter's context: the rows its next blocks can see (the window before the prompt's end)
        if (e.drafter) |dr| {
            const from = @max(start, ids.len -| (dr.w.c.window + window_rows));
            if (from < start + rows) {
                try dr.absorb(&f, &e.cur.draft.?, e.bp.taps + (from - start) * 5 * e.c.hidden * 2, start + rows - from, from);
                const q = e.cur;
                if (q.drafted[1] != from) q.drafted[0] = from; // a gap: the context starts again here
                q.drafted[1] = start + rows;
            }
        }
        return first;
    }

    /// A verify window (pending, then drafts) at the committed position; `bits`: rank 0's grammar words, else 0.
    pub fn verify(e: *Engine, ids: []const u32, out: []u32, bits: u64) !void {
        if (ids.len == 0 or ids.len > window_rows or out.len < ids.len) return error.WindowTooWide;
        if (e.cur.st.pos + ids.len + 1 > e.capacity) return error.ContextFull;
        const f = e.forward(&e.bd);
        const seg = [_]fw.Seg{.{ .st = &e.cur.st, .row0 = 0, .rows = ids.len }};
        const at = e.cur.st.pos;
        try f.run(@ptrCast(ids), &seg);
        try e.stashTaps(&e.bd, 0, ids.len);
        try e.draw(&e.bd, 0, ids.len, at + 1, bits, out[0..ids.len]);
    }

    /// Several streams' windows in one forward, each stream's rows on its own sequence.
    pub fn verifyShared(e: *Engine, parts: []const Part) !void {
        if (parts.len == 0 or parts.len > max_shared) return error.WindowTooWide;
        var segs: [max_shared]fw.Seg = undefined;
        var ids: [shared_rows]u32 = undefined;
        var rows: usize = 0;
        for (parts, 0..) |p, i| {
            if (p.ids.len == 0 or p.ids.len > window_rows or rows + p.ids.len > shared_rows) return error.WindowTooWide;
            for (parts[0..i]) |q| if (q.slot == p.slot) return error.SlotTwice;
            try e.bind(p.slot);
            if (e.cur.st.pos + p.ids.len + 1 > e.capacity) return error.ContextFull;
            segs[i] = .{ .st = &e.cur.st, .row0 = rows, .rows = p.ids.len };
            @memcpy(ids[rows..][0..p.ids.len], p.ids);
            rows += p.ids.len;
        }
        const f = e.forward(&e.bs);
        try f.run(@ptrCast(ids[0..rows]), segs[0..parts.len]);
        for (parts, segs[0..parts.len]) |p, g| {
            try e.bind(p.slot);
            try e.stashTaps(&e.bs, g.row0, g.rows);
            try e.draw(&e.bs, g.row0, g.rows, g.st.pos + 1, p.bits, p.out[0..g.rows]);
        }
    }

    /// A grammar step's draw: whole rows on every rank, rank 0 masks and draws (sampled, or greedy's lowest id).
    pub fn drawWhole(e: *Engine, logits: u64, rows: usize, first: u64, bits: u64, out: []u32) !void {
        try e.smp.gather(e.stream, e.exchange, e.spans, &.{.{ .rank = e.share.rank, .logits = logits }}, rows, e.smp.full.ptr);
        if (e.share.rank != 0) {
            try e.stream.synchronize();
            return e.gathered();
        }
        if (bits != 0) try e.smp.mask(e.stream, rows, bits, (e.c.vocab + 31) / 32);
        if (e.cur.sampling) |sm| {
            try e.smp.drawFull(e.stream, rows, first, sampler.rule(sm), out);
            return e.gathered();
        }
        for (0..rows) |r| {
            const vals = e.picks.ptr;
            try kern.top1(&e.m.k, e.stream, e.smp.full.ptr + r * e.c.vocab * 2, 1, e.c.vocab, 0, vals, vals + 4);
            var pick: [2]u32 = undefined;
            try e.stream.synchronize();
            try e.picks.download(0, std.mem.asBytes(&pick));
            out[r] = pick[1];
        }
        try e.gathered();
    }

    /// After a synchronize: a RoCE gather that failed since the last one (its rows hold nothing valid) is an error.
    fn gathered(e: *const Engine) !void {
        switch (e.exchange) {
            .local => {},
            .nccl => |x| if (x.group) |g| try g.check(),
        }
    }

    /// The first token drawn again from the prompt's last row (kept by its pass), among `bits`' tokens.
    pub fn redrawFirst(e: *Engine, len: usize, bits: u64) !u32 {
        var t: [1]u32 = .{0};
        try e.drawWhole(e.cur.first.?.ptr, 1, len, bits, &t);
        return t[0];
    }

    /// Bytes a state kept at `at` takes on this rank: every layer's rows below it, the draft caches' too.
    pub fn stateBytes(e: *const Engine, at: usize) u64 {
        const draft: u64 = if (e.drafter) |dr| @as(u64, dr.w.c.layers) * e.share.kv.len() * 128 * 2 * 2 else 0;
        return @as(u64, at) * (@as(u64, e.c.layers) * state.rowBytes(&e.c, e.share.kv.len()) + draft);
    }

    /// The bound sequence's rows below `at` copied under `id` (every rank keeps the same id).
    pub fn keepAs(e: *Engine, id: u64, at: usize) !void {
        if (at == 0 or at > e.cur.st.pos or e.snaps.contains(id)) return error.BadKeep;
        var buf = try cuda.DeviceBuffer.alloc(e.d, e.stateBytes(at));
        errdefer buf.free();
        try e.copyState(&e.cur.*, buf.ptr, at, true);
        const d0 = @min(e.cur.drafted[0], at);
        try e.snaps.put(e.gpa, id, .{ .buf = buf, .at = at, .drafted = .{ d0, @min(e.cur.drafted[1], at) } });
    }

    /// The bound sequence restored to state `id`: its rows below the state's position, its length there.
    pub fn restoreState(e: *Engine, id: u64) !void {
        const x = e.snaps.get(id) orelse return error.SnapshotStale;
        try e.copyState(&e.cur.*, x.buf.ptr, x.at, false);
        e.cur.st.pos = x.at;
        e.cur.st.pending = 0;
        e.cur.drafted = x.drafted;
    }

    pub fn holds(e: *const Engine, id: u64) bool {
        return e.snaps.contains(id);
    }

    /// Where kept state `id` stands (null: none kept under it).
    pub fn keptAt(e: *const Engine, id: u64) ?usize {
        return if (e.snaps.get(id)) |x| x.at else null;
    }

    pub fn dropState(e: *Engine, id: u64) void {
        const kv = e.snaps.fetchRemove(id) orelse return;
        e.stream.synchronize() catch {};
        var b = kv.value.buf;
        b.free();
    }

    /// Rows below `at` of each layer's caches between the sequence and `buf` (out: to buf), layer after layer.
    fn copyState(e: *Engine, q: *Seq, buf: u64, at: usize, out: bool) !void {
        const hk = e.share.kv.len();
        const kvb = at * hk * e.c.head_dim * 2;
        const ikb = at * e.c.index_dim * 2;
        var off: u64 = 0;
        for (0..e.c.layers) |l| {
            for ([_]u64{ q.st.keys(l), q.st.values(&e.c, l), q.st.indexKeys(&e.c, l) }, [_]usize{ kvb, kvb, ikb }) |src, n| {
                try e.copy(if (out) buf + off else src, if (out) src else buf + off, n);
                off += n;
            }
        }
        const draft_layers: usize = if (e.drafter) |dr| dr.w.c.layers else 0;
        if (q.draft) |*dc| for (0..draft_layers) |l| {
            const n = at * hk * 128 * 2;
            for ([_]u64{ dc.keys(l), dc.values(l) }) |src| {
                try e.copy(if (out) buf + off else src, if (out) src else buf + off, n);
                off += n;
            }
        };
        try e.stream.synchronize();
    }

    fn copy(e: *const Engine, dst: u64, src: u64, bytes: usize) !void {
        try e.d.check(e.d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, e.stream.handle), "cuMemcpyDtoDAsync");
    }

    /// Each request's kept rows absorbed below its committed length, then the blocks, as many to a pass as it holds.
    pub fn draftBlock(e: *Engine, reqs: []const Req) !void {
        const dr = e.drafter orelse return error.NoDrafter;
        var blocks: [max_shared]dspark.Block = undefined;
        for (reqs, blocks[0..reqs.len]) |r, *blk| {
            try e.bind(r.slot);
            const f = e.forward(&e.bd);
            const draft = &e.cur.draft.?;
            const q = e.cur;
            const at = q.st.pos;
            if (r.n > 0) {
                try dr.absorb(&f, draft, q.stash.?.ptr, r.n, at - r.n);
                if (q.drafted[1] != at - r.n) q.drafted[0] = at - r.n; // rows kept without a draft: start again
                q.drafted[1] = at;
            }
            if (q.drafted[1] != at) q.drafted = .{ at, at };
            blk.* = .{ .caches = draft, .anchor = r.anchor, .p = at, .from = q.drafted[0], .out = r.out };
        }
        const f = e.forward(&e.bd);
        var i: usize = 0;
        while (i < reqs.len) : (i += dr.perPass()) try dr.blocks(&f, blocks[i..@min(reqs.len, i + dr.perPass())]);
    }

    /// The last window's first `n` rows kept on the bound sequence (the rest of its writes are past pos: harmless).
    pub fn keep(e: *Engine, n: usize) !void {
        const seg = [_]fw.Seg{.{ .st = &e.cur.st, .row0 = 0, .rows = e.cur.st.pending }};
        try fw.Forward.commit(&seg, &.{n});
    }

    /// Rows of this rank's logits to tokens at first, first + 1, ...: keyed sampler on whole rows, or ranks' greedy.
    pub fn draw(e: *Engine, b: *state.Buffers, row0: usize, rows: usize, first: u64, bits: u64, out: []u32) !void {
        const world = e.share.world;
        const n = e.share.vocab.len();
        const logits = b.logits + row0 * n * 2;
        if (e.structured) return e.drawWhole(logits, rows, first, bits, out);
        if (e.cur.sampling) |sm| {
            try e.smp.draw(e.stream, e.exchange, e.spans, &.{.{ .rank = e.share.rank, .logits = logits }}, rows, first, sampler.rule(sm), out);
            return e.gathered();
        }
        const vals = e.picks.ptr + e.share.rank * rows * 8;
        try kern.top1(&e.m.k, e.stream, logits, rows, n, e.share.vocab.lo, vals, vals + rows * 4);
        switch (e.exchange) {
            .local => {},
            .nccl => |x| {
                if (x.group) |g| try g.allGather(e.stream, vals, e.picks.ptr, rows * 8) else try x.lib.check(x.lib.api.ncclAllGather(vals, e.picks.ptr, rows * 8, .u8, x.comm, e.stream.handle), "ncclAllGather");
            },
        }
        try e.stream.synchronize();
        try e.gathered();
        const bytes = e.picks_host[0 .. world * rows * 8];
        try e.picks.download(0, bytes);
        for (0..rows) |r| {
            var best: f32 = -std.math.inf(f32);
            var id: u32 = 0;
            for (0..world) |q| {
                const at = q * rows * 8;
                const v: f32 = @bitCast(std.mem.readInt(u32, bytes[at + r * 4 ..][0..4], .little));
                const i = std.mem.readInt(u32, bytes[at + rows * 4 + r * 4 ..][0..4], .little);
                if (v > best or (v == best and i < id)) {
                    best = v;
                    id = i;
                }
            }
            out[r] = id;
        }
    }
};
