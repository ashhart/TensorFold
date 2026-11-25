//! One TP rank's Inkling 2 engine: streams by slot, prompt chunks, decode rows (shared, graphed) and their draws.
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("shape.zig");
const shard = @import("shard.zig");
const weights = @import("pack_weights.zig");
const fw = @import("forward.zig");
const ops = @import("ops.zig");
const sampler = @import("sampler.zig");
const kept = @import("kept.zig");
const vision = @import("vision.zig");
const image = @import("image.zig");
const pictures = @import("pictures.zig");

/// Rows a prompt chunk holds (the prompt cache's mark grid) and a stream's window at most (one row: no drafts).
pub const prompt_rows = 2048;
pub const window_rows = 1;
/// A shared decode forward's streams and rows at most.
pub const max_shared = 16;
pub const shared_rows = max_shared * window_rows;
/// Partial slots a prompt call writes at most: 2,048 rows over 8 global chunks (deeper calls take fewer rows).
const prompt_parts = 8;

/// Rank 0's own word on its request (cancelled or not); the other ranks have none and hear rank 0's.
pub const Cancel = struct { ptr: ?*anyopaque, check: *const fn (?*anyopaque) bool };

/// One stream's window in a shared forward: its slot, its ids (the pending token), its draws.
pub const Part = struct { slot: u32, ids: []const u32, out: []u32 };

/// A slot's request state: its sampling (null: greedy).
const Seq = struct { sampling: ?sampler.Sampling = null };

pub const Options = struct { capacity: usize, vision: ?pictures.Limits = null };

/// A graph's key: a decode call's rows, streams, attention tiles and partial stride (its global chunks).
const Key = struct { n: u32, ni: u32, nt: u32, stride: u32 };

pub const Engine = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    d: *const cuda.Driver,
    c: cfg.Config,
    m: fw.Modules,
    stream: cuda.Stream,
    w: weights.Weights,
    shape_p: fw.Shape,
    shape_d: fw.Shape,
    bp: fw.Buffers,
    bd: fw.Buffers,
    parts: std.ArrayList(cuda.DeviceBuffer) = .empty, // the partials bp and bd share (one stream runs both in order)
    st: ?fw.State = null, // every slot's caches, made by reserve
    seqs: []Seq = &.{},
    slot: u32 = 0,
    capacity: usize,
    exchange: fw.Exchange = .local,
    picks: cuda.DeviceBuffer, // every rank's (value, id) of each row's greedy pick: fp32 then i32, [world][rows]
    picks_host: []u8,
    smp: sampler.Sampler,
    spans: []shard.Span, // every rank's vocabulary span
    graphs: cuda.graph_cache.Cache(Key),
    gengine: cuda.graph_cache.CudaEngine,
    states: kept.Store,
    checkpoint_digest: [32]u8 = @splat(0), // shard bytes, config and stream shape (native.readyDigest); ranks compare it
    vision: ?pictures.Limits = null, // images on (--vision): every rank's rows a slot, rank 0's tower
    tower: ?*vision.Tower = null,
    rows: []pictures.Rows = &.{},

    pub fn init(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, rank: usize, world: usize, o: Options) !*Engine {
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        const c = try cfg.Config.read(gpa, io, dir);
        var stream = try cuda.Stream.init(d, false);
        errdefer stream.deinit();
        e.* = .{ .gpa = gpa, .io = io, .d = d, .c = c, .m = try fw.Modules.load(d), .stream = stream, .w = undefined, .shape_p = undefined, .shape_d = undefined, .bp = undefined, .bd = undefined, .capacity = o.capacity, .picks = undefined, .picks_host = &.{}, .smp = undefined, .spans = &.{}, .graphs = undefined, .gengine = .{ .d = d }, .states = .{ .gpa = gpa } };
        errdefer e.m.deinit(); // e's copy: it holds the prompt projections and the scratch reserve grows
        e.w = try weights.load(gpa, io, d, &e.m.dense, stream, dir, c, rank, world);
        errdefer e.w.deinit();
        const heads = e.w.share.heads.len();
        const kv = e.w.share.kv.len();
        e.shape_p = .{ .rows = prompt_rows, .context = o.capacity, .heads = heads, .kv = kv };
        e.shape_d = .{ .rows = shared_rows, .context = o.capacity, .heads = heads, .kv = kv, .ring_rows = prompt_rows };
        errdefer {
            for (e.parts.items) |*b| b.free();
            e.parts.deinit(gpa);
        }
        const slots = @max(prompt_rows * heads * prompt_parts, shared_rows * heads * fw.stride(&e.c, o.capacity - 1));
        const shared = try fw.partialsFor(d, &e.parts, gpa, slots);
        e.bp = try fw.Buffers.init(gpa, d, &e.w, e.shape_p, shared);
        errdefer e.bp.deinit();
        e.bd = try fw.Buffers.init(gpa, d, &e.w, e.shape_d, shared);
        errdefer e.bd.deinit();
        for (e.w.layers) |l| for ([_]cuda.exl3.Weight{ l.attn.q, l.attn.k, l.attn.v, l.attn.r, l.attn.gate, l.attn.o }) |x| try e.m.dense.reserve(stream, x);
        try e.m.dense.reserve(stream, e.w.head); // scratch grown now, so nothing allocates inside a graph capture
        e.m.prompt = @import("prompt_dense.zig").Prompt.of(gpa, d, e.m.ops.prompt_tail, prompt_rows, e.w.layers) catch |err| blk: { // without cuBLASLt, prompt rows keep the decode kernels
            std.log.warn("inkling2: prompt projections on the decode kernels ({s})", .{@errorName(err)});
            break :blk null;
        };
        for (e.w.layers) |l| for ([_]cuda.exl3.Weight{ l.attn.k, l.attn.v, l.attn.r }) |x| try e.m.side.dense.reserve(stream, x);
        e.vision = o.vision;
        if (o.vision != null and rank == 0) e.tower = try vision.Tower.load(gpa, io, d, dir, c.vision, pictures.picture_tokens);
        errdefer if (e.tower) |t| t.deinit();
        e.picks = try cuda.DeviceBuffer.alloc(d, world * pickStride(shared_rows));
        errdefer e.picks.free();
        e.picks_host = try gpa.alloc(u8, world * pickStride(shared_rows));
        errdefer gpa.free(e.picks_host);
        e.smp = try sampler.Sampler.init(d, e.w.padded_vocab, shared_rows);
        errdefer e.smp.deinit();
        e.spans = try gpa.alloc(shard.Span, world);
        errdefer gpa.free(e.spans);
        for (e.spans, 0..) |*sp, q| sp.* = (try shard.Share.of(&e.c, e.w.padded_vocab, q, world)).vocab;
        e.graphs = .init(gpa, e.gengine.engine(), .{});
        errdefer e.graphs.deinit();
        e.graphs.name = "inkling2 decode";
        try e.stream.synchronize();
        return e;
    }

    pub fn deinit(e: *Engine) void {
        e.stream.synchronize() catch {};
        e.graphs.deinit();
        e.states.deinit();
        if (e.tower) |t| t.deinit();
        pictures.freeRows(e.gpa, e.rows);
        if (e.st) |*s| s.deinit();
        e.gpa.free(e.seqs);
        e.gpa.free(e.spans);
        e.smp.deinit();
        e.gpa.free(e.picks_host);
        e.picks.free();
        e.bd.deinit();
        e.bp.deinit();
        for (e.parts.items) |*b| b.free();
        e.parts.deinit(e.gpa);
        e.w.deinit();
        e.m.deinit();
        e.stream.deinit();
        const gpa = e.gpa;
        gpa.destroy(e);
    }

    /// Bytes one more stream takes on this rank: its slot's rings and conv states, its image rows.
    pub fn seqBytes(e: *const Engine) u64 {
        const rows: u64 = if (e.vision) |l| pictures.slotBytes(l, e.c.hidden) else 0;
        return fw.slotBytes(&e.c, e.shape_d) + rows;
    }

    /// The largest all-gather a rank sends (a prompt call's fp32 partial): RoCE writes take every gather up to it.
    pub fn roceBytes(e: *const Engine) usize {
        return prompt_rows * @as(usize, e.c.hidden) * 4;
    }

    fn convBytes(e: *const Engine) usize {
        return @as(usize, e.c.layers) * 2 * 3 * (e.shape_d.kv * e.c.head_dim + e.c.hidden) * 4;
    }

    /// Every slot's caches for `slots` streams, made once (the host's admission decides how many).
    pub fn reserve(e: *Engine, slots: usize) !void {
        if (e.st != null) return if (e.seqs.len == slots) {} else error.Inkling2Reserved;
        e.st = try fw.State.init(e.gpa, e.d, &e.c, e.shape_d, slots);
        errdefer {
            e.st.?.deinit();
            e.st = null;
        }
        if (e.vision) |l| e.rows = try pictures.makeRows(e.gpa, e.d, l, slots, e.c.hidden);
        e.seqs = try e.gpa.alloc(Seq, slots);
        @memset(e.seqs, .{});
    }

    pub fn slotCount(e: *const Engine) usize {
        return e.seqs.len;
    }

    pub fn bind(e: *Engine, slot: u32) !void {
        if (slot >= e.seqs.len) return error.Inkling2NoSlot;
        e.slot = slot;
    }

    /// The bound slot's committed length: the pending token sits there, a window's row r draws position pos + 1 + r.
    pub fn pos(e: *const Engine) usize {
        return e.st.?.pos[e.slot];
    }

    /// A new request on the bound slot with `sampling` (null or temperature 0: greedy).
    pub fn begin(e: *Engine, s: ?sampler.Sampling) void {
        e.seqs[e.slot] = .{ .sampling = sampler.check(s) };
        if (e.rows.len > 0) e.rows[e.slot].spans.clearRetainingCapacity(); // a new request splices nothing until told
    }

    /// The bound slot's pictures for its next prompt pass: rank 0 encodes them (`pics`), every rank gets the rows.
    pub fn images(e: *Engine, pics: ?[]const image.Prepared, spans: []const vision.Span) !void {
        const l = e.vision orelse return error.NoVision;
        if (e.rows.len == 0) return error.Inkling2Reserved;
        const total = try pictures.checkSpans(spans, l);
        const r = &e.rows[e.slot];
        r.spans.clearRetainingCapacity();
        if (e.tower) |t| {
            const ps = pics orelse return error.ImagePlaceholders;
            if (ps.len != spans.len) return error.ImagePlaceholders;
            for (ps, spans) |p, sp| try t.encode(e.stream, p, r.buf.ptr + @as(u64, sp.row) * e.c.hidden * 2);
        } else if (e.exchange == .local) return error.NoVision;
        switch (e.exchange) {
            .local => {},
            .nccl => |x| try x.lib.check(x.lib.api.ncclBroadcast(r.buf.ptr, r.buf.ptr, @as(usize, total) * e.c.hidden * 2, .u8, 0, x.comm, e.stream.handle), "ncclBroadcast"),
        }
        try r.spans.appendSlice(e.gpa, spans);
    }

    fn forward(e: *Engine, b: *fw.Buffers, shape: fw.Shape) fw.Forward {
        return .{ .d = e.d, .m = &e.m, .w = &e.w, .b = b, .shape = shape, .stream = e.stream, .exchange = e.exchange };
    }

    /// One chunk of the bound slot's prompt from `start` (0: a fresh pass); the first token once it ends the prompt.
    pub fn prefillChunk(e: *Engine, ids: []const u32, start: usize) !?u32 {
        const st = &e.st.?;
        if (ids.len == 0 or ids.len + window_rows >= e.capacity) return error.PromptTooLong;
        if (start >= ids.len or start % prompt_rows != 0) return error.BadChunk;
        if (start == 0) try st.clearSlot(&e.c, e.d, e.slot, e.stream) else if (start != st.pos[e.slot]) return error.BadChunk;
        e.states.overwrite(e.slot, start); // states on this slot past `start` lose their rows to this pass
        const end = @min(ids.len, start + prompt_rows);
        var f = e.forward(&e.bp, e.shape_p);
        f.last_logits = true;
        // a deep call writes partials for every global chunk: fewer rows a call, the same bits a row
        const most = @max(1, e.bp.parts.slots / (e.shape_p.heads * fw.stride(&e.c, end - 1)));
        var at = start;
        var n: usize = 0;
        while (at < end) : (at += n) {
            n = @min(end - at, most);
            const item = [_]fw.Item{.{ .slot = e.slot, .ids = @ptrCast(ids[at..][0..n]) }};
            var g = f;
            if (e.rows.len > 0 and e.rows[e.slot].spans.items.len > 0) g.images = .{ .rows = e.rows[e.slot].buf.ptr, .spans = e.rows[e.slot].spans.items, .lo = at };
            try g.run(st, &item);
        }
        if (end < ids.len) return null;
        var t: [1]u32 = undefined;
        try e.draw(&e.bp, n - 1, 1, ids.len, e.seqs[e.slot].sampling, &t);
        return t[0];
    }

    /// Several streams' rows in one forward (each on its own slot), graphed.
    pub fn verifyShared(e: *Engine, parts: []const Part) !void {
        const st = &e.st.?;
        if (parts.len == 0 or parts.len > max_shared) return error.WindowTooWide;
        var items: [max_shared]fw.Item = undefined;
        var ids: [shared_rows]i32 = undefined;
        var rows: usize = 0;
        for (parts, 0..) |p, i| {
            if (p.ids.len == 0 or p.ids.len > window_rows or rows + p.ids.len > shared_rows) return error.WindowTooWide;
            for (parts[0..i]) |q| if (q.slot == p.slot) return error.SlotTwice;
            if (p.slot >= e.seqs.len) return error.Inkling2NoSlot;
            if (st.pos[p.slot] + p.ids.len + 1 > e.capacity) return error.ContextFull;
            for (p.ids, 0..) |t, r| ids[rows + r] = @intCast(t);
            items[i] = .{ .slot = p.slot, .ids = ids[rows..][0..p.ids.len] };
            rows += p.ids.len;
        }
        const f = e.forward(&e.bd, e.shape_d);
        const plan = try f.stage(st, items[0..parts.len]);
        const Body = struct {
            f: fw.Forward,
            st: *const fw.State,
            p: fw.Plan,
            fn run(ctx: *anyopaque, _: cuda.Stream) anyerror!void {
                const b: *@This() = @ptrCast(@alignCast(ctx));
                try b.f.launch(b.st, b.p);
            }
        };
        var body: Body = .{ .f = f, .st = st, .p = plan };
        const key: Key = .{ .n = @intCast(plan.n), .ni = @intCast(plan.ni), .nt = @intCast(plan.nt), .stride = plan.stride };
        const print = cuda.graph_cache.fingerprintOf(&.{ st.kc[0], e.bd.ids.ptr, e.bd.parts.o });
        _ = try e.graphs.run(key, e.stream, print, .{ .ctx = &body, .run = Body.run });
        fw.Forward.advance(st, items[0..parts.len]);
        var row: usize = 0;
        for (parts) |p| {
            e.slot = p.slot;
            try e.draw(&e.bd, row, p.ids.len, st.pos[p.slot] - p.ids.len + 1, e.seqs[p.slot].sampling, p.out[0..p.ids.len]);
            row += p.ids.len;
        }
    }

    /// Rows of this rank's logits (from row `row0` of `b`) to tokens at first, first + 1, ...: sampled or greedy.
    fn draw(e: *Engine, b: *fw.Buffers, row0: usize, rows: usize, first: u64, sm: ?sampler.Sampling, out: []u32) !void {
        const sh = e.w.share;
        const n = sh.vocab.len();
        const real = @min(n, @as(usize, e.c.real_vocab) -| sh.vocab.lo);
        const logits = b.logits.ptr + row0 * n * 4;
        if (sm) |s| {
            try ops.tobf(&e.m.ops, e.stream, logits, rows, n, real, e.smp.full.ptr, e.smp.vocab, sh.vocab.lo, n);
            try e.smp.draw(e.stream, e.exchange, e.spans, rows, first, sampler.rule(s), out);
            return e.gathered();
        }
        const stride = pickStride(rows);
        const vals = e.picks.ptr + sh.rank * stride;
        try ops.top1(&e.m.ops, e.stream, logits, rows, n, real, sh.vocab.lo, vals, vals + rows * 4);
        switch (e.exchange) {
            .local => {},
            .nccl => |x| if (x.group) |g| try g.allGather(e.stream, vals, e.picks.ptr, stride) else try x.lib.check(x.lib.api.ncclAllGather(vals, e.picks.ptr, stride, .u8, x.comm, e.stream.handle), "ncclAllGather"),
        }
        try e.stream.synchronize();
        try e.gathered();
        const bytes = e.picks_host[0 .. sh.world * stride];
        try e.picks.download(0, bytes);
        for (0..rows) |r| {
            var best: f32 = -std.math.inf(f32);
            var id: u32 = 0;
            for (0..sh.world) |q| {
                const at = q * stride;
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

    /// After a synchronize: a RoCE gather that failed since the last one (its rows hold nothing valid) is an error.
    fn gathered(e: *const Engine) !void {
        switch (e.exchange) {
            .local => {},
            .nccl => |x| if (x.group) |g| try g.check(),
        }
    }

    /// Bytes a kept state takes on this rank: the slot's sliding rings and conv states (global rows stay in place).
    pub fn stateBytes(e: *const Engine) u64 {
        return kept.bytes(&e.c, &e.st.?, e.convBytes());
    }

    /// The bound slot's state at its length kept under `id` (every rank keeps the same id).
    pub fn keepAs(e: *Engine, id: u64, at: usize) !void {
        if (at == 0 or at != e.pos()) return error.BadKeep;
        try e.states.save(e.d, e.stream, &e.c, &e.st.?, e.slot, id, at, e.convBytes());
    }

    /// The bound slot restored to state `id` (it must be the state's slot, still holding its rows).
    pub fn restoreState(e: *Engine, id: u64) !void {
        try e.states.restore(e.d, e.stream, &e.c, &e.st.?, e.slot, id);
    }

    pub fn dropState(e: *Engine, id: u64) void {
        e.stream.synchronize() catch {};
        e.states.drop(id);
    }
};

/// A rank's greedy picks of `rows` rows (fp32 values, then i32 ids) padded to 16 bytes: the RoCE gather takes them.
fn pickStride(rows: usize) usize {
    return std.mem.alignForward(usize, rows * 8, 16);
}
