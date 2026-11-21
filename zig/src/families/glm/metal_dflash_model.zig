//! A native GLM draft model: incoai's drafter checkpoint, the shared frame backend and its committed context.
const std = @import("std");
const core = @import("core");
const mtl = @import("metal");
const cfg = @import("../qwen27/dflash/config.zig");
const weights = @import("metal_dflash_weights.zig");
const draft_cfg = @import("metal_dflash_config.zig");
const backend_mod = @import("metal_dflash_backend.zig");
const Backend = backend_mod.Backend;
const Target = backend_mod.Target;
const Session = @import("../qwen27/dflash/session.zig").Session;
const op = @import("../qwen27/dflash/operators.zig");
const selector = @import("../qwen27/dflash/selector.zig");
const catchup = @import("../qwen27/dflash/catchup.zig");
const execution = @import("../qwen27/dflash/execution.zig");
const Cache = backend_mod.Cache;
const prepared = @import("metal_dflash_prepared.zig");
const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
/// --drafter-bits: 0 reads the bf16 weights as published, 4 quantizes each linear at load.
pub const Bits = enum { bf16, q4 };
pub const max_nodes = 63; // selected tree nodes, independent of the unchanged 16-row drafter block
/// The widest tap span one catch-up frame absorbs: a whole 2,048 ring (the positions buffer's size).
const catchup_cap = 2048;
pub const Model = struct {
    allocator: std.mem.Allocator,
    checkpoint: core.checkpoint_metal.Checkpoint,
    backend: Backend,
    graph: weights.Graph,
    session: Session,
    preparation: op.Preparation,
    tree_block: u32 = 16,
    pending: mtl.Buffer,
    committed_end: u64 = 0,
    /// Device bytes the drafter's checkpoint buffers, caches and pending ring hold.
    bytes: usize = 0,
    /// The drafter of `dir`'s checkpoint against `target`'s embedding and head: bf16 as published, or 4-bit copies.
    pub fn load(a: std.mem.Allocator, io: std.Io, target: *Target, dir: []const u8, bits: Bits) !*Model {
        const path = try std.fs.path.join(a, &.{ dir, "config.json" });
        defer a.free(path);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20));
        defer a.free(text);
        const c = try draft_cfg.parse(a, text);
        try draft_cfg.bind(c, .{ .hidden = target.c.hidden, .vocab = target.c.vocab, .layers = target.c.layers });
        const shards = try core.checkpoint_host.shardFiles(a, io, dir);
        defer core.checkpoint_host.freeShardFiles(a, shards);
        return open(a, target, c, shards, bits);
    }
    /// The draft model of config `c` from its checkpoint `shards`; 4-bit linears stay in the file until copied.
    pub fn open(a: std.mem.Allocator, target: *Target, c: cfg.Config, shards: []const [:0]const u8, bits: Bits) !*Model {
        const model = try a.create(Model);
        errdefer a.destroy(model);
        model.allocator = a;
        model.tree_block = 16;
        model.committed_end = 0;
        model.checkpoint = core.checkpoint_metal.Checkpoint.init(a);
        errdefer model.checkpoint.deinit();
        const later: ?core.checkpoint_metal.Later = if (bits == .q4) prepared.linear else null;
        for (shards) |file| try model.checkpoint.addFileSplit(target.device, file, "", null, later);
        model.backend = try Backend.init(a, target, &model.checkpoint, c);
        errdefer model.backend.deinit();
        model.graph = try loadGraph(a, &model.backend);
        errdefer model.graph.deinit();
        if (bits == .q4) model.backend.copies = try prepared.Copies.prepare(a, target.device, &model.checkpoint, c.linearCount());
        model.checkpoint.closeFiles();
        model.pending = try target.device.buffer(@as(usize, c.window) * c.tapWidth() * 2, opts);
        errdefer model.pending.deinit();
        model.session = try Session.init(c, 1, 0);
        model.preparation = if (bits == .q4) .{ .mode = .prepared_q4_reference, .quantization_verified = true, .source_sha256 = core.affine4.signature() } else .{ .mode = .bf16_reference };
        model.bytes = model.checkpointBytes() + model.pending.length() + model.backendBytes();
        return model;
    }
    /// Device bytes the checkpoint's loaded shard buffers hold.
    fn checkpointBytes(m: *const Model) usize {
        var n: usize = 0;
        for (m.checkpoint.shards.items) |s| n += s.bytes;
        return n;
    }
    /// Device bytes the drafter's caches, scratch frame and 4-bit copies hold.
    fn backendBytes(m: *const Model) usize {
        var n: usize = 0;
        if (m.backend.copies) |c| n += c.bytes();
        for (m.backend.frame.pool.items) |b| n += b.length();
        for (m.backend.caches) |c| n += c.keys.length() + c.values.length();
        return n;
    }
    pub fn deinit(m: *Model) void {
        m.graph.deinit();
        m.backend.deinit();
        m.pending.deinit();
        m.checkpoint.deinit();
        m.allocator.destroy(m);
    }
    /// The model's own taps ring gets `rows` committed rows from `taps` (the synthetic check feeds it this way).
    pub fn absorb(m: *Model, taps: backend_mod.Ref, rows: u32) !void {
        if (m.backend.failed or m.backend.frame.active or m.backend.transaction != null or rows == 0 or rows > 128) return error.BadCommittedTaps;
        const ring: @import("../qwen27/tap_ring.zig").Ring = .{ .buf = m.pending, .end = &m.committed_end, .window = m.graph.config.window, .stride = @as(usize, m.graph.config.tapWidth()) * 2 };
        try ring.absorb(taps.buf, taps.off, rows);
    }
    /// Every committed tap the session lacks absorbed, 128 rows a frame, from where the window still holds them.
    pub fn flush(m: *Model) !void {
        if (m.backend.failed or m.backend.frame.active or m.backend.transaction != null or m.committed_end == 0 or m.session.context.end > m.committed_end) return error.BadDraftContext;
        const begin = m.committed_end -| (@as(u64, m.graph.config.window) - 1);
        if (m.session.context.end < begin) {
            m.session.context = .{ .cache = 1, .begin = begin, .end = begin };
            m.session.ready = false;
            m.backend.context = m.session.context;
        }
        const stride = @as(usize, m.graph.config.tapWidth()) * 2;
        while (m.session.context.end < m.committed_end) {
            const first = m.session.context.end;
            const slot: u32 = @intCast(first % m.graph.config.window);
            const rows: u32 = @intCast(@min(@min(m.committed_end - first, 128), m.graph.config.window - slot));
            var positions: [128]u64 = undefined;
            for (positions[0..rows], 0..) |*position, i| position.* = first + i;
            const view = try m.backend.frame.borrow(.{ .buf = m.pending, .off = @as(usize, slot) * stride }, rows, m.graph.config.tapWidth());
            try m.session.absorb(m.backend.ops(), &m.graph, m.preparation, .{ .values = view, .positions = positions[0..rows], .layers = m.graph.config.taps, .target_end = first + rows });
        }
    }
    /// Exchange the drafting state with `x` (no frame or transaction open): swap a stream in, draft, swap it back.
    pub fn swap(m: *Model, x: *Context) !void {
        if (m.backend.failed or m.backend.frame.active or m.backend.transaction != null or m.backend.staged != null or x.caches.len != m.backend.caches.len) return error.BadDraftContext;
        std.mem.swap(mtl.Buffer, &m.pending, &x.pending);
        std.mem.swap(u64, &m.committed_end, &x.committed_end);
        std.mem.swap([]Cache, &m.backend.caches, &x.caches);
        std.mem.swap(Session, &m.session, &x.session);
        std.mem.swap(op.Context, &m.backend.context, &x.context);
    }
    pub fn propose(m: *Model, pending: u32, nodes: u32) !selector.Tree {
        return m.proposeOptions(pending, .{ .nodes = nodes });
    }
    /// A tree from the masked block after `pending`, the taps the session lacks absorbed in the same frame.
    pub fn proposeOptions(m: *Model, pending: u32, options: selector.Options) !selector.Tree {
        const metric = mtl.pass_metrics.phase(.drafter);
        defer metric.end();
        const nodes = options.nodes;
        if (m.backend.failed or m.backend.frame.active or m.backend.transaction != null or pending >= m.graph.config.vocab or nodes > max_nodes or m.tree_block == 0 or m.tree_block > 16) return error.BadDraftOptions;
        if (m.committed_end == 0) return error.DraftContextNotReady;
        const block = @min(m.tree_block, nodes + 1);
        if (block < 2) return .{ .gpa = m.allocator, .tokens = &.{}, .parents = &.{}, .scores = &.{} };
        const window = m.graph.config.window;
        const stride = @as(usize, m.graph.config.tapWidth()) * 2;
        if (m.session.ready and m.session.context.end >= m.committed_end -| (@as(u64, window) - 1) and m.committed_end > m.session.context.end and window <= catchup_cap) {
            // the whole backlog sits contiguous in the ring: absorb it and draft the block in one frame
            const end = m.session.context.end;
            const behind: u32 = @intCast(m.committed_end - end);
            const slot: u32 = @intCast(end % window);
            if (slot + behind <= window) {
                var positions: [catchup_cap]u64 = undefined;
                for (positions[0..behind], 0..) |*position, i| position.* = end + i;
                const view = try m.backend.frame.borrow(.{ .buf = m.pending, .off = @as(usize, slot) * stride }, behind, m.graph.config.tapWidth());
                return m.session.absorbTreeBlock(m.allocator, m.backend.ops(), &m.graph, m.preparation, .{ .values = view, .positions = positions[0..behind], .layers = m.graph.config.taps, .target_end = m.committed_end }, pending, block, options);
            }
        }
        if (catchup.fresh(m.session.ready, m.session.context.begin, m.session.context.end, m.committed_end, window)) |span| {
            const rows = span.rows;
            var positions: [128]u64 = undefined;
            for (positions[0..rows], 0..) |*position, i| position.* = m.session.context.end + i;
            const view = try m.backend.frame.borrow(.{ .buf = m.pending, .off = @as(usize, span.slot) * stride }, rows, m.graph.config.tapWidth());
            return m.session.absorbTreeBlock(m.allocator, m.backend.ops(), &m.graph, m.preparation, .{ .values = view, .positions = positions[0..rows], .layers = m.graph.config.taps, .target_end = m.session.context.end + rows }, pending, block, options);
        }
        try m.flush();
        const first = try m.session.firstDraw();
        return m.session.treeBlock(m.allocator, m.backend.ops(), &m.graph, m.preparation, pending, first, block, options);
    }
    /// One stream of a batched pass: its parked context (null: the model's own), pending token and tree options.
    pub const Proposal = struct { context: ?*Context, pending: u32, options: selector.Options };

    /// The block rows a stream's tree of `nodes` drafts takes: a batch's streams share one block size.
    pub fn blockFor(m: *const Model, nodes: u32) u32 {
        return @min(m.tree_block, nodes + 1);
    }

    /// Trees for 2 to 8 streams of one block size from one frame: their new taps, then one pass over all their rows.
    pub fn proposeBatch(m: *Model, entries: []const Proposal, out: []selector.Tree) !void {
        return m.proposeBatchOps(m.backend.ops(), entries, out);
    }
    /// proposeBatch through `ops`: the backend's own, or a wrapper over them (the speed harness times each call).
    pub fn proposeBatchOps(m: *Model, ops: op.Ops, entries: []const Proposal, out: []selector.Tree) !void {
        const metric = mtl.pass_metrics.phase(.drafter);
        defer metric.end();
        const n = entries.len;
        if (n < 2 or n > 8 or out.len != n or m.backend.failed or m.backend.frame.active or m.backend.transaction != null) return error.BadDraftOptions;
        const c = m.graph.config;
        const block = m.blockFor(entries[0].options.nodes);
        if (block < 2 or n * block > 128) return error.BadDraftOptions;
        var streams: [8]execution.Stream = undefined;
        var caches: [8][]Cache = undefined;
        var news: [8]?Fresh = @splat(null);
        for (entries, 0..) |en, k| {
            if (en.pending >= c.vocab or en.options.nodes > max_nodes or m.blockFor(en.options.nodes) != block) return error.BadDraftOptions;
            if (en.context) |x| try m.swap(x);
            const ready = m.committed_end > 0;
            news[k] = if (ready) m.fresh() else null;
            const flushed: anyerror!void = if (!ready) error.DraftContextNotReady else if (news[k] == null) m.flush() else {};
            if (en.context) |x| m.swap(x) catch {
                m.backend.failed = true; // the model holds this stream's context: recovery resets both
                return error.DraftContextSwap;
            };
            try flushed;
            const current = if (en.context) |x| x.session.context else m.session.context;
            streams[k] = .{ .context = if (news[k]) |f| f.next else current, .anchor = en.pending };
            caches[k] = if (en.context) |x| x.caches else m.backend.caches;
        }
        m.backend.batch = caches[0..n];
        m.backend.batch_block = block;
        defer {
            m.backend.batch = &.{};
            m.backend.batch_block = 0;
        }
        const frame = try ops.vtable.begin(ops.ptr);
        defer ops.vtable.release(ops.ptr, frame);
        try m.absorbAll(ops, frame, news[0..n]);
        const absorbs: [8]?execution.Absorb = @splat(null); // absorbed above, every stream's taps in one product
        var lattices: [8]op.Lattice = undefined;
        try execution.absorbBlocks(ops, &m.graph, m.preparation, frame, absorbs[0..n], streams[0..n], block, lattices[0..n]);
        defer for (lattices[0..n]) |*l| l.deinit();
        // the frame completed, so an absorbing stream's context now covers its new taps
        for (entries, news[0..n]) |en, f| if (f) |x| {
            if (en.context) |ctx| {
                ctx.session.context = x.next;
                ctx.session.ready = true;
                ctx.context = x.next;
            } else {
                m.session.context = x.next;
                m.session.ready = true;
                m.backend.context = x.next;
            }
        };
        const select = mtl.pass_metrics.phase(.selector);
        defer select.end();
        var made: usize = 0;
        errdefer for (out[0..made]) |*t| t.deinit();
        for (entries, streams[0..n], lattices[0..n], out) |en, s2, l, *t| {
            t.* = try selector.propose(m.allocator, &m.graph, l, en.pending, s2.context.end + 1, en.options);
            made += 1;
        }
    }
    /// Every stream's new taps through the fusion and each layer's keys and values, one product each, into its caches.
    fn absorbAll(m: *Model, ops: op.Ops, frame: u64, news: []const ?Fresh) !void {
        const c = m.graph.config;
        const g = &m.graph;
        var total: u32 = 0;
        for (news) |f| if (f) |x| {
            total += x.rows;
        };
        if (total == 0) return;
        const b = &m.backend;
        var positions: [8 * 128]u64 = undefined;
        const taps = try b.frame.alloc(total, c.tapWidth(), .bf16);
        const to = try b.frame.ref(taps);
        const e = try b.encoder(frame);
        const stride = @as(usize, c.tapWidth()) * 2;
        var at: u32 = 0;
        for (news) |f| if (f) |x| {
            try b.copy(e, .{ .buf = x.ring, .off = @as(usize, x.slot) * stride }, .{ .buf = to.buf, .off = to.off + @as(usize, at) * stride }, x.rows, c.tapWidth());
            for (positions[at..][0..x.rows], 0..) |*p, i| p.* = x.first + i;
            at += x.rows;
        };
        const pos = positions[0..total];
        const mode = m.preparation.mode;
        const fusion = try ops.vtable.project(ops.ptr, frame, taps, g.fusion, mode);
        const hidden = try ops.vtable.norm(ops.ptr, frame, fusion, .{ .weight = g.hidden_norm, .eps = c.eps, .heads = 1, .head_dim = c.hidden });
        for (g.layers, 0..) |layer, i| {
            const raw = try ops.vtable.project(ops.ptr, frame, hidden, layer.attention.k, mode);
            const normed = try ops.vtable.norm(ops.ptr, frame, raw, .{ .weight = layer.attention.k_norm, .eps = c.eps, .heads = c.kv_heads, .head_dim = c.head_dim });
            const keys = try ops.vtable.rope(ops.ptr, frame, normed, .{ .positions = pos, .heads = c.kv_heads, .dim = c.head_dim, .theta = c.theta });
            const values = try ops.vtable.project(ops.ptr, frame, hidden, layer.attention.v, mode);
            var first: u32 = 0;
            for (news, 0..) |f, k| if (f) |x| {
                const kr = try b.frame.ref(keys);
                const vr = try b.frame.ref(values);
                const ks = try b.frame.add(.{ .buf = kr.buf, .off = kr.off + @as(usize, first) * keys.stride * 2 }, x.rows, keys.width, keys.stride, .bf16);
                const vs = try b.frame.add(.{ .buf = vr.buf, .off = vr.off + @as(usize, first) * values.stride * 2 }, x.rows, values.width, values.stride, .bf16);
                try ops.vtable.stream_append(ops.ptr, frame, @intCast(k), @intCast(i), ks, vs, pos[first..][0..x.rows]);
                first += x.rows;
            };
        }
    }
    /// A short run of committed taps the session lacks, for a batched frame to absorb (null: none, or flush first).
    const Fresh = struct { ring: mtl.Buffer, slot: u32, rows: u32, first: u64, next: op.Context };
    fn fresh(m: *const Model) ?Fresh {
        const span = catchup.fresh(m.session.ready, m.session.context.begin, m.session.context.end, m.committed_end, m.graph.config.window) orelse return null;
        return .{ .ring = m.pending, .slot = span.slot, .rows = span.rows, .first = span.first, .next = .{ .cache = m.session.context.cache, .begin = span.begin, .end = m.committed_end } };
    }
};
/// A stream's own drafter state, parked while another stream drafts: its taps ring, rolling K/V and session.
pub const Context = struct {
    pending: mtl.Buffer,
    committed_end: u64 = 0,
    caches: []Cache,
    session: Session,
    context: op.Context = .{ .cache = 1, .begin = 0, .end = 0 },

    /// The stream's taps ring and every layer's rolling K/V, empty.
    pub fn init(a: std.mem.Allocator, m: *const Model) !*Context {
        const c = m.graph.config;
        const device = m.backend.target.device;
        const x = try a.create(Context);
        errdefer a.destroy(x);
        const pending = try device.buffer(@as(usize, c.window) * c.tapWidth() * 2, opts);
        errdefer pending.deinit();
        const caches = try backend_mod.newCaches(a, device, c, c.layers);
        errdefer backend_mod.freeCaches(a, caches);
        x.* = .{ .pending = pending, .caches = caches, .session = try Session.init(c, 1, 0) };
        return x;
    }
    pub fn deinit(x: *Context, a: std.mem.Allocator) void {
        backend_mod.freeCaches(a, x.caches);
        x.pending.deinit();
        a.destroy(x);
    }
    /// Empty, starting at `at`: the parked context and its session agree there, so the swapped-in backend accepts it.
    pub fn reset(x: *Context, c: cfg.Config, at: u64) !void {
        x.committed_end = at;
        x.session = try Session.init(c, 1, 0);
        x.session.context = .{ .cache = 1, .begin = at, .end = at };
        x.session.ready = false;
        x.context = x.session.context;
    }
};

/// The drafter graph from the backend's checkpoint; the embedding record is the target's table's shape alone.
pub fn loadGraph(a: std.mem.Allocator, b: *Backend) !weights.Graph {
    var names: std.ArrayList([]const u8) = .empty;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const local = arena.allocator();
    var iterator = b.checkpoint.tensors.keyIterator();
    while (iterator.next()) |name| try names.append(local, name.*);
    var later = b.checkpoint.deferred.keyIterator();
    while (later.next()) |name| try names.append(local, name.*);
    return weights.load(a, .{ .source = b.source(), .names = names.items }, b.config, weights.embedBinding(b.config));
}
