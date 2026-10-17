//! A native draft model borrows target bindings and owns its checkpoint, callback backend and committed context.
const std = @import("std");
const core = @import("core");
const mtl = @import("metal");
const cfg = @import("config.zig");
const ck = @import("../checkpoint.zig");
const weights = @import("weights.zig");
const Backend = @import("runtime_backend.zig").Backend;
const Target = @import("../model.zig").Model;
const Session = @import("session.zig").Session;
const op = @import("operators.zig");
const execution = @import("execution.zig");
const selector = @import("selector.zig");
const backend_mod = @import("runtime_backend.zig");
const Cache = backend_mod.Cache;
const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
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
    pub fn load(a: std.mem.Allocator, io: std.Io, target: *Target, dir: []const u8, mode: op.Mode) !*Model {
        if (mode == .python_q4g64) return error.DraftPreparationUnqualified;
        const path = try std.fs.path.join(a, &.{ dir, "config.json" });
        defer a.free(path);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20));
        defer a.free(text);
        const c = try cfg.parse(a, text);
        if (target.config.hidden != c.hidden or target.config.vocab != c.vocab or target.config.layers != c.target_layers) return error.TargetBinding;
        const shards = try core.checkpoint_host.shardFiles(a, io, dir);
        defer core.checkpoint_host.freeShardFiles(a, shards);
        return open(a, target, c, shards, mode);
    }
    /// The draft model of config `c` from its checkpoint `shards`, prepared for `mode`.
    pub fn open(a: std.mem.Allocator, target: *Target, c: cfg.Config, shards: []const [:0]const u8, mode: op.Mode) !*Model {
        if (mode == .python_q4g64) return error.DraftPreparationUnqualified;
        const model = try a.create(Model);
        errdefer a.destroy(model);
        model.allocator = a;
        model.tree_block = 16;
        model.committed_end = 0;
        model.checkpoint = core.checkpoint_metal.Checkpoint.init(a);
        errdefer model.checkpoint.deinit();
        // q4 drafts read their linears only through prepared copies: those stay in the file until prepare reads them
        const later: ?core.checkpoint_metal.Later = if (mode == .prepared_q4_reference) Backend.linear else null;
        for (shards) |file| try model.checkpoint.addFileSplit(target.device, file, "", null, later);
        model.backend = try Backend.init(a, target, &model.checkpoint, c);
        errdefer model.backend.deinit();
        model.graph = try loadGraph(a, &model.backend);
        errdefer model.graph.deinit();
        if (mode == .prepared_q4_reference) try model.backend.prepare();
        model.checkpoint.closeFiles();
        model.pending = try target.device.buffer(@as(usize, c.window) * c.tapWidth() * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer model.pending.deinit();
        model.session = try Session.init(c, 1, 0);
        model.preparation = .{ .mode = mode, .quantization_verified = mode == .prepared_q4_reference, .source_sha256 = if (mode == .prepared_q4_reference) core.affine4.signature() else @splat(0) };
        return model;
    }
    pub fn deinit(m: *Model) void {
        m.graph.deinit();
        m.backend.deinit();
        m.pending.deinit();
        m.checkpoint.deinit();
        m.allocator.destroy(m);
    }
    /// After a GPU failure: the frame and any open transaction are dropped, and the context starts empty.
    pub fn recover(m: *Model) !void {
        m.backend.frame.release();
        m.backend.transaction = null;
        m.backend.staged = null;
        m.backend.failed = false;
        try m.reset();
    }
    pub fn reset(m: *Model) !void {
        if (m.backend.failed or m.backend.frame.active or m.backend.transaction != null) return error.BadDraftContext;
        m.backend.context = .{ .cache = 1, .begin = 0, .end = 0 };
        m.backend.frame.external = null;
        m.session = try Session.init(m.graph.config, 1, 0);
        m.committed_end = 0;
    }
    pub fn absorb(m: *Model, taps: @import("runtime_backend.zig").Ref, rows: u32) !void {
        if (m.backend.failed or m.backend.frame.active or m.backend.transaction != null or rows == 0 or rows > 128) return error.BadCommittedTaps;
        try m.ring().absorb(taps.buf, taps.off, rows);
    }
    /// The committed taps the drafter reads, as a ring a lane's state copies.
    pub fn ring(m: *Model) @import("../tap_ring.zig").Ring {
        return .{ .buf = m.pending, .end = &m.committed_end, .window = m.graph.config.window, .stride = @as(usize, m.graph.config.tapWidth()) * 2 };
    }
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
    pub fn setTreeBlock(m: *Model, block: u32) !void {
        if (block == 0 or block > 16 or m.backend.failed or m.backend.frame.active or m.backend.transaction != null) return error.BadRuntimeBlock;
        m.tree_block = block;
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
    /// One stream of a batch: its parked context (null: the model's own), pending token and node budget.
    pub const Proposal = struct { context: ?*Context, pending: u32, nodes: u32 };
    /// Trees for 2 to 8 streams from one frame: each stream's new taps into its own caches, then one block pass.
    pub fn proposeBatch(m: *Model, entries: []const Proposal, out: []selector.Tree) !void {
        const n = entries.len;
        if (n < 2 or n > 8 or out.len != n or m.tree_block < 2) return error.BadDraftOptions;
        const c = m.graph.config;
        var streams: [8]execution.Stream = undefined;
        var caches: [8][]Cache = undefined;
        var news: [8]?Fresh = @splat(null);
        for (entries, 0..) |en, k| {
            if (en.pending >= c.vocab or en.nodes == 0 or en.nodes > 15) return error.BadDraftOptions;
            if (en.context) |x| try m.swap(x);
            news[k] = m.fresh();
            const flushed: anyerror!void = if (news[k] == null) m.flush() else {};
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
        defer m.backend.batch = &.{};
        // the block spans the deepest tree asked for: drafter rows cost even when a stream's tree is small
        var deepest: u32 = 1;
        for (entries) |en| deepest = @max(deepest, en.nodes);
        const block = @min(m.tree_block, deepest + 1);
        const ops = m.backend.ops();
        const frame = try ops.vtable.begin(ops.ptr);
        defer ops.vtable.release(ops.ptr, frame);
        var absorbs: [8]?execution.Absorb = @splat(null);
        var positions: [8][128]u64 = undefined;
        const stride = @as(usize, c.tapWidth()) * 2;
        for (news[0..n], absorbs[0..n], 0..) |f, *a, k| if (f) |x| {
            for (positions[k][0..x.rows], 0..) |*p, i| p.* = x.first + i;
            a.* = .{ .taps = try m.backend.frame.add(.{ .buf = x.ring, .off = @as(usize, x.slot) * stride }, x.rows, c.tapWidth(), c.tapWidth(), .bf16), .positions = positions[k][0..x.rows] };
        };
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
        var made: usize = 0;
        errdefer for (out[0..made]) |*t| t.deinit();
        for (entries, streams[0..n], lattices[0..n], out) |en, s2, l, *t| {
            t.* = try selector.propose(m.allocator, &m.graph, l, en.pending, s2.context.end + 1, .{ .nodes = en.nodes });
            made += 1;
        }
    }
    /// A short run of committed taps the session lacks, for a batched frame to absorb (null: none, or flush first).
    const Fresh = struct { ring: mtl.Buffer, slot: u32, rows: u32, first: u64, next: op.Context };
    fn fresh(m: *const Model) ?Fresh {
        const window = m.graph.config.window;
        if (!m.session.ready or m.committed_end <= m.session.context.end) return null;
        const behind = m.committed_end - m.session.context.end;
        const slot: u32 = @intCast(m.session.context.end % window);
        if (behind > 128 or slot + behind > window or m.session.context.end < m.committed_end -| (@as(u64, window) - 1)) return null;
        const begin = @max(m.session.context.begin, m.committed_end -| (@as(u64, window) - 1));
        return .{ .ring = m.pending, .slot = slot, .rows = @intCast(behind), .first = m.session.context.end, .next = .{ .cache = m.session.context.cache, .begin = begin, .end = m.committed_end } };
    }
    pub fn propose(m: *Model, pending: u32, nodes: u32) !@import("selector.zig").Tree {
        if (m.backend.failed or m.backend.frame.active or m.backend.transaction != null or pending >= m.graph.config.vocab or nodes > 15 or m.tree_block == 0 or m.tree_block > 16) return error.BadDraftOptions;
        if (m.committed_end == 0) return error.DraftContextNotReady;
        const block = @min(m.tree_block, nodes + 1);
        if (block < 2) return .{ .gpa = m.allocator, .tokens = &.{}, .parents = &.{}, .scores = &.{} };
        const window = m.graph.config.window;
        const behind = m.committed_end - m.session.context.end;
        const slot: u32 = @intCast(m.session.context.end % window);
        if (m.session.ready and behind > 0 and behind <= 128 and slot + behind <= window and m.session.context.end >= m.committed_end -| (@as(u64, window) - 1)) {
            const rows: u32 = @intCast(behind);
            var positions: [128]u64 = undefined;
            for (positions[0..rows], 0..) |*position, i| position.* = m.session.context.end + i;
            const stride = @as(usize, m.graph.config.tapWidth()) * 2;
            const view = try m.backend.frame.borrow(.{ .buf = m.pending, .off = @as(usize, slot) * stride }, rows, m.graph.config.tapWidth());
            return m.session.absorbTreeBlock(m.allocator, m.backend.ops(), &m.graph, m.preparation, .{ .values = view, .positions = positions[0..rows], .layers = m.graph.config.taps, .target_end = m.session.context.end + rows }, pending, block, .{ .nodes = nodes });
        }
        try m.flush();
        const first = try m.session.firstDraw();
        return m.session.treeBlock(m.allocator, m.backend.ops(), &m.graph, m.preparation, pending, first, block, .{ .nodes = nodes });
    }
};
/// A stream's own drafter state, parked while another stream drafts: its taps ring, rolling K/V and session.
pub const Context = struct {
    pending: mtl.Buffer,
    committed_end: u64 = 0,
    caches: []Cache,
    session: Session,
    context: op.Context = .{ .cache = 1, .begin = 0, .end = 0 },

    /// `drafting`: the stream drafts its own trees, so it gets K/V caches; else only its ring.
    pub fn init(a: std.mem.Allocator, m: *const Model, drafting: bool) !*Context {
        const c = m.graph.config;
        const device = m.backend.target.device;
        const x = try a.create(Context);
        errdefer a.destroy(x);
        const pending = try device.buffer(@as(usize, c.window) * c.tapWidth() * 2, opts);
        errdefer pending.deinit();
        const caches = try backend_mod.newCaches(a, device, c, if (drafting) c.layers else 0);
        x.* = .{ .pending = pending, .caches = caches, .session = try Session.init(c, 1, 0) };
        return x;
    }
    pub fn deinit(x: *Context, a: std.mem.Allocator) void {
        backend_mod.freeCaches(a, x.caches);
        x.pending.deinit();
        a.destroy(x);
    }
    /// Empty, as Model.reset leaves the model's own.
    pub fn reset(x: *Context, c: cfg.Config) !void {
        x.committed_end = 0;
        x.session = try Session.init(c, 1, 0);
        x.context = .{ .cache = 1, .begin = 0, .end = 0 };
    }
    /// Its taps ring, where the lane backend absorbs the stream's kept rows while it is parked.
    pub fn ring(x: *Context, c: cfg.Config) @import("../tap_ring.zig").Ring {
        return .{ .buf = x.pending, .end = &x.committed_end, .window = c.window, .stride = @as(usize, c.tapWidth()) * 2 };
    }
    /// Its buffers, for a residency set.
    pub fn buffers(x: *const Context, list: *std.ArrayList(mtl.Buffer), a: std.mem.Allocator) !void {
        try list.append(a, x.pending);
        for (x.caches) |cache| try list.appendSlice(a, &.{ cache.keys, cache.values });
    }
};

pub fn loadGraph(a: std.mem.Allocator, b: *Backend) !weights.Graph {
    const target = b.target;
    const prefix: []const u8 = if (target.checkpoint.has("language_model.model.embed_tokens.weight")) "language_model." else "";
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const local = arena.allocator();
    const embed = try binding(local, target, prefix, "model.embed_tokens", .{ .bits = target.weights.embed.bits, .group_size = target.weights.embed.group });
    var names: std.ArrayList([]const u8) = .empty;
    var iterator = b.checkpoint.tensors.keyIterator();
    while (iterator.next()) |name| try names.append(local, name.*);
    var later = b.checkpoint.deferred.keyIterator();
    while (later.next()) |name| try names.append(local, name.*);
    return weights.load(a, .{ .source = b.source(), .names = names.items }, b.config, embed);
}
fn binding(a: std.mem.Allocator, target: *Target, prefix: []const u8, module: []const u8, format: @import("../affine.zig").Spec) !ck.Linear {
    const stem = try std.mem.concat(a, u8, &.{ prefix, module });
    const w = Backend.cpu(try target.checkpoint.get(try std.mem.concat(a, u8, &.{ stem, ".weight" })));
    const s = Backend.cpu(try target.checkpoint.get(try std.mem.concat(a, u8, &.{ stem, ".scales" })));
    const b = Backend.cpu(try target.checkpoint.get(try std.mem.concat(a, u8, &.{ stem, ".biases" })));
    return ck.validate(w, s, b, format, target.config.vocab, target.config.hidden);
}
