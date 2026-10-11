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
        const model = try a.create(Model);
        errdefer a.destroy(model);
        model.allocator = a;
        model.tree_block = 16;
        model.committed_end = 0;
        model.checkpoint = core.checkpoint_metal.Checkpoint.init(a);
        errdefer model.checkpoint.deinit();
        const shards = try core.checkpoint_host.shardFiles(a, io, dir);
        defer core.checkpoint_host.freeShardFiles(a, shards);
        for (shards) |file| try model.checkpoint.addFile(target.device, file, "");
        model.backend = try Backend.init(a, target, &model.checkpoint, c);
        errdefer model.backend.deinit();
        model.graph = try loadGraph(a, &model.backend);
        errdefer model.graph.deinit();
        if (mode == .prepared_q4_reference) try model.backend.prepare();
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
pub fn loadGraph(a: std.mem.Allocator, b: *Backend) !weights.Graph {
    const target = b.target;
    const prefix: []const u8 = if (target.checkpoint.has("language_model.model.embed_tokens.weight")) "language_model." else "";
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const local = arena.allocator();
    const embed = try binding(local, target, prefix, "model.embed_tokens");
    const head = try binding(local, target, prefix, "lm_head");
    var names: std.ArrayList([]const u8) = .empty;
    var iterator = b.checkpoint.tensors.keyIterator();
    while (iterator.next()) |name| try names.append(local, name.*);
    return weights.load(a, .{ .source = b.source(), .names = names.items }, b.config, embed, head);
}
fn binding(a: std.mem.Allocator, target: *Target, prefix: []const u8, module: []const u8) !ck.Linear {
    const stem = try std.mem.concat(a, u8, &.{ prefix, module });
    const w = Backend.cpu(try target.checkpoint.get(try std.mem.concat(a, u8, &.{ stem, ".weight" })));
    const s = Backend.cpu(try target.checkpoint.get(try std.mem.concat(a, u8, &.{ stem, ".scales" })));
    const b = Backend.cpu(try target.checkpoint.get(try std.mem.concat(a, u8, &.{ stem, ".biases" })));
    return ck.validate(w, s, b, .{ .bits = 4, .group_size = 64 }, target.config.vocab, target.config.hidden);
}
