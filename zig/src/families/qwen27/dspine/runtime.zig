//! DSpine on Metal: its checkpoint, prepared linears, rolling K/V caches, bridge weights and TG table, on the target.
const std = @import("std");
const core = @import("core");
const mtl = @import("metal");
const ck = @import("../checkpoint.zig");
const config = @import("config.zig");
const weights = @import("weights.zig");
const Kernels = @import("kernels.zig").Kernels;
const Frame = @import("../dflash/runtime_frame.zig").Frame;
const Tiled = @import("../dflash/runtime_backend.zig").Tiled;
const Target = @import("../model.zig").Model;
const Tree = @import("../dflash/selector.zig").Tree;
pub const Ref = core.draft_ops.Ref;
const simd = core.row_projection.simd;
const options = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

/// The linears: all affine-4 (the default), only the body's (the bridge BF16 as dspine_mlx keeps it), or BF16.
pub const Mode = enum { q4, q4_body, bf16 };
/// The bridge's linears stay BF16, as dspine_mlx keeps them; `q4` prepares only the body's.
pub fn backbone(name: []const u8, rank: usize) bool {
    return rank == 2 and std.mem.endsWith(u8, name, ".weight") and !std.mem.startsWith(u8, name, "selected_prefix_bridge.");
}

pub const Cache = struct { keys: mtl.Buffer, values: mtl.Buffer };

pub const Model = struct {
    allocator: std.mem.Allocator,
    target: *Target,
    checkpoint: core.checkpoint_metal.Checkpoint,
    graph: weights.Graph,
    mode: Mode,
    primitives: core.draft_ops.Ops,
    attention: core.shared_attention.Attention,
    topk: core.bf16_topk_gpu.Ops,
    kernels: Kernels,
    frame: Frame,
    caches: []Cache,
    /// [3 rank, code]: gate, predecessor_gate and prev stacked, as dspine_mlx's packed_weight.
    packed_w: mtl.Buffer,
    /// TG [vocab, rank] bf16: the final write's projection of each head row through the final norm.
    tg: mtl.Buffer,
    /// The committed context the caches hold: absolute rows [begin, end).
    begin: u64 = 0,
    committed_end: u64 = 0,
    broken: bool = false,
    residency: ?mtl.ResidencySet.Follow = null,
    prepared: std.AutoHashMapUnmanaged(usize, core.affine4_lane.Adapter) = .empty,
    tiled: std.AutoHashMapUnmanaged(usize, Tiled) = .empty,
    simd: ?simd.Pipelines = null,
    /// The target's LM head in its MLX layout, read again from its files: the base dequantizes candidates' rows.
    head_copy: ?core.checkpoint_metal.Checkpoint = null,
    head_rows: ?@import("kernels.zig").Head = null,

    pub fn shape(m: *const Model) config.Shape {
        return m.graph.shape;
    }

    pub fn load(a: std.mem.Allocator, io: std.Io, target: *Target, dir: []const u8, mode: Mode, target_dir: ?[]const u8) !*Model {
        const path = try std.fs.path.join(a, &.{ dir, "config.json" });
        defer a.free(path);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20));
        defer a.free(text);
        const s = try config.parse(a, text);
        try config.bind(s, target.config.hidden, target.config.vocab, target.config.layers);
        const shards = try core.checkpoint_host.shardFiles(a, io, dir);
        defer core.checkpoint_host.freeShardFiles(a, shards);
        const m = try a.create(Model);
        errdefer a.destroy(m);
        m.* = .{ .allocator = a, .target = target, .checkpoint = core.checkpoint_metal.Checkpoint.init(a), .graph = undefined, .mode = mode, .primitives = undefined, .attention = undefined, .topk = undefined, .kernels = undefined, .frame = .{ .allocator = a, .device = target.device, .queue = target.queue }, .caches = &.{}, .packed_w = undefined, .tg = undefined };
        errdefer m.checkpoint.deinit();
        const later: ?core.checkpoint_metal.Later = if (mode != .bf16) backbone else null;
        for (shards) |file| try m.checkpoint.addFileSplit(target.device, file, "", null, later);
        m.primitives = try core.draft_ops.Ops.init(target.device);
        errdefer m.primitives.deinit();
        m.attention = try core.shared_attention.Attention.init(a, target.device, .{ .query_heads = @intCast(s.heads), .kv_heads = @intCast(s.kv_heads), .head_dim = @intCast(s.head_dim), .rotary_dim = @intCast(s.head_dim), .capacity = @intCast(s.window + s.block), .output = .bf16 });
        errdefer m.attention.deinit();
        m.topk = try core.bf16_topk_gpu.Ops.init(target.device);
        errdefer m.topk.deinit();
        m.kernels = try Kernels.init(target.device);
        errdefer m.kernels.deinit();
        m.graph = try loadGraph(a, m, s);
        errdefer m.graph.deinit();
        errdefer m.dropPrepared();
        if (mode != .bf16) try m.prepare();
        m.checkpoint.closeFiles();
        m.caches = try newCaches(a, target.device, s);
        errdefer freeCaches(a, m.caches);
        m.packed_w = try m.packWeights();
        errdefer m.packed_w.deinit();
        if (mode == .q4) try m.prepareBridge();
        m.tg = try m.buildTg();
        errdefer m.tg.deinit();
        try m.copyHead(io, target_dir orelse return error.DSpineNeedsTargetDir);
        return m;
    }

    fn dropPrepared(m: *Model) void {
        var prepared = m.prepared.valueIterator();
        while (prepared.next()) |adapter| adapter.deinit();
        m.prepared.deinit(m.allocator);
        var tiled = m.tiled.valueIterator();
        while (tiled.next()) |t| t.deinit();
        m.tiled.deinit(m.allocator);
        if (m.simd) |*p| p.deinit();
    }

    pub fn deinit(m: *Model) void {
        if (m.head_copy) |*c| c.deinit();
        m.frame.deinit();
        m.dropPrepared();
        freeCaches(m.allocator, m.caches);
        m.tg.deinit();
        m.packed_w.deinit();
        m.graph.deinit();
        m.kernels.deinit();
        m.topk.deinit();
        m.attention.deinit();
        m.primitives.deinit();
        m.checkpoint.deinit();
        m.allocator.destroy(m);
    }

    /// Generation's drafter interface (dflash/generation.zig GenerationOf): DSpine reads `target`'s taps.
    pub fn binds(m: *const Model, target: *const Target) bool {
        return m.target == target and target.config.hidden == m.graph.shape.hidden and target.config.layers == m.graph.shape.target_layers;
    }
    pub fn tapIds(m: *const Model) []const u32 {
        return m.graph.shape.tapIds();
    }
    pub fn tapWidth(m: *const Model) usize {
        return config.tapWidth(m.graph.shape);
    }
    pub fn failed(m: *const Model) bool {
        return m.broken;
    }
    pub fn idle(m: *const Model) bool {
        return !m.broken and !m.frame.active;
    }
    /// Committed taps [rows, taps * hidden] into the context, at once.
    pub fn absorb(m: *Model, taps: Ref, rows: u32) !void {
        return @import("forward.zig").append(m, taps, rows);
    }
    /// A best-first tree of `nodes` drafts by the shared walker, the table times each row's norm scale as logits.
    pub fn propose(m: *Model, pending: u32, nodes: u32) !Tree {
        const dt = @import("core").lanes.draft_tree;
        const p = try @import("forward.zig").propose(m, pending);
        var tokens: [64]u32 = undefined;
        var parents: [64]i32 = undefined;
        var own: [64]f64 = undefined;
        const want = @min(nodes, tokens.len);
        var n: usize = 0;
        if (try @import("sys").environment.read(bool, "TF_DSPINE_CHAIN", false)) { // the greedy chain alone
            n = @min(want, p.drafts.len);
            @memcpy(tokens[0..n], p.drafts[0..n]);
            for (parents[0..n], 0..) |*v, i| v.* = @as(i32, @intCast(i)) - 1;
            @memset(own[0..n], 0);
        } else {
            // each row: its 16 ids ascending, then 16 unused unary logits (the table scores every candidate)
            var words: [15 * 32]u32 = @splat(0);
            var logits: [15][16][16]f32 = undefined;
            for (0..15) |r| {
                @memcpy(words[r * 32 ..][0..16], &p.candidates[r]);
                for (0..16) |s| for (0..16) |j| {
                    logits[r][s][j] = p.table[r][s][j] * p.scales[r];
                };
            }
            const c: dt.Candidates = .{ .words = std.mem.sliceAsBytes(&words), .projected = std.mem.asBytes(&logits), .topk = 16, .rank = 0, .predecessor = &.{}, .successor = &.{}, .table = true, .layout = .{ .words = 32, .ids = 0, .logits = 16 } };
            n = dt.grow(64, c, pending, 15, want, tokens[0..want], parents[0..want], own[0..want]);
        }
        const out_tokens = try m.allocator.dupe(u32, tokens[0..n]);
        errdefer m.allocator.free(out_tokens);
        const out_parents = try m.allocator.dupe(i32, parents[0..n]);
        errdefer m.allocator.free(out_parents);
        return .{ .gpa = m.allocator, .tokens = out_tokens, .parents = out_parents, .scores = try m.allocator.dupe(f64, own[0..n]) };
    }

    /// An empty context; after a GPU failure too.
    pub fn reset(m: *Model) void {
        m.frame.release();
        m.broken = false;
        m.begin = 0;
        m.committed_end = 0;
    }

    /// The body's linears as affine-4 copies (tensor-unit adapters, else SIMD tiles), keyed by their host address.
    fn prepare(m: *Model) !void {
        const units = m.target.device.tensorUnits();
        const order = if (units) try core.lane_projection.RegOrder.init(m.target.device) else null;
        defer if (order) |o| o.deinit();
        if (!units) m.simd = try simd.Pipelines.load(m.target.device);
        var rows: std.ArrayList(u16) = .empty;
        defer rows.deinit(m.allocator);
        var later = m.checkpoint.deferred.iterator();
        var count: usize = 0;
        while (later.next()) |entry| {
            const d = entry.value_ptr.*;
            if (d.dtype != .bf16 or d.shape[0] % 32 != 0 or d.shape[1] % 64 != 0) return error.DraftPreparationUnqualified;
            try rows.resize(m.allocator, d.bytes / 2);
            try m.checkpoint.read(entry.key_ptr.*, 0, std.mem.sliceAsBytes(rows.items));
            try m.prepareRows(order, @intFromPtr((try m.checkpoint.view(entry.key_ptr.*)).ptr), rows.items, d.shape[0], d.shape[1]);
            count += 1;
        }
        // fc and seven a layer (q, k, v, o, gate, up, down)
        if (count != 1 + 7 * m.graph.shape.layers) return error.DraftTensorCoverage;
    }

    /// One bf16 [n][k] matrix's affine-4 copy under `key`: a tensor-unit adapter, or (no `order`) SIMD tiles.
    fn prepareRows(m: *Model, order: ?core.lane_projection.RegOrder, key: usize, rows: []align(1) const u16, n: usize, k: usize) !void {
        if (n % 32 != 0 or k % 64 != 0 or rows.len != n * k) return error.DraftPreparationUnqualified;
        const copy_rows = try m.allocator.alloc(u16, rows.len);
        defer m.allocator.free(copy_rows);
        @memcpy(copy_rows, rows);
        var p = try core.affine4.prepare(m.allocator, copy_rows, n, k);
        defer p.deinit();
        if (order) |o| {
            const adapter = try core.affine4_lane.Adapter.init(m.allocator, m.target.device, p, try @import("../projection.zig").split(n, k), o);
            try @import("../residency.zig").putPrepared(m, key, adapter);
        } else {
            try @import("../residency.zig").putTiled(m, key, try Tiled.init(m.target.device, p));
        }
    }

    /// The bridge's reads and writes, the packed gates and the TSV pair [predecessor_gate; prev], affine-4 too.
    fn prepareBridge(m: *Model) !void {
        const s = m.graph.shape;
        const order = if (m.target.device.tensorUnits()) try core.lane_projection.RegOrder.init(m.target.device) else null;
        defer if (order) |o| o.deinit();
        for (0..s.layers) |l| {
            const read = m.graph.readOf(l);
            try m.prepareRows(order, @intFromPtr(read.ptr), std.mem.bytesAsSlice(u16, read), s.code, s.hidden);
            const write = m.graph.writeOf(l);
            try m.prepareRows(order, @intFromPtr(write.ptr), std.mem.bytesAsSlice(u16, write), s.hidden, s.rank);
        }
        const all = m.packed_w.slice(u16, 3 * s.rank * s.code);
        try m.prepareRows(order, @intFromPtr(all.ptr), all, 3 * s.rank, s.code);
        try m.prepareRows(order, @intFromPtr(all[s.rank * s.code ..].ptr), all[s.rank * s.code ..], 2 * s.rank, s.code);
    }

    pub fn source(m: *Model) ck.Source {
        return .{ .ptr = m, .getFn = get };
    }

    fn get(ptr: *anyopaque, name: []const u8) !ck.Tensor {
        const m: *Model = @ptrCast(@alignCast(ptr));
        const d = m.checkpoint.deferred.get(name) orelse return cpu(try m.checkpoint.get(name));
        var dims: @FieldType(ck.Tensor, "shape") = @splat(1);
        @memcpy(dims[0..d.rank], d.shape[0..d.rank]);
        return .{ .dtype = d.dtype, .rank = @intCast(d.rank), .shape = dims, .bytes = try m.checkpoint.view(name) };
    }

    pub fn cpu(t: core.checkpoint_metal.Tensor) ck.Tensor {
        var dims: @FieldType(ck.Tensor, "shape") = @splat(1);
        @memcpy(dims[0..t.rank], t.shape[0..t.rank]);
        return .{ .dtype = t.dtype, .rank = @intCast(t.rank), .shape = dims, .bytes = t.buffer.contents()[t.offset .. t.offset + t.bytes] };
    }

    /// A resident tensor's (or a slice of one's) GPU address.
    pub fn weight(m: *const Model, bytes: []const u8) !Ref {
        var it = m.checkpoint.tensors.valueIterator();
        while (it.next()) |t| {
            const base = t.buffer.contents() + t.offset;
            if (@intFromPtr(bytes.ptr) >= @intFromPtr(base) and @intFromPtr(bytes.ptr) + bytes.len <= @intFromPtr(base) + t.bytes) return .{ .buf = t.buffer, .off = t.offset + (@intFromPtr(bytes.ptr) - @intFromPtr(base)) };
        }
        return error.UnownedDraftWeight;
    }

    /// y [rows, n] = x [rows, k] w^T: prepared affine-4 when the body runs q4, else the BF16 checkpoint rows.
    pub fn linear(m: *Model, e: mtl.ComputeEncoder, x: Ref, w: ck.Tensor, y: Ref, rows: u32) !void {
        return m.matmul(e, x, w.bytes, null, y, rows, @intCast(w.shape[0]), @intCast(w.shape[1]));
    }

    /// y = x w^T for the matrix at host `bytes`: its affine-4 copy if prepared, else BF16 at `w` (or its tensor).
    pub fn matmul(m: *Model, e: mtl.ComputeEncoder, x: Ref, bytes: []const u8, w: ?Ref, y: Ref, rows: u32, n: u32, k: u32) !void {
        const key = @intFromPtr(bytes.ptr);
        if (m.tiled.get(key)) |t| {
            simd.encode(e, try simd.call(&m.simd.?, t.n, t.k, rows, false, false, 0), t.matrix(), x.buf, x.off, y.buf, y.off);
        } else if (m.prepared.get(key)) |adapter| {
            var first: u32 = 0;
            while (first < rows) : (first += 16) try adapter.encode(e, .{ .buf = x.buf, .off = x.off + @as(usize, first) * k * 2 }, .{ .buf = y.buf, .off = y.off + @as(usize, first) * n * 2 }, @min(rows - first, 16));
        } else {
            try m.primitives.linear(e, x, w orelse try m.weight(bytes), y, .{ .rows = rows, .n = n, .k = k, .x_stride = k, .w_stride = k, .y_stride = n });
        }
        e.barrier();
    }

    /// The target's lm_head (weight, scales, biases in MLX's affine-4 layout) from its files.
    fn copyHead(m: *Model, io: std.Io, dir: []const u8) !void {
        const shards = try core.checkpoint_host.shardFiles(m.allocator, io, dir);
        defer core.checkpoint_host.freeShardFiles(m.allocator, shards);
        var c = core.checkpoint_metal.Checkpoint.init(m.allocator);
        errdefer c.deinit();
        const prefix: []const u8 = if (m.target.checkpoint.has("language_model.lm_head.weight")) "language_model.lm_head" else "lm_head";
        for (shards) |file| try c.addFileSelected(m.target.device, file, "", prefix);
        var name: [64]u8 = undefined;
        const w = try c.get(try std.fmt.bufPrint(&name, "{s}.weight", .{prefix}));
        const sc = try c.get(try std.fmt.bufPrint(&name, "{s}.scales", .{prefix}));
        const bi = try c.get(try std.fmt.bufPrint(&name, "{s}.biases", .{prefix}));
        const s = m.graph.shape;
        if (w.dtype != .u32 or w.shape[0] != s.vocab or w.shape[1] * 8 != s.hidden or sc.shape[1] * 64 != s.hidden or sc.dtype != .bf16 or bi.dtype != .bf16) return error.TargetBinding;
        c.closeFiles();
        m.head_copy = c;
        m.head_rows = .{ .words = .{ .buf = w.buffer, .off = w.offset }, .scales = .{ .buf = sc.buffer, .off = sc.offset }, .biases = .{ .buf = bi.buffer, .off = bi.offset } };
    }

    /// The target head over `rows` rows, through the target's own head kernel.
    pub fn head(m: *Model, e: mtl.ComputeEncoder, x: Ref, y: Ref, rows: u32) !void {
        try m.target.kernels.quant(e, m.target.weights.head, .{ .buffer = x.buf, .offset = x.off }, null, m.target.frame.get(.dims), .{ .buffer = y.buf, .offset = y.off }, rows);
        e.barrier();
    }

    /// rows x width bf16 from x to y, ordered behind a barrier.
    pub fn copy(m: *Model, e: mtl.ComputeEncoder, x: Ref, y: Ref, rows: usize, width: usize) !void {
        try m.target.glue.unstack(e, .{ .buffer = x.buf, .offset = x.off }, .{ .buffer = y.buf, .offset = y.off }, .{ .rows = @intCast(rows), .width = @intCast(width), .stride = @intCast(width), .offset = 0 });
        e.barrier();
    }

    /// gate, predecessor_gate and prev stacked into one [3 rank, code] matrix.
    fn packWeights(m: *Model) !mtl.Buffer {
        const b = m.graph.bridge;
        const each = b.gate.bytes.len;
        const out = try m.target.device.buffer(3 * each, options);
        for ([_]ck.Tensor{ b.gate, b.predecessor_gate, b.prev }, 0..) |t, i| @memcpy(out.contents()[i * each ..][0..each], t.bytes);
        return out;
    }

    /// TG through the target head: chunk rows x_r = bf16(norm * write_last[:, r]) give logits = TG^T rows.
    fn buildTg(m: *Model) !mtl.Buffer {
        const s = m.graph.shape;
        const h = s.hidden;
        const r = s.rank;
        const v = s.vocab;
        const tg = try m.target.device.buffer(v * r * 2, options);
        errdefer tg.deinit();
        const chunk = 16;
        const x = try m.target.device.buffer(chunk * h * 2, options);
        defer x.deinit();
        const y = try m.target.device.buffer(chunk * v * 2, options);
        defer y.deinit();
        const norm = std.mem.bytesAsSlice(u16, m.graph.norm.bytes);
        const write = std.mem.bytesAsSlice(u16, m.graph.writeOf(s.layers - 1));
        const host = @import("host.zig");
        const xs = x.slice(u16, chunk * h);
        const ys = y.slice(u16, chunk * v);
        const out = tg.slice(u16, v * r);
        try m.target.kernels.prepareDecode(m.target.weights.head);
        var first: usize = 0;
        while (first < r) : (first += chunk) {
            for (0..chunk) |c| for (0..h) |i| {
                xs[c * h + i] = host.round(host.float(norm[i]) * host.float(write[i * r + first + c]));
            };
            const cb = m.target.queue.commandBuffer();
            const e = cb.compute(.serial);
            try m.target.kernels.quant(e, m.target.weights.head, .{ .buffer = x, .offset = 0 }, null, m.target.frame.get(.dims), .{ .buffer = y, .offset = 0 }, chunk);
            e.end();
            cb.commit();
            cb.wait();
            if (cb.failure()) |msg| {
                std.log.err("dspine: TG build failed on the GPU: {s}", .{msg});
                return error.DraftGpuFailure;
            }
            for (0..chunk) |c| for (0..v) |id| {
                out[id * r + first + c] = ys[c * v + id];
            };
        }
        return tg;
    }
};

fn loadGraph(a: std.mem.Allocator, m: *Model, s: config.Shape) !weights.Graph {
    const target = m.target;
    const prefix: []const u8 = if (target.checkpoint.has("language_model.model.embed_tokens.weight")) "language_model." else "";
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const local = arena.allocator();
    const stem = try std.mem.concat(local, u8, &.{ prefix, "model.embed_tokens" });
    const w = Model.cpu(try target.checkpoint.get(try std.mem.concat(local, u8, &.{ stem, ".weight" })));
    const sc = Model.cpu(try target.checkpoint.get(try std.mem.concat(local, u8, &.{ stem, ".scales" })));
    const bi = Model.cpu(try target.checkpoint.get(try std.mem.concat(local, u8, &.{ stem, ".biases" })));
    const embed = try ck.validate(w, sc, bi, .{ .bits = target.weights.embed.bits, .group_size = target.weights.embed.group }, target.config.vocab, target.config.hidden);
    var names: std.ArrayList([]const u8) = .empty;
    var it = m.checkpoint.tensors.keyIterator();
    while (it.next()) |name| try names.append(local, name.*);
    var later = m.checkpoint.deferred.keyIterator();
    while (later.next()) |name| try names.append(local, name.*);
    return weights.load(a, .{ .source = m.source(), .names = names.items }, s, embed);
}

/// A layer's rolling K/V: the window plus one block, so a block's own rows sit behind an unwrapped context.
fn newCaches(a: std.mem.Allocator, device: mtl.Device, s: config.Shape) ![]Cache {
    const caches = try a.alloc(Cache, s.layers);
    var made: usize = 0;
    errdefer {
        for (caches[0..made]) |c| {
            c.keys.deinit();
            c.values.deinit();
        }
        a.free(caches);
    }
    const bytes = (s.window + s.block) * config.kvWidth(s) * 2;
    for (caches) |*c| {
        const keys = try device.buffer(bytes, options);
        const values = device.buffer(bytes, options) catch |err| {
            keys.deinit();
            return err;
        };
        c.* = .{ .keys = keys, .values = values };
        made += 1;
    }
    return caches;
}

fn freeCaches(a: std.mem.Allocator, caches: []Cache) void {
    for (caches) |c| {
        c.keys.deinit();
        c.values.deinit();
    }
    a.free(caches);
}
