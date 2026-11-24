//! Fake operators prove GLM's own five-layer draft schedule over the shared execution contract.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const ck = tf.qwen27.checkpoint;
const draft = tf.qwen27.dflash.config;
const op = tf.qwen27.dflash.operators;
const execution = tf.qwen27.dflash.execution;
const session = tf.qwen27.dflash.session;
const selector = tf.qwen27.dflash.selector;
const weights = tf.glm.metal_dflash_weights;
const dcfg = tf.glm.metal_dflash_config;

// GLM's toy drafter: the stock incoai contract's geometry at fixture scale, five taps into 45 layers.
const toy = draft.Config{ .hidden = 64, .intermediate = 128, .vocab = 64, .heads = 8, .kv_heads = 4, .head_dim = 8, .mask = 63, .window = 16, .rank = 16, .topk = 2, .target_layers = 45, .taps = dcfg.stock.taps, .tap_count = dcfg.stock.tap_count };
const Dimensions = struct { values: @FieldType(ck.Tensor, "shape"), rank: usize };

const Store = struct {
    arena: std.heap.ArenaAllocator,
    keys: [][]const u8 = undefined,
    tensors: []ck.Tensor = undefined,
    fn init(gpa: std.mem.Allocator) !Store {
        var s = Store{ .arena = std.heap.ArenaAllocator.init(gpa) };
        errdefer s.deinit();
        const a = s.arena.allocator();
        s.keys = try weights.names(a, toy);
        s.tensors = try a.alloc(ck.Tensor, s.keys.len);
        for (s.keys, s.tensors) |name, *t| {
            const shape = dims(name);
            var count: usize = 2;
            for (shape.values[0..shape.rank]) |n| count = try std.math.mul(usize, count, n);
            const data = try a.alloc(u8, count);
            @memset(data, 0);
            t.* = .{ .dtype = .bf16, .rank = @intCast(shape.rank), .shape = shape.values, .bytes = data };
        }
        return s;
    }
    pub fn deinit(s: *Store) void {
        s.arena.deinit();
    }
    pub fn graph(s: *Store, gpa: std.mem.Allocator) !weights.Graph {
        return weights.load(gpa, .{ .source = .{ .ptr = s, .getFn = get }, .names = s.keys }, toy, weights.embedBinding(toy));
    }
    fn get(ptr: *anyopaque, name: []const u8) !ck.Tensor {
        const s: *Store = @ptrCast(@alignCast(ptr));
        for (s.keys, s.tensors) |key, t| if (std.mem.eql(u8, key, name)) return t;
        return error.MissingTensor;
    }
    fn dims(name: []const u8) Dimensions {
        if (std.mem.eql(u8, name, "fc.weight")) return .{ .values = .{ toy.hidden, toy.tapWidth(), 1, 1, 1 }, .rank = 2 };
        if (std.mem.eql(u8, name, "candidate_selector.hidden_projection.weight")) return .{ .values = .{ toy.rank, toy.hidden, 1, 1, 1 }, .rank = 2 };
        if (std.mem.startsWith(u8, name, "candidate_selector.")) return .{ .values = .{ toy.vocab, toy.rank, 1, 1, 1 }, .rank = 2 };
        if (std.mem.endsWith(u8, name, "base_kernel")) return .{ .values = .{ 2, 2, toy.hidden, 1, 1 }, .rank = 3 };
        if (std.mem.endsWith(u8, name, "kernel_projection.weight")) return .{ .values = .{ toy.dynamicWidth(), toy.hidden, 1, 1, 1 }, .rank = 2 };
        if (std.mem.endsWith(u8, name, "q_proj.weight")) return .{ .values = .{ toy.qWidth(), toy.hidden, 1, 1, 1 }, .rank = 2 };
        if (std.mem.endsWith(u8, name, "k_proj.weight") or std.mem.endsWith(u8, name, "v_proj.weight")) return .{ .values = .{ toy.kvWidth(), toy.hidden, 1, 1, 1 }, .rank = 2 };
        if (std.mem.endsWith(u8, name, "o_proj.weight")) return .{ .values = .{ toy.hidden, toy.qWidth(), 1, 1, 1 }, .rank = 2 };
        if (std.mem.endsWith(u8, name, "gate_proj.weight") or std.mem.endsWith(u8, name, "up_proj.weight")) return .{ .values = .{ toy.intermediate, toy.hidden, 1, 1, 1 }, .rank = 2 };
        if (std.mem.endsWith(u8, name, "down_proj.weight")) return .{ .values = .{ toy.hidden, toy.intermediate, 1, 1, 1 }, .rank = 2 };
        if (std.mem.endsWith(u8, name, "q_norm.weight") or std.mem.endsWith(u8, name, "k_norm.weight")) return .{ .values = .{ toy.head_dim, 1, 1, 1, 1 }, .rank = 1 };
        return .{ .values = .{ toy.hidden, 1, 1, 1, 1 }, .rank = 1 };
    }
};

const Fake = struct {
    handle: u64 = 1,
    projects: u32 = 0,
    convolutions: u32 = 0,
    attends: u32 = 0,
    commits: u32 = 0,
    stages: u32 = 0,
    appends: u32 = 0,
    aborts: u32 = 0,
    fail_append: bool = false,
    ids: [16]u32 = undefined,
    positions: [16]u64 = undefined,
    mask_first: u32 = 0,
    fn self(ptr: *anyopaque) *Fake {
        return @ptrCast(@alignCast(ptr));
    }
    fn view(f: *Fake, rows: u32, width: u32) op.View {
        f.handle += 1;
        return .{ .handle = f.handle, .rows = rows, .width = width, .stride = width, .dtype = .bf16 };
    }
    fn begin(ptr: *anyopaque) !u64 {
        self(ptr).handle += 1;
        return self(ptr).handle;
    }
    fn release(_: *anyopaque, _: u64) void {}
    fn project(ptr: *anyopaque, _: u64, x: op.View, t: ck.Tensor, mode: op.Mode) !op.View {
        if (mode != .bf16_reference) return error.NotFakeReference;
        const f = self(ptr);
        f.projects += 1;
        return f.view(x.rows, @intCast(t.shape[0]));
    }
    fn norm(_: *anyopaque, _: u64, x: op.View, n: op.Norm) !op.View {
        if (x.width != n.heads * n.head_dim) return error.FakeNormShape;
        return .{ .handle = 1, .rows = x.rows, .width = x.width, .stride = x.width, .dtype = .bf16 };
    }
    fn embed(ptr: *anyopaque, _: u64, _: ck.Linear, ids: []const u32) !op.View {
        @memcpy(self(ptr).ids[0..ids.len], ids);
        return self(ptr).view(@intCast(ids.len), toy.hidden);
    }
    fn conv(ptr: *anyopaque, _: u64, x: op.View, a: op.Conv) !op.View {
        const f = self(ptr);
        if (a.branch != f.convolutions % 2 or a.block != x.rows or a.group != 16 or ((a.residual != null) != (a.branch == 1))) return error.FakeConvOrder;
        f.convolutions += 1;
        return f.view(x.rows, x.width);
    }
    fn rope(ptr: *anyopaque, _: u64, x: op.View, a: op.Rotary) !op.View {
        if (a.positions.len <= 16) @memcpy(self(ptr).positions[0..a.positions.len], a.positions);
        return self(ptr).view(x.rows, x.width);
    }
    fn attention(ptr: *anyopaque, _: u64, a: op.Attention) !op.View {
        if (a.mask != .all_slots or a.layer != self(ptr).attends or (a.query.rows < 2 or a.query.rows > 16 or a.query.rows != a.positions.len) or a.positions[0] != a.context.end) return error.FakeAttentionOrder;
        self(ptr).attends += 1;
        return self(ptr).view(a.query.rows, a.config.qWidth());
    }
    fn swiglu(_: *anyopaque, _: u64, a: op.View, b: op.View) !op.View {
        if (a.width != b.width or a.rows != b.rows) return error.FakeMlpShape;
        return .{ .handle = 1, .rows = a.rows, .width = a.width, .stride = a.width, .dtype = .bf16 };
    }
    fn mask(ptr: *anyopaque, _: u64, x: op.View, first: u32, count: u32) !op.View {
        if (first != 1 or count != x.rows - 1) return error.FakeAnchorShift;
        self(ptr).mask_first = first;
        return self(ptr).view(count, x.width);
    }
    fn head(_: *anyopaque, _: u64, x: op.View) !op.Head {
        return .{ .logits = .{ .handle = 1, .rows = x.rows, .width = toy.vocab, .stride = toy.vocab, .dtype = .bf16 } };
    }
    fn lattices(_: *anyopaque, _: u64, _: op.Head, _: op.View, _: draft.Config, _: u32, _: []op.Lattice) !void {
        return error.FakeBatchedHead;
    }
    fn streamAppend(_: *anyopaque, _: u64, _: u32, _: u32, _: op.View, _: op.View, _: []const u64) !void {
        return error.FakeBatchedHead;
    }
    fn lattice(_: *anyopaque, _: u64, _: op.Head, projected: op.View, c: draft.Config) !op.Lattice {
        if (projected.rows == 0 or projected.rows > 15 or projected.width != c.rank) return error.FakeProjectionShape;
        const a = std.testing.allocator;
        const ids = try a.alloc(u32, projected.rows * c.topk);
        errdefer a.free(ids);
        const unary = try a.alloc(f32, ids.len);
        errdefer a.free(unary);
        const hp = try a.alloc(f32, projected.rows * c.rank);
        for (ids, unary, 0..) |*id, *value, i| {
            id.* = @intCast(1 + i % c.topk);
            value.* = 0;
        }
        @memset(hp, 0);
        return .{ .gpa = a, .depth = projected.rows, .topk = c.topk, .rank = c.rank, .candidates = ids, .unary = unary, .projected = hp };
    }
    fn contextBegin(_: *anyopaque, _: op.Context) !u64 {
        return 1;
    }
    fn append(ptr: *anyopaque, _: u64, _: u64, layer: u32, k: op.View, v: op.View, positions: []const u64) !void {
        const f = self(ptr);
        if (f.fail_append) return error.InjectedAppendFailure;
        if (layer != f.appends or k.rows != positions.len or v.rows != k.rows) return error.FakeAppendOrder;
        f.appends += 1;
    }
    fn commit(ptr: *anyopaque, _: u64, _: u64, _: op.Context) !void {
        self(ptr).commits += 1;
    }
    fn stage(ptr: *anyopaque, _: u64, _: u64, _: op.Context) !void {
        self(ptr).stages += 1;
    }
    fn abort(ptr: *anyopaque, _: u64) void {
        self(ptr).aborts += 1;
    }
    fn ops(f: *Fake) op.Ops {
        return .{ .ptr = f, .vtable = &.{ .begin = begin, .release = release, .project = project, .norm = norm, .embed = embed, .conv = conv, .rope = rope, .attention = attention, .swiglu = swiglu, .mask_rows = mask, .head = head, .lattice = lattice, .lattices = lattices, .context_begin = contextBegin, .context_append = append, .context_commit = commit, .context_stage = stage, .context_abort = abort, .stream_append = streamAppend } };
    }
};

fn batch() session.Batch {
    return .{ .values = .{ .handle = 99, .rows = 2, .width = toy.tapWidth(), .stride = toy.tapWidth(), .dtype = .bf16 }, .positions = &.{ 0, 1 }, .layers = dcfg.stock.taps, .target_end = 2 };
}

test "GLM's five-tap schedule consumes anchor plus masks and emits the seven masked positions" {
    const a = std.testing.allocator;
    var store = try Store.init(a);
    defer store.deinit();
    var g = try store.graph(a);
    defer g.deinit();
    var fake = Fake{};
    var state = try session.Session.init(toy, 1, 0);
    const p = op.Preparation{ .mode = .bf16_reference };
    try state.absorb(fake.ops(), &g, p, batch());
    var tree = try state.tree(a, fake.ops(), &g, p, 5, 3, .{});
    defer tree.deinit();
    try std.testing.expectEqual(@as(usize, 15), tree.tokens.len);
    try std.testing.expectEqualSlices(u32, &.{ 5, 63, 63, 63, 63, 63, 63, 63 }, fake.ids[0..8]);
    try std.testing.expectEqualSlices(u64, &.{ 2, 3, 4, 5, 6, 7, 8, 9 }, fake.positions[0..8]);
    try std.testing.expectEqual(@as(u32, 20), fake.convolutions);
    try std.testing.expectEqual(@as(u32, 5), fake.attends);
    try std.testing.expectEqual(@as(u32, 5), fake.appends);
    try std.testing.expectEqual(@as(u32, 1), fake.commits);
    try std.testing.expectEqual(@as(u32, 1), fake.mask_first);
}

test "GLM's staged block path matches the separate path and commits through one transaction" {
    const a = std.testing.allocator;
    var store = try Store.init(a);
    defer store.deinit();
    var g = try store.graph(a);
    defer g.deinit();
    const p = op.Preparation{ .mode = .bf16_reference };
    var split = Fake{};
    var first = try session.Session.init(toy, 1, 0);
    try first.absorb(split.ops(), &g, p, .{ .values = batch().values, .positions = &.{ 0, 1 }, .layers = dcfg.stock.taps, .target_end = 2 });
    var merged = Fake{};
    var second = try session.Session.init(toy, 1, 0);
    try second.absorb(merged.ops(), &g, p, .{ .values = batch().values, .positions = &.{ 0, 1 }, .layers = dcfg.stock.taps, .target_end = 2 });
    const more = session.Batch{ .values = batch().values, .positions = &.{ 2, 3 }, .layers = dcfg.stock.taps, .target_end = 4 };
    split.appends = 0; // the fake checks layer order within one transaction
    merged.appends = 0;
    try first.absorb(split.ops(), &g, p, more);
    var want = try first.treeBlock(a, split.ops(), &g, p, 5, try first.firstDraw(), 8, .{});
    defer want.deinit();
    var got = try second.absorbTreeBlock(a, merged.ops(), &g, p, more, 5, 8, .{});
    defer got.deinit();
    try std.testing.expectEqualSlices(u32, want.tokens, got.tokens);
    try std.testing.expectEqualSlices(i32, want.parents, got.parents);
    try std.testing.expectEqual(first.context.end, second.context.end);
    try std.testing.expectEqual(@as(u32, 1), merged.commits);
    try std.testing.expectEqual(@as(u32, 1), merged.stages);
    try std.testing.expectEqual(@as(u32, 5), merged.appends);
}

test "GLM's append failure aborts before session metadata or readiness advances" {
    const a = std.testing.allocator;
    var store = try Store.init(a);
    defer store.deinit();
    var g = try store.graph(a);
    defer g.deinit();
    var fake = Fake{ .fail_append = true };
    var state = try session.Session.init(toy, 1, 0);
    try std.testing.expectError(error.InjectedAppendFailure, state.absorb(fake.ops(), &g, .{ .mode = .bf16_reference }, batch()));
    try std.testing.expectEqual(@as(u64, 0), state.context.end);
    try std.testing.expect(!state.ready);
    try std.testing.expectEqual(@as(u32, 1), fake.aborts);
    try std.testing.expectEqual(@as(u32, 0), fake.commits);
}

test "GLM's draft refusals cover the stock names and qwen27's 64-layer tap contract stays out" {
    try std.testing.expectEqualSlices(u32, &dcfg.tap_ids, dcfg.stock.tapIds());
    var c = dcfg.stock;
    c.target_layers = 64;
    c.taps = tf.qwen27.dflash.config.tap_ids;
    try std.testing.expectError(error.GlmDflash2TargetLayers, dcfg.check(c));
}

const Taps = tf.glm.metal_dflash_taps.Taps;

test "tap ring offsets wrap by window and pack planes by layer order" {
    const taps = Taps{ .ring = undefined, .window = 16, .hidden = 64 };
    const stride = 5 * 64 * 2;
    try std.testing.expectEqual(null, taps.offset(0, 3));
    try std.testing.expectEqual(@as(?usize, 3 * stride + 0 * 64 * 2), taps.offset(5, 3));
    try std.testing.expectEqual(@as(?usize, 3 * stride + 1 * 64 * 2), taps.offset(14, 3));
    try std.testing.expectEqual(@as(?usize, (3 + 13) % 16 * stride + 4 * 64 * 2), taps.offset(42, 16));
    try std.testing.expectEqual(@as(?usize, 0 * stride + 2 * 64 * 2), taps.offset(24, 32));
}

test "per-stream routes: a shared window's segs capture into their own stream's ring by its state" {
    var routes: [3]Taps.Route = @splat(.{ .s = null, .ring = undefined });
    const taps = Taps{ .ring = undefined, .window = 16, .hidden = 64, .routes = &routes };
    var states: [3]tf.glm.state.State = undefined;
    for (&states, 0..) |*s, i| s.* = .{ .cap = 64, .pos = @intCast(i) };
    var t = taps;
    t.setRoute(0, &states[0], undefined);
    t.setRoute(1, &states[1], undefined);
    try std.testing.expect(t.routeRing(&states[0]) != null);
    t.setActive(0, false);
    try std.testing.expect(t.routeRing(&states[0]) == null);
    t.setActive(0, true);
    try std.testing.expect(t.routeRing(&states[0]) != null);
    try std.testing.expect(t.routeRing(&states[2]) == null); // a stream drafting no chain: its seg rows are skipped
    t.clearRoute(&states[0]);
    try std.testing.expect(t.routeRing(&states[0]) == null);
    try std.testing.expect(t.routeRing(&states[1]) != null); // the other stream keeps its route
}

test "a stream above the drafter's caps captures no taps: no route, no ring, so its prefill launches no tap copies" {
    var routes: [3]Taps.Route = @splat(.{ .s = null, .ring = undefined });
    var t = Taps{ .ring = .{ .id = @ptrFromInt(0x1000) }, .window = 16, .hidden = 64, .routes = &routes };
    var states: [2]tf.glm.state.State = undefined;
    for (&states, 0..) |*s, i| s.* = .{ .cap = 64, .pos = @intCast(i) };
    t.setRoute(0, &states[0], .{ .id = @ptrFromInt(0x2000) });
    try std.testing.expect(t.ringFor(&states[0]) != null);
    try std.testing.expect(t.ringFor(&states[1]) == null); // never routed: a capped stream (begin clears its route)
    t.clearRoute(&states[0]);
    try std.testing.expect(t.ringFor(&states[0]) == null);
    const lone = Taps{ .ring = .{ .id = @ptrFromInt(0x1000) }, .window = 16, .hidden = 64 };
    try std.testing.expect(lone.ringFor(&states[1]) != null); // no routes at all: one drafter context, the shared ring
    const P = tf.glm.metal_dflash_rounds.Policy;
    try std.testing.expect(!(P{}).ok(65536)); // a 64K prompt is over the default cap: begin gives it no route
}

/// A backend over undefined slots whose engine has an MTP head and `drafter` (null: none), no GPU opened.
fn facts(drafter: bool) !tf.lanes.Model {
    const objc = mtl.objc;
    const no_arch = objc.msg(objc.Id, objc.class("NSObject"), "new", .{}); // facts() asks it for tensor units, no GPU
    defer objc.release(no_arch);
    var w: tf.glm.weights.Weights = undefined;
    w.mtp = @as(tf.glm.weights.Mtp, undefined);
    var e: tf.glm.engine.Engine = undefined;
    e.w = &w;
    e.ep = null;
    e.device = .{ .id = no_arch };
    e.dflash = if (drafter) @ptrFromInt(@alignOf(tf.glm.metal_dflash_load.Drafter)) else null; // never read
    var slots: [16]tf.glm.slots.Slot = undefined;
    var sl: tf.glm.slots.Slots = undefined;
    sl.e = &e;
    sl.slots = &slots;
    var b: tf.glm.backend.Backend = .{ .gpa = std.testing.allocator, .sl = &sl };
    return b.facts();
}

test "without --drafter the MTP head's facts stand; with it DFlash2's chains and chances" {
    const head = try facts(false);
    try std.testing.expectEqual(@as(u32, 4), head.drafts);
    try std.testing.expect(!head.draft_probabilities);
    try std.testing.expect(head.allocate_shared_drafts and head.gpu_tokens);
    try std.testing.expectEqual(@as(usize, 16), head.window_costs.len);
    const drafted = try facts(true);
    try std.testing.expectEqual(tf.glm.metal_dflash_rounds.max_held, drafted.drafts);
    // shared rounds fit row slope across DFlash2/MTP and follow draft_chances instead of default_prior depth caps
    try std.testing.expect(!drafted.fixed_draft_block and drafted.allocate_shared_drafts and !drafted.limit_shared_drafts);
    try std.testing.expectEqual(head.shared_costs.len, drafted.shared_costs.len);
    try std.testing.expect(drafted.draft_probabilities and drafted.mtp and drafted.gpu_tokens == head.gpu_tokens);
    // the drafter's verify windows are priced as the head's, and learned as the head's are off an M5 Ultra
    try std.testing.expectEqualSlices(tf.lanes.config.Cost, head.window_costs, drafted.window_costs);
    try std.testing.expectEqual(head.learn_row_costs, drafted.learn_row_costs);
}

test "a cache hit's drafter context starts where the restored pass does, its session and parked context agreeing" {
    var x: tf.glm.metal_dflash_model.Context = .{ .pending = undefined, .caches = &.{}, .session = try session.Session.init(dcfg.stock, 1, 0) };
    x.context = .{ .cache = 1, .begin = 3, .end = 9 }; // a previous request's
    try x.reset(dcfg.stock, 4096);
    try std.testing.expectEqual(@as(u64, 4096), x.committed_end);
    try std.testing.expectEqual(op.Context{ .cache = 1, .begin = 4096, .end = 4096 }, x.context);
    try std.testing.expectEqual(x.context, x.session.context);
    try std.testing.expect(!x.session.ready);
}

test "the per-stream policy: prompts past 32768 use the MTP head, inclusive bound allows 32768" {
    const P = tf.glm.metal_dflash_rounds.Policy;
    try std.testing.expectEqual(@as(u64, 32768), (P{}).max_ctx);
    try std.testing.expect((P{}).ok(32768));
    try std.testing.expect(!(P{}).ok(32769));
    try std.testing.expect((P{ .max_ctx = 0 }).ok(1 << 40));
    try std.testing.expect((P{ .max_ctx = 8192 }).ok(8192));
    try std.testing.expect(!(P{ .max_ctx = 8192 }).ok(8193));
}

test "the draft head runs on tensor units at 4 or 8 bits in groups of 64, and on the row kernels otherwise" {
    const backend = tf.glm.metal_dflash_backend;
    const base = tf.glm.weights.Affine{ .w = undefined, .s = undefined, .b = undefined, .n = 154880, .k = 4096, .bits = 8, .group = 64 };
    try std.testing.expectEqual(backend.HeadPath.block8, backend.headPath(true, base, true));
    var q = base;
    q.bits = 4;
    try std.testing.expectEqual(backend.HeadPath.block4, backend.headPath(true, q, true));
    try std.testing.expectEqual(backend.HeadPath.rows, backend.headPath(false, q, true));
    try std.testing.expectEqual(backend.HeadPath.rows, backend.headPath(true, base, false));
    for ([_]u8{ 2, 5, 6 }) |bits| {
        q.bits = bits;
        try std.testing.expectEqual(backend.HeadPath.rows, backend.headPath(true, q, true));
    }
    q = base;
    q.group = 32;
    try std.testing.expectEqual(backend.HeadPath.rows, backend.headPath(true, q, true));
    q = base;
    q.n = 154880 + 32;
    try std.testing.expectEqual(backend.HeadPath.rows, backend.headPath(true, q, true));
}

test "heldBytes returns drafter model bytes without removed tap-stage buffer" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const ring = try device.buffer(64 * 5 * 512 * 2, mtl.ResourceOptions.shared);
    defer ring.deinit();
    var taps = try tf.glm.metal_dflash_taps.Taps.init(device, ring, 64, 512, std.testing.allocator, 1);
    defer taps.deinit(std.testing.allocator);
    var m: tf.glm.metal_dflash_model.Model = undefined;
    m.bytes = 1024;
    var d: tf.glm.metal_dflash_load.Drafter = .{ .gpa = std.testing.allocator, .model = &m, .taps = &taps, .target = undefined };
    var e: tf.glm.engine.Engine = undefined;
    e.dflash = &d;
    try std.testing.expectEqual(@as(u64, 1024), tf.glm.metal_dflash_load.heldBytes(&e));
}

test "chain flatten takes the best root and best children, chances are the per-step own" {
    const a = std.testing.allocator;
    const rounds = tf.glm.metal_dflash_rounds;
    const tokens = [_]u32{ 3, 8, 8, 8, 8, 8, 8, 8, 8 };
    const parents = [_]i32{ -1, 0, 0, 0, 1, 1, 2, 2, 2 };
    const scores = [_]f64{ -0.5, -0.7, -1.4, -3.0, -0.9, -1.1, -1.5, -1.6, -2.5 };
    const tree: selector.Tree = .{ .gpa = a, .tokens = @constCast(&tokens), .parents = @constCast(&parents), .scores = @constCast(&scores) };
    var chain: [7]u32 = undefined;
    var chances: [7]f64 = undefined;
    const n = rounds.flatten(&tree, &chain, &chances);
    try std.testing.expectEqual(@as(u32, 3), n);
    try std.testing.expectEqual(@as(u32, 3), chain[0]);
    try std.testing.expectEqual(@as(u32, 8), chain[1]);
    try std.testing.expectEqual(@as(u32, 8), chain[2]);
    try std.testing.expectApproxEqAbs(@exp(-0.5), chances[0], 1e-9);
    try std.testing.expectApproxEqAbs(@exp(-0.2), chances[1], 1e-9);
    try std.testing.expectApproxEqAbs(@exp(-0.2), chances[2], 1e-9);
}

test "chain flatten stops where the tree has no child at the chain's depth" {
    const a = std.testing.allocator;
    const rounds = tf.glm.metal_dflash_rounds;
    const tokens = [_]u32{ 5, 6 };
    const parents = [_]i32{ -1, 0 };
    const scores = [_]f64{ -0.3, -0.4 };
    const tree: selector.Tree = .{ .gpa = a, .tokens = @constCast(&tokens), .parents = @constCast(&parents), .scores = @constCast(&scores) };
    var chain: [7]u32 = undefined;
    var chances: [7]f64 = undefined;
    try std.testing.expectEqual(@as(u32, 2), rounds.flatten(&tree, &chain, &chances));
}

test "confidence cut and settledMtp track active prefixes and safe MTP tap pausing" {
    const rounds = tf.glm.metal_dflash_rounds;
    const src = tf.glm.metal_dflash_source;
    var chances = [_]f64{ 0.9, 0.9, 0.9, 0.9 };
    rounds.cutChances(&chances, 0.7);
    try std.testing.expectEqual(@as(f64, 0.9), chances[0]);
    try std.testing.expectEqual(@as(f64, 0), chances[3]);
    try std.testing.expectEqual(@as(u32, 3), rounds.pricedPrefix(&chances));
    var tiny = [_]f64{ 0.1, 0.9, 0.9 };
    rounds.cutChances(&tiny, 0.7);
    try std.testing.expectEqual(@as(u32, 1), rounds.pricedPrefix(&tiny));
    try std.testing.expectApproxEqAbs(rounds.default_confidence, try rounds.confidence(), 1e-9);
    var c: src.Chooser = .{};
    const rule: src.Rule = .{ .first_dflash = 2, .probe_mtp = 2, .back_every = 8, .probe_len = 2, .watch = 2 };
    for (0..4) |_| {
        _ = c.next(rule, true);
        c.observe(.dflash, 1, 40);
        c.observe(.mtp, 1, 20);
    }
    try std.testing.expect(!c.settledMtp(rule)); // still chosen/probing until settled
    while (!c.settledMtp(rule)) {
        _ = c.next(rule, true);
        c.observe(.mtp, 1, 20);
    }
    try std.testing.expect(c.settledMtp(rule));
    while (c.settledMtp(rule)) {
        _ = c.next(rule, true);
        c.observe(.mtp, 1, 20);
    }
    try std.testing.expect(c.since + 2 >= rule.back_every); // wakes up before probe_len triggers
}
