//! Fake operators prove the forward schedule, dtype boundaries and commit failure handling.
const std = @import("std");
const ck = @import("../checkpoint.zig");
const cfg = @import("config.zig");
const op = @import("operators.zig");
const execution = @import("execution.zig");
const fixtures = @import("fixtures.zig");
const session = @import("session.zig");
const selector = @import("selector.zig");
const Fake = struct {
    handle: u64 = 1,
    projects: u32 = 0,
    convolutions: u32 = 0,
    attends: u32 = 0,
    commits: u32 = 0,
    stages: u32 = 0,
    aborts: u32 = 0,
    appends: u32 = 0,
    fail_append: bool = false,
    wrong_dtype: bool = false,
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
        var y = f.view(x.rows, @intCast(t.shape[0]));
        if (f.wrong_dtype) y.dtype = .f32;
        return y;
    }
    fn norm(ptr: *anyopaque, _: u64, x: op.View, n: op.Norm) !op.View {
        if (x.width != n.heads * n.head_dim) return error.FakeNormShape;
        return self(ptr).view(x.rows, x.width);
    }
    fn embed(ptr: *anyopaque, _: u64, _: ck.Linear, ids: []const u32) !op.View {
        @memcpy(self(ptr).ids[0..ids.len], ids);
        return self(ptr).view(@intCast(ids.len), fixtures.toy.hidden);
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
    fn swiglu(ptr: *anyopaque, _: u64, a: op.View, b: op.View) !op.View {
        if (a.width != b.width or a.rows != b.rows) return error.FakeMlpShape;
        return self(ptr).view(a.rows, a.width);
    }
    fn mask(ptr: *anyopaque, _: u64, x: op.View, first: u32, count: u32) !op.View {
        if (first != 1 or count != x.rows - 1) return error.FakeAnchorShift;
        self(ptr).mask_first = first;
        return self(ptr).view(count, x.width);
    }
    fn head(ptr: *anyopaque, _: u64, _: ck.Linear, x: op.View) !op.Head {
        return .{ .logits = self(ptr).view(x.rows, fixtures.toy.vocab) };
    }
    fn lattice(_: *anyopaque, _: u64, _: op.Head, projected: op.View, c: cfg.Config) !op.Lattice {
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
        return .{ .ptr = f, .vtable = &.{ .begin = begin, .release = release, .project = project, .norm = norm, .embed = embed, .conv = conv, .rope = rope, .attention = attention, .swiglu = swiglu, .mask_rows = mask, .head = head, .lattice = lattice, .context_begin = contextBegin, .context_append = append, .context_commit = commit, .context_stage = stage, .context_abort = abort } };
    }
};
fn batch() session.Batch {
    return .{ .values = .{ .handle = 99, .rows = 2, .width = fixtures.toy.tapWidth(), .stride = fixtures.toy.tapWidth(), .dtype = .bf16 }, .positions = &.{ 0, 1 }, .layers = cfg.tap_ids, .target_end = 2 };
}
test "actual five-layer schedule consumes anchor plus masks and only emits seven mask positions" {
    const a = std.testing.allocator;
    var store = try fixtures.Store.init(a);
    defer store.deinit();
    var g = try store.graph(a);
    defer g.deinit();
    var fake = Fake{};
    var state = try session.Session.init(fixtures.toy, 1, 0);
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
test "taps absorbed in the block's frame give the separate path's tree, one staged append and no commit" {
    const a = std.testing.allocator;
    var store = try fixtures.Store.init(a);
    defer store.deinit();
    var g = try store.graph(a);
    defer g.deinit();
    const p = op.Preparation{ .mode = .bf16_reference };
    var split = Fake{};
    var first = try session.Session.init(fixtures.toy, 1, 0);
    try first.absorb(split.ops(), &g, p, .{ .values = batch().values, .positions = &.{ 0, 1 }, .layers = cfg.tap_ids, .target_end = 2 });
    var merged = Fake{};
    var second = try session.Session.init(fixtures.toy, 1, 0);
    try second.absorb(merged.ops(), &g, p, .{ .values = batch().values, .positions = &.{ 0, 1 }, .layers = cfg.tap_ids, .target_end = 2 });
    const more = session.Batch{ .values = batch().values, .positions = &.{ 2, 3 }, .layers = cfg.tap_ids, .target_end = 4 };
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
test "context append failure aborts before session metadata or readiness advances" {
    const a = std.testing.allocator;
    var store = try fixtures.Store.init(a);
    defer store.deinit();
    var g = try store.graph(a);
    defer g.deinit();
    var fake = Fake{ .fail_append = true };
    var state = try session.Session.init(fixtures.toy, 1, 0);
    try std.testing.expectError(error.InjectedAppendFailure, state.absorb(fake.ops(), &g, .{ .mode = .bf16_reference }, batch()));
    try std.testing.expectEqual(@as(u64, 0), state.context.end);
    try std.testing.expect(!state.ready);
    try std.testing.expectEqual(@as(u32, 1), fake.aborts);
    try std.testing.expectEqual(@as(u32, 0), fake.commits);
}
test "default preparation and wrong operator dtype refuse before claiming native parity" {
    const a = std.testing.allocator;
    var store = try fixtures.Store.init(a);
    defer store.deinit();
    var g = try store.graph(a);
    defer g.deinit();
    var fake = Fake{};
    try std.testing.expectError(error.DraftPreparationUnqualified, execution.forward(fake.ops(), &g, .{}, .{ .cache = 1, .begin = 0, .end = 1 }, 5));
    try std.testing.expectEqual(@as(u32, 0), fake.projects);
    fake.wrong_dtype = true;
    try std.testing.expectError(error.DraftOperatorShape, execution.forward(fake.ops(), &g, .{ .mode = .bf16_reference }, .{ .cache = 1, .begin = 0, .end = 1 }, 5));
}
test "selector conditions descendants on parent tokens and shifts the native verify root exactly once" {
    const a = std.testing.allocator;
    var store = try fixtures.Store.init(a);
    defer store.deinit();
    var g = try store.graph(a);
    defer g.deinit();
    for ([_]struct { t: *ck.Tensor, row: usize, bits: u16 }{ .{ .t = &g.predecessor, .row = 3, .bits = 0x3f80 }, .{ .t = &g.predecessor, .row = 1, .bits = 0x3f80 }, .{ .t = &g.predecessor, .row = 2, .bits = 0xbf80 }, .{ .t = &g.successor, .row = 1, .bits = 0x3f80 }, .{ .t = &g.successor, .row = 2, .bits = 0x4000 } }) |change| std.mem.writeInt(u16, @constCast(change.t.bytes)[change.row * fixtures.toy.rank * 2 ..][0..2], change.bits, .little);
    var fake = Fake{};
    var lattice = try execution.forward(fake.ops(), &g, .{ .mode = .bf16_reference }, .{ .cache = 1, .begin = 0, .end = 1 }, 3);
    defer lattice.deinit();
    for (0..lattice.depth) |d| lattice.projected[d * lattice.rank] = 1;
    var tree = try selector.propose(a, &g, lattice, 3, 2, .{ .nodes = 15, .children = 1, .edge = 1, .tau = 1 });
    defer tree.deinit();
    try std.testing.expectEqualSlices(u32, &.{ 2, 1, 2, 1, 2, 1, 2 }, tree.tokens);
    try std.testing.expectEqualSlices(i32, &.{ -1, 0, 1, 2, 3, 4, 5 }, tree.parents);
    const parents = try tree.verifyParents(a);
    defer a.free(parents);
    try std.testing.expectEqualSlices(i32, &.{ -1, 0, 1, 2, 3, 4, 5, 6 }, parents);
}

test "forward sink retains the complete checked schedule without host lattice readback" {
    const a = std.testing.allocator;
    var store = try fixtures.Store.init(a);
    defer store.deinit();
    var graph = try store.graph(a);
    defer graph.deinit();
    var fake = Fake{};
    const Sink = struct {
        called: bool = false,
        fn consume(ptr: *anyopaque, frame: u64, output: execution.Output, c: cfg.Config) !void {
            const sink: *@This() = @ptrCast(@alignCast(ptr));
            try std.testing.expect(frame != 0);
            try output.head.logits.check(c.depth(), c.vocab, .bf16);
            try output.projected.check(c.depth(), c.rank, .bf16);
            sink.called = true;
        }
    };
    var sink = Sink{};
    try execution.forwardTo(fake.ops(), &graph, .{ .mode = .bf16_reference }, .{ .cache = 1, .begin = 0, .end = 2 }, 5, .{ .ptr = &sink, .call = Sink.consume });
    try std.testing.expect(sink.called);
    try std.testing.expectEqual(@as(u32, 20), fake.convolutions);
    try std.testing.expectEqual(@as(u32, 5), fake.attends);
}

test "normalized sibling ties preserve candidate order before heap token tie-breaking" {
    const a = std.testing.allocator;
    var store = try fixtures.Store.init(a);
    defer store.deinit();
    var graph = try store.graph(a);
    defer graph.deinit();
    var fake = Fake{};
    var lattice = try Fake.lattice(&fake, 1, undefined, .{ .handle = 1, .rows = 7, .width = fixtures.toy.rank, .stride = fixtures.toy.rank, .dtype = .bf16 }, fixtures.toy);
    defer lattice.deinit();
    for (lattice.candidates, 0..) |*id, i| id.* = @intCast(lattice.topk - i % lattice.topk);
    var tree = try selector.propose(a, &graph, lattice, 3, 2, .{ .nodes = 1, .children = 1, .edge = 0 });
    defer tree.deinit();
    try std.testing.expectEqual(lattice.topk, tree.tokens[0]);
}

test "runtime blocks2 through16 preserve checkpoint block8 and uninterrupted convolution geometry" {
    const a = std.testing.allocator;
    var store = try fixtures.Store.init(a);
    defer store.deinit();
    var graph = try store.graph(a);
    defer graph.deinit();
    for (2..17) |block| {
        var fake = Fake{};
        var lattice = try execution.forwardBlock(fake.ops(), &graph, .{ .mode = .bf16_reference }, .{ .cache = 1, .begin = 0, .end = 2 }, 5, @intCast(block));
        defer lattice.deinit();
        try std.testing.expectEqual(@as(u32, @intCast(block - 1)), lattice.depth);
        try std.testing.expectEqual(@as(u32, 8), graph.config.block);
        try std.testing.expectEqual(@as(u32, 20), fake.convolutions);
        for (0..block) |i| {
            try std.testing.expectEqual(@as(u32, if (i == 0) 5 else 63), fake.ids[i]);
            try std.testing.expectEqual(@as(u64, i + 2), fake.positions[i]);
        }
    }
    for ([_]u32{ 0, 1, 17 }) |block| {
        var fake = Fake{};
        try std.testing.expectError(error.BadRuntimeBlock, execution.forwardBlock(fake.ops(), &graph, .{ .mode = .bf16_reference }, .{ .cache = 1, .begin = 0, .end = 2 }, 5, block));
        try std.testing.expectEqual(@as(u32, 0), fake.projects);
    }
}

test "tiny conditional score differences collapse before sibling selection" {
    const a = std.testing.allocator;
    var store = try fixtures.Store.init(a);
    defer store.deinit();
    var graph = try store.graph(a);
    defer graph.deinit();
    graph.config.topk = 16;
    std.mem.writeInt(u16, @constCast(graph.predecessor.bytes)[3 * graph.config.rank * 2 ..][0..2], 0x3f80, .little);
    for (0..16) |id| {
        const value: f32 = @floatFromInt(id);
        std.mem.writeInt(u16, @constCast(graph.successor.bytes)[id * graph.config.rank * 2 ..][0..2], @truncate(@as(u32, @bitCast(value)) >> 16), .little);
    }
    var fake = Fake{};
    var lattice = try Fake.lattice(&fake, 1, undefined, .{ .handle = 1, .rows = 1, .width = graph.config.rank, .stride = graph.config.rank, .dtype = .bf16 }, graph.config);
    defer lattice.deinit();
    for (lattice.candidates, 0..) |*id, i| id.* = @intCast(i);
    lattice.projected[0] = 0x1p-60;
    var tree = try selector.propose(a, &graph, lattice, 3, 2, .{ .nodes = 4, .children = 4 });
    defer tree.deinit();
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 2, 3 }, tree.tokens);
    try std.testing.expectEqualSlices(i32, &.{ -1, -1, -1, -1 }, tree.parents);
}
