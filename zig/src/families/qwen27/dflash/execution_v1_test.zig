//! Fake operators prove the DFlash v1 schedule: a plain qwen3 body, two residual adds a layer, no conv, no selector.
const std = @import("std");
const ck = @import("../checkpoint.zig");
const cfg = @import("config.zig");
const op = @import("operators.zig");
const execution = @import("execution.zig");
const fixtures = @import("fixtures.zig");
const session = @import("session.zig");
const selector = @import("selector.zig");
const toy = fixtures.toy_v1;
const Fake = struct {
    handle: u64 = 1,
    projects: u32 = 0,
    adds: u32 = 0,
    attends: u32 = 0,
    appends: u32 = 0,
    commits: u32 = 0,
    ids: [16]u32 = undefined,
    positions: [16]u64 = undefined,
    headed: u32 = 0,
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
    fn project(ptr: *anyopaque, _: u64, x: op.View, t: ck.Tensor, _: op.Mode) !op.View {
        if (t.bytes.len == 0) return error.FakeAbsentWeight;
        self(ptr).projects += 1;
        return self(ptr).view(x.rows, @intCast(t.shape[0]));
    }
    fn norm(ptr: *anyopaque, _: u64, x: op.View, n: op.Norm) !op.View {
        if (x.width != n.heads * n.head_dim) return error.FakeNormShape;
        return self(ptr).view(x.rows, x.width);
    }
    fn embed(ptr: *anyopaque, _: u64, _: ck.Linear, ids: []const u32) !op.View {
        @memcpy(self(ptr).ids[0..ids.len], ids);
        return self(ptr).view(@intCast(ids.len), toy.hidden);
    }
    fn conv(_: *anyopaque, _: u64, _: op.View, _: op.Conv) !op.View {
        return error.FakeV1HasNoConv;
    }
    fn rope(ptr: *anyopaque, _: u64, x: op.View, a: op.Rotary) !op.View {
        if (a.positions.len <= 16) @memcpy(self(ptr).positions[0..a.positions.len], a.positions);
        return self(ptr).view(x.rows, x.width);
    }
    fn attention(ptr: *anyopaque, _: u64, a: op.Attention) !op.View {
        if (a.mask != .all_slots or a.layer != self(ptr).attends or a.query.rows != a.positions.len or a.positions[0] != a.context.end) return error.FakeAttentionOrder;
        self(ptr).attends += 1;
        return self(ptr).view(a.query.rows, a.config.qWidth());
    }
    fn swiglu(ptr: *anyopaque, _: u64, a: op.View, b: op.View) !op.View {
        if (a.width != b.width or a.rows != b.rows) return error.FakeMlpShape;
        return self(ptr).view(a.rows, a.width);
    }
    fn add(ptr: *anyopaque, _: u64, x: op.View, y: op.View) !op.View {
        if (x.rows != y.rows or x.width != y.width or x.width != toy.hidden) return error.FakeAddShape;
        self(ptr).adds += 1;
        return self(ptr).view(x.rows, x.width);
    }
    fn mask(ptr: *anyopaque, _: u64, x: op.View, first: u32, count: u32) !op.View {
        if (first != 1 or count != x.rows - 1) return error.FakeAnchorShift;
        return self(ptr).view(count, x.width);
    }
    fn head(ptr: *anyopaque, _: u64, x: op.View) !op.Head {
        self(ptr).headed = x.rows;
        return .{ .logits = self(ptr).view(x.rows, toy.vocab) };
    }
    fn lattices(_: *anyopaque, _: u64, _: op.Head, _: op.View, _: cfg.Config, _: u32, _: []op.Lattice) !void {
        return error.FakeBatchedHead;
    }
    fn streamAppend(_: *anyopaque, _: u64, _: u32, _: u32, _: op.View, _: op.View, _: []const u64) !void {
        return error.FakeBatchedHead;
    }
    /// A chain: depth d's best candidate is token 10 + d, its runner-up far behind.
    fn lattice(_: *anyopaque, _: u64, h: op.Head, _: op.View, c: cfg.Config) !op.Lattice {
        const a = std.testing.allocator;
        const ids = try a.alloc(u32, h.logits.rows * c.topk);
        errdefer a.free(ids);
        const unary = try a.alloc(f32, ids.len);
        errdefer a.free(unary);
        for (0..h.logits.rows) |d| {
            ids[d * c.topk] = @intCast(10 + d);
            unary[d * c.topk] = 0;
            ids[d * c.topk + 1] = @intCast(40 + d);
            unary[d * c.topk + 1] = -20;
        }
        return .{ .gpa = a, .depth = h.logits.rows, .topk = c.topk, .rank = c.rank, .candidates = ids, .unary = unary, .projected = try a.alloc(f32, 0) };
    }
    fn contextBegin(_: *anyopaque, _: op.Context) !u64 {
        return 1;
    }
    fn append(ptr: *anyopaque, _: u64, _: u64, layer: u32, k: op.View, v: op.View, positions: []const u64) !void {
        const f = self(ptr);
        if (layer != f.appends or k.rows != positions.len or v.rows != k.rows) return error.FakeAppendOrder;
        f.appends += 1;
    }
    fn commit(ptr: *anyopaque, _: u64, _: u64, _: op.Context) !void {
        self(ptr).commits += 1;
    }
    fn stage(_: *anyopaque, _: u64, _: u64, _: op.Context) !void {}
    fn abort(_: *anyopaque, _: u64) void {}
    fn ops(f: *Fake) op.Ops {
        return .{ .ptr = f, .vtable = &.{ .begin = begin, .release = release, .project = project, .norm = norm, .embed = embed, .conv = conv, .rope = rope, .attention = attention, .swiglu = swiglu, .mask_rows = mask, .head = head, .lattice = lattice, .lattices = lattices, .context_begin = contextBegin, .context_append = append, .context_commit = commit, .context_stage = stage, .context_abort = abort, .stream_append = streamAppend, .add = add } };
    }
};
fn taps() session.Batch {
    return .{ .values = .{ .handle = 99, .rows = 2, .width = toy.tapWidth(), .stride = toy.tapWidth(), .dtype = .bf16 }, .positions = &.{ 0, 1 }, .layers = toy.taps, .target_end = 2 };
}

test "a v1 block runs attention and an MLP a layer, two residual adds each, and heads its fifteen mask rows" {
    const a = std.testing.allocator;
    var store = try fixtures.Store.initConfig(a, toy);
    defer store.deinit();
    var g = try store.graph(a);
    defer g.deinit();
    var fake = Fake{};
    var state = try session.Session.init(toy, 1, 0);
    const p = op.Preparation{ .mode = .bf16_reference };
    try state.absorb(fake.ops(), &g, p, taps());
    try std.testing.expectEqual(@as(u32, 3), fake.appends);
    // the context append: fc, then k and v a layer
    try std.testing.expectEqual(@as(u32, 1 + 2 * 3), fake.projects);
    fake.projects = 0;
    var tree = try state.treeBlock(a, fake.ops(), &g, p, 5, try state.firstDraw(), 16, .{ .nodes = 15 });
    defer tree.deinit();
    try std.testing.expectEqualSlices(u32, &.{ 5, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62 }, &fake.ids);
    try std.testing.expectEqualSlices(u64, &.{ 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 }, &fake.positions);
    try std.testing.expectEqual(@as(u32, 3), fake.attends);
    try std.testing.expectEqual(@as(u32, 6), fake.adds);
    // q, k, v, o, gate, up and down a layer, and no selector projection
    try std.testing.expectEqual(@as(u32, 7 * 3), fake.projects);
    try std.testing.expectEqual(@as(u32, 15), fake.headed);
    // with no selector, a confident lattice is a chain of each depth's best
    try std.testing.expectEqual(@as(usize, 15), tree.tokens.len);
    for (tree.tokens, tree.parents, 0..) |token, parent, d| {
        try std.testing.expectEqual(@as(u32, @intCast(10 + d)), token);
        try std.testing.expectEqual(@as(i32, @intCast(d)) - 1, parent);
    }
}
test "a v1 graph holds no conv or selector tensor" {
    const a = std.testing.allocator;
    var store = try fixtures.Store.initConfig(a, toy);
    defer store.deinit();
    var g = try store.graph(a);
    defer g.deinit();
    try std.testing.expectEqual(@as(usize, 11 * 3 + 3), store.keys.len);
    try std.testing.expectEqual(@as(usize, 0), g.hidden_projection.bytes.len);
    try std.testing.expectEqual(@as(usize, 0), g.predecessor.bytes.len);
    for (g.layers) |l| try std.testing.expectEqual(@as(usize, 0), l.attention_conv.projection.bytes.len + l.mlp_conv.base.bytes.len);
    try std.testing.expect(try g.nativeBytes() > 0);
}
test "a v1 block without an add operator is refused, not run without its residuals" {
    const a = std.testing.allocator;
    var store = try fixtures.Store.initConfig(a, toy);
    defer store.deinit();
    var g = try store.graph(a);
    defer g.deinit();
    var fake = Fake{};
    var vtable = fake.ops().vtable.*;
    vtable.add = null;
    const ops = op.Ops{ .ptr = &fake, .vtable = &vtable };
    try std.testing.expectError(error.DraftOperatorShape, execution.forwardBlock(ops, &g, .{ .mode = .bf16_reference }, .{ .cache = 1, .begin = 0, .end = 2 }, 5, 16));
}
