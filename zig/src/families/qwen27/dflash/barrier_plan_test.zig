//! A recording backend: the concurrent draft plan runs the same ops and never reads a write across a missing barrier.
const std = @import("std");
const ck = @import("../checkpoint.zig");
const cfg = @import("config.zig");
const op = @import("operators.zig");
const execution = @import("execution.zig");
const fixtures = @import("fixtures.zig");

const Kind = enum { project, norm, embed, conv, rope, attention, swiglu, head };
const Plan = struct {
    concurrent: bool,
    handle: u64 = 1,
    fence: u32 = 0, // barriers so far
    written: [4096]u32 = @splat(0), // the fence count a view's writer closed under
    open: [4096]bool = @splat(false), // written, with no barrier since
    pending: [8]u64 = @splat(0),
    pending_count: usize = 0,
    overlap: bool = false,
    counts: [@typeInfo(Kind).@"enum".field_names.len]u32 = @splat(0),
    barriers: u32 = 0,
    hazards: u32 = 0,
    fn self(ptr: *anyopaque) *Plan {
        return @ptrCast(@alignCast(ptr));
    }
    fn read(p: *Plan, v: op.View) void {
        if (p.open[v.handle]) p.hazards += 1;
    }
    fn write(p: *Plan, kind: Kind, rows: u32, width: u32) op.View {
        p.handle += 1;
        p.counts[@intFromEnum(kind)] += 1;
        p.open[p.handle] = true;
        p.pending[p.pending_count] = p.handle;
        p.pending_count += 1;
        const y = op.View{ .handle = p.handle, .rows = rows, .width = width, .stride = width, .dtype = .bf16 };
        if (p.overlap) p.overlap = false else p.barrier();
        return y;
    }
    fn barrier(p: *Plan) void {
        p.barriers += 1;
        for (p.pending[0..p.pending_count]) |h| p.open[h] = false;
        p.pending_count = 0;
    }
    fn begin(ptr: *anyopaque) !u64 {
        self(ptr).overlap = false;
        return 1;
    }
    fn release(_: *anyopaque, _: u64) void {}
    fn project(ptr: *anyopaque, _: u64, x: op.View, t: ck.Tensor, _: op.Mode) !op.View {
        self(ptr).read(x);
        return self(ptr).write(.project, x.rows, @intCast(t.shape[0]));
    }
    fn norm(ptr: *anyopaque, _: u64, x: op.View, _: op.Norm) !op.View {
        self(ptr).read(x);
        return self(ptr).write(.norm, x.rows, x.width);
    }
    fn embed(ptr: *anyopaque, _: u64, _: ck.Linear, ids: []const u32) !op.View {
        return self(ptr).write(.embed, @intCast(ids.len), fixtures.toy.hidden);
    }
    fn conv(ptr: *anyopaque, _: u64, x: op.View, a: op.Conv) !op.View {
        self(ptr).read(x);
        self(ptr).read(a.dynamic);
        if (a.residual) |r| self(ptr).read(r);
        return self(ptr).write(.conv, x.rows, x.width);
    }
    fn rope(ptr: *anyopaque, _: u64, x: op.View, _: op.Rotary) !op.View {
        self(ptr).read(x);
        return self(ptr).write(.rope, x.rows, x.width);
    }
    fn attention(ptr: *anyopaque, _: u64, a: op.Attention) !op.View {
        for ([_]op.View{ a.query, a.keys, a.values }) |v| self(ptr).read(v);
        return self(ptr).write(.attention, a.query.rows, a.config.qWidth());
    }
    fn swiglu(ptr: *anyopaque, _: u64, g: op.View, u: op.View) !op.View {
        self(ptr).read(g);
        self(ptr).read(u);
        return self(ptr).write(.swiglu, g.rows, g.width);
    }
    fn mask(ptr: *anyopaque, _: u64, x: op.View, _: u32, count: u32) !op.View {
        self(ptr).read(x);
        self(ptr).handle += 1;
        return .{ .handle = self(ptr).handle, .rows = count, .width = x.width, .stride = x.width, .dtype = .bf16 };
    }
    fn head(ptr: *anyopaque, _: u64, x: op.View) !op.Head {
        self(ptr).read(x);
        return .{ .logits = self(ptr).write(.head, x.rows, fixtures.toy.vocab) };
    }
    fn lattice(ptr: *anyopaque, _: u64, h: op.Head, projected: op.View, c: cfg.Config) !op.Lattice {
        self(ptr).read(h.logits);
        self(ptr).read(projected);
        const a = std.testing.allocator;
        const ids = try a.alloc(u32, projected.rows * c.topk);
        errdefer a.free(ids);
        const unary = try a.alloc(f32, ids.len);
        errdefer a.free(unary);
        const hp = try a.alloc(f32, projected.rows * c.rank);
        for (ids, 0..) |*id, i| id.* = @intCast(1 + i % c.topk);
        @memset(unary, 0);
        @memset(hp, 0);
        return .{ .gpa = a, .depth = projected.rows, .topk = c.topk, .rank = c.rank, .candidates = ids, .unary = unary, .projected = hp };
    }
    fn lattices(_: *anyopaque, _: u64, _: op.Head, _: op.View, _: cfg.Config, _: u32, _: []op.Lattice) !void {
        return error.NotInThisPlan;
    }
    fn contextBegin(_: *anyopaque, _: op.Context) !u64 {
        return error.NotInThisPlan;
    }
    fn append(_: *anyopaque, _: u64, _: u64, _: u32, _: op.View, _: op.View, _: []const u64) !void {
        return error.NotInThisPlan;
    }
    fn commit(_: *anyopaque, _: u64, _: u64, _: op.Context) !void {
        return error.NotInThisPlan;
    }
    fn abort(_: *anyopaque, _: u64) void {}
    fn streamAppend(_: *anyopaque, _: u64, _: u32, _: u32, _: op.View, _: op.View, _: []const u64) !void {
        return error.NotInThisPlan;
    }
    fn overlaps(ptr: *anyopaque) bool {
        return self(ptr).concurrent;
    }
    fn overlapNext(ptr: *anyopaque, _: u64) !void {
        self(ptr).overlap = self(ptr).concurrent;
    }
    const vtable = op.Ops.VTable{ .begin = begin, .release = release, .project = project, .norm = norm, .embed = embed, .conv = conv, .rope = rope, .attention = attention, .swiglu = swiglu, .mask_rows = mask, .head = head, .lattice = lattice, .lattices = lattices, .context_begin = contextBegin, .context_append = append, .context_commit = commit, .context_stage = commit, .context_abort = abort, .stream_append = streamAppend, .overlaps = overlaps, .overlap_next = overlapNext };
    fn ops(p: *Plan) op.Ops {
        return .{ .ptr = p, .vtable = &vtable };
    }
};

fn run(g: anytype, concurrent: bool) !Plan {
    var plan = Plan{ .concurrent = concurrent };
    var lattice = try execution.forward(plan.ops(), g, .{ .mode = .bf16_reference }, .{ .cache = 1, .begin = 0, .end = 2 }, 1);
    lattice.deinit();
    return plan;
}

test "concurrent draft plan: same ops, only barriers between independent linears removed, no read crosses a write" {
    const a = std.testing.allocator;
    var store = try fixtures.Store.init(a);
    defer store.deinit();
    var g = try store.graph(a);
    defer g.deinit();
    const off = try run(&g, false);
    const on = try run(&g, true);
    try std.testing.expectEqualSlices(u32, &off.counts, &on.counts);
    try std.testing.expectEqual(@as(u32, 0), off.hazards);
    try std.testing.expectEqual(@as(u32, 0), on.hazards);
    // per layer: q and k leave theirs to v, gate to up; the head leaves its own to the hidden projection
    try std.testing.expectEqual(off.barriers - @as(u32, @intCast(g.layers.len * 3 + 1)), on.barriers);
}

test "concurrent draft plan keeps the barrier between linears that share a weight's scratch" {
    const a = std.testing.allocator;
    var store = try fixtures.Store.init(a);
    defer store.deinit();
    var g = try store.graph(a);
    defer g.deinit();
    const separate = try run(&g, true);
    @constCast(g.layers)[0].mlp.up = g.layers[0].mlp.gate;
    const tied = try run(&g, true);
    try std.testing.expectEqual(separate.barriers + 1, tied.barriers);
    try std.testing.expectEqual(@as(u32, 0), tied.hazards);
}
